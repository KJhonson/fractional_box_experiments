# loads.py
"""
Right-hand side (RHS) assembly functions using dual FVM (Finite Volume Method).
Vertex-centered barycentric dual control volume approach for RHS assembly.
"""
import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import petsc
# ============================================================================
# Quadrature rules
# ============================================================================

def tri_rule(deg: int):
    """Simple triangle quadrature rules (reference triangle)"""
    if deg <= 1:
        # 1-point barycenter (exact for linear)
        lambdas = np.array([[1/3, 1/3, 1/3]])
        weights = np.array([1.0])
    elif deg == 2:
        # 3-point (exact for quadratics)
        a, b = 2/3, 1/6
        lambdas = np.array([[a, b, b], [b, a, b], [b, b, a]])
        weights = np.array([1/3, 1/3, 1/3])
    else:
        # 4-point degree-3 rule
        lambdas = np.array([[1/3, 1/3, 1/3],
                            [0.6, 0.2, 0.2],
                            [0.2, 0.6, 0.2],
                            [0.2, 0.2, 0.6]])
        weights = np.array([-27/48, 25/48, 25/48, 25/48])
    return lambdas, weights


def gauss_legendre(n):
    """Gauss-Legendre quadrature rule on [-1,1]"""
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x, w
# ============================================================================
# Geometry helpers
# ============================================================================
def tri_area(A, B, C):
    """Compute area of triangle with vertices A, B, C"""
    return 0.5 * abs(np.cross(B - A, C - A))

def midpoint(A, B):
    """Compute midpoint between points A and B"""
    return 0.5 * (A + B)

def barycenter_triangle(P):
    """Compute barycenter of triangle with vertices P (shape: (3,2))"""
    return np.mean(P, axis=0)

# ============================================================================
# Core FVM assembly
# ============================================================================
def assemble_dual_fvm_rhs(V: fem.FunctionSpace, f_expr, quad_degree: int = 3) -> PETSc.Vec:
    """
    Vertex-centered barycentric dual load:
      F_i = ∫_{b_i} f dx  ≈  sum over cells K incident to i of
            ( ∫_{Δ(v_i, m_{i,i+1}, c_K)} f + ∫_{Δ(v_i, c_K, m_{i,i-1})} f )

    - Works for 1D segments and 2D triangles (P1).
    - quad_degree controls accuracy: (1,2,3,...) for triangles; for 1D we use max(2,quad_degree).
    """
    domain = V.mesh
    tdim = domain.topology.dim
    gdim = domain.geometry.dim
    assert tdim in (1, 2), "This routine currently supports 1D segments and 2D triangles."

    # P1 vertex dofs
    Vdm = V.dofmap
    owned = Vdm.index_map.size_local
    ghosts = Vdm.index_map.num_ghosts

    X = domain.geometry.x  # vertex coordinates (local+ghost)
    vert_ids = np.arange(X.shape[0], dtype=np.int32)
    
    # Create necessary connectivity
    topo = domain.topology
    topo.create_connectivity(0, 1)  # vertices -> edges
    topo.create_connectivity(1, 0)  # edges -> vertices
    
    v2d = fem.locate_dofs_topological(V, 0, vert_ids)

    # Create a function to evaluate f_expr at given points
    def evaluate_f(pts):
        """Evaluate f_expr at given points"""
        try:
            # Try to evaluate as UFL expression
            if hasattr(f_expr, 'ufl_domains'):
                # It's a UFL expression
                f_func = fem.Function(V)
                f_func.interpolate(f_expr)
                return f_func.eval(pts, np.zeros(pts.shape[1], dtype=np.int32))
            else:
                # It's a Python function
                return np.array([f_expr(pt) for pt in pts.T])
        except:
            # Fallback: assume it's a Python function
            return np.array([f_expr(pt) for pt in pts.T])

    vload_local = np.zeros(X.shape[0], dtype=np.float64)
    if tdim == 1:
        # ---- 1D: each cell K = (v0, v1); b_vi ∩ K is segment [vi, midpoint]
        topo.create_connectivity(1, 0)           # cells -> vertices
        c2v = topo.connectivity(1, 0)
        num_cells = c2v.num_nodes
        cell_vertices = c2v.array.reshape(num_cells, -1)  # (ncells, 2)

        # N-point Gauss–Legendre mapped to [0,1], then to [vi, mid]
        n1d = max(2, quad_degree)
        xi, wi = gauss_legendre(n1d)
        # map [-1,1] -> [0,1]
        s = 0.5 * (xi + 1.0)
        ws = 0.5 * wi

        for c in range(num_cells):
            vi, vj = cell_vertices[c]
            P0, P1 = X[vi, :], X[vj, :]
            mid = midpoint(P0, P1)

            # segments for vi and vj
            for v_idx, A, B in ((vi, P0, mid), (vj, P1, mid)):
                L = np.linalg.norm(B - A)
                if L == 0.0:
                    continue
                # quadrature points along segment
                pts = (A[None, :] + s[:, None] * (B - A)[None, :]).T  # (gdim, npts)
                cells = np.full(s.shape[0], c, dtype=np.int32)
                # evaluate and accumulate
                vals = evaluate_f(pts)
                vload_local[v_idx] += L * np.dot(ws, vals)

    else:
        # ---- 2D: triangles; Q_{vi}^K = quad (vi, m(i,i+1), cK, m(i,i-1)) split into two triangles
        topo.create_connectivity(2, 0)           # cells -> vertices
        c2v = topo.connectivity(2, 0)
        num_cells = c2v.num_nodes
        cell_vertices = c2v.array.reshape(num_cells, -1)  # (ncells, 3)

        lambdas, w = tri_rule(quad_degree)
        nq = lambdas.shape[0]

        for c in range(num_cells):
            verts = cell_vertices[c]       # [i0,i1,i2]
            P = X[verts, :]                # (3,2)
            C = barycenter_triangle(P)
            M01 = midpoint(P[0], P[1])
            M12 = midpoint(P[1], P[2])
            M20 = midpoint(P[2], P[0])

            # For each vertex, build the quad and split along (vertex, C):
            # v0: triangles (v0, M01, C) and (v0, C, M20)
            # v1: triangles (v1, M12, C) and (v1, C, M01)
            # v2: triangles (v2, M20, C) and (v2, C, M12)
            sub = {
                verts[0]: [(P[0], M01, C), (P[0], C, M20)],
                verts[1]: [(P[1], M12, C), (P[1], C, M01)],
                verts[2]: [(P[2], M20, C), (P[2], C, M12)],
            }

            for v_idx, tris in sub.items():
                for (A, B, D) in tris:
                    area = tri_area(A, B, D)
                    if area <= 0.0:
                        continue
                    # physical quadrature points via barycentric rule on triangle (A,B,D)
                    phys_q = lambdas @ np.vstack([A, B, D])            # (nq,2)
                    pts = phys_q.T.reshape(gdim, nq)                 # (2,nq)
                    cells = np.full(nq, c, dtype=np.int32)
                    # evaluate f and accumulate
                    vals = evaluate_f(pts)
                    vload_local[v_idx] += area * np.dot(w, vals)

    # Map vertex-accumulated loads to P1 dofs (local+ghost), then insert owned part into PETSc Vec
    dof_load_local = np.zeros(owned + ghosts, dtype=np.float64)
    for local_vert, dof in enumerate(v2d):
        dof_load_local[dof] += vload_local[local_vert]

    # Create PETSc vector using DOLFINx approach
    # Create a dummy form to get the proper vector structure
    v = ufl.TestFunction(V)
    dummy_form = v * ufl.dx
    b = petsc.assemble_vector(fem.form(dummy_form))
    b.setValues(np.arange(owned, dtype=np.int32), dof_load_local[:owned])
    b.assemblyBegin()
    b.assemblyEnd()
    return b


# ============================================================================
# Dual FVM RHS assembly functions
# ============================================================================

def assemble_rhs_dual_fvm(V, f_expr, quad_degree=3, bc=None, combined_form_for_lifting=None):
    """
    Assemble RHS using dual FVM (Finite Volume Method) approach with optional Dirichlet lifting.

    Parameters
    ----------
    V : fem.FunctionSpace
        The function space
    f_expr : callable
        Source function f(x) -> scalar
    quad_degree : int, optional
        Quadrature degree for integration (default: 3)
    bc : fem.DirichletBC or None
        Optional Dirichlet boundary condition to enforce on the load via lifting
    combined_form_for_lifting : UFL bilinear form
        Combined bilinear form (e.g., a + κ m) required if bc is not None

    Returns
    -------
    PETSc.Vec
        The assembled RHS vector using dual FVM approach
    """
    b = assemble_dual_fvm_rhs(V, f_expr, quad_degree)

    if bc is not None:
        if combined_form_for_lifting is None:
            raise ValueError("Provide combined_form_for_lifting when using Dirichlet bc with FVM RHS.")
        petsc.apply_lifting(b, [fem.form(combined_form_for_lifting)], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

    return b


def assemble_rhs_dual_fvm_exact(V, alpha: float, bc=None, combined_form_for_lifting=None):
    """
    Assemble RHS using exact integration of f(x)=x^{-alpha} over 1D dual control volumes (vertex-centered).

    Notes
    -----
    - Assumes 1D domain [0,1] and P1 space; integrates each vertex's dual cell exactly.
    - If `bc` is provided (Dirichlet), lifting is applied using `combined_form_for_lifting`.
    
    Parameters
    ----------
    V : fem.FunctionSpace
        The function space (1D P1)
    alpha : float
        Exponent for source function f(x) = x^{-alpha}
    bc : fem.DirichletBC or None
        Optional Dirichlet boundary condition
    combined_form_for_lifting : UFL bilinear form
        Combined bilinear form (e.g., a + κ m) required if bc is not None
    """
    domain = V.mesh
    tdim = domain.topology.dim
    if tdim != 1:
        raise NotImplementedError("assemble_rhs_dual_fvm_exact currently supports only 1D domains.")

    # Geometry data
    X = domain.geometry.x[:, 0]
    n_vertices = X.shape[0]

    # Sort vertices by coordinate to define dual intervals correctly
    sorted_pairs = sorted([(i, float(x)) for i, x in enumerate(X)], key=lambda p: p[1])
    sorted_indices = [i for i, _ in sorted_pairs]
    xs = np.array([x for _, x in sorted_pairs], dtype=float)

    # Dual cell boundaries per sorted vertex
    vals_per_vertex = np.zeros(n_vertices, dtype=float)
    one_minus_alpha = 1.0 - float(alpha)

    def exact_int(xl: float, xr: float) -> float:
        xl_clip = max(0.0, xl)
        xr_clip = max(0.0, xr)
        if xr_clip <= xl_clip:
            return 0.0
        if xl_clip > 1e-15:
            return (xr_clip**one_minus_alpha - xl_clip**one_minus_alpha) / one_minus_alpha
        return (xr_clip**one_minus_alpha) / one_minus_alpha

    n = n_vertices
    for k in range(n):
        xi = xs[k]
        if k == 0:
            xl = 0.0
        else:
            xl = 0.5 * (xs[k-1] + xs[k])
        if k == n - 1:
            xr = 1.0
        else:
            xr = 0.5 * (xs[k] + xs[k+1])
        integ = exact_int(xl, xr)
        orig_idx = sorted_indices[k]
        vals_per_vertex[orig_idx] = integ

    # Map vertex values to dofs
    v2d = fem.locate_dofs_topological(V, 0, np.arange(n_vertices, dtype=np.int32))
    Vdm = V.dofmap
    owned = Vdm.index_map.size_local
    ghosts = Vdm.index_map.num_ghosts
    dof_load_local = np.zeros(owned + ghosts, dtype=float)
    for local_vert, dof in enumerate(v2d):
        dof_load_local[dof] += vals_per_vertex[local_vert]

    # Create PETSc vector consistent with V using a dummy form
    v = ufl.TestFunction(V)
    dummy_form = v * ufl.dx
    b = petsc.assemble_vector(fem.form(dummy_form))
    b.setValues(np.arange(owned, dtype=np.int32), dof_load_local[:owned])
    b.assemblyBegin()
    b.assemblyEnd()

    # Optional lifting for Dirichlet
    if bc is not None:
        if combined_form_for_lifting is None:
            raise ValueError("Provide combined_form_for_lifting when using Dirichlet bc with exact FVM RHS.")
        petsc.apply_lifting(b, [fem.form(combined_form_for_lifting)], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

    return b


def assemble_rhs(V, f_expr, bc=None, combined_form_for_lifting=None):
    """
    Assemble standard FEM RHS vector.
    
    Parameters
    ----------
    V : fem.FunctionSpace
        The function space
    f_expr : callable or UFL expression
        Source function. Can be:
        - A UFL expression (e.g., ufl.conditional(...))
        - A Python function that takes SpatialCoordinate and returns a UFL expression
        - A Python function that takes numpy array and returns scalar values
    bc : fem.DirichletBC or None
        Optional Dirichlet boundary condition
    combined_form_for_lifting : UFL bilinear form
        Combined bilinear form (e.g., a + κ m) required if bc is not None
        
    Returns
    -------
    PETSc.Vec
        The assembled RHS vector using standard FEM approach
    """
    v = ufl.TestFunction(V)
    mesh = V.mesh
    
    # Check if f_expr is a UFL expression or a Python function
    # Try to use it as UFL expression first
    try:
        # If it's already a UFL expression, use it directly
        if hasattr(f_expr, 'ufl_domains') or isinstance(f_expr, ufl.core.expr.Expr):
            L = f_expr * v * ufl.dx
        else:
            # It's a callable - try calling it with SpatialCoordinate
            x = ufl.SpatialCoordinate(mesh)
            f_ufl = f_expr(x)
            # Check if the result is a UFL expression
            if hasattr(f_ufl, 'ufl_domains') or isinstance(f_ufl, ufl.core.expr.Expr):
                L = f_ufl * v * ufl.dx
            else:
                raise TypeError("f_expr must return a UFL expression when called with SpatialCoordinate")
        
        # Try to create the form to verify it's valid UFL
        _ = fem.form(L)
    except (TypeError, AttributeError, ValueError):
        # If that fails, it's likely a Python function that takes numpy arrays
        # Interpolate it first
        f_func = fem.Function(V)
        f_func.interpolate(f_expr)
        L = f_func * v * ufl.dx
    
    b = petsc.assemble_vector(fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    
    if bc is not None:
        if combined_form_for_lifting is None:
            raise ValueError("Provide combined_form_for_lifting when using Dirichlet bc with standard FEM RHS.")
        petsc.apply_lifting(b, [fem.form(combined_form_for_lifting)], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
    
    return b
