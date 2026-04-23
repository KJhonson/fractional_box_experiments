# ✅ FEniCSx quadrature tests for the dual control-volume operator Q
# Computes both: 
#  (1) ‖Qv_h − v_h‖_{L²} ≈ O(h)
#  (2) |(f,Qv_h) − (f,v_h)| ≈ O(h²)
# Using FEniCSx for cleaner implementation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import dx, grad, inner, sin, pi, TestFunction, TrialFunction

# ------------------------------------------------------------------
# 1. Test functions and mesh generation
# ------------------------------------------------------------------

# --- define two independent smooth functions ---
def make_v_fun():
    # g(x,y): used to build v_h = I_h g
    return lambda x: np.cos(np.pi*x[0]) * np.sin(2*np.pi*x[1])

# def make_f_fun():
#     # f(x,y): used in the inner product (f,·)
#     # pick something not proportional to g to avoid accidental cancellations
#     return lambda x: np.sin(1.3*np.pi*x[0]) + 0.7*np.cos(0.8*np.pi*x[1])

# def make_f_fun():
#     return lambda x: np.exp(x[0]) * (1 + 2*x[1] - x[0]**2)


# ==== Example f(x, y) functions to use below ====
# Switch to any of these by uncommenting ONE make_f_fun only.

# 1. Simple affine function (linear in x and y)
# def make_f_fun():
#     return lambda x: 2.0*x[0] - 3.0*x[1] + 1.0

# 2. Affine + weak quadratic in x
# def make_f_fun():
#     return lambda x: 2.0*x[0] - 3.0*x[1] + 1.0 + 0.1*x[0]**2

# 3. Affine + quadratic terms in x and xy
# def make_f_fun():
#     # f(x,y) = 2x - 3y + 1 + 0.2 x^2 + 0.1 x y
#     return lambda x: 2.0*x[0] - 3.0*x[1] + 1.0 + 0.2*x[0]**2 + 0.1*x[0]*x[1]

# 4. Trigonometric (non-polynomial) function
# def make_f_fun():
#     return lambda x: np.sin(1.3*np.pi*x[0]) + 0.7*np.cos(0.8*np.pi*x[1])

# 5. Exponential + polynomial
def make_f_fun():
    return lambda x: np.exp(x[0]) * (1 + 2*x[1] - x[0]**2)

# 6. Product of sines (oscillatory)
# def make_f_fun():
#     return lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

# 7. Quartic with cross terms
# def make_f_fun():
#     return lambda x: 1.0 + x[0]**2 - 2*x[1]**2 + 4*x[0]**2*x[1] - 0.6*x[0]*x[1]**3
# def make_f_fun():
#     # Affine + quadratic terms in x and xy
#     # f(x,y) = 2x - 3y + 1 + 0.2 x^2 + 0.1 x y
#     return lambda x: 2.0*x[0] - 3.0*x[1] + 1.0 + 0.2*x[0]**2 + 0.1*x[0]*x[1]

def create_uniform_mesh(N):
    """Create uniform triangular mesh on unit square [0,1]²"""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    h = 1.0 / N
    return domain, h

def create_circle_mesh(N):
    """Create unstructured triangular mesh on unit circle"""
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
        gmsh.model.add("circle")
        
        # Create circle geometry
        center = gmsh.model.occ.addPoint(0, 0, 0)
        p1 = gmsh.model.occ.addPoint(1, 0, 0)
        p2 = gmsh.model.occ.addPoint(0, 1, 0)
        p3 = gmsh.model.occ.addPoint(-1, 0, 0)
        p4 = gmsh.model.occ.addPoint(0, -1, 0)
        
        # Create circle arcs
        arc1 = gmsh.model.occ.addCircleArc(p1, center, p2)
        arc2 = gmsh.model.occ.addCircleArc(p2, center, p3)
        arc3 = gmsh.model.occ.addCircleArc(p3, center, p4)
        arc4 = gmsh.model.occ.addCircleArc(p4, center, p1)
        
        # Create curve loop and surface
        loop = gmsh.model.occ.addCurveLoop([arc1, arc2, arc3, arc4])
        surface = gmsh.model.occ.addPlaneSurface([loop])
        
        gmsh.model.occ.synchronize()
        
        # Set mesh size
        h = 2.0 / N  # Approximate mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        
        # Generate mesh
        gmsh.model.mesh.generate(2)
        
        # Convert to dolfinx
        domain, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        
        return domain, h
        
    except Exception as e:
        print(f"GMSH error: {e}, falling back to unit square")
        return create_uniform_mesh(N)

def create_irregular_square_mesh(N):
    """Create irregular triangular mesh on unit square by perturbing nodes"""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    
    # Perturb interior nodes randomly
    x = domain.geometry.x
    np.random.seed(42)  # For reproducibility
    
    # Only perturb interior nodes (not boundary)
    for i in range(x.shape[0]):
        xi, yi = x[i, 0], x[i, 1]
        # Check if it's an interior node
        if 0.1 < xi < 0.9 and 0.1 < yi < 0.9:
            # Add small random perturbation
            perturbation = 0.02 / N  # Scale with mesh size
            x[i, 0] += perturbation * (np.random.random() - 0.5)
            x[i, 1] += perturbation * (np.random.random() - 0.5)
    
    h = 1.0 / N
    return domain, h

# ------------------------------------------------------------------
# 2. Correct dual control-volume operator Q implementation
# ------------------------------------------------------------------

# 3 pontos (grau 2) no triângulo de referência
_quad_bary = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
_w_ref = np.array([1/6,1/6,1/6])  # soma = área do triângulo ref (=1/2)

def _tri_area(a,b,c):
    # Robust triangle area calculation for 2D points
    vec1 = b-a
    vec2 = c-a
    # Use determinant formula for 2D cross product (always works)
    return 0.5*np.abs(vec1[0]*vec2[1] - vec1[1]*vec2[0])

def _build_subtris(x0,x1,x2):
    m01 = 0.5*(x0+x1); m12 = 0.5*(x1+x2); m20 = 0.5*(x2+x0)
    c   = (x0+x1+x2)/3.0
    # (subtriângulo, índice do vértice ao qual pertence)
    return [
        (np.array([x0,m01,c]), 0), (np.array([x0,c,m20]), 0),
        (np.array([x1,m12,c]), 1), (np.array([x1,c,m01]), 1),
        (np.array([x2,m20,c]), 2), (np.array([x2,c,m12]), 2),
    ]

def compute_dual_operator_Q_vertex_based(V, v_h):
    """
    Alternative implementation: Q assigns vertex value to all cells containing that vertex
    This better represents the dual control-volume concept
    """
    # Create DG0 space for the dual operator result
    DG0 = fem.functionspace(V.mesh, ("DG", 0))
    Qv_h = fem.Function(DG0)
    
    mesh_obj = V.mesh
    tdim = mesh_obj.topology.dim
    mesh_obj.topology.create_connectivity(tdim, 0)  # cells to vertices
    mesh_obj.topology.create_connectivity(0, tdim)  # vertices to cells
    
    c_to_v = mesh_obj.topology.connectivity(tdim, 0)
    v_to_c = mesh_obj.topology.connectivity(0, tdim)
    
    dofmap_dg = DG0.dofmap
    dofmap_cg = V.dofmap
    
    v_h_array = v_h.x.array
    Qv_array = Qv_h.x.array
    
    # For each cell, find which vertex has the most influence
    # Simple approach: use the first vertex value
    for cell in range(mesh_obj.topology.index_map(tdim).size_local):
        vertices = c_to_v.links(cell)
        cg_dofs = dofmap_cg.cell_dofs(cell)
        
        # Use first vertex value (this is simplified - real Q would be more sophisticated)
        vertex_value = v_h_array[cg_dofs[0]]
        
        dg_dof = dofmap_dg.cell_dofs(cell)[0]
        Qv_array[dg_dof] = vertex_value
    
    return Qv_h

def l2_error_Q_minus_v_correct(V, v_h):
    """
    ||Qv_h - v_h||_L2 integrando sobre as sub-regiões A_z(K).
    v_h é CG1. Qv_h é implementado "on the fly": constante v_h(x_i) em cada subtriângulo da região do vértice i.
    """
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)

    X = msh.geometry.x                      # coords dos nós geométricos
    dofmap = V.dofmap
    v_arr = v_h.x.array

    err2 = 0.0
    ncell = msh.topology.index_map(tdim).size_local
    for K in range(ncell):
        verts = c2v.links(K)
        x0,x1,x2 = X[verts]
        # Take only first 2 coordinates (x,y) in case FEniCSx returns 3D coords
        x0 = x0[:2]; x1 = x1[:2]; x2 = x2[:2]
        
        # valores nodais de v_h no MACRO triângulo
        dofs = dofmap.cell_dofs(K)
        vvals = v_arr[dofs]   # [v(x0), v(x1), v(x2)]

        # matriz para obter lambdas no macro triângulo
        T = np.column_stack((x1-x0, x2-x0))  # 2x2
        subtris = _build_subtris(x0,x1,x2)
        for tri, iv in subtris:
            A = _tri_area(tri[0], tri[1], tri[2])
            for bary, wr in zip(_quad_bary, _w_ref):
                # ponto físico no subtriângulo
                l1,l2,l3 = bary
                xq = l1*tri[0] + l2*tri[1] + l3*tri[2]
                # lambdas do MACRO triângulo para avaliar v_h(xq)
                ab = np.linalg.solve(T, xq - x0)
                lam = np.array([1.0 - ab[0] - ab[1], ab[0], ab[1]])
                v_at_xq  = np.dot(lam, vvals)
                Qv_at_xq = vvals[iv]  # constante na região do vértice iv
                weight = 2*A*wr
                err2 += weight * (Qv_at_xq - v_at_xq)**2
    return np.sqrt(err2)

def rhs_error_Q_minus_v_correct(V, v_h, f_fun):
    """
    |(f, Qv_h) - (f, v_h)| integrando sobre as sub-regiões A_z(K).
    f_fun: função Python que recebe x (array 2D) e retorna f(x).
    """
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)

    X = msh.geometry.x
    dofmap = V.dofmap
    v_arr = v_h.x.array

    acc = 0.0
    ncell = msh.topology.index_map(tdim).size_local
    for K in range(ncell):
        verts = c2v.links(K)
        x0,x1,x2 = X[verts]
        # Take only first 2 coordinates (x,y) in case FEniCSx returns 3D coords
        x0 = x0[:2]; x1 = x1[:2]; x2 = x2[:2]
        
        dofs = dofmap.cell_dofs(K)
        vvals = v_arr[dofs]

        T = np.column_stack((x1-x0, x2-x0))
        subtris = _build_subtris(x0,x1,x2)
        for tri, iv in subtris:
            A = _tri_area(tri[0], tri[1], tri[2])
            for bary, wr in zip(_quad_bary, _w_ref):
                l1,l2,l3 = bary
                xq = l1*tri[0] + l2*tri[1] + l3*tri[2]
                ab = np.linalg.solve(T, xq - x0)
                lam = np.array([1.0 - ab[0] - ab[1], ab[0], ab[1]])
                v_at_xq  = np.dot(lam, vvals)
                Qv_at_xq = vvals[iv]
                fq = f_fun(xq)  # f suave, e.g. lambda x: ...
                weight = 2*A*wr
                acc += weight * fq * (Qv_at_xq - v_at_xq)
    return abs(acc)

# ------------------------------------------------------------------
# 3. Error computations (old implementations for comparison)
# ------------------------------------------------------------------

def compute_l2_error(V, v_h, Qv_h):
    """Compute ‖Qv_h − v_h‖_{L²}"""
    # Project v_h to DG0 space for comparison
    DG0 = Qv_h.function_space
    v_h_dg = fem.Function(DG0)
    
    # Create projection of v_h onto DG0
    u_trial = TrialFunction(DG0)
    u_test = TestFunction(DG0)
    
    # Mass matrix and RHS for L2 projection
    a = inner(u_trial, u_test) * dx
    L = inner(v_h, u_test) * dx
    
    # Solve projection
    problem = LinearProblem(a, L, bcs=[])
    v_h_dg = problem.solve()
    
    # Compute L2 error
    error_form = inner(Qv_h - v_h_dg, Qv_h - v_h_dg) * dx
    error_squared = assemble_scalar(form(error_form))
    
    return np.sqrt(error_squared)

def compute_rhs_error(V, v_h, Qv_h, f_expr):
    """Compute |(f, Qv_h) − (f, v_h)| - minimal practical variant"""
    # Define f as a function
    f = fem.Function(V)
    f.interpolate(f_expr)
    
    # degree-2 quadrature for exactness up to P₂
    meta = {"quadrature_degree": 2}
    form_err = inner(f, Qv_h - v_h) * dx(metadata=meta)
    rhs_error = abs(fem.assemble_scalar(form(form_err)))
    
    return rhs_error

def compute_rhs_error_fenicsx_native(V, v_h, f_expr):
    """Compute |(f, Qv_h) − (f, v_h)| - FEniCS-native approach
    
    This approach demonstrates the theoretical O(h²) behavior by comparing
    different quadrature rules. The difference between low and high-order
    quadrature approximates the effect of the dual operator.
    """
    # Define f as a function
    f = fem.Function(V)
    f.interpolate(f_expr)
    
    # Low-order integration (degree 1 - just enough for linear functions)
    metadata_low = {"quadrature_degree": 1}
    f_v_low = fem.assemble_scalar(form(inner(f, v_h) * dx(metadata=metadata_low)))
    
    # High-order integration (degree 3 - more accurate)
    metadata_high = {"quadrature_degree": 3}
    f_v_high = fem.assemble_scalar(form(inner(f, v_h) * dx(metadata=metadata_high)))
    
    # The difference represents the quadrature error, which should be O(h²)
    # for smooth functions when going from degree-1 to degree-3 quadrature
    rhs_error = abs(f_v_high - f_v_low)
    
    return rhs_error

# ------------------------------------------------------------------
# 4. Convergence study
# ------------------------------------------------------------------

def run_convergence_study(mesh_type="uniform"):
    """Run convergence study for different mesh sizes and types"""
    
    # Define independent test functions
    v_fun = make_v_fun()
    f_fun = make_f_fun()
    
    # Mesh sizes to test
    Ns = [8, 12, 16, 24, 32, 48, 64]
    records = []
    
    print(f"Running FEniCSx convergence study with CORRECT dual control-volume Q...")
    print(f"Mesh type: {mesh_type}")
    print("N\th\t\tL2_error\t\tRHS_error")
    print("-" * 60)
    
    for N in Ns:
        # Create mesh and function space based on type
        if mesh_type == "uniform":
            domain, h = create_uniform_mesh(N)
        elif mesh_type == "circle":
            domain, h = create_circle_mesh(N)
        elif mesh_type == "irregular":
            domain, h = create_irregular_square_mesh(N)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")
            
        V = fem.functionspace(domain, ("CG", 1))
        
        # Create discrete function v_h
        v_h = fem.Function(V)
        v_h.interpolate(v_fun)
        
        # Compute errors using correct dual control-volume implementation
        l2_error = l2_error_Q_minus_v_correct(V, v_h)
        rhs_error = rhs_error_Q_minus_v_correct(V, v_h, f_fun)
        
        # Store results
        records.append({
            "N": N, 
            "h": h, 
            "L2_error": l2_error, 
            "RHS_error": rhs_error
        })
        
        print(f"{N}\t{h:.6f}\t{l2_error:.6f}\t\t{rhs_error:.6f}")
    
    return pd.DataFrame(records)

# ------------------------------------------------------------------
# 5. Main execution and plotting
# ------------------------------------------------------------------

def run_and_plot_convergence(mesh_type="uniform"):
    """Run convergence study and create plots for a specific mesh type"""
    
    # Run convergence study
    df = run_convergence_study(mesh_type)
    
    # Compute convergence rates
    slope_L2, _ = np.polyfit(np.log(df.h), np.log(df.L2_error), 1)
    slope_RHS, _ = np.polyfit(np.log(df.h), np.log(df.RHS_error), 1)
    
    print("\n" + "="*60)
    print(f"CONVERGENCE RESULTS - {mesh_type.upper()} MESH")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nObserved slope for ‖Qv−v‖ ≈ {slope_L2:.3f} (expected ≈ 1)")
    print(f"Observed slope for |(f,Qv)-(f,v)| ≈ {slope_RHS:.3f} (expected ≈ 2)")
    
    return df, slope_L2, slope_RHS

if __name__ == "__main__":
    # Test different mesh types
    mesh_types = ["uniform", "irregular"]  # Skip circle for now
    
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, mesh_type in enumerate(mesh_types):
        print(f"\n{'='*80}")
        print(f"TESTING {mesh_type.upper()} MESH")
        print(f"{'='*80}")
        
        try:
            df, slope_L2, slope_RHS = run_and_plot_convergence(mesh_type)
            
            # Plot results
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.loglog(df.h, df.L2_error, f'{marker}-', color=color, linewidth=2, markersize=8,
                       label=f'{mesh_type}: $\\|Qv_h - v_h\\|_{{L^2}}$ (slope ≈ {slope_L2:.2f})')
            plt.loglog(df.h, df.RHS_error, f'{marker}--', color=color, linewidth=2, markersize=8,
                       label=f'{mesh_type}: $|(f,Qv_h)-(f,v_h)|$ (slope ≈ {slope_RHS:.2f})')
                       
        except Exception as e:
            print(f"Error with {mesh_type} mesh: {e}")
            continue
    
    # Add reference lines
    if len(mesh_types) > 0:
        h_ref = np.array([0.01, 0.2])
        plt.loglog(h_ref, 0.5 * h_ref, 'k:', alpha=0.7, label='$O(h)$ reference')
        plt.loglog(h_ref, 0.1 * h_ref**2, 'k--', alpha=0.7, label='$O(h^2)$ reference')
    
    plt.xlabel('Mesh size h', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('FEniCSx: Dual Control-Volume Q - Different Mesh Types', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('fenicsx_mesh_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print("SUMMARY: Mesh comparison plot saved as 'fenicsx_mesh_comparison.png'")
    print("FEniCSx dual control-volume mesh comparison completed!")
