import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh as dmesh

# ---------- helpers ----------

def _all_boundary_facets(mesh):
    """Return the indices of all exterior facets (works in 1D/2D/3D)."""
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    exterior = fem.locate_entities_boundary(mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    return exterior.astype(np.int32)

def _hom_dirichlet_bc(V, facets):
    """Homogeneous Dirichlet bc (u=0) on the given facets."""
    fdim = V.mesh.topology.dim - 1
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    uD = fem.Function(V)  # zero
    return fem.dirichletbc(uD, dofs), dofs

def _assemble_K_M(V, quadrature_degree=None):
    """Assemble stiffness K and consistent mass M (no BCs)."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.dx(metadata={"quadrature_degree": quadrature_degree} if quadrature_degree else {})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    m = ufl.inner(u, v) * dx
    K = fem.petsc.assemble_matrix(fem.form(a)); K.assemble()
    M = fem.petsc.assemble_matrix(fem.form(m)); M.assemble()
    return K, M, a, m

def _lumped_from_rowsum(M):
    """Return lumped mass matrix (diagonal) and its diagonal as Vec."""
    rowsum = M.getRowSum()                     # PETSc.Vec with row sums
    m, n = M.getSize()
    Ml = PETSc.Mat().createAIJ([m, n], nnz=1)  # diagonal-only AIJ
    Ml.setUp()
    Ml.setDiagonal(rowsum); Ml.assemble()
    return Ml, rowsum

def _apply_dirichlet_on_B(B, dofs, diag_value=1.0):
    """Apply homogeneous Dirichlet to an already-assembled PETSc Mat."""
    # zero rows+cols at constrained dofs and set unit diagonal
    iset = PETSc.IS().createGeneral(dofs, comm=B.comm)
    B.zeroRowsColumnsIS(iset, diag_value)

# ---------- main utility ----------

def fem_utils(mesh,
              bdr_condition="dirichlet",          # "dirichlet" or "neumann" (homogeneous)
              p=1,
              kappa=None,                         # scalar float, fem.Function, or callable for 'weighted'
              kappa_mode="scalar",                # "scalar" | "scalar_lumped" | "weighted" | "nodal_diag" | "nodal_diag_lumped"
              kappa_field=None,                   # for "weighted": fem.Function(V) or callable f(x)->UFL
              kappa_nodal=None,                   # for nodal_diag(_lumped): fem.Function(V) with nodal values
              quadrature_degree=None,
              attach_nullspace_for_neumann=True):
    """
    Returns:
        V, bc, B, K, M, Mlump
    Notes:
        - Homogeneous BCs only (Dirichlet u=0 on boundary; Neumann natural).
        - K, M, Mlump are assembled WITHOUT BCs (reusable).
        - B is assembled according to kappa_mode; if Dirichlet and B is not a pure UFL operator,
          Dirichlet is imposed once on B via zeroRowsColumns (safe).
    """
    # Function space
    V = fem.functionspace(mesh, ("Lagrange", p))

    # Assemble base matrices (no BCs)
    K, M, a_form, m_form = _assemble_K_M(V, quadrature_degree)
    Mlump, Mlump_diag = _lumped_from_rowsum(M)

    # Build B, depending on kappa_mode
    km = kappa_mode.lower()

    def _weighted_mass_matrix():
        """Assemble ∫ κ u v dx from a field κ (Function or callable)."""
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)
        dx = ufl.dx(metadata={"quadrature_degree": quadrature_degree} if quadrature_degree else {})
        if isinstance(kappa_field, fem.Function):
            kappa_expr = kappa_field
        elif callable(kappa_field):
            # callable f(x) expected to return UFL expr; pass coordinates
            x = ufl.SpatialCoordinate(V.mesh)
            kappa_expr = kappa_field(x)
        elif isinstance(kappa, (int, float)):
            kappa_expr = float(kappa)
        else:
            raise ValueError("Provide kappa_field (Function/callable) or scalar kappa for 'weighted'.")
        W_form = ufl.inner(kappa_expr * u, v) * dx
        W = fem.petsc.assemble_matrix(fem.form(W_form)); W.assemble()
        return W, W_form

    # build B_noBC and also keep a UFL combined form if we can
    B_noBC = None
    combined_form = None   # for Dirichlet cases where we can assemble with bcs directly

    if km == "scalar":
        if kappa is None:
            kappa = 0.0
        B_noBC = K.copy(); B_noBC.axpy(float(kappa), M, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B_noBC.assemble()
        combined_form = a_form + float(kappa) * m_form

    elif km == "scalar_lumped":
        if kappa is None:
            kappa = 0.0
        B_noBC = K.copy(); B_noBC.axpy(float(kappa), Mlump, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B_noBC.assemble()
        combined_form = None  # no UFL for lumped

    elif km == "weighted":
        W, W_form = _weighted_mass_matrix()
        B_noBC = K.copy(); B_noBC.axpy(1.0, W, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B_noBC.assemble()
        combined_form = a_form + W_form

    elif km == "nodal_diag":
        if kappa_nodal is None or not isinstance(kappa_nodal, fem.Function):
            raise ValueError("Provide kappa_nodal = fem.Function(V) for 'nodal_diag'.")
        # Mk = diag(kappa_nodal) * M  (row-scale M)
        Mk = M.copy()
        vec = kappa_nodal.vector
        Mk.diagonalScale(vec, None)  # left scale by vec
        Mk.assemble()
        B_noBC = K.copy(); B_noBC.axpy(1.0, Mk, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B_noBC.assemble()
        combined_form = None

    elif km == "nodal_diag_lumped":
        if kappa_nodal is None or not isinstance(kappa_nodal, fem.Function):
            raise ValueError("Provide kappa_nodal = fem.Function(V) for 'nodal_diag_lumped'.")
        # B = K + diag(kappa_nodal .* Mlump_diag)
        diag_vec = Mlump_diag.copy()
        # multiply elementwise by kappa nodal values
        with kappa_nodal.vector.localForm() as lf, diag_vec.localForm() as dv:
            dv.array[:] *= lf.array
        Mk = PETSc.Mat().createAIJ(B_noBC.getSize() if B_noBC else K.getSize(), nnz=1)
        Mk.setUp(); Mk.setDiagonal(diag_vec); Mk.assemble()
        B_noBC = K.copy(); B_noBC.axpy(1.0, Mk, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B_noBC.assemble()
        combined_form = None

    else:
        raise ValueError("kappa_mode must be one of {'scalar','scalar_lumped','weighted','nodal_diag','nodal_diag_lumped'}.")

    # Boundary conditions (homogeneous only)
    bc = None
    if bdr_condition.lower() == "dirichlet":
        facets = _all_boundary_facets(mesh)
        bc, dofs = _hom_dirichlet_bc(V, facets)

        if combined_form is not None:
            # We can assemble B with BCs in one go from UFL
            B = fem.petsc.assemble_matrix(fem.form(combined_form), bcs=[bc]); B.assemble()
        else:
            # Non-UFL B (lumped or nodal-diag variants): impose Dirichlet directly on B_noBC
            B = B_noBC.copy()
            _apply_dirichlet_on_B(B, dofs, diag_value=1.0)
    elif bdr_condition.lower() == "neumann":
        # Natural BCs: use B_noBC as-is
        B = B_noBC
        bc = None
        # Pure Neumann Laplacian: attach constant nullspace for kappa==0 (any mode that yields K only)
        if attach_nullspace_for_neumann:
            is_pure_K = (km in {"scalar", "weighted"} and (kappa in [0.0, 0, None])) or \
                        (km in {"nodal_diag", "nodal_diag_lumped"} and
                         isinstance(kappa_nodal, fem.Function) and
                         np.allclose(kappa_nodal.vector.array, 0.0))
            if is_pure_K:
                ns = PETSc.NullSpace().create(constant=True)
                B.setNullSpace(ns)
    else:
        raise ValueError("bdr_condition must be 'dirichlet' or 'neumann' (homogeneous).")

    return V, bc, B, K, M, Mlump




from mpi4py import MPI
from dolfinx import mesh as dmesh

comm = MPI.COMM_WORLD
mesh2d = dmesh.create_rectangle(comm, [np.array([0,0]), np.array([1,1])], [64,64], cell_type=dmesh.CellType.triangle)

# 1) Dirichlet, scalar kappa → B = K + kappa*M with BCs applied once
V, bc, B, K, M, Ml = fem_utils(mesh2d, bdr_condition="dirichlet", p=1, kappa=0.5, kappa_mode="scalar")

# 2) Neumann, weighted field kappa(x) → B = K + ∫ κ u v dx
kappa_field = lambda x: 1.0 + 0.2*ufl.sin(2*ufl.pi*x[0])
Vn, bcn, Bn, K_, M_, Ml_ = fem_utils(mesh2d, bdr_condition="neumann", p=1,
                                     kappa_mode="weighted", kappa_field=kappa_field)

# 3) Dirichlet, lumped mass with scalar kappa → B = K + kappa*Mlump (BCs enforced on B)
Vd, bcd, Bd, *_ = fem_utils(mesh2d, bdr_condition="dirichlet", p=1,
                            kappa=3.0, kappa_mode="scalar_lumped")

# 4) Dirichlet, nodal-diag with nodal kappa values
kfun = fem.Function(V); kfun.interpolate(lambda x: 1 + 0*x[0])  # for example, all ones
Vd2, bcd2, Bd2, *_ = fem_utils(mesh2d, bdr_condition="dirichlet", p=1,
                               kappa_mode="nodal_diag", kappa_nodal=kfun)
