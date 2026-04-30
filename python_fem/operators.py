# operators.py
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc

# -------- Base assemblies (no BCs) --------

def assemble_K_M(V, quadrature_degree=None):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.dx(metadata={"quadrature_degree": quadrature_degree} if quadrature_degree else {})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    m = ufl.inner(u, v) * dx
    K = petsc.assemble_matrix(fem.form(a)); K.assemble()
    M = petsc.assemble_matrix(fem.form(m)); M.assemble()
    return K, M, a, m

def lumped_from_rowsum(M):
    rows = M.getRowSum()
    m, n = M.getSize()
    Ml = PETSc.Mat().createAIJ([m, n], nnz=1)
    Ml.setUp(); Ml.setDiagonal(rows); Ml.assemble()
    return Ml, rows

# -------- Apply homogeneous Dirichlet to an already-assembled matrix --------

def apply_dirichlet_on_matrix(B, dofs, diag_value=1.0):
    iset = PETSc.IS().createGeneral(dofs, comm=B.comm)
    B.zeroRowsColumns(iset, diag_value)  # zeros both rows and columns for proper Dirichlet BC

# -------- Build operator B = K + κ·(variant of M) --------

def build_operator_B(V,
                     bc_type="dirichlet",                 # 'dirichlet' or 'neumann' (homogeneous)
                     facet_tags=None,                     # needed if bc_type='dirichlet'
                     ids=(1,),                            # which facet ids to clamp, usually (1,)
                     kappa=None,                          # float or None for 0
                     kappa_mode="scalar",                 # 'scalar'|'scalar_lumped'|'weighted'|'nodal_diag'|'nodal_diag_lumped'
                     kappa_field=None,                    # for 'weighted': fem.Function(V) or callable x->UFL
                     kappa_nodal=None,                    # for 'nodal_diag'/_lumped: fem.Function(V)
                     quadrature_degree=None,
                     attach_nullspace_for_neumann=True):
    """
    Returns:
        B, K, M, Mlump, bc (bc is None for Neumann)
    """
    # Base matrices
    K, M, a, m = assemble_K_M(V, quadrature_degree)
    Mlump, Mlump_diag = lumped_from_rowsum(M)

    # Boundary (Dirichlet homogeneous if requested)
    bc = None
    dofs = None
    if bc_type.lower() == "dirichlet":
        if facet_tags is None:
            raise ValueError("facet_tags required for Dirichlet (which facets to clamp).")
        bc = _make_dirichlet(V, facet_tags, ids)
        dofs = _dirichlet_dofs(V, facet_tags, ids)

    # Build B without BCs first
    km = kappa_mode.lower()
    if kappa is None:
        kappa = 0.0

    if km == "scalar":
        B = K.copy(); B.axpy(float(kappa), M, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B.assemble()
        combined_form = a + float(kappa)*m  # UFL for Dirichlet assembly path
    elif km == "scalar_lumped":
        B = K.copy(); B.axpy(float(kappa), Mlump, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B.assemble()
        combined_form = None
    elif km == "weighted":
        # W_ij = ∫ κ u v dx
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(metadata={"quadrature_degree": quadrature_degree} if quadrature_degree else {})
        if kappa_field is None:
            raise ValueError("kappa_field (Function or callable x->UFL) required for 'weighted'")
        if hasattr(kappa_field, "value_shape"):    # fem.Function
            kapp = kappa_field
        elif callable(kappa_field):
            x = ufl.SpatialCoordinate(V.mesh); kapp = kappa_field(x)
        else:
            raise ValueError("Invalid kappa_field.")
        W_form = ufl.inner(kapp*u, v) * dx
        W = petsc.assemble_matrix(fem.form(W_form)); W.assemble()
        B = K.copy(); B.axpy(1.0, W, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B.assemble()
        combined_form = a + W_form
    elif km == "nodal_diag":
        if kappa_nodal is None:
            raise ValueError("kappa_nodal = fem.Function(V) required for 'nodal_diag'")
        Mk = M.copy()
        Mk.diagonalScale(kappa_nodal.vector, None)  # row-scale
        Mk.assemble()
        B = K.copy(); B.axpy(1.0, Mk, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B.assemble()
        combined_form = None
    elif km == "nodal_diag_lumped":
        if kappa_nodal is None:
            raise ValueError("kappa_nodal = fem.Function(V) required for 'nodal_diag_lumped'")
        # diag(κ ⊙ diag(Mlump))
        diag = Mlump_diag.copy()
        with diag.localForm() as d, kappa_nodal.vector.localForm() as k:
            d.array[:] *= k.array
        m, n = K.getSize()
        Mk = PETSc.Mat().createAIJ([m, n], nnz=1); Mk.setUp(); Mk.setDiagonal(diag); Mk.assemble()
        B = K.copy(); B.axpy(1.0, Mk, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); B.assemble()
        combined_form = None
    else:
        raise ValueError("Unknown kappa_mode.")

    # Impose homogeneous BCs on B correctly
    if bc_type.lower() == "dirichlet":
        if combined_form is not None:
            # safest: assemble combined form with bcs=[bc]
            B = petsc.assemble_matrix(fem.form(combined_form), bcs=[bc]); B.assemble()
        else:
            # algebraic B → zero rows/cols on imposed dofs
            apply_dirichlet_on_matrix(B, dofs, diag_value=1.0)
    elif bc_type.lower() == "neumann":
        # natural BCs, no change; optional nullspace for κ=0
        if attach_nullspace_for_neumann and np.isclose(float(kappa), 0.0):
            ns = PETSc.NullSpace().create(constant=True)
            B.setNullSpace(ns)

    return B, K, M, Mlump, bc

# ---- local helpers (private) ----
def _dirichlet_dofs(V, facet_tags, ids):
    fdim = V.mesh.topology.dim - 1
    sel = np.concatenate([facet_tags.find(i) for i in np.atleast_1d(ids)])
    return fem.locate_dofs_topological(V, fdim, sel)

def _make_dirichlet(V, facet_tags, ids):
    dofs = _dirichlet_dofs(V, facet_tags, ids)
    uD = fem.Function(V)
    return fem.dirichletbc(uD, dofs)




#####

# - builds $V$ and homogeneous BCs (Dirichlet or Neumann) for $d=1,2,3$
# - assembles $K$ and (consistent + lumped) $M$ without BCs
# - builds the operator $B$ according to how you want to use $\kappa$ :
# 1. scalar: $B=K+\kappa M$
# 2. scalar_lumped: $B=K+\kappa M_{\text {lump }}$
# 3. weighted (field): $B=K+M_\kappa$ with $\left(M_\kappa\right)_{i j}=\int \kappa \phi_i \phi_j d x$
# 4. nodal_diag: $B=K+\operatorname{diag}\left(\kappa_{\text {nodal }}\right) M($ row-scales $M)$
# 5. nodal_diag_lumped: $B=K+\operatorname{diag}\left(\kappa_{\text {nodal }}\right) M_{\text {lump }}$ (purely diagonal)

#####