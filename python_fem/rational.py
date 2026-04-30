# rational.py - BRASIL rational approximation for fractional operators with PETSc/DOLFINx
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from baryrat import brasil

def fractional_operators_petsc(L: PETSc.Mat, beta: float, C: PETSc.Mat,
                               scale_factor: float = 1.0, m: int = 1,
                               tau: float | np.ndarray = 1.0,
                               interval=(1.0, 11.0)) -> tuple[PETSc.Mat, PETSc.Mat]:
    """
    BRASIL rational approximation (CODE 2) using PETSc-native matrices.
    Returns PETSc.Mat (Pl, Pr).
    """
    n, _ = L.getSize()

    # Normalize L: L_scaled = (1/scale_factor) * L
    L_scaled = L.copy(); L_scaled.scale(1.0 / scale_factor)

    # Lumped mass via row-sum: C_diag = rowsum(C)
    ones = PETSc.Vec().createMPI(n, comm=C.comm); ones.set(1.0)
    C_diag = C.createVecRight(); C.mult(ones, C_diag)
    # Build diagonal matrices C and Ci
    Ci_vec = C_diag.copy()
    arr = Ci_vec.getArray()
    arr[arr == 0.0] = 1.0
    arr[:] = 1.0 / arr
    Ci_vec.setArray(arr)

    C_diag_mat = PETSc.Mat().createAIJ([n, n], comm=C.comm)
    C_diag_mat.setUp(); C_diag_mat.setDiagonal(C_diag); C_diag_mat.assemble()
    Ci_diag_mat = PETSc.Mat().createAIJ([n, n], comm=C.comm)
    Ci_diag_mat.setUp(); Ci_diag_mat.setDiagonal(Ci_vec); Ci_diag_mat.assemble()

    # Identity
    I = PETSc.Mat().createAIJ([n, n], comm=L.comm)
    I.setUp(); I.setDiagonal(ones); I.assemble()

    # CiL = Ci * L_scaled (left diagonal scaling)
    CiL = L_scaled.copy(); CiL.diagonalScale(Ci_vec, None); CiL.assemble()

    # Decompose beta
    beta_floor = int(np.floor(beta))
    beta_frac = beta - beta_floor

    # BRASIL rational for fractional part
    if abs(beta_frac) > 1e-14:
        r = brasil(lambda x: x ** beta_frac, interval, m)
        zeros = np.array(r.zeros())
        poles = np.array(r.poles())
        gain = float(r.gain())
    else:
        zeros, poles, gain = [], [], 1.0

    # Helper to form (CiL - s I)
    def shift_factor(mat: PETSc.Mat, s: float) -> PETSc.Mat:
        B = mat.copy()
        B.axpy(-s, I, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        B.assemble()
        return B

    # Build Pr (product over poles) - highly optimized
    if len(poles) > 0:
        # Use polynomial evaluation approach for better stability
        Pr = I.copy()
        for p in poles:
            Fp = shift_factor(CiL, p)
            Pr = Pr.matMult(Fp)
    else:
        Pr = I.copy()

    # Build Pl (full product with gain and integer part) - highly optimized
    # Start with integer component: (C^{-1}L)^[β]
    if beta_floor > 0:
        # More efficient: compute C @ CiL^beta_floor directly
        A_int = I.copy()
        for _ in range(beta_floor):
            A_int = A_int.matMult(CiL)
        Pl = C_diag_mat.matMult(A_int)
        Pl.scale(gain)
    else:
        # For pure fractional case (0 < beta < 1) - start with C
        Pl = C_diag_mat.copy()
        Pl.scale(gain)
    
    # Apply zeros - highly optimized
    if len(zeros) > 0:
        # Direct multiplication for better numerical stability
        for z in zeros:
            Fz = shift_factor(CiL, z)
            Pl = Pl.matMult(Fz)
    
    # Add extra C @ multiplication for stability (only for beta >= 1)
    # This is equivalent to multiplying by (C @ CiL)^(beta_floor-1)
    extra_iterations = max(1, beta_floor) - 1
    if extra_iterations > 0:
        # More efficient: compute (C @ CiL)^extra_iterations
        C_CiL = C_diag_mat.matMult(CiL)
        for _ in range(extra_iterations):
            Pl = C_CiL.matMult(Pl)
    
    # Apply final scaling
    Pl.scale(scale_factor ** beta)
    Pl.assemble()

    # Tau scaling on Pr: Phi = diag(1/tau), Pr := Phi * Pr
    if np.isscalar(tau):
        phi_vec = PETSc.Vec().createMPI(n, comm=L.comm); phi_vec.set(1.0 / float(tau))
    else:
        tau_arr = np.asarray(tau, dtype=float)
        phi_vec = PETSc.Vec().createMPI(n, comm=L.comm)
        phi_vec.set(0.0)
        # Set array
        phi_vec.setArray(1.0 / tau_arr)
    Pr.diagonalScale(phi_vec, None)
    Pr.assemble()

    return Pl, Pr

def rational_solve(B, M, b, V, beta=0.5, m=2, interval=(0.01, 10), scale_factor=1.0, ksp_opts=None):
    """BRASIL solver for Neumann problems (PETSc-native)."""
    print(f"=== BRASIL Rational Solver (β={beta}, m={m}) ===")

    # Build operators (PETSc mats)
    Pl, Pr = fractional_operators_petsc(B, beta, M, scale_factor=scale_factor, m=m, interval=interval)

    # Solve Pl v = b with PETSc KSP
    ksp = PETSc.KSP().create(B.comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=10000)
    ksp.setOperators(Pl)
    v = b.duplicate(); v.set(0.0)
    ksp.solve(b, v)

    # u = Pr v
    u_vec = b.duplicate(); Pr.mult(v, u_vec)

    # Convert to DOLFINx function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array
    u_h.x.scatter_forward()

    print(f"BRASIL solver completed (β={beta})")
    return u_h, ksp

def rational_solve_unified(B, M, b, V, bc=None, beta=0.5, m=2, interval=(0.01, 10), scale_factor=1.0, ksp_opts=None):
    """Unified BRASIL solver (Neumann only for now)."""
    if bc is None:
        return rational_solve(B, M, b, V, beta=beta, m=m, interval=interval, scale_factor=scale_factor, ksp_opts=ksp_opts)
    else:
        return rational_solve_dirichlet(B, M, b, V, bc, beta=beta, m=m, interval=interval, scale_factor=scale_factor, ksp_opts=ksp_opts)


# ------------------------ Dirichlet support ------------------------
def _dirichlet_free_is(A: PETSc.Mat, bc: fem.DirichletBC) -> PETSc.IS:
    """
    Create an index set of free DOFs for homogeneous Dirichlet BC.
    Returns PETSc.IS for free DOFs.
    """
    n, _ = A.getSize()
    if n == 0:
        raise RuntimeError("Empty matrix.")

    dof_indices_tuple = bc.dof_indices()
    if isinstance(dof_indices_tuple, tuple) and len(dof_indices_tuple) >= 1:
        bdy = np.array(dof_indices_tuple[0], dtype=np.int32)
    else:
        bdy = np.array(dof_indices_tuple, dtype=np.int32)

    bdy = bdy.ravel()
    mask = np.ones(n, dtype=bool)
    mask[bdy] = False
    free = np.flatnonzero(mask).astype(np.int32)
    IS_f = PETSc.IS().createGeneral(free, comm=A.getComm())
    return IS_f


def rational_solve_dirichlet(B, M, b, V, bc, beta=0.5, m=2, interval=(0.01, 10), scale_factor=1.0, ksp_opts=None):
    """BRASIL solver for Dirichlet problems (restrict to free DOFs)."""
    print(f"=== BRASIL Rational Solver Dirichlet (β={beta}, m={m}) ===")

    # Index set of free DOFs
    IS_f = _dirichlet_free_is(B, bc)

    # Submatrices on free DOFs and restricted RHS
    B_ff = B.createSubMatrix(IS_f, IS_f)
    M_ff = M.createSubMatrix(IS_f, IS_f)
    b_f = b.getSubVector(IS_f)

    # Build operators on free DOFs (PETSc-native)
    Pl_ff, Pr_ff = fractional_operators_petsc(B_ff, beta, M_ff, scale_factor=scale_factor, m=m, interval=interval)

    # Solve on free DOFs with KSP: Pl_ff v = b_f
    ksp = PETSc.KSP().create(B.comm)
    ksp.setType("cg"); ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=10000)
    ksp.setOperators(Pl_ff)
    v_f = b_f.duplicate(); v_f.set(0.0)
    ksp.solve(b_f, v_f)
    u_f = b_f.duplicate(); Pr_ff.mult(v_f, u_f)

    # Scatter back to full vector (zeros on constrained DOFs)
    n, _ = B.getSize()
    u_full = PETSc.Vec().createMPI(n, comm=B.comm); u_full.set(0.0)
    free_idx = np.array(IS_f.getIndices(), dtype=np.int32)
    u_full.setValues(free_idx, u_f.getArray())
    u_full.assemble()

    # Convert to DOLFINx function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_full.array
    u_h.x.scatter_forward()

    print(f"BRASIL Dirichlet solver completed (β={beta})")
    return u_h, ksp