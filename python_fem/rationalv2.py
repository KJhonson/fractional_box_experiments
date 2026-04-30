# rationalv2.py - Alternative rational approximation for fractional operators with PETSc/DOLFINx
# Uses the approach from fractional.operators.py but adapted for PETSc matrices
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from baryrat import brasil

def fractional_operators_petsc(L: PETSc.Mat, beta: float, C: PETSc.Mat,
                               scale_factor: float = 1.0, m: int = 1,
                               tau: float | np.ndarray = 1.0,
                               interval=(1.0, 11.0)) -> tuple[PETSc.Mat, PETSc.Mat]:
    """
    Alternative fractional operator approximation using the approach from fractional.operators.py
    but adapted for PETSc matrices.
    Returns PETSc.Mat (Pl, Pr).
    """
    n, _ = L.getSize()

    # Normalize L: L_scaled = (1/scale_factor) * L
    L_scaled = L.copy()
    L_scaled.scale(1.0 / scale_factor)

    # Lumped mass via row-sum: C_diag = rowsum(C)
    ones = PETSc.Vec().createMPI(n, comm=C.comm)
    ones.set(1.0)
    C_diag = C.createVecRight()
    C.mult(ones, C_diag)
    
    # Build diagonal matrices C and Ci
    Ci_vec = C_diag.copy()
    arr = Ci_vec.getArray()
    arr[arr == 0.0] = 1.0
    arr[:] = 1.0 / arr
    Ci_vec.setArray(arr)

    C_diag_mat = PETSc.Mat().createAIJ([n, n], comm=C.comm)
    C_diag_mat.setUp()
    C_diag_mat.setDiagonal(C_diag)
    C_diag_mat.assemble()
    
    Ci_diag_mat = PETSc.Mat().createAIJ([n, n], comm=C.comm)
    Ci_diag_mat.setUp()
    Ci_diag_mat.setDiagonal(Ci_vec)
    Ci_diag_mat.assemble()

    # Identity
    I = PETSc.Mat().createAIJ([n, n], comm=L.comm)
    I.setUp()
    I.setDiagonal(ones)
    I.assemble()

    # CiL = Ci * L_scaled (left diagonal scaling)
    CiL = L_scaled.copy()
    CiL.diagonalScale(Ci_vec, None)
    CiL.assemble()

    # Check if beta is integer
    if beta % 1 == 0:
        # Integer beta case
        Pr = I.copy()
        Pl = L_scaled.copy()
        
        # Pl = L_scaled @ (CiL)^(beta-1)
        for _ in range(int(beta) - 1):
            Pl = Pl.matMult(CiL)
        
        # Apply final scaling
        Pl.scale(scale_factor ** beta)
        Pl.assemble()
    else:
        # Fractional beta case
        beta_floor = int(np.floor(beta))
        beta_frac = beta - beta_floor

        # BRASIL rational approximation for fractional part
        r = brasil(lambda x: x ** beta_frac, interval, m)
        rb = np.array(r.poles())
        rc = np.array(r.zeros())
        gain = float(r.gain())

        # Helper to form (I - s * CiL)
        def shift_factor(s: float) -> PETSc.Mat:
            B = I.copy()
            B.axpy(-s, CiL, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            B.assemble()
            return B

        # Construct Pl using poles
        if len(rb) > 0:
            Pl = shift_factor(rb[0])
            for root in rb[1:]:
                Pl = Pl.matMult(shift_factor(root))
        else:
            Pl = I.copy()

        # Apply integer part: (C @ CiL)^(max(1, beta_floor) - 1)
        for _ in range(max(1, beta_floor) - 1):
            Pl = C_diag_mat.matMult(CiL.matMult(Pl))

        # Final C multiplication
        Pl = C_diag_mat.matMult(Pl)
        
        # Apply scaling and gain
        Pl.scale(scale_factor ** beta / gain)
        Pl.assemble()

        # Construct Pr using zeros
        if len(rc) > 0:
            Pr = shift_factor(rc[0])
            for root in rc[1:]:
                Pr = Pr.matMult(shift_factor(root))
        else:
            Pr = I.copy()

    # Tau scaling on Pr: Phi = diag(1/tau), Pr := Phi * Pr
    if np.isscalar(tau):
        phi_vec = PETSc.Vec().createMPI(n, comm=L.comm)
        phi_vec.set(1.0 / float(tau))
    else:
        tau_arr = np.asarray(tau, dtype=float)
        phi_vec = PETSc.Vec().createMPI(n, comm=L.comm)
        phi_vec.set(0.0)
        phi_vec.setArray(1.0 / tau_arr)
    
    Pr.diagonalScale(phi_vec, None)
    Pr.assemble()

    return Pl, Pr

def rational_solve(B, M, b, V, beta=0.5, m=2, interval=(0.01, 10), scale_factor=1.0, ksp_opts=None):
    """BRASIL solver for Neumann problems (PETSc-native)."""
    print(f"=== BRASIL Rational Solver v2 (β={beta}, m={m}) ===")

    # Build operators (PETSc mats)
    Pl, Pr = fractional_operators_petsc(B, beta, M, scale_factor=scale_factor, m=m, interval=interval)

    # Solve Pl v = b with PETSc KSP
    ksp = PETSc.KSP().create(B.comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=10000)
    ksp.setOperators(Pl)
    v = b.duplicate()
    v.set(0.0)
    ksp.solve(b, v)

    # u = Pr v
    u_vec = b.duplicate()
    Pr.mult(v, u_vec)

    # Convert to DOLFINx function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array
    u_h.x.scatter_forward()

    print(f"BRASIL solver v2 completed (β={beta})")
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
    print(f"=== BRASIL Rational Solver Dirichlet v2 (β={beta}, m={m}) ===")

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
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=10000)
    ksp.setOperators(Pl_ff)
    v_f = b_f.duplicate()
    v_f.set(0.0)
    ksp.solve(b_f, v_f)
    u_f = b_f.duplicate()
    Pr_ff.mult(v_f, u_f)

    # Scatter back to full vector (zeros on constrained DOFs)
    n, _ = B.getSize()
    u_full = PETSc.Vec().createMPI(n, comm=B.comm)
    u_full.set(0.0)
    free_idx = np.array(IS_f.getIndices(), dtype=np.int32)
    u_full.setValues(free_idx, u_f.getArray())
    u_full.assemble()

    # Convert to DOLFINx function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_full.array
    u_h.x.scatter_forward()

    print(f"BRASIL Dirichlet solver v2 completed (β={beta})")
    return u_h, ksp
