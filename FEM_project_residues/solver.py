# solver.py - Complete solver library for FEM problems
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from math import ceil, pi
from petsc4py import PETSc
from dolfinx import fem

# ============================================================================
# STANDARD PETSc PURE SOLVER
# ============================================================================

def solve_petsc(B, b, V, ksp_opts=None):
    """
    Solve B u = b with PETSc KSP, return fem.Function(V).
    ksp_opts: dict, e.g., {"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10}
    """
    ksp = PETSc.KSP().create(B.comm)
    ksp.setOperators(B)

    # defaults
    ksp.setType("cg")
    pc = ksp.getPC(); pc.setType("hypre")  # or "gamg"
    ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=10_000)

    if ksp_opts:
        # convenience: allow users to pass PETSc-style options
        if "ksp_type" in ksp_opts: ksp.setType(ksp_opts["ksp_type"])
        if "pc_type"  in ksp_opts: ksp.getPC().setType(ksp_opts["pc_type"])
        if "ksp_rtol" in ksp_opts: ksp.setTolerances(rtol=float(ksp_opts["ksp_rtol"]))
        if "ksp_atol" in ksp_opts: ksp.setTolerances(atol=float(ksp_opts["ksp_atol"]))
        if "max_it"   in ksp_opts: ksp.setTolerances(max_it=int(ksp_opts["max_it"]))

    u_vec = B.createVecRight()
    ksp.solve(b, u_vec)

    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array
    u_h.x.scatter_forward()
    return u_h, ksp

# ============================================================================
# SINC SOLVER FOR FRACTIONAL AND NON-FRACTIONAL PROBLEMS
# ============================================================================

def sinc_solve_standard(B, M, b, V, beta=1.0, k=0.25, ksp_opts=None):
    """
    Sinc-based solver for standard and fractional eigenvalue problems with Neumann boundary conditions.
    
    For beta=1.0, this is equivalent to solving: B u = b (standard case)
    For 0 < beta < 1, this solves: B^β u = b (fractional case)
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        Right-hand side vector
    V : dolfinx.fem.FunctionSpace
        Function space
    beta : float, optional
        Power of the operator (default: 1.0 for standard case)
    k : float, optional
        Sinc step size parameter (default: 0.25)
    ksp_opts : dict, optional
        PETSc KSP options
        
    Returns:
    --------
    u_h : dolfinx.fem.Function
        Solution function
    ksp : PETSc.KSP
        KSP object used for solving
    """
    print(f"=== Sinc Standard Solver (β={beta}, Neumann BC) ===")
    
    def _make_ksp_from_opts(A: PETSc.Mat) -> PETSc.KSP:
        ksp_type = "cg"; pc_type = "hypre"; rtol = 1e-10; max_it = 500
        if ksp_opts:
            if "ksp_type" in ksp_opts: ksp_type = ksp_opts["ksp_type"]
            if "pc_type" in ksp_opts: pc_type = ksp_opts["pc_type"]
            if "ksp_rtol" in ksp_opts: rtol = float(ksp_opts["ksp_rtol"])
            if "ksp_atol" in ksp_opts: rtol = float(ksp_opts["ksp_atol"])
            if "max_it" in ksp_opts: max_it = int(ksp_opts["max_it"])
        return make_ksp(A, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, max_it=max_it)

    def _apply_binvm_power(B: PETSc.Mat, M: PETSc.Mat, v: PETSc.Vec, p: int) -> PETSc.Vec:
        if p <= 0:
            out = v.duplicate(); out.setArray(v.getArray().copy()); return out
        ksp_B = _make_ksp_from_opts(B)
        y = v.duplicate(); y.setArray(v.getArray().copy())
        tmp = v.duplicate()
        for _ in range(p):
            # tmp = M * y
            M.mult(y, tmp)
            # solve B x = tmp
            x = v.duplicate(); x.set(0.0)
            ksp_B.setOperators(B)
            ksp_B.solve(tmp, x)
            y = x
        return y

    if abs(beta - 1.0) < 1e-14:
        # Standard case: B u = b
        print("  Using standard case: B u = b")
        u_h, ksp = solve_petsc(B, b, V, ksp_opts)
    elif 0 < beta < 1:
        # Fractional case: use existing sinc_solve
        print(f"  Using fractional case: B^{beta} u = b")
        u_h, ksp = sinc_solve(B, M, b, V, beta=beta, k=k, ksp_opts=ksp_opts)
    elif beta >= 1 and abs(beta - round(beta)) < 1e-14:
        # Integer beta >= 1: apply (B^{-1} M)^beta M^{-1} b stably via solves
        p = int(round(beta))
        print(f"  Using integer case: (B^{-1} M)^{p} M^{-1} b")
        # y0 = M^{-1} b
        ksp_M = _make_ksp_from_opts(M)
        y0 = b.duplicate(); y0.set(0.0)
        ksp_M.setOperators(M)
        ksp_M.solve(b, y0)
        # y = (B^{-1} M)^p y0
        y = _apply_binvm_power(B, M, y0, p)
        # to Function
        u_h = fem.Function(V)
        u_h.x.array[:] = y.array
        u_h.x.scatter_forward()
        ksp = _make_ksp_from_opts(B)
    elif beta > 1:
        # Mixed case beta = floor + frac; use sinc for frac and then integer power
        p = int(np.floor(beta))
        frac = float(beta - p)
        print(f"  Using mixed case: u = (B^-1 M)^{p} (B^-{frac} b)")
        # u_frac = B^{-frac} b via sinc
        u_frac = sinc_apply(B, M, b, frac, k=k)
        # apply (B^{-1} M)^p to u_frac
        y = _apply_binvm_power(B, M, u_frac, p)
        u_h = fem.Function(V)
        u_h.x.array[:] = y.array
        u_h.x.scatter_forward()
        ksp = _make_ksp_from_opts(B)
    else:
        raise ValueError(f"Invalid beta value: {beta}. Must be > 0.")
    
    print(f"Sinc standard solver completed (β={beta})")
    return u_h, ksp

def sinc_solve_standard_dirichlet(B, M, b, V, bc, beta=1.0, k=0.25, ksp_opts=None):
    """
    Sinc-based solver for standard and fractional eigenvalue problems with Dirichlet boundary conditions.
    
    For beta=1.0, this is equivalent to solving: B u = b with Dirichlet BC (standard case)
    For 0 < beta < 1, this solves: B^β u = b with Dirichlet BC (fractional case)
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        Right-hand side vector
    V : dolfinx.fem.FunctionSpace
        Function space
    bc : dolfinx.fem.DirichletBC
        Dirichlet boundary condition
    beta : float, optional
        Power of the operator (default: 1.0 for standard case)
    k : float, optional
        Sinc step size parameter (default: 0.25)
    ksp_opts : dict, optional
        PETSc KSP options
        
    Returns:
    --------
    u_h : dolfinx.fem.Function
        Solution function
    ksp : PETSc.KSP
        KSP object used for solving
    """
    print(f"=== Sinc Standard Solver Dirichlet (β={beta}) ===")
    
    def _dirichlet_free_is_local(A: PETSc.Mat, bc_local: fem.DirichletBC) -> Tuple[PETSc.IS, PETSc.IS]:
        return _dirichlet_free_is(A, bc_local)

    def _make_ksp_from_opts(A: PETSc.Mat) -> PETSc.KSP:
        ksp_type = "cg"; pc_type = "hypre"; rtol = 1e-10; max_it = 500
        if ksp_opts:
            if "ksp_type" in ksp_opts: ksp_type = ksp_opts["ksp_type"]
            if "pc_type" in ksp_opts: pc_type = ksp_opts["pc_type"]
            if "ksp_rtol" in ksp_opts: rtol = float(ksp_opts["ksp_rtol"])
            if "ksp_atol" in ksp_opts: rtol = float(ksp_opts["ksp_atol"])
            if "max_it" in ksp_opts: max_it = int(ksp_opts["max_it"])
        return make_ksp(A, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, max_it=max_it)

    def _apply_binvm_power_dirichlet(B: PETSc.Mat, M: PETSc.Mat, v_full: PETSc.Vec, p: int) -> PETSc.Vec:
        if p <= 0:
            out = v_full.duplicate(); out.setArray(v_full.getArray().copy()); return out
        IS_f, _ = _dirichlet_free_is_local(B, bc)
        B_ff = B.createSubMatrix(IS_f, IS_f)
        M_ff = M.createSubMatrix(IS_f, IS_f)
        v_f = v_full.getSubVector(IS_f)
        ksp_B = _make_ksp_from_opts(B_ff)
        y_f = v_f.duplicate(); y_f.setArray(v_f.getArray().copy())
        tmp_f = v_f.duplicate()
        for _ in range(p):
            M_ff.mult(y_f, tmp_f)
            x_f = v_f.duplicate(); x_f.set(0.0)
            ksp_B.setOperators(B_ff)
            ksp_B.solve(tmp_f, x_f)
            y_f = x_f
        # scatter back
        n, _ = B.getSize()
        y_full = PETSc.Vec().createMPI(n, comm=B.comm); y_full.set(0.0)
        y_full.setValues(IS_f.indices, y_f.getArray())
        y_full.assemble()
        return y_full

    if abs(beta - 1.0) < 1e-14:
        # Standard case: B u = b with Dirichlet BC
        print("  Using standard case: B u = b with Dirichlet BC")
        u_h, ksp = solve_petsc(B, b, V, ksp_opts)
    elif 0 < beta < 1:
        # Fractional case: use existing sinc_solve_dirichlet
        print(f"  Using fractional case: B^{beta} u = b with Dirichlet BC")
        u_h, ksp = sinc_solve_dirichlet(B, M, b, V, bc, beta=beta, k=k, ksp_opts=ksp_opts)
    elif beta >= 1 and abs(beta - round(beta)) < 1e-14:
        # Integer beta >= 1: apply (B^{-1} M)^beta M^{-1} b on free DOFs
        p = int(round(beta))
        print(f"  Using integer case (Dirichlet): (B^{-1} M)^{p} M^{-1} b")
        IS_f, _ = _dirichlet_free_is_local(B, bc)
        B_ff = B.createSubMatrix(IS_f, IS_f)
        M_ff = M.createSubMatrix(IS_f, IS_f)
        b_f = b.getSubVector(IS_f)
        # y0_f = M_ff^{-1} b_f
        ksp_M = _make_ksp_from_opts(M_ff)
        y0_f = b_f.duplicate(); y0_f.set(0.0)
        ksp_M.setOperators(M_ff)
        ksp_M.solve(b_f, y0_f)
        # lift to full vector to reuse helper
        n, _ = B.getSize()
        y0_full = PETSc.Vec().createMPI(n, comm=B.comm); y0_full.set(0.0)
        y0_full.setValues(IS_f.indices, y0_f.getArray())
        y0_full.assemble()
        y_full = _apply_binvm_power_dirichlet(B, M, y0_full, p)
        u_h = fem.Function(V)
        u_h.x.array[:] = y_full.array
        u_h.x.scatter_forward()
        ksp = _make_ksp_from_opts(B)
    elif beta > 1:
        # Mixed case: use sinc for fractional part, then integer power on free DOFs
        p = int(np.floor(beta))
        frac = float(beta - p)
        print(f"  Using mixed case (Dirichlet): u = (B^-1 M)^{p} (B^-{frac} b)")
        u_full_frac = sinc_apply_dirichlet(B, M, b, frac, bc=bc, k=k)
        y_full = _apply_binvm_power_dirichlet(B, M, u_full_frac, p)
        u_h = fem.Function(V)
        u_h.x.array[:] = y_full.array
        u_h.x.scatter_forward()
        ksp = _make_ksp_from_opts(B)
    else:
        raise ValueError(f"Invalid beta value: {beta}. Must be > 0.")
    
    print(f"Sinc standard solver Dirichlet completed (β={beta})")
    return u_h, ksp

def sinc_solve_standard_unified(B, M, b, V, bc=None, beta=1.0, k=0.25, ksp_opts=None):
    """
    Unified Sinc-based solver for both standard and fractional problems.
    
    Automatically handles both Neumann and Dirichlet boundary conditions:
    - If bc is None: Uses Neumann boundary conditions
    - If bc is provided: Uses Dirichlet boundary conditions
    
    For beta=1.0, this is equivalent to solving: B u = b (standard case)
    For 0 < beta < 1, this solves: B^β u = b (fractional case)
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        Right-hand side vector
    V : dolfinx.fem.FunctionSpace
        Function space
    bc : dolfinx.fem.DirichletBC, optional
        Dirichlet boundary condition (None for Neumann)
    beta : float, optional
        Power of the operator (default: 1.0 for standard case)
    k : float, optional
        Sinc step size parameter (default: 0.25)
    ksp_opts : dict, optional
        PETSc KSP options
        
    Returns:
    --------
    u_h : dolfinx.fem.Function
        Solution function
    ksp : PETSc.KSP
        KSP object used for solving
    """
    if bc is None:
        return sinc_solve_standard(B, M, b, V, beta=beta, k=k, ksp_opts=ksp_opts)
    else:
        return sinc_solve_standard_dirichlet(B, M, b, V, bc, beta=beta, k=k, ksp_opts=ksp_opts)

# ============================================================================
# SINC SOLVER FOR FRACTIONAL PROBLEMS (ORIGINAL)
# ============================================================================

# ------- helpers ----------------------------------------------------
def balanced_MN(beta: float, k: float, s: float = 0.0) -> Tuple[int, int]:
    """
    Balanced truncation sizes for sinc quadrature (resolvent form).
      N ≈ π^2 / (2 (β - s+) k^2),   M ≈ π^2 / (2 (1-β) k^2)
    Use s=0 unless you explicitly measure error in D(B^s).
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")
    s_plus = max(0.0, s)
    if beta <= s_plus:
        raise ValueError("Use s < beta if you measure in D(B^s); otherwise keep s=0.")
    N = ceil(pi**2 / (2.0 * (beta - s_plus) * k**2))
    M = ceil(pi**2 / (2.0 * (1.0 - beta) * k**2))
    return N, M

def make_ksp(A: PETSc.Mat,
             ksp_type: str = "cg",
             pc_type: str = "hypre",
             rtol: float = 1e-10,
             max_it: int = 200) -> PETSc.KSP:
    """
    Build a PETSc KSP with common sensible defaults.
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    return ksp

# ------- core: apply B^{-beta} to b via sinc-resolvent sum ----------
def sinc_apply(
    B: PETSc.Mat,
    M: PETSc.Mat,
    b: PETSc.Vec,
    beta: float,
    *,
    k: float = 0.25,
    N_terms: Optional[int] = None,
    M_terms: Optional[int] = None,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
    rtol: float = 1e-10,
    max_it: int = 500,
) -> PETSc.Vec:
    """
    Returns u ≈ B^{-β} b using the resolvent sinc quadrature:
        u = (sin(πβ)/π) k Σ_{ℓ=-M}^{N} e^{(1-β) y_ℓ} x_ℓ,
        (B + e^{y_ℓ} M) x_ℓ = b,  y_ℓ = ℓ k.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")

    if N_terms is None or M_terms is None:
        N_terms, M_terms = balanced_MN(beta, k, s=0.0)

    # Prepare work vectors
    u = b.duplicate(); u.set(0.0)
    x = b.duplicate()
    rhs = b.duplicate()

    # Constant prefactor
    pref = (np.sin(pi * beta) / pi) * k

    # Single KSP object reused across terms
    ksp = make_ksp(B, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, max_it=max_it)

    # Main loop: ℓ = -M_terms ... N_terms
    for ell in range(-M_terms, N_terms + 1):
        y = ell * k
        ey = float(np.exp(y))

        # Build S = B + ey * M
        S = B.copy()
        S.axpy(ey, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        S.assemble()

        # Solve S x = b
        ksp.setOperators(S)
        b.copy(rhs)
        x.set(0.0)
        ksp.solve(rhs, x)

        # Accumulate with weight e^{(1-β) y}
        weight = pref * np.exp((1.0 - beta) * y)
        u.axpy(weight, x)

    return u

# ---------- Dirichlet restriction (homogeneous) ---------------------
def _dirichlet_free_is(A: PETSc.Mat, bc: fem.DirichletBC) -> Tuple[PETSc.IS, PETSc.IS]:
    """
    Create index sets for free and constrained DOFs for homogeneous Dirichlet BC.
    Returns (IS_f, IS_c) where IS_f are free DOFs and IS_c are constrained DOFs.
    """
    n, _ = A.getSize()
    if n == 0:
        raise RuntimeError("Empty matrix.")
    
    # Get DOF indices - bc.dof_indices() returns (array, count)
    dof_indices_tuple = bc.dof_indices()
    if isinstance(dof_indices_tuple, tuple) and len(dof_indices_tuple) >= 1:
        # Extract the array of DOF indices (first element of tuple)
        bdy = np.array(dof_indices_tuple[0], dtype=np.int32)
    else:
        # Fallback: treat as direct array
        bdy = np.array(dof_indices_tuple, dtype=np.int32)
    
    bdy = bdy.ravel()
    mask = np.ones(n, dtype=bool); mask[bdy] = False
    free = np.flatnonzero(mask).astype(np.int32)
    IS_f = PETSc.IS().createGeneral(free, comm=A.getComm())
    IS_c = PETSc.IS().createGeneral(bdy,  comm=A.getComm())
    return IS_f, IS_c

# ---------- core: B^{-β} b with homogeneous Dirichlet --------------
def sinc_apply_dirichlet(
    B: PETSc.Mat,
    M: PETSc.Mat,
    b: PETSc.Vec,
    beta: float,
    *,
    bc: fem.DirichletBC,            # homogeneous Dirichlet (value=0)
    k: float = 0.25,
    N_terms: Optional[int] = None,
    M_terms: Optional[int] = None,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
    rtol: float = 1e-10,
    max_it: int = 500,
) -> PETSc.Vec:
    """
    Apply B^{-β} to load b under homogeneous Dirichlet BCs by restricting to free DOFs.
    For each ℓ: (B_ff + e^{yℓ} M_ff) x_f = b_f; accumulate with sinc weights.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")

    # Build restriction
    IS_f, _ = _dirichlet_free_is(B, bc)

    # Submatrices on free DOFs
    B_ff = B.createSubMatrix(IS_f, IS_f)
    M_ff = M.createSubMatrix(IS_f, IS_f)

    # Restricted RHS (free DOFs)
    b_f = b.getSubVector(IS_f)

    # Work vectors (free DOFs)
    u_f = b_f.duplicate(); u_f.set(0.0)
    x_f = b_f.duplicate()
    rhs_f = b_f.duplicate()

    # Truncation sizes
    if N_terms is None or M_terms is None:
        N_terms, M_terms = balanced_MN(beta, k, s=0.0)

    # Prefactor
    pref = (np.sin(pi * beta) / pi) * k

    # Main sinc loop
    for ell in range(-M_terms, N_terms + 1):
        y = ell * k
        ey = float(np.exp(y))

        # S_ff = B_ff + e^y M_ff
        S_ff = B_ff.copy()
        S_ff.axpy(ey, M_ff, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        S_ff.assemble()
        
        # Create KSP for this specific matrix - manual setup to avoid issues
        ksp = PETSc.KSP().create()
        ksp.setType(ksp_type)
        pc = ksp.getPC()
        pc.setType(pc_type)
        ksp.setTolerances(rtol=rtol, max_it=max_it)
        ksp.setOperators(S_ff)

        # Solve on free DOFs
        # Create a proper copy to avoid memory issues
        rhs_f.setArray(b_f.getArray().copy())
        x_f.set(0.0)
        ksp.solve(rhs_f, x_f)

        # Accumulate with weight
        weight = pref * np.exp((1.0 - beta) * y)
        u_f.axpy(weight, x_f)

    # Scatter back to full vector (zeros on constrained DOFs)
    n, _ = B.getSize()
    u = PETSc.Vec().createMPI(n, comm=B.comm); u.set(0.0)
    u.setValues(IS_f.indices, u_f.getArray())
    u.assemble()
    return u

# ---------- convenience: return fem.Function ------------------------
def sinc_apply_dirichlet_func(
    V: fem.FunctionSpace,
    B: PETSc.Mat,
    M: PETSc.Mat,
    b: PETSc.Vec,
    beta: float,
    *,
    bc: fem.DirichletBC,
    **kwargs,
) -> fem.Function:
    """Convenience function that returns a fem.Function instead of PETSc.Vec."""
    u_vec = sinc_apply_dirichlet(B, M, b, beta, bc=bc, **kwargs)
    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array
    u_h.x.scatter_forward()
    return u_h

# ============================================================================
# HIGH-LEVEL SINC SOLVERS
# ============================================================================

def sinc_solve(B, M, b, V, beta=0.5, k=0.25, ksp_opts=None):
    """
    Sinc-based solver for fractional eigenvalue problems with Neumann boundary conditions.
    
    Solves: B^β u = b using sinc resolvent method.
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix (will be modified to B^β)
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        RHS vector
    V : fem.FunctionSpace
        Function space
    beta : float
        Fractional power (0 < beta < 1)
    k : float, optional
        Sinc step size (default: 0.25)
    ksp_opts : dict, optional
        PETSc solver options
    """
    if beta <= 0 or beta >= 1:
        raise ValueError(f"beta must be in (0, 1), got beta={beta}")
    
    print(f"=== Sinc Solver (β={beta}, Neumann BC) ===")

    # Extract solver options
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    max_it = 500
    
    if ksp_opts:
        if "ksp_type" in ksp_opts: ksp_type = ksp_opts["ksp_type"]
        if "pc_type" in ksp_opts: pc_type = ksp_opts["pc_type"]
        if "ksp_rtol" in ksp_opts: rtol = float(ksp_opts["ksp_rtol"])
        if "ksp_atol" in ksp_opts: rtol = float(ksp_opts["ksp_atol"])
        if "max_it" in ksp_opts: max_it = int(ksp_opts["max_it"])

    # Apply sinc resolvent method: u ≈ B^{-β} b
    u_vec = sinc_apply(
        B, M, b, beta, k=k, ksp_type=ksp_type, pc_type=pc_type, 
        rtol=rtol, max_it=max_it
    )
    
    # Convert to dolfinx Function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array
    u_h.x.scatter_forward()
    
    # Create dummy KSP for compatibility
    ksp = make_ksp(B, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, max_it=max_it)
    
    print(f"Sinc solver completed (β={beta})")
    return u_h, ksp

def sinc_solve_dirichlet(B, M, b, V, bc, beta=0.5, k=0.25, ksp_opts=None):
    """
    Sinc-based solver for fractional eigenvalue problems with Dirichlet boundary conditions.
    
    Solves: B^β u = b using sinc resolvent method.
    Properly handles Dirichlet boundary conditions by restricting to free DOFs only.
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix (will be modified to B^β)
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        RHS vector
    V : fem.FunctionSpace
        Function space
    bc : fem.DirichletBC
        Dirichlet boundary condition
    beta : float
        Fractional power (0 < beta < 1)
    k : float, optional
        Sinc step size (default: 0.25)
    ksp_opts : dict, optional
        PETSc solver options
    """
    if beta <= 0 or beta >= 1:
        raise ValueError(f"beta must be in (0, 1), got beta={beta}")
    
    print(f"=== Sinc Solver Dirichlet (β={beta}) ===")

    # Extract solver options
    k = 0.25  # Sinc step size
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    max_it = 500
    
    if ksp_opts:
        if "ksp_type" in ksp_opts: ksp_type = ksp_opts["ksp_type"]
        if "pc_type" in ksp_opts: pc_type = ksp_opts["pc_type"]
        if "ksp_rtol" in ksp_opts: rtol = float(ksp_opts["ksp_rtol"])
        if "ksp_atol" in ksp_opts: rtol = float(ksp_opts["ksp_atol"])
        if "max_it" in ksp_opts: max_it = int(ksp_opts["max_it"])

    # Use the new Dirichlet-specific solver
    u_h = sinc_apply_dirichlet_func(
        V, B, M, b, beta, bc=bc, k=k, ksp_type=ksp_type, 
        pc_type=pc_type, rtol=rtol, max_it=max_it
    )
    
    # Create dummy KSP for compatibility
    ksp = make_ksp(B, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, max_it=max_it)
    
    print(f"Sinc solver Dirichlet completed (β={beta})")
    return u_h, ksp

def sinc_solve_unified(B, M, b, V, bc=None, beta=0.5, k=0.25, ksp_opts=None):
    """
    Unified Sinc-based solver for fractional eigenvalue problems.
    
    Automatically handles both Neumann and Dirichlet boundary conditions:
    - If bc is None: Uses Neumann boundary conditions
    - If bc is provided: Uses Dirichlet boundary conditions
    
    Solves: B^β u = b using sinc resolvent method.
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix (will be modified to B^β)
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        RHS vector
    V : fem.FunctionSpace
        Function space
    bc : fem.DirichletBC or None
        Dirichlet boundary condition (None for Neumann BC)
    beta : float
        Fractional power (0 < beta < 1)
    k : float, optional
        Sinc step size (default: 0.25)
    ksp_opts : dict, optional
        PETSc solver options
    
    Returns:
    --------
    u_h : fem.Function
        Solution function
    ksp : PETSc.KSP
        KSP solver object
    """
    if beta <= 0 or beta >= 1:
        raise ValueError(f"beta must be in (0, 1), got beta={beta}")
    
    # Determine boundary condition type and solve
    if bc is None:
        print(f"=== Unified Sinc Solver (β={beta}, Neumann BC) ===")
        # Use the existing Neumann solver
        return sinc_solve(B, M, b, V, beta=beta, ksp_opts=ksp_opts)
    else:
        print(f"=== Unified Sinc Solver (β={beta}, Dirichlet BC) ===")
        # Use the existing Dirichlet solver
        return sinc_solve_dirichlet(B, M, b, V, bc, beta=beta, ksp_opts=ksp_opts)