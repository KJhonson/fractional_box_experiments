# sinc_solver.py - Organized Sinc-based solver library for FEM problems
# This is a refactored version of solver.py with minimal changes to preserve behavior
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from math import ceil, floor, pi
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
# SHARED HELPER FUNCTIONS
# ============================================================================

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
             atol: float = 0.0,
             max_it: int = 200) -> PETSc.KSP:
    """
    Build a PETSc KSP with common sensible defaults.
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    return ksp

def _extract_ksp_opts(ksp_opts: Optional[dict], defaults: Optional[dict] = None) -> dict:
    """
    Extract KSP options from user dict with defaults.
    Returns dict with keys: ksp_type, pc_type, rtol, atol, max_it
    """
    if defaults is None:
        defaults = {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10, "atol": 0.0, "max_it": 500}
    
    opts = defaults.copy()
    if ksp_opts:
        if "ksp_type" in ksp_opts:
            opts["ksp_type"] = ksp_opts["ksp_type"]
        if "pc_type" in ksp_opts:
            opts["pc_type"] = ksp_opts["pc_type"]
        if "ksp_rtol" in ksp_opts:
            opts["rtol"] = float(ksp_opts["ksp_rtol"])
        elif "rtol" in ksp_opts:
            opts["rtol"] = float(ksp_opts["rtol"])
        if "ksp_atol" in ksp_opts:
            opts["atol"] = float(ksp_opts["ksp_atol"])
        elif "atol" in ksp_opts:
            opts["atol"] = float(ksp_opts["atol"])
        if "max_it" in ksp_opts:
            opts["max_it"] = int(ksp_opts["max_it"])
    return opts

def _make_ksp_from_opts(A: PETSc.Mat, ksp_opts: Optional[dict]) -> PETSc.KSP:
    """Create KSP from options dict (shared helper)."""
    opts = _extract_ksp_opts(ksp_opts)
    return make_ksp(A, ksp_type=opts["ksp_type"], pc_type=opts["pc_type"],
                    rtol=opts["rtol"], atol=opts["atol"], max_it=opts["max_it"])

def _vec_to_function(vec: PETSc.Vec, V: fem.FunctionSpace) -> fem.Function:
    """Convert PETSc vector to dolfinx Function (shared helper)."""
    u_h = fem.Function(V)
    u_h.x.array[:] = vec.array
    u_h.x.scatter_forward()
    return u_h

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
    mask = np.ones(n, dtype=bool)
    mask[bdy] = False
    free = np.flatnonzero(mask).astype(np.int32)
    IS_f = PETSc.IS().createGeneral(free, comm=A.getComm())
    IS_c = PETSc.IS().createGeneral(bdy, comm=A.getComm())
    return IS_f, IS_c

def _apply_binvm_power(B: PETSc.Mat, M: PETSc.Mat, v: PETSc.Vec, p: int,
                       ksp_opts: Optional[dict], bc: Optional[fem.DirichletBC] = None) -> PETSc.Vec:
    """
    Apply (B^{-1} M)^p to vector v.
    Supports both Neumann (bc=None) and Dirichlet cases.
    """
    if p <= 0:
        out = v.duplicate()
        out.setArray(v.getArray().copy())
        return out
    
    if bc is None:
        # Neumann case
        ksp_B = _make_ksp_from_opts(B, ksp_opts)
        ksp_B.setOperators(B)
        y = v.duplicate()
        y.setArray(v.getArray().copy())
        tmp = v.duplicate()
        for _ in range(p):
            # tmp = M * y
            M.mult(y, tmp)
            # solve B x = tmp
            x = v.duplicate()
            x.set(0.0)
            ksp_B.setOperators(B)
            ksp_B.solve(tmp, x)
            y = x
        return y
    else:
        # Dirichlet case: work on free DOFs only
        IS_f, _ = _dirichlet_free_is(B, bc)
        B_ff = B.createSubMatrix(IS_f, IS_f)
        M_ff = M.createSubMatrix(IS_f, IS_f)
        v_f = v.getSubVector(IS_f)
        
        ksp_B = _make_ksp_from_opts(B_ff, ksp_opts)
        ksp_B.setOperators(B_ff)
        y_f = v_f.duplicate()
        y_f.setArray(v_f.getArray().copy())
        tmp_f = v_f.duplicate()
        for _ in range(p):
            M_ff.mult(y_f, tmp_f)
            x_f = v_f.duplicate()
            x_f.set(0.0)
            ksp_B.setOperators(B_ff)
            ksp_B.solve(tmp_f, x_f)
            y_f = x_f
        
        # Scatter back to full vector
        n, _ = B.getSize()
        y_full = PETSc.Vec().createMPI(n, comm=B.comm)
        y_full.set(0.0)
        y_full.setValues(IS_f.indices, y_f.getArray())
        y_full.assemble()
        return y_full

# ============================================================================
# SINC SOLVER FOR FRACTIONAL PROBLEMS (0 < β < 1)
# ============================================================================
# Original implementation: First version for fractional powers with Neumann BC

# ------- core: apply B^{-beta} to b via sinc-resolvent sum ----------
def sinc_neum(
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

# ---------- core: B^{-β} b with homogeneous Dirichlet --------------
def sinc_dir(
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
def sinc_dir_func(
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
    u_vec = sinc_dir(B, M, b, beta, bc=bc, **kwargs)
    return _vec_to_function(u_vec, V)

# ============================================================================
# UNIFIED SINC SOLVER (handles all beta > 0)
# ============================================================================

def sinc_solver(B, M, b, V, bc=None, beta=0.5, k=0.25, ksp_opts=None):
    """
    Unified Sinc-based solver for all beta > 0.
    
    Automatically handles both Neumann and Dirichlet boundary conditions:
    - If bc is None: Uses Neumann boundary conditions
    - If bc is provided: Uses Dirichlet boundary conditions
    
    Handles all beta > 0:
    - beta = 1.0: Standard case (B u = b) - uses solve_petsc
    - 0 < beta < 1: Fractional case - uses sinc quadrature
    - beta >= 1 (integer): Uses (B^{-1} M)^beta M^{-1} b
    - beta > 1 (mixed): Uses sinc for fractional part + integer power
    
    Parameters:
    -----------
    B : PETSc.Mat
        Stiffness matrix
    M : PETSc.Mat
        Mass matrix
    b : PETSc.Vec
        RHS vector
    V : fem.FunctionSpace
        Function space
    bc : fem.DirichletBC or None
        Dirichlet boundary condition (None for Neumann BC)
    beta : float
        Power of the operator (must be > 0)
    k : float, optional
        Sinc step size (default: 0.25, only used for fractional parts)
    ksp_opts : dict, optional
        PETSc solver options
    
    Returns:
    --------
    u_h : fem.Function
        Solution function
    ksp : PETSc.KSP
        KSP solver object
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got beta={beta}")
    
    # Case 1: Standard case (beta = 1.0)
    if abs(beta - 1.0) < 1e-14:
        return solve_petsc(B, b, V, ksp_opts)
    
    # Case 2: Fractional case (0 < beta < 1) - use sinc
    elif 0 < beta < 1:
        # Extract solver options
        opts = _extract_ksp_opts(ksp_opts)
        
        if bc is None:
            print(f"=== Sinc Solver (β={beta}, Neumann BC) ===")
            # Apply sinc resolvent method: u ≈ B^{-β} b
            u_vec = sinc_neum(
                B, M, b, beta, k=k, ksp_type=opts["ksp_type"], pc_type=opts["pc_type"], 
                rtol=opts["rtol"], max_it=opts["max_it"]
            )
            u_h = _vec_to_function(u_vec, V)
            ksp = make_ksp(B, ksp_type=opts["ksp_type"], pc_type=opts["pc_type"], 
                          rtol=opts["rtol"], atol=opts["atol"], max_it=opts["max_it"])
            print(f"Sinc solver completed (β={beta})")
            return u_h, ksp
        else:
            print(f"=== Sinc Solver (β={beta}, Dirichlet BC) ===")
            # Standard default step; caller may pass k (e.g. mesh-based dynamic_k).
            if k is None:
                k = 0.25
            u_h = sinc_dir_func(
                V, B, M, b, beta, bc=bc, k=k, ksp_type=opts["ksp_type"],
                pc_type=opts["pc_type"], rtol=opts["rtol"], max_it=opts["max_it"]
            )
            ksp = make_ksp(B, ksp_type=opts["ksp_type"], pc_type=opts["pc_type"], 
                          rtol=opts["rtol"], atol=opts["atol"], max_it=opts["max_it"])
            print(f"Sinc solver Dirichlet completed (β={beta})")
            return u_h, ksp
    
    # Case 3: Integer beta >= 1
    elif beta >= 1 and abs(beta - round(beta)) < 1e-14:
        p = int(round(beta))
        print(f"=== Integer Power Solver (β={p}) ===")
        if bc is None:
            # Neumann: (B^{-1} M)^p M^{-1} b
            ksp_M = _make_ksp_from_opts(M, ksp_opts)
            ksp_M.setOperators(M)
            y0 = b.duplicate()
            y0.set(0.0)
            ksp_M.solve(b, y0)
            y = _apply_binvm_power(B, M, y0, p, ksp_opts, bc=None)
            u_h = _vec_to_function(y, V)
            ksp = _make_ksp_from_opts(B, ksp_opts)
        else:
            # Dirichlet: work on free DOFs
            IS_f, _ = _dirichlet_free_is(B, bc)
            B_ff = B.createSubMatrix(IS_f, IS_f)
            M_ff = M.createSubMatrix(IS_f, IS_f)
            b_f = b.getSubVector(IS_f)
            ksp_M = _make_ksp_from_opts(M_ff, ksp_opts)
            ksp_M.setOperators(M_ff)
            y0_f = b_f.duplicate()
            y0_f.set(0.0)
            ksp_M.solve(b_f, y0_f)
            n, _ = B.getSize()
            y0_full = PETSc.Vec().createMPI(n, comm=B.comm)
            y0_full.set(0.0)
            y0_full.setValues(IS_f.indices, y0_f.getArray())
            y0_full.assemble()
            y_full = _apply_binvm_power(B, M, y0_full, p, ksp_opts, bc=bc)
            u_h = _vec_to_function(y_full, V)
            ksp = _make_ksp_from_opts(B, ksp_opts)
        return u_h, ksp
    
    # Case 4: Mixed case (beta > 1, not integer)
    elif beta > 1:
        p = int(np.floor(beta))
        frac = float(beta - p)
        print(f"=== Mixed Power Solver (β={beta} = {p} + {frac}) ===")
        if bc is None:
            # Neumann: use sinc for fractional part, then integer power
            u_frac = sinc_neum(B, M, b, frac, k=k)
            y = _apply_binvm_power(B, M, u_frac, p, ksp_opts, bc=None)
        else:
            # Dirichlet: use sinc for fractional part, then integer power
            u_full_frac = sinc_dir(B, M, b, frac, bc=bc, k=k)
            y = _apply_binvm_power(B, M, u_full_frac, p, ksp_opts, bc=bc)
        u_h = _vec_to_function(y, V)
        ksp = _make_ksp_from_opts(B, ksp_opts)
        return u_h, ksp
    
    else:
        raise ValueError(f"Invalid beta value: {beta}. Must be > 0.")