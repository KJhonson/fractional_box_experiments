# %%

"""
Fractional Dirichlet problem experiment with a validated spectral reference.
Problem:
    Find u such that:
        (-d²/dx²)^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0            (Dirichlet boundary conditions)

    where:
        f(x) = x^(-α)
        α ∈ (0, 1/2) (singularity exponent)
        β > 0 (fractional power)

    The reference solution uses exact sine coefficients
        f_n = sqrt(2) ∫_0^1 x^(-α) sin(nπx) dx
    computed with the substitution x = t^(1/(1-α)) (integrand without endpoint singularity).
    A legacy DST-on-a-grid helper is kept only for optional diagnostics.
"""

import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import matplotlib.pyplot as plt
import sys
from functools import lru_cache
from pathlib import Path

try:
    from scipy import integrate as scipy_integrate
except ImportError:  # pragma: no cover
    scipy_integrate = None
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.norm import get_norm
from utils.sinc_solver import sinc_solver
from utils.loads import assemble_rhs_dual_fvm

# --- L2 reference validation quadrature (NumPy 2.0 name) ---
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

# Cache: exact sine coefficients f_n for f(x)=x^{-α} (same α, N reused across β)
_COEFF_CACHE = {}


def clear_xalpha_coefficient_cache():
    _COEFF_CACHE.clear()


def compute_xalpha_sine_coefficients(
    alpha,
    N_modes,
    method="gauss",
    quad_order=None,
    block_size=512,
    epsabs=1e-12,
    epsrel=1e-12,
    gauss_composite_divisor=0,
):
    """
    Sine coefficients f_n = sqrt(2) ∫_0^1 x^{-α} sin(nπx) dx, n = 1..N_modes.

    Uses x = t^{1/(1-α)} so x^{-α} dx = dt/(1-α) and
    f_n = sqrt(2)/(1-α) ∫_0^1 sin(nπ t^{1/(1-α)}) dt.

    For method=\"gauss\", the integrand is highly oscillatory for large n. A single
    Gauss rule on [0, 1] with modest ``quad_order`` gives wrong f_n and a spurious
    high-frequency ringing in the summed u(x). Set ``gauss_composite_divisor > 0``
    to use composite Gauss on subintervals (about ceil(n/divisor) panels per mode).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    if abs(alpha - 1.0) < 1e-14:
        raise ValueError("alpha ≈ 1 is not handled here (use a log-based formulation).")
    N_modes = int(N_modes)
    if N_modes < 1:
        raise ValueError("N_modes must be >= 1.")

    qo = int(quad_order) if quad_order is not None else 0
    gcd = int(gauss_composite_divisor)
    cache_key = (
        round(float(alpha), 14),
        N_modes,
        str(method),
        qo,
        int(block_size),
        float(epsabs),
        float(epsrel),
        gcd,
    )
    if cache_key in _COEFF_CACHE:
        return _COEFF_CACHE[cache_key].copy()

    one_m_a = 1.0 - alpha
    gamma = 1.0 / one_m_a
    pref = np.sqrt(2.0) / one_m_a
    f_out = np.zeros(N_modes, dtype=np.float64)

    if method == "quad":
        if scipy_integrate is None:
            raise ImportError("scipy.integrate is required for method='quad'.")

        for n in range(1, N_modes + 1):
            val, _ = scipy_integrate.quad(
                lambda t, nn: np.sin(nn * np.pi * (t**gamma)),
                0.0,
                1.0,
                args=(n,),
                epsabs=epsabs,
                epsrel=epsrel,
            )
            f_out[n - 1] = pref * val

    elif method == "gauss":
        order = int(quad_order) if quad_order is not None else 256
        nodes11, weights11 = leggauss(order)
        if gcd <= 0:
            if N_modes > 6 * order and MPI.COMM_WORLD.rank == 0:
                warnings.warn(
                    "Spectral coefficients: single-interval Gauss with "
                    f"N_modes={N_modes} and quad_order={order} is unreliable (oscillatory "
                    "integrand sin(nπ t^γ)). Set reference_gauss_composite_divisor (e.g. 32–64) "
                    "in HYPERPARAMETERS to remove spurious ringing in the reference curve.",
                    UserWarning,
                    stacklevel=2,
                )
            t_nodes = 0.5 * (nodes11 + 1.0)
            w = 0.5 * weights11
            t_pow = t_nodes**gamma
            bs = int(block_size)
            for start in range(0, N_modes, bs):
                end = min(start + bs, N_modes)
                nc = end - start
                n_vec = np.arange(start + 1, end + 1, dtype=np.float64).reshape(1, nc)
                sin_block = np.sin(np.pi * t_pow[:, np.newaxis] * n_vec)
                integrals = w @ sin_block
                f_out[start:end] = pref * integrals
        else:
            for n in range(1, N_modes + 1):
                n_pan = max(1, int(np.ceil(n / float(gcd))))
                edges = np.linspace(0.0, 1.0, n_pan + 1)
                acc = 0.0
                for ip in range(n_pan):
                    a, b = edges[ip], edges[ip + 1]
                    half = 0.5 * (b - a)
                    mid = 0.5 * (b + a)
                    tloc = half * nodes11 + mid
                    acc += half * float(np.dot(weights11, np.sin(n * np.pi * (tloc**gamma))))
                f_out[n - 1] = pref * acc
    else:
        raise ValueError("method must be 'quad' or 'gauss'.")

    _COEFF_CACHE[cache_key] = f_out.copy()
    return f_out


class XAlphaSpectralReference:
    """
    High-accuracy spectral reference for (-d²/dx²)^β u = x^{-α}, u(0)=u(1)=0:
        u(x) = sum_{n=1}^{N_modes} (nπ)^{-2β} f_n sqrt(2) sin(nπx),
    evaluated directly (no interpolation grid).
    """

    def __init__(
        self,
        alpha,
        beta,
        N_modes,
        coeff_method="gauss",
        quad_order=None,
        coeff_block_size=512,
        epsabs=1e-12,
        epsrel=1e-12,
        eval_block_size=512,
        gauss_composite_divisor=0,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.N_modes = int(N_modes)
        self.coeff_method = str(coeff_method)
        self.eval_block_size = int(eval_block_size)

        self.f_n = compute_xalpha_sine_coefficients(
            alpha,
            N_modes,
            method=coeff_method,
            quad_order=quad_order,
            block_size=coeff_block_size,
            epsabs=epsabs,
            epsrel=epsrel,
            gauss_composite_divisor=gauss_composite_divisor,
        )
        n_idx = np.arange(1, self.N_modes + 1, dtype=np.float64)
        self._mode_amp = (n_idx * np.pi) ** (-2.0 * self.beta) * self.f_n * np.sqrt(2.0)

    def evaluate_array(self, x_interior_1d):
        """Evaluate u on points strictly in (0, 1); shape (K,) -> (K,)."""
        x = np.asarray(x_interior_1d, dtype=np.float64).ravel()
        out = np.zeros_like(x)
        bs = self.eval_block_size
        start_n = 0
        while start_n < self.N_modes:
            end_n = min(start_n + bs, self.N_modes)
            n_slice = np.arange(start_n + 1, end_n + 1, dtype=np.float64)
            amp = self._mode_amp[start_n:end_n]
            sines = np.sin(np.pi * np.outer(x, n_slice))
            out += sines @ amp
            start_n = end_n
        return out

    def __call__(self, x):
        """DOLFINx-style coordinates x with x[0] giving the spatial coordinate(s)."""
        x_vals = x[0]
        x_vals = np.atleast_1d(x_vals).astype(np.float64, copy=False)
        out = np.zeros_like(x_vals, dtype=np.float64)
        inside = (x_vals > 0.0) & (x_vals < 1.0)
        if np.any(inside):
            out[inside] = self.evaluate_array(x_vals[inside])
        if x_vals.size == 1:
            return float(out[0])
        return out


def resolve_reference_with_validation(
    alpha,
    beta,
    N_ref_initial,
    N_ref_max,
    reference_tol,
    validation_grid_size,
    coeff_method="gauss",
    quad_order=None,
    coeff_block_size=512,
    epsabs=1e-12,
    epsrel=1e-12,
    eval_block_size=512,
    gauss_composite_divisor=0,
):
    """
    Increase truncation N until ||u_N - u_{2N}||_{L2(0,1)} <= reference_tol or N reaches N_ref_max.
    Returns (reference instance with accepted truncation, info dict).
    """
    N = int(N_ref_initial)
    N_max = int(N_ref_max)
    if N < 1 or N_max < N:
        raise ValueError("Require 1 <= N_ref_initial <= N_ref_max.")

    xg = np.linspace(0.0, 1.0, int(validation_grid_size) + 2)[1:-1]
    info = {
        "beta": float(beta),
        "alpha": float(alpha),
        "N_ref_used": None,
        "N_pair_compared": None,
        "coeff_method": coeff_method,
        "validation_L2_diff": None,
        "reference_tol": float(reference_tol),
        "tolerance_met": False,
        "warnings": [],
    }

    while True:
        N2 = min(2 * N, N_max)
        ref_N = XAlphaSpectralReference(
            alpha,
            beta,
            N_modes=N,
            coeff_method=coeff_method,
            quad_order=quad_order,
            coeff_block_size=coeff_block_size,
            epsabs=epsabs,
            epsrel=epsrel,
            eval_block_size=eval_block_size,
            gauss_composite_divisor=gauss_composite_divisor,
        )
        ref_2N = XAlphaSpectralReference(
            alpha,
            beta,
            N_modes=N2,
            coeff_method=coeff_method,
            quad_order=quad_order,
            coeff_block_size=coeff_block_size,
            epsabs=epsabs,
            epsrel=epsrel,
            eval_block_size=eval_block_size,
            gauss_composite_divisor=gauss_composite_divisor,
        )
        uN = ref_N.evaluate_array(xg)
        u2 = ref_2N.evaluate_array(xg)
        diff_l2 = float(np.sqrt(_trapz((uN - u2) ** 2, xg)))

        info["validation_L2_diff"] = diff_l2
        info["N_pair_compared"] = (N, N2)

        if (
            MPI.COMM_WORLD.rank == 0
            and diff_l2 > reference_tol
            and N2 < N_max
        ):
            print(
                f"  Spectral reference: ||u_{N}-u_{N2}||_L2={diff_l2:.3e} > tol={reference_tol:.3e}; "
                f"increasing truncation to N={N2}."
            )

        if diff_l2 <= reference_tol:
            info["N_ref_used"] = N
            info["tolerance_met"] = True
            return ref_N, info

        if N2 >= N_max:
            info["N_ref_used"] = N_max
            info["tolerance_met"] = False
            w = (
                f"Validated spectral reference: ||u_{N} - u_{N2}||_L2 = {diff_l2:.3e} > tol={reference_tol:.3e}. "
                f"Using N={N_max} modes (increase N_ref_max or reference_tol if needed). "
                f"Small β may need more modes."
            )
            info["warnings"].append(w)
            if MPI.COMM_WORLD.rank == 0:
                print(f"  WARNING: {w}")
            return ref_2N, info

        N = N2


def dynamic_k(beta, dofs):
    """
    Calculate dynamic sinc parameter k based on beta and number of degrees of freedom.
    
    Formula: k = -pi^2 / (4 * beta * dynamic_k_beta_factor * log(h))
    where h is derived from the number of DOFs.
    
    For a 1D interval [0,1] with Dirichlet BC:
    - With n elements, we have approximately n-1 free DOFs
    - Mesh size h ≈ 1/n ≈ 1/(dofs+1)
    
    Parameters:
    -----------
    beta : float
        Fractional power (> 0)
    dofs : int
        Number of degrees of freedom (free DOFs)
    
    Returns:
    --------
    k : float
        Sinc step size parameter
    """
    # For 1D interval [0,1] with Dirichlet BC: h ≈ 1/(dofs+1)
    # More precisely: with n elements, h = 1/n, and free DOFs ≈ n-1
    # So: n ≈ dofs + 1, hence h ≈ 1/(dofs + 1)
    h = 1.0 / (dofs + 1)
    
    # k = -pi^2 / (4 * beta * dynamic_k_beta_factor * log(h))
    # Note: log(h) is negative since h < 1, so k will be positive
    log_h = np.log(h)
    k = -np.pi**2 / (10 * beta * float(dynamic_k_beta_factor) * log_h)
    
    return k


def solve_fractional_dirichlet_on_mesh(n_elements, beta=0.5, kappa_func=None, f_source_func=None, use_dual_fvm=False, use_fvm_exact_xalpha=False, use_fem_exact_xalpha=False, alpha=None, k=None):
    """
    Solve fractional Dirichlet problem on a mesh.
    
    Parameters:
    -----------
    n_elements : int
        Number of elements in the mesh
    beta : float
        Fractional power
    kappa_func : callable, optional
        Mass term coefficient function
    f_source_func : callable, optional
        Source function
    use_dual_fvm : bool
        Use dual FVM RHS assembly
    use_fvm_exact_xalpha : bool
        Use exact FVM load for f(x) = x^(-alpha)
    use_fem_exact_xalpha : bool
        Use exact FEM load for f(x) = x^(-alpha)
    alpha : float, optional
        Exponent for x^(-alpha) source (required if use_fvm_exact_xalpha or use_fem_exact_xalpha=True)
    k : float, optional
        Sinc step size parameter. If None, uses dynamic_k based on beta and DOFs
    """
    mesh_local = dmesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, 1.0])
    V_local = fem.functionspace(mesh_local, ("CG", 1))
    
    x = ufl.SpatialCoordinate(mesh_local)
    if kappa_func is None:
        kappa_val = fem.Constant(mesh_local, 0.0)  # No mass term for this problem
    else:
        kappa_val = kappa_func(x)
    
    u = ufl.TrialFunction(V_local)
    v = ufl.TestFunction(V_local)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + kappa_val * ufl.inner(u, v) * ufl.dx
    m = ufl.inner(u, v) * ufl.dx
    
    boundary_facets = dmesh.locate_entities_boundary(mesh_local, mesh_local.topology.dim - 1, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
    boundary_dofs = fem.locate_dofs_topological(V_local, mesh_local.topology.dim - 1, boundary_facets)
    bc = fem.dirichletbc(0.0, boundary_dofs, V_local)
    
    B = petsc.assemble_matrix(fem.form(a), bcs=[bc])
    B.assemble()
    
    M = petsc.assemble_matrix(fem.form(m), bcs=[bc])
    M.assemble()
    
    # Choose RHS assembly method
    if use_fem_exact_xalpha:
        # Use FEM-exact load for f(x) = x^(-alpha)
        if alpha is None:
            raise ValueError("alpha must be provided when using use_fem_exact_xalpha")
        x_coords = mesh_local.geometry.x[:, 0]
        F_numpy = assemble_fem_load_xalpha(x_coords, alpha)
        
        # Map vertex values to dofs
        Vdm = V_local.dofmap
        owned = Vdm.index_map.size_local
        ghosts = Vdm.index_map.num_ghosts
        v2d = fem.locate_dofs_topological(V_local, 0, np.arange(len(x_coords), dtype=np.int32))
        dof_load_local = np.zeros(owned + ghosts, dtype=np.float64)
        for local_vert, dof in enumerate(v2d):
            if local_vert < len(F_numpy):
                dof_load_local[dof] += F_numpy[local_vert]
        
        # Create PETSc vector consistent with V using a dummy form
        dummy_form = v * ufl.dx
        b = petsc.assemble_vector(fem.form(dummy_form))
        b.setValues(np.arange(owned, dtype=np.int32), dof_load_local[:owned])
        b.assemblyBegin()
        b.assemblyEnd()
    elif use_fvm_exact_xalpha:
        # Use FVM-exact load for f(x) = x^(-alpha)
        if alpha is None:
            raise ValueError("alpha must be provided when using use_fvm_exact_xalpha")
        x_coords = mesh_local.geometry.x[:, 0]
        F_numpy = assemble_fv_load_xalpha(x_coords, alpha)
        
        # Map vertex values to dofs (similar to assemble_rhs_dual_fvm_exact)
        Vdm = V_local.dofmap
        owned = Vdm.index_map.size_local
        ghosts = Vdm.index_map.num_ghosts
        v2d = fem.locate_dofs_topological(V_local, 0, np.arange(len(x_coords), dtype=np.int32))
        dof_load_local = np.zeros(owned + ghosts, dtype=np.float64)
        for local_vert, dof in enumerate(v2d):
            if local_vert < len(F_numpy):
                dof_load_local[dof] += F_numpy[local_vert]
        
        # Create PETSc vector consistent with V using a dummy form
        dummy_form = v * ufl.dx
        b = petsc.assemble_vector(fem.form(dummy_form))
        b.setValues(np.arange(owned, dtype=np.int32), dof_load_local[:owned])
        b.assemblyBegin()
        b.assemblyEnd()
    elif use_dual_fvm:
        # Use dual FVM RHS
        b = assemble_rhs_dual_fvm(V_local, f_source_func if f_source_func else lambda x: np.sin(np.pi * x[0]), quad_degree=3)
    else:
        # Use standard FEM RHS
        f_func_local = fem.Function(V_local)
        f_func_local.interpolate(f_source_func if f_source_func else lambda x: np.sin(np.pi * x[0]))
        L = f_func_local * v * ufl.dx
        b = petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    
    fem.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    
    fem.set_bc(b, [bc])
    
    if k is None:
        free_dofs = V_local.dofmap.index_map.size_local - len(boundary_dofs)
        k = dynamic_k(beta, free_dofs)
    
    u_h_local, _ = sinc_solver(B, M, b, V_local, bc=bc, beta=beta, k=k)
    
    return mesh_local, V_local, u_h_local


def f_xalpha(x, alpha):
    """Source function: f(x) = x^(-alpha)"""
    x_vals = x[0]
    eps = 1e-12  # to avoid division by zero
    safe_x = np.maximum(x_vals, eps)
    return safe_x ** (-alpha)


def assemble_fem_load_xalpha(x, alpha):
    """
    Compute FEM load vector F_i = ∫ x^{-α} φ_i(x) dx
    for linear hat functions on a nonuniform 1D mesh.
    
    Parameters
    ----------
    x : array_like
        Mesh nodes [x0, x1, ..., xN], length N+1.
    alpha : float
        Exponent in f(x) = x^{-alpha}, with 0 < alpha < 1.

    Returns
    -------
    F : np.ndarray, shape (N+1,)
        Nodal load entries; boundary values are overwritten by Dirichlet BCs in the solver.
    """
    x = np.asarray(x)
    N = len(x) - 1
    F = np.zeros(N + 1)
    
    # Handle special cases for alpha
    if abs(alpha - 1.0) < 1e-10:
        # For alpha = 1, integrals involve ln(x)
        for i in range(1, N):
            hL = x[i] - x[i-1]
            hR = x[i+1] - x[i] if i < N else 0
            
            # left integral
            if hL > 0:
                IL = (1/hL) * (
                    x[i] * (np.log(max(x[i], 1e-15)) - np.log(max(x[i-1], 1e-15)))
                    - (x[i] - x[i-1])
                )
            else:
                IL = 0.0
            
            # right integral (only if not last internal node)
            if i < N and hR > 0:
                IR = (1/hR) * (
                    x[i+1] * (np.log(max(x[i+1], 1e-15)) - np.log(max(x[i], 1e-15)))
                    - (x[i+1] - x[i])
                )
            else:
                IR = 0.0
            
            F[i] = IL + IR
    else:
        # Standard case: alpha != 1
        one_minus_alpha = 1.0 - alpha
        two_minus_alpha = 2.0 - alpha
        
        for i in range(1, N):
            hL = x[i] - x[i-1]
            hR = x[i+1] - x[i] if i < N else 0
            
            # left integral
            if hL > 0:
                IL = (1/hL) * (
                    (x[i]**two_minus_alpha - x[i-1]**two_minus_alpha) / two_minus_alpha
                    - x[i-1] * (x[i]**one_minus_alpha - x[i-1]**one_minus_alpha) / one_minus_alpha
                )
            else:
                IL = 0.0
            
            # right integral (only if not last internal node)
            if i < N and hR > 0:
                IR = (1/hR) * (
                    x[i+1] * (x[i+1]**one_minus_alpha - x[i]**one_minus_alpha) / one_minus_alpha
                    - (x[i+1]**two_minus_alpha - x[i]**two_minus_alpha) / two_minus_alpha
                )
            else:
                IR = 0.0
            
            F[i] = IL + IR
    
    return F


def assemble_fv_load_xalpha(x, alpha):
    """
    Assemble the exact FVM load vector for f(x) = x^(-alpha)
    on an arbitrary non-uniform 1D mesh.
    
    Uses the exact formula:
        F_i = ∫_{x_{i-1/2}}^{x_{i+1/2}} x^(-alpha) dx 
            = (x_{i+1/2}^{1-alpha} - x_{i-1/2}^{1-alpha}) / (1 - alpha)
    
    Parameters
    ----------
    x : array_like
        1D array of node coordinates [x0, x1, ..., xN]
    alpha : float
        Exponent for source function f(x) = x^(-alpha), where 0 < alpha < 1/2

    Returns
    -------
    F : ndarray
        Global load vector of length N+1 (integrals over dual cells)
    """
    N = len(x) - 1
    F = np.zeros(N + 1)
    
    # midpoints between nodes
    xm = 0.5 * (x[:-1] + x[1:])
    
    # build dual cell boundaries
    x_dual = np.zeros(N + 2)
    x_dual[0] = x[0]
    x_dual[1:-1] = xm
    x_dual[-1] = x[-1]
    
    # Handle the case alpha = 1 (though not expected for this problem)
    if abs(alpha - 1.0) < 1e-10:
        # For alpha = 1, integral is ln(x)
        for i in range(N + 1):
            left = max(x_dual[i], 1e-15)  # avoid log(0)
            right = max(x_dual[i + 1], 1e-15)
            if right > left:
                F[i] = np.log(right) - np.log(left)
    else:
        # Standard case: F_i = (x_{i+1/2}^{1-alpha} - x_{i-1/2}^{1-alpha}) / (1 - alpha)
        one_minus_alpha = 1.0 - alpha
        for i in range(N + 1):
            left = max(x_dual[i], 0.0)
            right = max(x_dual[i + 1], 0.0)
            if right > left:
                F[i] = (right**one_minus_alpha - left**one_minus_alpha) / one_minus_alpha
    
    return F


def dst_type1(x):
    """
    DST-I helper for reference_solution_dst_sampled_deprecated only.
    Builds a dense M×M sine matrix — keep M modest.

    For a length-M vector x:
        y_k = 2 * sum_{j=1}^{M} x_j * sin(pi * j * k / (M+1)),  k = 1..M.
    """
    M = len(x)
    k = np.arange(1, M + 1)
    j = np.arange(1, M + 1)
    
    # Transformation matrix: sin(pi * j * k / (M+1))
    j_mat = j[:, np.newaxis]  # shape (M, 1)
    k_mat = k[np.newaxis, :]   # shape (1, M)
    sin_matrix = np.sin(np.pi * j_mat * k_mat / (M + 1))
    
    # Apply transform with factor 2 to match SciPy
    y = 2.0 * (sin_matrix.T @ x)  # shape (M,)
    
    return y


def reference_solution_dst_sampled_deprecated(alpha, beta, M=500, N=None, x_eval=None):
    """
    DEPRECATED diagnostic only: samples f(x)=x^{-α} on a uniform grid and uses DST-I.
    Not singularity-aware; for x^{-α} with α near 1/2 prefer XAlphaSpectralReference + exact f_n.
    Warning: dst_type1 builds a dense M×M matrix — avoid large M.
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 500)[1:-1]
    x_eval = np.asarray(x_eval)

    if N is None or N > M:
        N = M

    j = np.arange(1, M + 1)
    x = j / (M + 1)
    dx = 1.0 / (M + 1)
    f_vals = x ** (-alpha)
    y = dst_type1(f_vals)
    n_vals = np.arange(1, N + 1)
    f_n = np.sqrt(2.0) * dx * 0.5 * y[:N]
    u_n = (n_vals * np.pi) ** (-2.0 * beta) * f_n
    Sx = np.sin(np.pi * np.outer(x_eval, n_vals))
    u_vals = np.sqrt(2.0) * (Sx @ u_n)

    return x_eval, u_vals


# %%
# ============================================================================
# HYPERPARAMETERS - Adjust these values to change experiment settings
# ============================================================================

# Mesh refinement levels
n_list_xalpha = [8, 16, 32, 64, 128, 256, 512]

# Fractional power values to test
beta_values_xalpha = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Singularity exponent for source function f(x) = x^(-alpha)
alpha_xalpha = 0.499  # Must satisfy: 0 < alpha < 1/2

# --- Spectral reference (cost vs accuracy) ---
# Fast default: fixed mode count, no N vs 2N doubling loop (much cheaper per β).
reference_use_adaptive_validation = False
N_ref_modes = 8000  # +1000 modes vs 1200: sharper Fourier tail (esp. small β)

# Adaptive mode (reference_use_adaptive_validation = True): increase N until
# ||u_N - u_{2N}||_L2 <= reference_tol or N_ref_max is reached.
N_ref_initial = 10000
N_ref_max = 13000
reference_tol = 1e-5
reference_validation_grid = 2048

reference_coeff_method = "gauss"  # "gauss" (default) or "quad" (SciPy; slow per mode)
reference_quad_order = 250  # GL nodes per subinterval (composite mode)
# Panels per mode ≈ ceil(n / divisor). Required for large N_modes (oscillatory ∫ sin(nπ t^γ) dt).
reference_gauss_composite_divisor = 40
reference_coeff_block_size = 512
reference_eval_block_size = 512

# Fine mesh for L2 error quadrature (numerical vs spectral reference)
n_fine_mesh = 10000

# Error norm type
norm_choice_xalpha = "L2"

# Only sinc-related knob here: scales β in dynamic_k (see function dynamic_k above).
# Standard value 1.0 gives k = -π² / (4 β log h).
dynamic_k_beta_factor = 1.0

# ============================================================================
# Run experiments
# ============================================================================

results_xalpha_fem_exact = {}
results_xalpha_fvm = {}
results_xalpha_fvm_exact = {}
reference_info_by_beta = {}
_spectral_ref_cache = {}


def _get_spectral_reference(beta):
    if beta not in _spectral_ref_cache:
        if reference_use_adaptive_validation:
            _spectral_ref_cache[beta] = resolve_reference_with_validation(
                alpha_xalpha,
                beta,
                N_ref_initial,
                N_ref_max,
                reference_tol,
                reference_validation_grid,
                coeff_method=reference_coeff_method,
                quad_order=reference_quad_order,
                coeff_block_size=reference_coeff_block_size,
                eval_block_size=reference_eval_block_size,
                gauss_composite_divisor=reference_gauss_composite_divisor,
            )
        else:
            ref = XAlphaSpectralReference(
                alpha_xalpha,
                beta,
                N_modes=int(N_ref_modes),
                coeff_method=reference_coeff_method,
                quad_order=reference_quad_order,
                coeff_block_size=reference_coeff_block_size,
                eval_block_size=reference_eval_block_size,
                gauss_composite_divisor=reference_gauss_composite_divisor,
            )
            info = {
                "beta": float(beta),
                "alpha": float(alpha_xalpha),
                "N_ref_used": int(N_ref_modes),
                "N_pair_compared": (int(N_ref_modes), int(N_ref_modes)),
                "coeff_method": reference_coeff_method,
                "validation_L2_diff": None,
                "reference_tol": float(reference_tol),
                "tolerance_met": None,
                "warnings": [],
                "fixed_truncation": True,
            }
            _spectral_ref_cache[beta] = (ref, info)
    return _spectral_ref_cache[beta]


for beta in beta_values_xalpha:
    spectral_ref, ref_info = _get_spectral_reference(beta)
    reference_info_by_beta[beta] = ref_info
    u_exact_func = spectral_ref

    if MPI.COMM_WORLD.rank == 0:
        N_a, N_b = ref_info["N_pair_compared"]
        print("\n" + "-" * 60)
        print("Spectral reference")
        print(f"  β             = {beta}")
        print(f"  α             = {alpha_xalpha}")
        print(f"  N_ref used    = {ref_info['N_ref_used']}")
        print(f"  coeff method  = {ref_info['coeff_method']}")
        if ref_info.get("fixed_truncation"):
            print(f"  mode          = fixed N={ref_info['N_ref_used']} (adaptive validation off)")
        else:
            print(f"  compared pair = (N={N_a}, N={N_b})")
            vd = ref_info["validation_L2_diff"]
            print(
                f"  ||u_N - u_2N||_L2 (validation grid) = {vd:.6e}"
                if vd is not None
                else "  ||u_N - u_2N||_L2 = n/a"
            )
            print(f"  reference_tol = {ref_info['reference_tol']:.6e}")
            print(f"  tolerance met = {ref_info['tolerance_met']}")

    meshf_xalpha = dmesh.create_interval(MPI.COMM_WORLD, n_fine_mesh, [0.0, 1.0])
    Vf_xalpha = fem.functionspace(meshf_xalpha, ("CG", 1))
    
    errors_xalpha_fem_exact = []
    errors_xalpha_fvm = []
    errors_xalpha_fvm_exact = []
    mesh_sizes_xalpha = []
    
    for n in n_list_xalpha:
        mesh_local_xalpha_fem_exact, _, u_h_local_xalpha_fem_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_fem_exact_xalpha=True, alpha=alpha_xalpha
        )
        mesh_local_xalpha_fvm, _, u_h_local_xalpha_fvm = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_dual_fvm=True
        )
        mesh_local_xalpha_fvm_exact, _, u_h_local_xalpha_fvm_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_fvm_exact_xalpha=True, alpha=alpha_xalpha
        )
        
        h = 1.0 / n
        mesh_sizes_xalpha.append(h)
        
        error_fem_exact = get_norm(u_h_local_xalpha_fem_exact, u_exact_func, mesh_local_xalpha_fem_exact, meshf_xalpha, Vf_xalpha, norm_choice_xalpha)
        errors_xalpha_fem_exact.append(error_fem_exact)
        
        error_fvm = get_norm(u_h_local_xalpha_fvm, u_exact_func, mesh_local_xalpha_fvm, meshf_xalpha, Vf_xalpha, norm_choice_xalpha)
        errors_xalpha_fvm.append(error_fvm)
        
        error_fvm_exact = get_norm(u_h_local_xalpha_fvm_exact, u_exact_func, mesh_local_xalpha_fvm_exact, meshf_xalpha, Vf_xalpha, norm_choice_xalpha)
        errors_xalpha_fvm_exact.append(error_fvm_exact)
    
    results_xalpha_fem_exact[beta] = (np.array(mesh_sizes_xalpha), np.array(errors_xalpha_fem_exact))
    results_xalpha_fvm[beta] = (np.array(mesh_sizes_xalpha), np.array(errors_xalpha_fvm))
    results_xalpha_fvm_exact[beta] = (np.array(mesh_sizes_xalpha), np.array(errors_xalpha_fvm_exact))
    
    if meshf_xalpha.comm.rank == 0:
        ref_info = reference_info_by_beta[beta]
        ref_tag = f"spectral ref, N={ref_info['N_ref_used']}, validated"
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) — vs {ref_tag} — FEM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_xalpha, mesh_sizes_xalpha, errors_xalpha_fem_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) — vs {ref_tag} — FVM:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_xalpha, mesh_sizes_xalpha, errors_xalpha_fvm):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) — vs {ref_tag} — FVM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_xalpha, mesh_sizes_xalpha, errors_xalpha_fvm_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

dofs_list_xalpha = []
for n in n_list_xalpha:
    mesh_temp_xalpha = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp_xalpha = fem.functionspace(mesh_temp_xalpha, ("CG", 1))
    # Global mesh dofs (P1: n+1 vertices). Use size_global, not size_local, under MPI.
    dofs_list_xalpha.append(V_temp_xalpha.dofmap.index_map.size_global)

dofs_array_xalpha = np.array(dofs_list_xalpha)

if MPI.COMM_WORLD.rank == 0:
    colors_xalpha = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_xalpha = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_xalpha):
        _, L2_errors_xalpha_fem_exact = results_xalpha_fem_exact[beta]
        _, L2_errors_xalpha_fvm = results_xalpha_fvm[beta]
        _, L2_errors_xalpha_fvm_exact = results_xalpha_fvm_exact[beta]
        
        color = colors_xalpha[i % len(colors_xalpha)]
        marker = markers_xalpha[i % len(markers_xalpha)]
        
        log_dofs_xalpha = np.log10(dofs_array_xalpha)
        log_errors_fem_exact = np.log10(L2_errors_xalpha_fem_exact)
        log_errors_fvm = np.log10(L2_errors_xalpha_fvm)
        log_errors_fvm_exact = np.log10(L2_errors_xalpha_fvm_exact)
        
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_dofs_xalpha, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = -slope_fem_exact
        
        slope_fvm, intercept_fvm = np.polyfit(log_dofs_xalpha, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_dofs_xalpha, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = -slope_fvm_exact
        
        # Plot FEM-exact (dotted), FVM (solid), and FVM-exact (dashdot) with same color and marker
        plt.loglog(dofs_array_xalpha, L2_errors_xalpha_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_xalpha, L2_errors_xalpha_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_xalpha, L2_errors_xalpha_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-{convergence_rate_fvm:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title(
        f'L2 error vs DOFs (vs spectral reference) — f(x)=x^(-α), α={alpha_xalpha}',
        fontsize=14,
    )
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Convergence plot: Error vs Mesh Size (h)

if MPI.COMM_WORLD.rank == 0:
    # Calculate mesh sizes
    mesh_sizes_array = 1.0 / np.array(n_list_xalpha)
    
    colors_xalpha = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_xalpha = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_xalpha):
        _, L2_errors_xalpha_fem_exact = results_xalpha_fem_exact[beta]
        _, L2_errors_xalpha_fvm = results_xalpha_fvm[beta]
        _, L2_errors_xalpha_fvm_exact = results_xalpha_fvm_exact[beta]
        
        color = colors_xalpha[i % len(colors_xalpha)]
        marker = markers_xalpha[i % len(markers_xalpha)]
        
        log_h = np.log10(mesh_sizes_array)
        log_errors_fem_exact = np.log10(L2_errors_xalpha_fem_exact)
        log_errors_fvm = np.log10(L2_errors_xalpha_fvm)
        log_errors_fvm_exact = np.log10(L2_errors_xalpha_fvm_exact)
        
        # vs h: if ||e|| ~ C h^p then log||e|| = p log h + const => slope = p (positive).
        # (Do not negate — unlike DOF plot where error ~ DOF^{-p} gives negative slope.)
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_h, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = slope_fem_exact
        
        slope_fvm, intercept_fvm = np.polyfit(log_h, log_errors_fvm, 1)
        convergence_rate_fvm = slope_fvm
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_h, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = slope_fvm_exact
        
        # Plot FEM-exact (dotted), FVM (solid), and FVM-exact (dashdot) with same color and marker
        plt.loglog(mesh_sizes_array, L2_errors_xalpha_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(mesh_sizes_array, L2_errors_xalpha_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(mesh_sizes_array, L2_errors_xalpha_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-{convergence_rate_fvm:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Maximum Mesh Size (h)', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title(
        f'L2 error vs mesh size h (vs spectral reference) — f(x)=x^(-α), α={alpha_xalpha}',
        fontsize=14,
    )
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Solution comparison plot

# Parameters for this plot
beta_plot = 0.3  # Beta value for comparison plot
N_plot = n_list_xalpha[-1]  # Number of elements for comparison plot (uses last value from n_list_xalpha)
n_fine_plot = 1000  # Fine mesh for comparison plot interpolation

# Plot visualization parameters (for this plot only)
figsize_plot = (10, 6)  # Figure size for plots
linewidth_plot = 2      # Line width for plot lines
markersize_plot = 4      # Marker size for DOF markers
alpha_line = 0.8         # Transparency for solution lines
alpha_marker = 0.6       # Transparency for DOF markers

spectral_ref_plot, spectral_ref_plot_info = _get_spectral_reference(beta_plot)

print("\n" + "="*70)
print("Solution comparison plot (spectral reference)")
print("="*70)
print(f"  α = {alpha_xalpha}")
print(f"  β = {beta_plot}")
print(f"  N (elements) = {N_plot}")
print(f"  spectral N_modes = {spectral_ref_plot_info['N_ref_used']}")
if spectral_ref_plot_info.get("fixed_truncation"):
    print("  reference mode = fixed truncation (faster; lower accuracy than adaptive)")
else:
    print(f"  tolerance met  = {spectral_ref_plot_info['tolerance_met']}")
    vd = spectral_ref_plot_info["validation_L2_diff"]
    if vd is not None:
        print(f"  validation ||u_N-u_2N||_L2 = {vd:.6e}")
print("="*70)

# Comparison plot: spectral reference vs numerical solutions
mesh_xalpha_plot_fem_exact, V_xalpha_plot_fem_exact, u_h_xalpha_plot_fem_exact = solve_fractional_dirichlet_on_mesh(
    N_plot, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_fem_exact_xalpha=True, alpha=alpha_xalpha
)
mesh_xalpha_plot_fvm, V_xalpha_plot_fvm, u_h_xalpha_plot_fvm = solve_fractional_dirichlet_on_mesh(
    N_plot, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_dual_fvm=True
)
mesh_xalpha_plot_fvm_exact, V_xalpha_plot_fvm_exact, u_h_xalpha_plot_fvm_exact = solve_fractional_dirichlet_on_mesh(
    N_plot, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), use_fvm_exact_xalpha=True, alpha=alpha_xalpha
)

meshf_xalpha_plot = dmesh.create_interval(MPI.COMM_WORLD, n_fine_plot, [0.0, 1.0])
Vf_xalpha_plot = fem.functionspace(meshf_xalpha_plot, ("CG", 1))

if mesh_xalpha_plot_fvm.comm.rank == 0:
    x_fine_xalpha = Vf_xalpha_plot.tabulate_dof_coordinates()[:, 0]
    
    x_coarse_xalpha_fem_exact = mesh_xalpha_plot_fem_exact.geometry.x[:, 0]
    u_h_xalpha_vals_fem_exact = u_h_xalpha_plot_fem_exact.x.array
    sort_idx_xalpha_fem_exact = np.argsort(x_coarse_xalpha_fem_exact)
    u_h_xalpha_interp_fem_exact = np.interp(x_fine_xalpha, x_coarse_xalpha_fem_exact[sort_idx_xalpha_fem_exact], u_h_xalpha_vals_fem_exact[sort_idx_xalpha_fem_exact])
    
    x_coarse_xalpha_fvm = mesh_xalpha_plot_fvm.geometry.x[:, 0]
    u_h_xalpha_vals_fvm = u_h_xalpha_plot_fvm.x.array
    sort_idx_xalpha_fvm = np.argsort(x_coarse_xalpha_fvm)
    u_h_xalpha_interp_fvm = np.interp(x_fine_xalpha, x_coarse_xalpha_fvm[sort_idx_xalpha_fvm], u_h_xalpha_vals_fvm[sort_idx_xalpha_fvm])
    
    x_coarse_xalpha_fvm_exact = mesh_xalpha_plot_fvm_exact.geometry.x[:, 0]
    u_h_xalpha_vals_fvm_exact = u_h_xalpha_plot_fvm_exact.x.array
    sort_idx_xalpha_fvm_exact = np.argsort(x_coarse_xalpha_fvm_exact)
    u_h_xalpha_interp_fvm_exact = np.interp(x_fine_xalpha, x_coarse_xalpha_fvm_exact[sort_idx_xalpha_fvm_exact], u_h_xalpha_vals_fvm_exact[sort_idx_xalpha_fvm_exact])
    
    u_exact_xalpha_vals = spectral_ref_plot(np.asarray([x_fine_xalpha]))
    u_exact_xalpha_vals = np.atleast_1d(u_exact_xalpha_vals)

    plt.figure(figsize=figsize_plot)
    plt.plot(
        x_fine_xalpha,
        u_exact_xalpha_vals,
        'b-',
        label=f'Spectral reference (N={spectral_ref_plot_info["N_ref_used"]}, α={alpha_xalpha}, β={beta_plot})',
        linewidth=linewidth_plot,
        alpha=alpha_line,
    )
    plt.plot(x_fine_xalpha, u_h_xalpha_interp_fem_exact, 'm:', label=f'FEM-exact solution (N={N_plot})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_fine_xalpha, u_h_xalpha_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_fine_xalpha, u_h_xalpha_interp_fvm_exact, 'c-.', label=f'FVM-exact solution (N={N_plot})', linewidth=linewidth_plot, alpha=alpha_line)
    # plt.plot(x_fine_xalpha, f_xalpha_vals, 'y:', label=f'Source f(x)=x^(-α), α={alpha_xalpha}', linewidth=linewidth_plot, alpha=alpha_marker)
    plt.plot(x_coarse_xalpha_fem_exact, u_h_xalpha_vals_fem_exact, 's', markersize=markersize_plot, label='FEM-exact DOFs', alpha=alpha_marker)
    plt.plot(x_coarse_xalpha_fvm, u_h_xalpha_vals_fvm, 'x', markersize=markersize_plot, label='FVM DOFs', alpha=alpha_marker)
    plt.plot(x_coarse_xalpha_fvm_exact, u_h_xalpha_vals_fvm_exact, '^', markersize=markersize_plot, label='FVM-exact DOFs', alpha=alpha_marker)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(
        f'Comparison: spectral reference vs numerical — f(x)=x^(-α), α={alpha_xalpha}, β={beta_plot}',
        fontsize=14,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# Save error tables to files

if MPI.COMM_WORLD.rank == 0:
    # Use DOF values instead of h values
    # dofs_array_xalpha is already computed above
    
    # Prepare data for FEM-exact errors
    fem_exact_data = []
    fem_exact_data.append(['DOF'] + [str(beta) for beta in beta_values_xalpha])
    
    for dof_idx, dof in enumerate(dofs_array_xalpha):
        row = [str(int(dof))]
        for beta in beta_values_xalpha:
            _, errors = results_xalpha_fem_exact[beta]
            row.append(str(errors[dof_idx]))
        fem_exact_data.append(row)
    
    # Prepare data for FVM-exact errors
    fvm_exact_data = []
    fvm_exact_data.append(['DOF'] + [str(beta) for beta in beta_values_xalpha])
    
    for dof_idx, dof in enumerate(dofs_array_xalpha):
        row = [str(int(dof))]
        for beta in beta_values_xalpha:
            _, errors = results_xalpha_fvm_exact[beta]
            row.append(str(errors[dof_idx]))
        fvm_exact_data.append(row)
    
    # Create output directory based on experiment file name (without .py extension)
    experiment_name = Path(__file__).stem  # Gets filename without extension: 'f_ex_xalpha'
    output_dir = Path(__file__).parent / experiment_name  # Save inside 1d_experiments folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FEM-exact errors
    fem_exact_file = output_dir / 'errors_1dfem.dat'
    with open(fem_exact_file, 'w') as f:
        for row in fem_exact_data:
            f.write('\t'.join(row) + '\n')
    print(f"\nSaved FEM-exact errors to: {fem_exact_file}")
    
    # Save FVM-exact errors
    fvm_exact_file = output_dir / 'errors_1dfvm.dat'
    with open(fvm_exact_file, 'w') as f:
        for row in fvm_exact_data:
            f.write('\t'.join(row) + '\n')
    print(f"Saved FVM-exact errors to: {fvm_exact_file}")
    
    # Calculate and save beta vs slope tables
    # Use DOF-based calculation (log(error) vs log(DOF))
    # dofs_array_xalpha is already computed above
    
    fem_exact_slopes = []
    fvm_exact_slopes = []
    
    for beta in beta_values_xalpha:
        _, L2_errors_fem_exact = results_xalpha_fem_exact[beta]
        _, L2_errors_fvm_exact = results_xalpha_fvm_exact[beta]
        
        # Use DOF-based calculation: log(error) vs log(DOF)
        log_dofs = np.log10(dofs_array_xalpha)
        log_errors_fem_exact = np.log10(L2_errors_fem_exact)
        log_errors_fvm_exact = np.log10(L2_errors_fvm_exact)
        
        # Fit log(error) = slope * log(DOF) + intercept
        # If error ~ DOF^(-p), then slope = -p (negative because error decreases as DOF increases)
        # We negate to get positive convergence rate p
        slope_fem_exact, _ = np.polyfit(log_dofs, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = -slope_fem_exact  # Negate because error decreases as DOF increases
        
        slope_fvm_exact, _ = np.polyfit(log_dofs, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = -slope_fvm_exact  # Negate because error decreases as DOF increases
        
        fem_exact_slopes.append(convergence_rate_fem_exact)
        fvm_exact_slopes.append(convergence_rate_fvm_exact)
    
    # Save FEM-exact beta vs slope
    betaxslope_fem_file = output_dir / 'betaxslope_1dfem.dat'
    with open(betaxslope_fem_file, 'w') as f:
        f.write('x\ty\n')
        for beta, slope in zip(beta_values_xalpha, fem_exact_slopes):
            f.write(f'{beta}\t{slope}\n')
    print(f"Saved FEM-exact beta vs slope to: {betaxslope_fem_file}")
    
    # Save FVM-exact beta vs slope
    betaxslope_fvm_file = output_dir / 'betaxslope_1dfvm.dat'
    with open(betaxslope_fvm_file, 'w') as f:
        f.write('x\ty\n')
        for beta, slope in zip(beta_values_xalpha, fvm_exact_slopes):
            f.write(f'{beta}\t{slope}\n')
    print(f"Saved FVM-exact beta vs slope to: {betaxslope_fvm_file}")

# %%

