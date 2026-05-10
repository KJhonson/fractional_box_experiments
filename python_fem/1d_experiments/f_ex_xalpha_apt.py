# %%

"""
Fractional Dirichlet problem experiment with adaptive mesh refinement.
Problem:
    Find u such that:
        (-d²/dx²)^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0            (Dirichlet boundary conditions)
    
    where:
        f(x) = x^(-α)
        α ∈ (0, 1/2) (singularity exponent)
        β > 0 (fractional power)
    
    This file focuses on adaptive mesh refinement analysis.
    The exact solution is given by a Fourier series expansion computed using
    DST-I (Discrete Sine Transform) - the fastest and most accurate method according to benchmark.
    Implementation uses authentic DST-I directly in the code (no scipy dependency).
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.norm import get_norm
from utils.sinc_solver import sinc_solver
from utils.loads import assemble_rhs_dual_fvm


def dynamic_k(beta, dofs):
    """
    Calculate dynamic sinc parameter k based on beta and number of degrees of freedom.
    
    Formula: k = -pi^2 / (4 * beta * log(h))
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
    
    # Calculate k using the formula: k = -pi^2 / (4 * beta * log(h))
    # Note: log(h) is negative since h < 1, so k will be positive
    log_h = np.log(h)
    k = -np.pi**2 / (4 * beta * log_h)
    
    return k


def solve_fractional_dirichlet_on_mesh(n_elements=None, mesh_local=None, beta=0.5, kappa_func=None, f_source_func=None, use_dual_fvm=False, use_fvm_exact_xalpha=False, use_fem_exact_xalpha=False, alpha=None, k=None):
    """
    Solve fractional Dirichlet problem on a mesh.
    
    Parameters:
    -----------
    n_elements : int, optional
        Number of elements in the mesh (used if mesh_local is None)
    mesh_local : dolfinx.mesh.Mesh, optional
        Mesh to use (if provided, n_elements is ignored)
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
    if mesh_local is None:
        if n_elements is None:
            raise ValueError("Either n_elements or mesh_local must be provided")
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
    
    # Calculate dynamic k if not provided
    if k is None:
        # Get number of free DOFs (after applying Dirichlet BC)
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
    alpha : float
        Exponent in f(x) = x^{-alpha}, with 0 < alpha < 1.
    x : array_like
        Mesh nodes [x0, x1, ..., xN].
        
    Returns
    -------
    F : np.ndarray, shape (N+1,)
        Load vector for all nodes (boundary nodes will be set to 0 by Dirichlet BCs).
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
    Direct DST-I implementation matching scipy.fftpack.dst(x, type=1).
    
    DST-I de um vetor x de tamanho M:
        y_k = 2 * sum_{j=1}^{M} x_j * sin(pi * j * k / (M+1))
    
    The factor 2 ensures compatibility with scipy.fftpack.dst(type=1).
    
    Parameters:
    -----------
    x : array_like
        Input vector of length M.
    
    Returns:
    --------
    y : np.ndarray
        DST-I transform of length M, compatible with scipy.
    """
    M = len(x)
    k = np.arange(1, M + 1)
    j = np.arange(1, M + 1)
    
    # Transform matrix: sin(pi * j * k / (M+1)).
    j_mat = j[:, np.newaxis]  # shape (M, 1)
    k_mat = k[np.newaxis, :]   # shape (1, M)
    sin_matrix = np.sin(np.pi * j_mat * k_mat / (M + 1))
    
    # Apply the transform with factor 2 to match scipy.
    y = 2.0 * (sin_matrix.T @ x)  # shape (M,)
    
    return y


def reference_solution(alpha, beta, M=500, N=None, x_eval=None):
    """
    High-accuracy spectral solution for (-d^2/dx^2)^beta u = x^{-alpha}, u(0)=u(1)=0.
    Uses DST-I (Discrete Sine Transform), which is fast and accurate in the benchmark.
    The DST-I implementation is included directly in the code.
    
    Parameters:
    ------------
    alpha : float
        Source exponent (0 < alpha < 1/2).
    beta : float
        Fractional order (> 0).
    M : int
        Number of interior mesh points (x_j = j/(M+1)). Default: 500.
    N : int, optional
        Number of modes used in the solution (<= M). If None, uses N = M.
    x_eval : array-like, optional
        Points in (0, 1) where the solution is evaluated. If None, uses 200 uniform points.
    
    Returns:
    --------
    x_eval : np.ndarray
        Evaluation points.
    u_vals : np.ndarray
        Solution values u(x) at those points.
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 500)[1:-1]  # avoid endpoints 0 and 1, where u=0
    x_eval = np.asarray(x_eval)

    if N is None or N > M:
        N = M

    # Mesh points.
    j = np.arange(1, M + 1)
    x = j / (M + 1)
    dx = 1.0 / (M + 1)

    # f(x) = x^{-alpha}
    f_vals = x**(-alpha)

    # Unnormalized DST-I compatible with scipy.
    y = dst_type1(f_vals)   # len M

    # Continuous coefficients f_n approx sqrt(2)/(2(M+1)) * y_{n-1}.
    n_vals = np.arange(1, N + 1)
    f_n = np.sqrt(2.0) * dx * 0.5 * y[:N]   # = sqrt(2)/(2(M+1)) * y

    # coeficientes de u
    u_n = (n_vals * np.pi)**(-2.0 * beta) * f_n

    # avalia u(x) em x_eval
    Sx = np.sin(np.pi * np.outer(x_eval, n_vals))
    u_vals = np.sqrt(2.0) * (Sx @ u_n)

    return x_eval, u_vals


def create_adaptive_mesh_1d_smooth(N_base, alpha=0.49):
    """
    Create a smooth adaptive 1D mesh that follows the singularity x^(-α).
    
    The mesh size h(x) should be proportional to x^(α) to capture the singularity
    behavior properly. This creates a smooth transition from fine to coarse mesh.
    
    Parameters:
    -----------
    N_base : int
        Base number of elements
    alpha : float
        Singularity exponent (0 < alpha < 1)
    
    Returns:
    --------
    mesh : dolfinx.mesh.Mesh
        Adaptive mesh with refinement near x=0
    """
    def smooth_mapping(t, alpha):
        """
        Smooth mapping from [0,1] to [0,1] that creates refinement near x=0.
        The mapping should be smooth and create smaller elements near x=0.
        """
        # Use a power function that creates smooth refinement
        # For t ∈ [0,1], we want x ∈ [0,1] with more points near x=0
        return t**(1.0 / (1.0 - alpha))
    
    # Generate smooth coordinates
    N_total = N_base * 3  # More elements for smoothness
    t_coords = np.linspace(0.0, 1.0, N_total + 1)
    x_coords = np.array([smooth_mapping(t, alpha) for t in t_coords])
    
    # Ensure we start at 0 and end at 1
    x_coords[0] = 0.0
    x_coords[-1] = 1.0
    
    # Create mesh with these coordinates
    mesh_adaptive = dmesh.create_interval(MPI.COMM_WORLD, N_total, [0.0, 1.0])
    # Modify coordinates to create adaptive mesh
    # Sort the mesh coordinates and assign adaptive coordinates in sorted order
    order = np.argsort(mesh_adaptive.geometry.x[:, 0])
    mesh_adaptive.geometry.x[order, 0] = x_coords
    
    return mesh_adaptive


# %%
# ============================================================================
# HYPERPARAMETERS - Adjust these values to change experiment settings
# ============================================================================

# Mesh refinement levels for adaptive mesh
n_list_adaptive = [8, 16, 32, 64, 128]  # Base number of elements for adaptive mesh (added more levels for better convergence analysis)

# Fractional power values to test
beta_values_xalpha = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Singularity exponent for source function f(x) = x^(-alpha)
alpha_xalpha = 0.499  # Must satisfy: 0 < alpha < 1/2

# DST parameters for reference solution (M = number of internal mesh points, N = number of modes)
# Higher M values give better accuracy but are slower
M_dst = 10000  # Number of internal mesh points for DST (x_j = j/(M+1))
N_modes_map = {0.3: None, 0.5: None, 0.8: None, 1.0: None, 1.5: None, 2.0: None}  # None means use M (all modes)

# Fine mesh for reference solution evaluation
n_fine_mesh = 10000

# Error norm type
norm_choice_xalpha = "L2"

# ============================================================================
# Adaptive Mesh Convergence Analysis: Error vs DOF
# ============================================================================

results_adaptive_fem_exact = {}
results_adaptive_fvm = {}
results_adaptive_fvm_exact = {}

# Compute DOFs once (they're the same for all beta values since we use the same mesh refinement levels)
dofs_adaptive_all = []
for n_base in n_list_adaptive:
    mesh_temp_adaptive = create_adaptive_mesh_1d_smooth(n_base, alpha=alpha_xalpha)
    V_temp_adaptive = fem.functionspace(mesh_temp_adaptive, ("CG", 1))
    dofs_adaptive_all.append(V_temp_adaptive.dofmap.index_map.size_local)
dofs_array_adaptive = np.array(dofs_adaptive_all)

for beta in beta_values_xalpha:
    N_modes = N_modes_map.get(beta, None)
    
    # Choose exact/reference solution based on beta
    if abs(beta - 1.0) < 1e-12:
        # Closed-form exact solution for -u'' = x^(-alpha) with Dirichlet BC
        def u_exact_func(x):
            # Extract x-coordinate from input (handles tuple, array, or scalar)
            if isinstance(x, tuple):
                x_coord = float(np.asarray(x[0]).flatten()[0])
            elif isinstance(x, np.ndarray):
                x_flat = x.flatten()
                x_coord = float(x_flat[0])  # Take first coordinate (x in 1D, x-coord in 2D/3D)
            else:
                x_coord = float(x)
            
            # Compute solution value
            if x_coord > 1e-15:
                denom = (1.0 - alpha_xalpha) * (2.0 - alpha_xalpha)
                result = (x_coord - x_coord**(2.0 - alpha_xalpha)) / denom
            else:
                result = 0.0
            
            return result
        ref_label = 'Exact (closed-form)'
        ref_name_print = 'exact'
    else:
        # Fractional case: spectral reference using DST method
        x_ref, u_ref_vals = reference_solution(alpha_xalpha, beta, M=M_dst, N=N_modes)
        
        # Create a callable function from the reference solution
        def u_exact_func(x):
            x_vals = x[0]
            x_vals = np.atleast_1d(x_vals)
            # Interpolate from reference solution
            u_interp = np.interp(x_vals, x_ref, u_ref_vals)
            # Handle boundary points
            u_interp = np.where(x_vals <= 0.0, 0.0, u_interp)
            u_interp = np.where(x_vals >= 1.0, 0.0, u_interp)
            if x_vals.size == 1:
                return float(u_interp[0])
            return u_interp
        ref_label = f'Reference spectral (DST: M={M_dst}, N={N_modes if N_modes is not None else M_dst})'
        ref_name_print = 'spectral'
    
    meshf_adaptive = dmesh.create_interval(MPI.COMM_WORLD, n_fine_mesh, [0.0, 1.0])
    Vf_adaptive = fem.functionspace(meshf_adaptive, ("CG", 1))
    
    errors_adaptive_fem_exact = []
    errors_adaptive_fvm = []
    errors_adaptive_fvm_exact = []
    h_max_values = []  # Store maximum mesh size for each refinement level
    
    for n_base in n_list_adaptive:
        # Create adaptive mesh
        mesh_adaptive = create_adaptive_mesh_1d_smooth(n_base, alpha=alpha_xalpha)
        
        # Compute maximum mesh size h_max
        x_coords = np.sort(mesh_adaptive.geometry.x[:, 0])
        h_max = np.max(np.diff(x_coords))
        h_max_values.append(h_max)
        
        # Solve on adaptive mesh with different methods
        mesh_adaptive_fem_exact, V_adaptive_fem_exact, u_h_adaptive_fem_exact = solve_fractional_dirichlet_on_mesh(
            mesh_local=mesh_adaptive, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
            use_fem_exact_xalpha=True, alpha=alpha_xalpha
        )
        mesh_adaptive_fvm, V_adaptive_fvm, u_h_adaptive_fvm = solve_fractional_dirichlet_on_mesh(
            mesh_local=mesh_adaptive, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
            use_dual_fvm=True
        )
        mesh_adaptive_fvm_exact, V_adaptive_fvm_exact, u_h_adaptive_fvm_exact = solve_fractional_dirichlet_on_mesh(
            mesh_local=mesh_adaptive, beta=beta, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
            use_fvm_exact_xalpha=True, alpha=alpha_xalpha
        )
        
        # Compute errors
        error_fem_exact = get_norm(u_h_adaptive_fem_exact, u_exact_func, mesh_adaptive_fem_exact, meshf_adaptive, Vf_adaptive, norm_choice_xalpha)
        errors_adaptive_fem_exact.append(error_fem_exact)
        
        error_fvm = get_norm(u_h_adaptive_fvm, u_exact_func, mesh_adaptive_fvm, meshf_adaptive, Vf_adaptive, norm_choice_xalpha)
        errors_adaptive_fvm.append(error_fvm)
        
        error_fvm_exact = get_norm(u_h_adaptive_fvm_exact, u_exact_func, mesh_adaptive_fvm_exact, meshf_adaptive, Vf_adaptive, norm_choice_xalpha)
        errors_adaptive_fvm_exact.append(error_fvm_exact)
    
    results_adaptive_fem_exact[beta] = (dofs_array_adaptive.copy(), np.array(errors_adaptive_fem_exact), np.array(h_max_values))
    results_adaptive_fvm[beta] = (dofs_array_adaptive.copy(), np.array(errors_adaptive_fvm), np.array(h_max_values))
    results_adaptive_fvm_exact[beta] = (dofs_array_adaptive.copy(), np.array(errors_adaptive_fvm_exact), np.array(h_max_values))
    
    if meshf_adaptive.comm.rank == 0:
        if abs(beta - 1.0) < 1e-12:
            ref_info = f"[{ref_name_print}]"
        else:
            N_display = N_modes if N_modes is not None else M_dst
            ref_info = f"[DST: M={M_dst}, N={N_display}]"
        
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) - Adaptive Mesh {ref_info} - FEM-exact:")
        print(f"{'N_base':<8} {'DOFs':<8} {'h_max':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 42)
        for n_base, dof, h_max, err in zip(n_list_adaptive, dofs_array_adaptive, h_max_values, errors_adaptive_fem_exact):
            print(f"{n_base:<8} {dof:<8} {h_max:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) - Adaptive Mesh {ref_info} - FVM:")
        print(f"{'N_base':<8} {'DOFs':<8} {'h_max':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 42)
        for n_base, dof, h_max, err in zip(n_list_adaptive, dofs_array_adaptive, h_max_values, errors_adaptive_fvm):
            print(f"{n_base:<8} {dof:<8} {h_max:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet x^(-α) (α={alpha_xalpha}, β={beta}) - Adaptive Mesh {ref_info} - FVM-exact:")
        print(f"{'N_base':<8} {'DOFs':<8} {'h_max':<12} {f'{norm_choice_xalpha} Error':<12}")
        print("-" * 42)
        for n_base, dof, h_max, err in zip(n_list_adaptive, dofs_array_adaptive, h_max_values, errors_adaptive_fvm_exact):
            print(f"{n_base:<8} {dof:<8} {h_max:<12.6e} {err:<12.6e}")
        
        # Compute convergence rates using first few points (where convergence is better)
        # For beta=1, we expect order 2 convergence
        # Use first 3 points for better rate estimation (avoid plateauing region)
        n_points_for_rate = min(3, len(errors_adaptive_fem_exact))  # Use first 3 points
        if n_points_for_rate >= 2:
            dofs_subset = dofs_array_adaptive[:n_points_for_rate]
            errors_subset_fem = errors_adaptive_fem_exact[:n_points_for_rate]
            errors_subset_fvm = errors_adaptive_fvm[:n_points_for_rate]
            errors_subset_fvm_exact = errors_adaptive_fvm_exact[:n_points_for_rate]
            h_max_subset = np.array(h_max_values[:n_points_for_rate])
            
            # Rate vs DOFs (using first few points)
            log_dofs_subset = np.log10(dofs_subset)
            log_err_fem_subset = np.log10(errors_subset_fem)
            log_err_fvm_subset = np.log10(errors_subset_fvm)
            log_err_fvm_exact_subset = np.log10(errors_subset_fvm_exact)
            
            slope_fem_subset = np.polyfit(log_dofs_subset, log_err_fem_subset, 1)[0]
            slope_fvm_subset = np.polyfit(log_dofs_subset, log_err_fvm_subset, 1)[0]
            slope_fvm_exact_subset = np.polyfit(log_dofs_subset, log_err_fvm_exact_subset, 1)[0]
            
            rate_fem_subset = -slope_fem_subset
            rate_fvm_subset = -slope_fvm_subset
            rate_fvm_exact_subset = -slope_fvm_exact_subset
            
            # Rate vs h_max (using first few points)
            log_h_subset = np.log10(h_max_subset)
            slope_fem_h = np.polyfit(log_h_subset, log_err_fem_subset, 1)[0]
            slope_fvm_h = np.polyfit(log_h_subset, log_err_fvm_subset, 1)[0]
            slope_fvm_exact_h = np.polyfit(log_h_subset, log_err_fvm_exact_subset, 1)[0]
            
            rate_fem_h = slope_fem_h  # Positive slope means error decreases as h decreases
            rate_fvm_h = slope_fvm_h
            rate_fvm_exact_h = slope_fvm_exact_h
            
            print(f"\nConvergence rates (using first {n_points_for_rate} points, where convergence is better):")
            print(f"  vs DOFs: FEM-exact={rate_fem_subset:.3f}, FVM={rate_fvm_subset:.3f}, FVM-exact={rate_fvm_exact_subset:.3f}")
            print(f"  vs h_max: FEM-exact={rate_fem_h:.3f}, FVM={rate_fvm_h:.3f}, FVM-exact={rate_fvm_exact_h:.3f}")
            if abs(beta - 1.0) < 1e-10:
                print(f"  Expected for β=1: order ~2.0 (O(h²) in L2 norm)")
                print(f"  Note: Using closed-form exact solution. Error may plateau due to adaptive mesh strategy.")
            else:
                # Check if errors are plateauing (suggesting DST reference accuracy limit)
                if len(errors_adaptive_fem_exact) >= 4:
                    error_ratio = errors_adaptive_fem_exact[-1] / errors_adaptive_fem_exact[-2]
                    if error_ratio > 0.95:  # Error decreased by less than 5%
                        print(f"  ⚠️  Warning: Errors appear to plateau (ratio={error_ratio:.3f}).")
                        print(f"     This may indicate DST reference solution accuracy limit (M={M_dst}).")
                        print(f"     Consider increasing M_dst or checking DST convergence.")

if MPI.COMM_WORLD.rank == 0:
    colors_adaptive = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_adaptive = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_xalpha):
        dofs_adaptive_beta, L2_errors_adaptive_fem_exact, h_max_beta = results_adaptive_fem_exact[beta]
        _, L2_errors_adaptive_fvm, _ = results_adaptive_fvm[beta]
        _, L2_errors_adaptive_fvm_exact, _ = results_adaptive_fvm_exact[beta]
        
        color = colors_adaptive[i % len(colors_adaptive)]
        marker = markers_adaptive[i % len(markers_adaptive)]
        
        log_dofs_adaptive = np.log10(dofs_adaptive_beta)
        log_errors_fem_exact = np.log10(L2_errors_adaptive_fem_exact)
        log_errors_fvm = np.log10(L2_errors_adaptive_fvm)
        log_errors_fvm_exact = np.log10(L2_errors_adaptive_fvm_exact)
        
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_dofs_adaptive, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = -slope_fem_exact
        
        slope_fvm, intercept_fvm = np.polyfit(log_dofs_adaptive, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_dofs_adaptive, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = -slope_fvm_exact
        
        # Plot FEM-exact (dotted), FVM (solid), and FVM-exact (dashdot) with same color and marker
        plt.loglog(dofs_adaptive_beta, L2_errors_adaptive_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_adaptive_beta, L2_errors_adaptive_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_adaptive_beta, L2_errors_adaptive_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-{convergence_rate_fvm:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title(f'Convergence: Error vs DOFs (log-log scale) - Adaptive Mesh - Fractional Dirichlet with f(x)=x^(-α), α={alpha_xalpha}', fontsize=14)
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Convergence plot: Error vs maximum mesh size (h_max)
# If ||e|| ~ C h_max^p then log||e|| = p log h_max + const => polyfit slope = p (use slope, do not negate; unlike DOF plot).

if MPI.COMM_WORLD.rank == 0:
    colors_adaptive = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_adaptive = ['o', 's', '^', 'v', 'D', 'p']

    plt.figure(figsize=(10, 6))

    legend_handles = []
    legend_labels = []

    for i, beta in enumerate(beta_values_xalpha):
        _, L2_errors_adaptive_fem_exact, h_max_beta = results_adaptive_fem_exact[beta]
        _, L2_errors_adaptive_fvm, _ = results_adaptive_fvm[beta]
        _, L2_errors_adaptive_fvm_exact, _ = results_adaptive_fvm_exact[beta]

        color = colors_adaptive[i % len(colors_adaptive)]
        marker = markers_adaptive[i % len(markers_adaptive)]

        log_h_adaptive = np.log10(h_max_beta)
        log_errors_fem_exact = np.log10(L2_errors_adaptive_fem_exact)
        log_errors_fvm = np.log10(L2_errors_adaptive_fvm)
        log_errors_fvm_exact = np.log10(L2_errors_adaptive_fvm_exact)

        slope_fem_exact, _ = np.polyfit(log_h_adaptive, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = slope_fem_exact

        slope_fvm, _ = np.polyfit(log_h_adaptive, log_errors_fvm, 1)
        convergence_rate_fvm = slope_fvm

        slope_fvm_exact, _ = np.polyfit(log_h_adaptive, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = slope_fvm_exact

        plt.loglog(h_max_beta, L2_errors_adaptive_fem_exact,
                   color=color, marker=marker, linestyle=':',
                   linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(h_max_beta, L2_errors_adaptive_fvm,
                   color=color, marker=marker, linestyle='-',
                   linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(h_max_beta, L2_errors_adaptive_fvm_exact,
                   color=color, marker=marker, linestyle='-.',
                   linewidth=2, markersize=8, alpha=0.8)

        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(
            f'β={beta}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-{convergence_rate_fvm:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}'
        )

    plt.xlabel(r'Maximum mesh size $h_{\mathrm{max}}$', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title(
        f'Convergence: Error vs $h_{{\\mathrm{{max}}}}$ (log-log) - Adaptive mesh - '
        f'$f(x)=x^{{-α}}$, α={alpha_xalpha}',
        fontsize=14,
    )
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Solution comparison plot with adaptive mesh

# Parameters for adaptive mesh plot
beta_plot = 0.3  # Beta value for comparison plot
N_base_adapt = 32  # Base number of elements for adaptive mesh
M_dst_plot = 2000  # Number of internal mesh points for DST in comparison plot reference solution (higher = better accuracy)
N_modes_plot = None  # Number of modes for comparison plot (None means use M_dst_plot)
n_fine_plot = 1000  # Fine mesh for comparison plot interpolation

# Plot visualization parameters (for this plot only)
figsize_plot = (10, 6)  # Figure size for plots
linewidth_plot = 2      # Line width for plot lines
markersize_plot = 4      # Marker size for DOF markers
alpha_line = 0.8         # Transparency for solution lines
alpha_marker = 0.6       # Transparency for DOF markers

print(f"\n" + "="*70)
print(f"Solution Comparison Plot - Adaptive Mesh")
print(f"="*70)
print(f"Hyperparameters:")
print(f"  α = {alpha_xalpha}")
print(f"  β = {beta_plot}")
print(f"  N_base = {N_base_adapt} (adaptive mesh)")
print(f"  DST: M = {M_dst_plot}, N = {N_modes_plot if N_modes_plot is not None else M_dst_plot}")
print(f"  n_fine = {n_fine_plot} (fine mesh for interpolation)")
print(f"="*70)

# Create adaptive mesh
mesh_adaptive = create_adaptive_mesh_1d_smooth(N_base_adapt, alpha=alpha_xalpha)
n_elements_adapt = mesh_adaptive.topology.index_map(mesh_adaptive.topology.dim).size_global
print(f"  N_actual = {n_elements_adapt} (actual elements in adaptive mesh)")
print(f"="*70)

# Solve on adaptive mesh with different methods
mesh_xalpha_adapt_fem_exact, V_xalpha_adapt_fem_exact, u_h_xalpha_adapt_fem_exact = solve_fractional_dirichlet_on_mesh(
    mesh_local=mesh_adaptive, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
    use_fem_exact_xalpha=True, alpha=alpha_xalpha
)
mesh_xalpha_adapt_fvm, V_xalpha_adapt_fvm, u_h_xalpha_adapt_fvm = solve_fractional_dirichlet_on_mesh(
    mesh_local=mesh_adaptive, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
    use_dual_fvm=True
)
mesh_xalpha_adapt_fvm_exact, V_xalpha_adapt_fvm_exact, u_h_xalpha_adapt_fvm_exact = solve_fractional_dirichlet_on_mesh(
    mesh_local=mesh_adaptive, beta=beta_plot, f_source_func=lambda x: f_xalpha(x, alpha_xalpha), 
    use_fvm_exact_xalpha=True, alpha=alpha_xalpha
)

# Get reference solution using DST method (same as third plot)
x_ref_plot_adapt, u_ref_plot_adapt_vals = reference_solution(alpha_xalpha, beta_plot, M=M_dst_plot, N=N_modes_plot)

def u_exact_func_plot_adapt(x):
    x_vals = x[0]
    x_vals = np.atleast_1d(x_vals)
    u_interp = np.interp(x_vals, x_ref_plot_adapt, u_ref_plot_adapt_vals)
    u_interp = np.where(x_vals <= 0.0, 0.0, u_interp)
    u_interp = np.where(x_vals >= 1.0, 0.0, u_interp)
    if x_vals.size == 1:
        return float(u_interp[0])
    return u_interp

meshf_xalpha_adapt = dmesh.create_interval(MPI.COMM_WORLD, n_fine_plot, [0.0, 1.0])
Vf_xalpha_adapt = fem.functionspace(meshf_xalpha_adapt, ("CG", 1))

if mesh_xalpha_adapt_fvm.comm.rank == 0:
    x_fine_xalpha_adapt = Vf_xalpha_adapt.tabulate_dof_coordinates()[:, 0]
    
    x_coarse_xalpha_adapt_fem_exact = mesh_xalpha_adapt_fem_exact.geometry.x[:, 0]
    u_h_xalpha_adapt_vals_fem_exact = u_h_xalpha_adapt_fem_exact.x.array
    sort_idx_xalpha_adapt_fem_exact = np.argsort(x_coarse_xalpha_adapt_fem_exact)
    u_h_xalpha_adapt_interp_fem_exact = np.interp(x_fine_xalpha_adapt, x_coarse_xalpha_adapt_fem_exact[sort_idx_xalpha_adapt_fem_exact], u_h_xalpha_adapt_vals_fem_exact[sort_idx_xalpha_adapt_fem_exact])
    
    x_coarse_xalpha_adapt_fvm = mesh_xalpha_adapt_fvm.geometry.x[:, 0]
    u_h_xalpha_adapt_vals_fvm = u_h_xalpha_adapt_fvm.x.array
    sort_idx_xalpha_adapt_fvm = np.argsort(x_coarse_xalpha_adapt_fvm)
    u_h_xalpha_adapt_interp_fvm = np.interp(x_fine_xalpha_adapt, x_coarse_xalpha_adapt_fvm[sort_idx_xalpha_adapt_fvm], u_h_xalpha_adapt_vals_fvm[sort_idx_xalpha_adapt_fvm])
    
    x_coarse_xalpha_adapt_fvm_exact = mesh_xalpha_adapt_fvm_exact.geometry.x[:, 0]
    u_h_xalpha_adapt_vals_fvm_exact = u_h_xalpha_adapt_fvm_exact.x.array
    sort_idx_xalpha_adapt_fvm_exact = np.argsort(x_coarse_xalpha_adapt_fvm_exact)
    u_h_xalpha_adapt_interp_fvm_exact = np.interp(x_fine_xalpha_adapt, x_coarse_xalpha_adapt_fvm_exact[sort_idx_xalpha_adapt_fvm_exact], u_h_xalpha_adapt_vals_fvm_exact[sort_idx_xalpha_adapt_fvm_exact])
    
    u_exact_xalpha_adapt_vals = np.array([u_exact_func_plot_adapt(np.array([x])) for x in x_fine_xalpha_adapt])
    
    plt.figure(figsize=figsize_plot)
    plt.plot(x_fine_xalpha_adapt, u_exact_xalpha_adapt_vals, 'b-', label=f'Exact solution (α={alpha_xalpha}, β={beta_plot}, DST: M={M_dst_plot})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_fine_xalpha_adapt, u_h_xalpha_adapt_interp_fem_exact, 'm:', label=f'FEM-exact solution (adaptive, N_base={N_base_adapt})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_fine_xalpha_adapt, u_h_xalpha_adapt_interp_fvm, 'g-', label=f'FVM solution (adaptive, N_base={N_base_adapt})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_fine_xalpha_adapt, u_h_xalpha_adapt_interp_fvm_exact, 'c-.', label=f'FVM-exact solution (adaptive, N_base={N_base_adapt})', linewidth=linewidth_plot, alpha=alpha_line)
    plt.plot(x_coarse_xalpha_adapt_fem_exact, u_h_xalpha_adapt_vals_fem_exact, 's', markersize=markersize_plot, label='FEM-exact DOFs (adaptive)', alpha=alpha_marker)
    plt.plot(x_coarse_xalpha_adapt_fvm, u_h_xalpha_adapt_vals_fvm, 'x', markersize=markersize_plot, label='FVM DOFs (adaptive)', alpha=alpha_marker)
    plt.plot(x_coarse_xalpha_adapt_fvm_exact, u_h_xalpha_adapt_vals_fvm_exact, '^', markersize=markersize_plot, label='FVM-exact DOFs (adaptive)', alpha=alpha_marker)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Comparison: Exact vs Numerical Solution (Adaptive Mesh) - Fractional Dirichlet with f(x)=x^(-α)\nα={alpha_xalpha}, β={beta_plot}, N_base={N_base_adapt}, N_actual={n_elements_adapt}, DST: M={M_dst_plot}, N_modes={N_modes_plot if N_modes_plot is not None else M_dst_plot}, n_fine={n_fine_plot}', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# ============================================================================
# Diagnostic: Check DST Reference Solution Convergence
# ============================================================================
# This section helps verify if the DST reference solution is accurate enough
# by comparing solutions with different M values (number of DST points)

if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*70)
    print("DST Reference Solution Convergence Check")
    print("="*70)
    print("Comparing DST solutions with different M values to assess accuracy.")
    print(f"Testing for α={alpha_xalpha}, various β values")
    print("="*70)
    
    # Test M values
    M_test_values = [5000, 10000, 20000]
    beta_test = [0.3, 0.5, 0.8, 1.5, 2.0]  # Skip beta=1 (has exact solution)
    
    # Evaluation points (use a moderate number for comparison)
    x_test = np.linspace(0.01, 0.99, 100)  # Avoid boundaries
    
    for beta in beta_test:
        print(f"\nβ={beta}:")
        solutions = {}
        for M_test in M_test_values:
            x_ref_test, u_ref_test = reference_solution(alpha_xalpha, beta, M=M_test, N=None, x_eval=x_test)
            solutions[M_test] = u_ref_test
        
        # Compare consecutive M values
        M_ref = M_test_values[0]
        for M_test in M_test_values[1:]:
            diff = np.abs(solutions[M_test] - solutions[M_ref])
            rel_diff = diff / (np.abs(solutions[M_ref]) + 1e-12)
            max_rel_diff = np.max(rel_diff)
            mean_rel_diff = np.mean(rel_diff)
            L2_diff = np.sqrt(np.trapz(diff**2, x_test))
            
            print(f"  M={M_ref} vs M={M_test}:")
            print(f"    Max relative diff: {max_rel_diff:.2e}")
            print(f"    Mean relative diff: {mean_rel_diff:.2e}")
            print(f"    L2 difference: {L2_diff:.2e}")
            
            # If differences are small, DST is converged
            if max_rel_diff < 1e-6:
                print(f"    ✓ DST converged (M={M_test} is sufficient)")
            elif max_rel_diff < 1e-4:
                print(f"    ⚠ DST may need more modes (M={M_test} might be borderline)")
            else:
                print(f"    ✗ DST not converged (need M > {M_test})")
            
            M_ref = M_test
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("  - If DST differences are large (>1e-4), the reference solution")
    print("    may be inaccurate, affecting convergence rate calculations.")
    print("  - For fractional β, more modes may be needed due to slower")
    print("    decay of Fourier coefficients: u_n ~ (n*π)^(-2β)")
    print("  - Smaller β values require more modes for the same accuracy.")
    print("="*70 + "\n")

# %%
