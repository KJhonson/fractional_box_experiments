# %%

"""
Fractional Dirichlet problem experiment with exact Fourier series solution.
Problem:
    Find u such that:
        (-d²/dx² + κ)^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0                (Dirichlet boundary conditions)
    
    where:
        f(x) = 1_[0,1/2](x) = {1 if x ∈ [0, 1/2), 0 if x ∈ [1/2, 1]}
        κ = 1.0 (constant coefficient)
        β > 0 (fractional power)
    
    The exact solution is given by the Fourier series expansion:
        u(x) = Σ_{n=1}^∞ a_n sin(nπx)
    
    where the coefficients are:
        a_n = (2(1 - cos(nπ/2))) / (nπ(λ_n)^β)
        λ_n = (nπ)² + 1  (eigenvalues of the operator -d²/dx² + 1)
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


def solve_fractional_dirichlet_on_mesh(n_elements, beta=0.5, kappa_func=None, f_source_func=None, use_dual_fvm=False, use_exact_fvm_indicator=False, use_fvm_exact=False):
    mesh_local = dmesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, 1.0])
    V_local = fem.functionspace(mesh_local, ("CG", 1))
    
    x = ufl.SpatialCoordinate(mesh_local)
    if kappa_func is None:
        kappa_val = fem.Constant(mesh_local, 1.0)
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
    if use_fvm_exact:
        # Use FVM-exact load (dual cell integration)
        x_coords = mesh_local.geometry.x[:, 0]
        F_numpy = assemble_fv_load_indicator(x_coords)
        # Convert to PETSc vector - create a dummy form to get the right vector structure
        dummy_form = v * ufl.dx
        b = petsc.assemble_vector(fem.form(dummy_form))
        # Set the values from our FVM-exact load
        b.array[:] = F_numpy
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    elif use_exact_fvm_indicator:
        # FEM-exact RHS: ∫ f φ_i dx via assemble_load_indicator (flag name is historical)
        x_coords = mesh_local.geometry.x[:, 0]
        F_numpy = assemble_load_indicator(x_coords)
        # Convert to PETSc vector - create a dummy form to get the right vector structure
        dummy_form = v * ufl.dx
        b = petsc.assemble_vector(fem.form(dummy_form))
        # Set the values from our FEM-exact load
        b.array[:] = F_numpy
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
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
    
    u_h_local, _ = sinc_solver(B, M, b, V_local, bc=bc, beta=beta)
    
    return mesh_local, V_local, u_h_local


def f_indicator_half_one(x):
    """Indicator function: f(x) = 1 for x < 0.5, 0.5 for x = 0.5, 0 for x > 0.5"""
    return np.where(x[0] <= 0.5, 1.0, 0.0)
    
    # x_vals = x[0]
    # result = np.where(x_vals < 0.5, 1.0, 0.0)
    # result = np.where(np.abs(x_vals - 0.5) < 1e-10, 0.5, result)
    # return result


def assemble_load_indicator(x):
    """
    Assemble the exact FVM load vector for f(x) = 1_{[0,1/2]}(x)
    on an arbitrary non-uniform 1D mesh.

    Parameters
    ----------
    x : array_like
        1D array of node coordinates [x0, x1, ..., xN]

    Returns
    -------
    F : ndarray
        Global load vector of length N+1
    """
    N = len(x) - 1
    F = np.zeros(N + 1)
    c = 0.5  # discontinuity point

    for j in range(N):
        a, b = x[j], x[j+1]

        # Case 1: element fully inside [0, 0.5]
        if b <= c:
            f_loc = 0.5 * (b - a) * np.array([1.0, 1.0])

        # Case 2: element fully outside [0, 0.5]
        elif a >= c:
            f_loc = np.array([0.0, 0.0])

        # Case 3: element cut by x = 0.5
        else:
            f1 = (1.0 / (b - a)) * (b * (c - a) - 0.5 * (c**2 - a**2))
            f2 = (1.0 / (b - a)) * (0.5 * (c**2 - a**2) - a * (c - a))
            f_loc = np.array([f1, f2])

        # Assemble
        F[j] += f_loc[0]
        F[j+1] += f_loc[1]

    return F


def assemble_fv_load_indicator(x):
    """
    Finite-volume-style load for f(x) = 1_{[0,1/2]}(x)
    on an arbitrary non-uniform 1D mesh.
    
    Returns integrals over each dual cell.

    Parameters
    ----------
    x : array_like
        1D array of node coordinates [x0, x1, ..., xN]

    Returns
    -------
    F : ndarray
        Global load vector of length N+1 (integrals over dual cells)
    """
    N = len(x) - 1
    F = np.zeros(N + 1)
    c = 0.5

    # midpoints between nodes
    xm = 0.5 * (x[:-1] + x[1:])

    # build dual cell boundaries
    x_dual = np.zeros(N + 2)
    x_dual[0] = x[0]
    x_dual[1:-1] = xm
    x_dual[-1] = x[-1]

    # integrate f(x)=1_{[0,0.5]} over each dual cell
    for i in range(N + 1):
        left = x_dual[i]
        right = x_dual[i + 1]

        # compute overlap with [0, 0.5]
        left_overlap = max(left, 0.0)
        right_overlap = min(right, c)

        F[i] = max(0.0, right_overlap - left_overlap)

    return F


def sol_exact_dirichlet_series(beta: float, N_modes: int = 4000):
    """
    Return a callable u_exact(x) giving the series solution of
    (-u'' + u)^beta u = 1_[0,1/2] on (0,1) with u(0)=u(1)=0.
    
    Parameters
    ----------
    beta : float
        Fractional power β > 0.
    N_modes : int
        Number of Fourier modes to use in the truncation.
        For β smaller (e.g. 0.3), the series converges more slowly → you might want N_modes=6000 or 8000.
        For β ≥ 1, convergence is very fast, and even N_modes=1000 is usually fine.
    
    Returns
    -------
    u_exact : callable
        Function that takes x as a numpy array (shape (1,) or (1, N))
        and returns u_beta(x) as a numpy array of the same length N.
    """
    n = np.arange(1, N_modes + 1, dtype=float)
    lam = (np.pi * n) ** 2 + 1.0  # eigenvalues λ_n = (nπ)^2 + 1
    coeff = 2.0 * (1.0 - np.cos(0.5 * np.pi * n)) / (np.pi * n * lam**beta)
    
    def u_exact(x):
        # x is expected to be either shape (1,) or (1, N) (Dolfinx style)
        x_vals = x[0]
        # handle scalar x or array x
        x_vals = np.atleast_1d(x_vals)
        # sin(nπx) for all n and x
        S = np.sin(np.pi * np.outer(n, x_vals))  # shape (N_modes, N_points)
        u_vals = coeff @ S  # shape (N_points,)
        # If original input was scalar, return scalar
        if x_vals.shape == () or x_vals.size == 1:
            return float(u_vals[0])
        return u_vals
    
    return u_exact


# %%
# Run experiments

n_list_frac_indicator_exact = [8, 16, 32, 64, 128, 256, 512]
beta_values_indicator_exact = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Adjust N_modes based on beta for better convergence
N_modes_map = {0.3: 8000, 0.4: 6000, 0.5: 4000, 0.8: 2000, 1.0: 1000}

results_frac_indicator_exact_fem = {}
results_frac_indicator_exact_fvm = {}
results_frac_indicator_exact_fem_exact = {}
results_frac_indicator_exact_fvm_exact = {}

for beta in beta_values_indicator_exact:
    N_modes = N_modes_map.get(beta, 4000)
    u_exact_func = sol_exact_dirichlet_series(beta, N_modes=N_modes)
    
    meshf_indicator_exact = dmesh.create_interval(MPI.COMM_WORLD, 2000, [0.0, 1.0])
    Vf_indicator_exact = fem.functionspace(meshf_indicator_exact, ("CG", 1))
    
    errors_indicator_exact_fem = []
    errors_indicator_exact_fvm = []
    errors_indicator_exact_fem_exact = []
    errors_indicator_exact_fvm_exact = []
    mesh_sizes_indicator_exact = []
    
    for n in n_list_frac_indicator_exact:
        mesh_local_indicator_exact_fem, _, u_h_local_indicator_exact_fem = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_dual_fvm=False
        )
        mesh_local_indicator_exact_fvm, _, u_h_local_indicator_exact_fvm = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_dual_fvm=True
        )
        mesh_local_indicator_exact_fem_exact, _, u_h_local_indicator_exact_fem_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_exact_fvm_indicator=True
        )
        mesh_local_indicator_exact_fvm_exact, _, u_h_local_indicator_exact_fvm_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_fvm_exact=True
        )
        
        h = 1.0 / n
        mesh_sizes_indicator_exact.append(h)
        norm_choice_exact = "L2"
        
        error_fem = get_norm(u_h_local_indicator_exact_fem, u_exact_func, mesh_local_indicator_exact_fem, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fem.append(error_fem)
        
        error_fvm = get_norm(u_h_local_indicator_exact_fvm, u_exact_func, mesh_local_indicator_exact_fvm, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fvm.append(error_fvm)
        
        error_fem_exact = get_norm(u_h_local_indicator_exact_fem_exact, u_exact_func, mesh_local_indicator_exact_fem_exact, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fem_exact.append(error_fem_exact)
        
        error_fvm_exact = get_norm(u_h_local_indicator_exact_fvm_exact, u_exact_func, mesh_local_indicator_exact_fvm_exact, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fvm_exact.append(error_fvm_exact)
    
    results_frac_indicator_exact_fem[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fem))
    results_frac_indicator_exact_fvm[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fvm))
    results_frac_indicator_exact_fem_exact[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fem_exact))
    results_frac_indicator_exact_fvm_exact[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fvm_exact))
    
    if meshf_indicator_exact.comm.rank == 0:
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Exact method [f=1_[0,1/2], N_modes={N_modes}] - FEM:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_exact} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator_exact, mesh_sizes_indicator_exact, errors_indicator_exact_fem):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Exact method [f=1_[0,1/2], N_modes={N_modes}] - FVM:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_exact} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator_exact, mesh_sizes_indicator_exact, errors_indicator_exact_fvm):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Exact method [f=1_[0,1/2], N_modes={N_modes}] - FEM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_exact} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator_exact, mesh_sizes_indicator_exact, errors_indicator_exact_fem_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Exact method [f=1_[0,1/2], N_modes={N_modes}] - FVM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice_exact} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator_exact, mesh_sizes_indicator_exact, errors_indicator_exact_fvm_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

dofs_list_frac_indicator_exact = []
for n in n_list_frac_indicator_exact:
    mesh_temp_indicator_exact = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp_indicator_exact = fem.functionspace(mesh_temp_indicator_exact, ("CG", 1))
    dofs_list_frac_indicator_exact.append(V_temp_indicator_exact.dofmap.index_map.size_global)

dofs_array_frac_indicator_exact = np.array(dofs_list_frac_indicator_exact)

if MPI.COMM_WORLD.rank == 0:
    colors_indicator_exact = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_indicator_exact = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_indicator_exact):
        h_values_indicator_exact, L2_errors_indicator_exact_fem = results_frac_indicator_exact_fem[beta]
        _, L2_errors_indicator_exact_fvm = results_frac_indicator_exact_fvm[beta]
        _, L2_errors_indicator_exact_fem_exact = results_frac_indicator_exact_fem_exact[beta]
        _, L2_errors_indicator_exact_fvm_exact = results_frac_indicator_exact_fvm_exact[beta]
        
        color = colors_indicator_exact[i % len(colors_indicator_exact)]
        marker = markers_indicator_exact[i % len(markers_indicator_exact)]
        
        log_dofs_indicator_exact = np.log10(dofs_array_frac_indicator_exact)
        log_errors_fem = np.log10(L2_errors_indicator_exact_fem)
        log_errors_fvm = np.log10(L2_errors_indicator_exact_fvm)
        log_errors_fem_exact = np.log10(L2_errors_indicator_exact_fem_exact)
        log_errors_fvm_exact = np.log10(L2_errors_indicator_exact_fvm_exact)
        
        slope_fem, intercept_fem = np.polyfit(log_dofs_indicator_exact, log_errors_fem, 1)
        convergence_rate_fem = -slope_fem
        
        slope_fvm, intercept_fvm = np.polyfit(log_dofs_indicator_exact, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_dofs_indicator_exact, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = -slope_fem_exact
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_dofs_indicator_exact, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = -slope_fvm_exact
        
        # Plot FEM (dashed), FVM (solid), FEM-exact (dotted), and FVM-exact (dashdot) with same color and marker
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fem, 
                  color=color, marker=marker, linestyle='--', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel(f'L2 Error', fontsize=12)
    plt.title('Convergence: Error vs DOFs (log-log scale) - Fractional Dirichlet with Indicator f=1_[0,1/2] (Exact Solution)', fontsize=14)
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Convergence plot: Error vs Mesh Size (h)

if MPI.COMM_WORLD.rank == 0:
    # Calculate mesh sizes
    mesh_sizes_array = 1.0 / np.array(n_list_frac_indicator_exact)
    
    colors_indicator_exact = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_indicator_exact = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_indicator_exact):
        _, L2_errors_indicator_exact_fem = results_frac_indicator_exact_fem[beta]
        _, L2_errors_indicator_exact_fvm = results_frac_indicator_exact_fvm[beta]
        _, L2_errors_indicator_exact_fem_exact = results_frac_indicator_exact_fem_exact[beta]
        _, L2_errors_indicator_exact_fvm_exact = results_frac_indicator_exact_fvm_exact[beta]
        
        color = colors_indicator_exact[i % len(colors_indicator_exact)]
        marker = markers_indicator_exact[i % len(markers_indicator_exact)]
        
        log_h = np.log10(mesh_sizes_array)
        log_errors_fem = np.log10(L2_errors_indicator_exact_fem)
        log_errors_fvm = np.log10(L2_errors_indicator_exact_fvm)
        log_errors_fem_exact = np.log10(L2_errors_indicator_exact_fem_exact)
        log_errors_fvm_exact = np.log10(L2_errors_indicator_exact_fvm_exact)
        
        # If ||e|| ~ C h^p then log||e|| = p log h + const => polyfit slope = p (positive).
        slope_fem, intercept_fem = np.polyfit(log_h, log_errors_fem, 1)
        convergence_rate_fem = slope_fem
        
        slope_fvm, intercept_fvm = np.polyfit(log_h, log_errors_fvm, 1)
        convergence_rate_fvm = slope_fvm
        
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_h, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = slope_fem_exact
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_h, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = slope_fvm_exact
        
        # Plot FEM (dashed), FVM (solid), FEM-exact (dotted), and FVM-exact (dashdot) with same color and marker
        plt.loglog(mesh_sizes_array, L2_errors_indicator_exact_fem, 
                  color=color, marker=marker, linestyle='--', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(mesh_sizes_array, L2_errors_indicator_exact_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(mesh_sizes_array, L2_errors_indicator_exact_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(mesh_sizes_array, L2_errors_indicator_exact_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Maximum Mesh Size (h)', fontsize=12)
    plt.ylabel(f'L2 Error', fontsize=12)
    plt.title('Convergence: Error vs Mesh Size (log-log scale) - Fractional Dirichlet with Indicator f=1_[0,1/2] (Exact Solution)', fontsize=14)
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Solution comparison plot

beta_plot = 0.5
N_plot = 128

print(f"\n" + "="*70)
print(f"Solution Comparison Plot")
print(f"="*70)
print(f"Hyperparameters:")
print(f"  β = {beta_plot}")
print(f"  N = {N_plot}")
print(f"="*70)

# Comparison plot: exact solution vs numerical solution
mesh_indicator_exact_plot_fem, V_indicator_exact_plot_fem, u_h_indicator_exact_plot_fem = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=f_indicator_half_one, use_dual_fvm=False)
mesh_indicator_exact_plot_fvm, V_indicator_exact_plot_fvm, u_h_indicator_exact_plot_fvm = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=f_indicator_half_one, use_dual_fvm=True)
mesh_indicator_exact_plot_fem_exact, V_indicator_exact_plot_fem_exact, u_h_indicator_exact_plot_fem_exact = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=f_indicator_half_one, use_exact_fvm_indicator=True)
mesh_indicator_exact_plot_fvm_exact, V_indicator_exact_plot_fvm_exact, u_h_indicator_exact_plot_fvm_exact = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=f_indicator_half_one, use_fvm_exact=True)
u_exact_indicator_exact_func = sol_exact_dirichlet_series(beta_plot, N_modes=4000)

meshf_indicator_exact_plot = dmesh.create_interval(MPI.COMM_WORLD, 1000, [0.0, 1.0])
Vf_indicator_exact_plot = fem.functionspace(meshf_indicator_exact_plot, ("CG", 1))

if mesh_indicator_exact_plot_fem.comm.rank == 0:
    x_fine_indicator_exact = Vf_indicator_exact_plot.tabulate_dof_coordinates()[:, 0]
    
    x_coarse_indicator_exact_fem = mesh_indicator_exact_plot_fem.geometry.x[:, 0]
    u_h_indicator_exact_vals_fem = u_h_indicator_exact_plot_fem.x.array
    sort_idx_indicator_exact_fem = np.argsort(x_coarse_indicator_exact_fem)
    u_h_indicator_exact_interp_fem = np.interp(x_fine_indicator_exact, x_coarse_indicator_exact_fem[sort_idx_indicator_exact_fem], u_h_indicator_exact_vals_fem[sort_idx_indicator_exact_fem])
    
    x_coarse_indicator_exact_fvm = mesh_indicator_exact_plot_fvm.geometry.x[:, 0]
    u_h_indicator_exact_vals_fvm = u_h_indicator_exact_plot_fvm.x.array
    sort_idx_indicator_exact_fvm = np.argsort(x_coarse_indicator_exact_fvm)
    u_h_indicator_exact_interp_fvm = np.interp(x_fine_indicator_exact, x_coarse_indicator_exact_fvm[sort_idx_indicator_exact_fvm], u_h_indicator_exact_vals_fvm[sort_idx_indicator_exact_fvm])
    
    x_coarse_indicator_exact_fem_exact = mesh_indicator_exact_plot_fem_exact.geometry.x[:, 0]
    u_h_indicator_exact_vals_fem_exact = u_h_indicator_exact_plot_fem_exact.x.array
    sort_idx_indicator_exact_fem_exact = np.argsort(x_coarse_indicator_exact_fem_exact)
    u_h_indicator_exact_interp_fem_exact = np.interp(x_fine_indicator_exact, x_coarse_indicator_exact_fem_exact[sort_idx_indicator_exact_fem_exact], u_h_indicator_exact_vals_fem_exact[sort_idx_indicator_exact_fem_exact])
    
    x_coarse_indicator_exact_fvm_exact = mesh_indicator_exact_plot_fvm_exact.geometry.x[:, 0]
    u_h_indicator_exact_vals_fvm_exact = u_h_indicator_exact_plot_fvm_exact.x.array
    sort_idx_indicator_exact_fvm_exact = np.argsort(x_coarse_indicator_exact_fvm_exact)
    u_h_indicator_exact_interp_fvm_exact = np.interp(x_fine_indicator_exact, x_coarse_indicator_exact_fvm_exact[sort_idx_indicator_exact_fvm_exact], u_h_indicator_exact_vals_fvm_exact[sort_idx_indicator_exact_fvm_exact])
    
    u_exact_indicator_exact_vals = np.array([u_exact_indicator_exact_func(np.array([x])) for x in x_fine_indicator_exact])
    
    f_indicator_exact_vals = np.array([f_indicator_half_one(np.array([x])) for x in x_fine_indicator_exact])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine_indicator_exact, u_exact_indicator_exact_vals, 'b-', label=f'Exact solution (β={beta_plot}, Fourier series)', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fem, 'r--', label=f'FEM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fem_exact, 'm:', label=f'FEM-exact solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fvm_exact, 'c-.', label=f'FVM-exact solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, f_indicator_exact_vals, 'y:', label='Source f=1_[0,1/2]', linewidth=2, alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fem, u_h_indicator_exact_vals_fem, 'o', markersize=4, label='FEM DOFs', alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fvm, u_h_indicator_exact_vals_fvm, 'x', markersize=4, label='FVM DOFs', alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fem_exact, u_h_indicator_exact_vals_fem_exact, 's', markersize=4, label='FEM-exact DOFs', alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fvm_exact, u_h_indicator_exact_vals_fvm_exact, '^', markersize=4, label='FVM-exact DOFs', alpha=0.6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Comparison: Exact vs Numerical Solution - Fractional Dirichlet with Indicator f=1_[0,1/2] (β={beta_plot})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# Save error tables to files

if MPI.COMM_WORLD.rank == 0:
    # Use DOF values instead of h values
    # dofs_array_frac_indicator_exact is already computed above
    
    # Prepare data for FEM-exact errors
    fem_exact_data = []
    fem_exact_data.append(['DOF'] + [str(beta) for beta in beta_values_indicator_exact])
    
    for dof_idx, dof in enumerate(dofs_array_frac_indicator_exact):
        row = [str(int(dof))]
        for beta in beta_values_indicator_exact:
            _, errors = results_frac_indicator_exact_fem_exact[beta]
            row.append(str(errors[dof_idx]))
        fem_exact_data.append(row)
    
    # Prepare data for FVM-exact errors
    fvm_exact_data = []
    fvm_exact_data.append(['DOF'] + [str(beta) for beta in beta_values_indicator_exact])
    
    for dof_idx, dof in enumerate(dofs_array_frac_indicator_exact):
        row = [str(int(dof))]
        for beta in beta_values_indicator_exact:
            _, errors = results_frac_indicator_exact_fvm_exact[beta]
            row.append(str(errors[dof_idx]))
        fvm_exact_data.append(row)
    
    # Create output directory based on experiment file name (without .py extension)
    experiment_name = Path(__file__).stem  # Gets filename without extension: 'f_ex_indicator_v2'
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
    # dofs_array_frac_indicator_exact is already computed above
    
    fem_exact_slopes = []
    fvm_exact_slopes = []
    
    for beta in beta_values_indicator_exact:
        _, L2_errors_fem_exact = results_frac_indicator_exact_fem_exact[beta]
        _, L2_errors_fvm_exact = results_frac_indicator_exact_fvm_exact[beta]
        
        # Use DOF-based calculation: log(error) vs log(DOF)
        log_dofs = np.log10(dofs_array_frac_indicator_exact)
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
        for beta, slope in zip(beta_values_indicator_exact, fem_exact_slopes):
            f.write(f'{beta}\t{slope}\n')
    print(f"Saved FEM-exact beta vs slope to: {betaxslope_fem_file}")
    
    # Save FVM-exact beta vs slope
    betaxslope_fvm_file = output_dir / 'betaxslope_1dfvm.dat'
    with open(betaxslope_fvm_file, 'w') as f:
        f.write('x\ty\n')
        for beta, slope in zip(beta_values_indicator_exact, fvm_exact_slopes):
            f.write(f'{beta}\t{slope}\n')
    print(f"Saved FVM-exact beta vs slope to: {betaxslope_fvm_file}")

# %%

