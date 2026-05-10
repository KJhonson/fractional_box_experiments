# %%

"""
Fractional Dirichlet problem experiment with exact Fourier series solution.
Problem:
    Find u such that:
        (-u''(x) + κu(x))^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0                    (Dirichlet boundary conditions)
    
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


def solve_fractional_dirichlet_on_mesh(n_elements, beta=0.5, kappa_func=None, f_source_func=None, use_dual_fvm=False):
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
    if use_dual_fvm:
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
    """Indicator function: f(x) = 1_[0,1/2](x)"""
    return np.where(x[0] <= 0.5, 1.0, 0.0)


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

n_list_frac_indicator_exact = [9, 17, 33, 65, 129]
beta_values_indicator_exact = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Adjust N_modes based on beta for better convergence
N_modes_map = {0.3: 8000, 0.4: 6000, 0.5: 4000, 0.8: 2000, 1.0: 1000}

results_frac_indicator_exact_fem = {}
results_frac_indicator_exact_fvm = {}

for beta in beta_values_indicator_exact:
    N_modes = N_modes_map.get(beta, 4000)
    u_exact_func = sol_exact_dirichlet_series(beta, N_modes=N_modes)
    
    meshf_indicator_exact = dmesh.create_interval(MPI.COMM_WORLD, 2000, [0.0, 1.0])
    Vf_indicator_exact = fem.functionspace(meshf_indicator_exact, ("CG", 1))
    
    errors_indicator_exact_fem = []
    errors_indicator_exact_fvm = []
    mesh_sizes_indicator_exact = []
    
    for n in n_list_frac_indicator_exact:
        mesh_local_indicator_exact_fem, _, u_h_local_indicator_exact_fem = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_dual_fvm=False
        )
        mesh_local_indicator_exact_fvm, _, u_h_local_indicator_exact_fvm = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=f_indicator_half_one, use_dual_fvm=True
        )
        
        h = 1.0 / n
        mesh_sizes_indicator_exact.append(h)
        norm_choice_exact = "L2"
        
        error_fem = get_norm(u_h_local_indicator_exact_fem, u_exact_func, mesh_local_indicator_exact_fem, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fem.append(error_fem)
        
        error_fvm = get_norm(u_h_local_indicator_exact_fvm, u_exact_func, mesh_local_indicator_exact_fvm, meshf_indicator_exact, Vf_indicator_exact, norm_choice_exact)
        errors_indicator_exact_fvm.append(error_fvm)
    
    results_frac_indicator_exact_fem[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fem))
    results_frac_indicator_exact_fvm[beta] = (np.array(mesh_sizes_indicator_exact), np.array(errors_indicator_exact_fvm))
    
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

dofs_list_frac_indicator_exact = []
for n in n_list_frac_indicator_exact:
    mesh_temp_indicator_exact = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp_indicator_exact = fem.functionspace(mesh_temp_indicator_exact, ("CG", 1))
    dofs_list_frac_indicator_exact.append(V_temp_indicator_exact.dofmap.index_map.size_local)

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
        
        color = colors_indicator_exact[i % len(colors_indicator_exact)]
        marker = markers_indicator_exact[i % len(markers_indicator_exact)]
        
        log_dofs_indicator_exact = np.log10(dofs_array_frac_indicator_exact)
        log_errors_fem = np.log10(L2_errors_indicator_exact_fem)
        log_errors_fvm = np.log10(L2_errors_indicator_exact_fvm)
        
        slope_fem, intercept_fem = np.polyfit(log_dofs_indicator_exact, log_errors_fem, 1)
        convergence_rate_fem = -slope_fem
        
        slope_fvm, intercept_fvm = np.polyfit(log_dofs_indicator_exact, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        # Plot FEM (dashed) and FVM (solid) with same color and marker
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fem, 
                  color=color, marker=marker, linestyle='--', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator_exact, L2_errors_indicator_exact_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel(f'L2 Error', fontsize=12)
    plt.title('Convergence: Error vs DOFs (log-log scale) - Fractional Dirichlet with Indicator f=1_[0,1/2] (Exact Solution)', fontsize=14)
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
    
    u_exact_indicator_exact_vals = np.array([u_exact_indicator_exact_func(np.array([x])) for x in x_fine_indicator_exact])
    
    f_indicator_exact_vals = np.array([f_indicator_half_one(np.array([x])) for x in x_fine_indicator_exact])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine_indicator_exact, u_exact_indicator_exact_vals, 'b-', label=f'Exact solution (β={beta_plot}, Fourier series)', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fem, 'r--', label=f'FEM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, u_h_indicator_exact_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator_exact, f_indicator_exact_vals, 'm:', label='Source f=1_[0,1/2]', linewidth=2, alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fem, u_h_indicator_exact_vals_fem, 'o', markersize=4, label='FEM DOFs', alpha=0.6)
    plt.plot(x_coarse_indicator_exact_fvm, u_h_indicator_exact_vals_fvm, 'x', markersize=4, label='FVM DOFs', alpha=0.6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Comparison: Exact vs Numerical Solution - Fractional Dirichlet with Indicator f=1_[0,1/2] (β={beta_plot})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
