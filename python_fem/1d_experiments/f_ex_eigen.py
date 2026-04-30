# %%

"""
Fractional Dirichlet problem experiment with exact solution.
Problem:
    Find u such that:
        (-u''(x) + κu(x))^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0                    (Dirichlet boundary conditions)
    
    where:
        f(x) = sin(πx)
        κ = 1.0 (constant coefficient)
        β > 0 (fractional power)
    
    The exact solution is:
        u(x) = sin(πx) / (1 + π²)^β
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
        f_source_func_actual = f_source_func if f_source_func else lambda x: np.sin(np.pi * x[0])
        combined_form = a + kappa_val * m
        b = assemble_rhs_dual_fvm(V_local, f_source_func_actual, quad_degree=3, bc=bc, combined_form_for_lifting=combined_form)
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


def sol_exact_dirichlet(beta):
    """Exact solution for fractional Dirichlet problem with f(x) = sin(πx)"""
    return lambda x: np.sin(np.pi * x[0]) / ((1 + np.pi**2)**beta)


def numerical_experiment_fractional_dirichlet(n_elements_list, beta=0.5, kappa_func=None, f_source_func=None, norm_type="L2", n_integration=1000, use_dual_fvm=False):
    meshf = dmesh.create_interval(MPI.COMM_WORLD, n_integration, [0.0, 1.0])
    Vf = fem.functionspace(meshf, ("CG", 1))
    
    u_exact_func = sol_exact_dirichlet(beta)
    
    errors = []
    mesh_sizes = []
    
    for n in n_elements_list:
        mesh_local, _, u_h_local = solve_fractional_dirichlet_on_mesh(n, beta, kappa_func, f_source_func, use_dual_fvm=use_dual_fvm)
        
        h = 1.0 / n
        mesh_sizes.append(h)
        
        error = get_norm(u_h_local, u_exact_func, mesh_local, meshf, Vf, norm_type)
        errors.append(error)
    
    return np.array(mesh_sizes), np.array(errors)


# %%
# Run experiments

n_list_frac = [8, 16, 32, 64, 128]
beta_values = [0.3, 0.4, 0.5, 0.8, 1.0]

results_frac_fem = {}
results_frac_fvm = {}

# FEM experiments
for beta in beta_values:
    h_values_frac, L2_errors_frac = numerical_experiment_fractional_dirichlet(n_list_frac, beta=beta, norm_type="L2", use_dual_fvm=False)
    results_frac_fem[beta] = (h_values_frac, L2_errors_frac)
    print(f"\nFractional Dirichlet (β={beta}) - FEM method:")
    print(f"{'N':<6} {'h':<12} {'L2 Error':<12}")
    print("-" * 30)
    for n, h, err in zip(n_list_frac, h_values_frac, L2_errors_frac):
        print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

# FVM experiments
for beta in beta_values:
    h_values_frac, L2_errors_frac = numerical_experiment_fractional_dirichlet(n_list_frac, beta=beta, norm_type="L2", use_dual_fvm=True)
    results_frac_fvm[beta] = (h_values_frac, L2_errors_frac)
    print(f"\nFractional Dirichlet (β={beta}) - FVM method:")
    print(f"{'N':<6} {'h':<12} {'L2 Error':<12}")
    print("-" * 30)
    for n, h, err in zip(n_list_frac, h_values_frac, L2_errors_frac):
        print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

dofs_list_frac = []
for n in n_list_frac:
    mesh_temp = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp = fem.functionspace(mesh_temp, ("CG", 1))
    dofs_list_frac.append(V_temp.dofmap.index_map.size_local)

dofs_array_frac = np.array(dofs_list_frac)

if MPI.COMM_WORLD.rank == 0:
    colors = ['b', 'g', 'r', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', 'D']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values):
        # FEM results (dashed lines)
        h_values_fem, L2_errors_fem = results_frac_fem[beta]
        log_dofs = np.log10(dofs_array_frac)
        log_errors_fem = np.log10(L2_errors_fem)
        slope_fem, intercept_fem = np.polyfit(log_dofs, log_errors_fem, 1)
        convergence_rate_fem = -slope_fem
        
        # FVM results (solid lines)
        h_values_fvm, L2_errors_fvm = results_frac_fvm[beta]
        log_errors_fvm = np.log10(L2_errors_fvm)
        slope_fvm, intercept_fvm = np.polyfit(log_dofs, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        # Plot FEM (dashed) - no label
        plt.loglog(dofs_array_frac, L2_errors_fem, 
                  color=colors[i % len(colors)], marker=markers[i % len(markers)],
                  linestyle='--', linewidth=2, markersize=8)
        
        # Plot FVM (solid) - no label
        plt.loglog(dofs_array_frac, L2_errors_fvm, 
                  color=colors[i % len(colors)], marker=markers[i % len(markers)],
                  linestyle='-', linewidth=2, markersize=8)
        
        # Create custom legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                   marker=markers[i % len(markers)], 
                                   linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title('Convergence: Error vs DOFs (log-log scale) - Fractional Dirichlet', fontsize=14)
    plt.legend(legend_handles, legend_labels, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Solution comparison plot

beta_plot = 0.3
N_plot = 32

print(f"\n" + "="*70)
print(f"Solution Comparison Plot")
print(f"="*70)
print(f"Hyperparameters:")
print(f"  β = {beta_plot}")
print(f"  N = {N_plot}")
print(f"="*70)

mesh_frac_plot, V_frac_plot, u_h_frac_plot = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, use_dual_fvm=False)
mesh_frac_plot_fvm, V_frac_plot_fvm, u_h_frac_plot_fvm = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, use_dual_fvm=True)
u_exact_frac_func = sol_exact_dirichlet(beta_plot)

meshf_frac_plot = dmesh.create_interval(MPI.COMM_WORLD, 1000, [0.0, 1.0])
Vf_frac_plot = fem.functionspace(meshf_frac_plot, ("CG", 1))

if mesh_frac_plot.comm.rank == 0:
    x_fine = Vf_frac_plot.tabulate_dof_coordinates()[:, 0]
    
    x_coarse = mesh_frac_plot.geometry.x[:, 0]
    u_h_frac_vals = u_h_frac_plot.x.array
    sort_idx = np.argsort(x_coarse)
    u_h_frac_interp = np.interp(x_fine, x_coarse[sort_idx], u_h_frac_vals[sort_idx])
    
    x_coarse_fvm = mesh_frac_plot_fvm.geometry.x[:, 0]
    u_h_frac_vals_fvm = u_h_frac_plot_fvm.x.array
    sort_idx_fvm = np.argsort(x_coarse_fvm)
    u_h_frac_interp_fvm = np.interp(x_fine, x_coarse_fvm[sort_idx_fvm], u_h_frac_vals_fvm[sort_idx_fvm])
    
    u_exact_frac_vals = np.array([u_exact_frac_func(np.array([x])) for x in x_fine])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, u_exact_frac_vals, 'b-', label=f'Exact solution (β={beta_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine, u_h_frac_interp, 'r--', label=f'FEM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine, u_h_frac_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_coarse, u_h_frac_vals, 'o', markersize=6, label='FEM DOFs', alpha=0.6)
    plt.plot(x_coarse_fvm, u_h_frac_vals_fvm, 'x', markersize=6, label='FVM DOFs', alpha=0.6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


# %%
