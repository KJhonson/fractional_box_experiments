# %%
"""
Problem:
    Find u such that:
        -u''(x) + κu(x) = f(x)    in (0, 1)
    
    where:
        f(x) = sin(πx)
        κ = 1.0 (constant coefficient)
    
    The exact solution is:
        u(x) = sin(πx) / (π² + κ)
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
from utils.loads import assemble_rhs_dual_fvm


def f_source(x):
    """Eigenfunction source: f(x) = sin(πx)"""
    return np.sin(np.pi * x[0])


def solve_problem_on_mesh(n_elements, kappa_func=None, f_source_func=None, use_dual_fvm=False):
    mesh_local = dmesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, 1.0])
    V_local = fem.functionspace(mesh_local, ("CG", 1))
    
    x = ufl.SpatialCoordinate(mesh_local)
    if kappa_func is None:
        kappa_val = fem.Constant(mesh_local, 1.0)
    else:
        kappa_val = kappa_func(x)
    
    if f_source_func is None:
        def f_source_func(x):
            return np.sin(np.pi * x[0])
    
    u = ufl.TrialFunction(V_local)
    v = ufl.TestFunction(V_local)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + kappa_val * ufl.inner(u, v) * ufl.dx
    
    K = petsc.assemble_matrix(fem.form(a))
    K.assemble()
    
    # Choose RHS assembly method
    if use_dual_fvm:
        # Use dual FVM RHS
        b = assemble_rhs_dual_fvm(V_local, f_source_func, quad_degree=3)
    else:
        # Use standard FEM RHS
        f_func_local = fem.Function(V_local)
        f_func_local.interpolate(f_source_func)
        L = f_func_local * v * ufl.dx
        b = petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    
    ksp = PETSc.KSP().create(K.comm)
    ksp.setOperators(K)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=10000)
    
    u_vec = K.createVecRight()
    ksp.solve(b, u_vec)
    
    u_h_local = fem.Function(V_local)
    u_h_local.x.array[:] = u_vec.array
    u_h_local.x.scatter_forward()
    
    return mesh_local, V_local, u_h_local


def numerical_experiment(n_elements_list, u_exact_func=None, n_overkill=None, kappa_func=None, f_source_func=None, norm_type="L2", n_integration=1000, use_dual_fvm=False):
    meshf = dmesh.create_interval(MPI.COMM_WORLD, n_integration, [0.0, 1.0])
    Vf = fem.functionspace(meshf, ("CG", 1))
    
    if u_exact_func is None:
        if n_overkill is None:
            n_overkill = max(n_elements_list) * 8
        _, _, u_overkill = solve_problem_on_mesh(n_overkill, kappa_func, f_source_func, use_dual_fvm=use_dual_fvm)
        u_exact_func = u_overkill
    else:
        pass
    errors = []
    mesh_sizes = []
    for n in n_elements_list:
        mesh_local, _, u_h_local = solve_problem_on_mesh(n, kappa_func, f_source_func, use_dual_fvm=use_dual_fvm)
        
        h = 1.0 / n
        mesh_sizes.append(h)
        
        error = get_norm(u_h_local, u_exact_func, mesh_local, meshf, Vf, norm_type)
        errors.append(error)
    
    return np.array(mesh_sizes), np.array(errors)


# %%
# Run experiments

n_list = [8, 16, 32, 64, 128]

# FEM experiments
h_values_exact_fem, L2_errors_exact_fem = numerical_experiment(n_list, u_exact_func=f_source, f_source_func=f_source, kappa_func=None, norm_type="H1", use_dual_fvm=False)
h_values_overkill_fem, L2_errors_overkill_fem = numerical_experiment(n_list, n_overkill=1000, f_source_func=f_source, kappa_func=None, norm_type="H1", use_dual_fvm=False)

# FVM experiments
h_values_exact_fvm, L2_errors_exact_fvm = numerical_experiment(n_list, u_exact_func=f_source, f_source_func=f_source, kappa_func=None, norm_type="H1", use_dual_fvm=True)
h_values_overkill_fvm, L2_errors_overkill_fvm = numerical_experiment(n_list, n_overkill=1000, f_source_func=f_source, kappa_func=None, norm_type="H1", use_dual_fvm=True)

print(f"\nExact method - FEM:")
print(f"{'N':<6} {'h':<12} {'H1 Error':<12}")
print("-" * 30)
for n, h, err in zip(n_list, h_values_exact_fem, L2_errors_exact_fem):
    print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

print(f"\nExact method - FVM:")
print(f"{'N':<6} {'h':<12} {'H1 Error':<12}")
print("-" * 30)
for n, h, err in zip(n_list, h_values_exact_fvm, L2_errors_exact_fvm):
    print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

print(f"\nOverkill method - FEM:")
print(f"{'N':<6} {'h':<12} {'H1 Error':<12}")
print("-" * 30)
for n, h, err in zip(n_list, h_values_overkill_fem, L2_errors_overkill_fem):
    print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

print(f"\nOverkill method - FVM:")
print(f"{'N':<6} {'h':<12} {'H1 Error':<12}")
print("-" * 30)
for n, h, err in zip(n_list, h_values_overkill_fvm, L2_errors_overkill_fvm):
    print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

dofs_list = []
for n in n_list:
    mesh_temp = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp = fem.functionspace(mesh_temp, ("CG", 1))
    dofs_list.append(V_temp.dofmap.index_map.size_local)

dofs_array = np.array(dofs_list)

if MPI.COMM_WORLD.rank == 0:
    log_dofs = np.log10(dofs_array)
    
    # FEM slopes
    log_errors_fem = np.log10(L2_errors_overkill_fem)
    slope_fem, intercept_fem = np.polyfit(log_dofs, log_errors_fem, 1)
    convergence_rate_fem = -slope_fem
    
    # FVM slopes
    log_errors_fvm = np.log10(L2_errors_overkill_fvm)
    slope_fvm, intercept_fvm = np.polyfit(log_dofs, log_errors_fvm, 1)
    convergence_rate_fvm = -slope_fvm
    
    plt.figure(figsize=(10, 6))
    
    # Plot FEM (dashed) and FVM (solid) - no labels
    plt.loglog(dofs_array, L2_errors_overkill_fem, 'b--', linewidth=2, markersize=8, marker='o', alpha=0.8)
    plt.loglog(dofs_array, L2_errors_overkill_fvm, 'b-', linewidth=2, markersize=8, marker='o', alpha=0.8)
    
    # Create custom legend entry with marker only
    marker_handle = plt.Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=8)
    plt.legend([marker_handle], [f'FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}'], fontsize=11)
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel('H1 Error', fontsize=12)
    plt.title('Convergence: Error vs DOFs (log-log scale)', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# %%
# Solution comparison plot

N_plot = 64

print(f"\n" + "="*70)
print(f"Solution Comparison Plot")
print(f"="*70)
print(f"Hyperparameters:")
print(f"  N = {N_plot}")
print(f"="*70)

mesh_overkill_plot, V_overkill_plot, u_overkill_plot = solve_problem_on_mesh(1000, f_source_func=f_source, use_dual_fvm=False)
mesh_coarse_plot_fem, V_coarse_plot_fem, u_coarse_plot_fem = solve_problem_on_mesh(N_plot, f_source_func=f_source, use_dual_fvm=False)
mesh_coarse_plot_fvm, V_coarse_plot_fvm, u_coarse_plot_fvm = solve_problem_on_mesh(N_plot, f_source_func=f_source, use_dual_fvm=True)

meshf_plot = dmesh.create_interval(MPI.COMM_WORLD, 1000, [0.0, 1.0])
Vf_plot = fem.functionspace(meshf_plot, ("CG", 1))

if mesh_overkill_plot.comm.rank == 0:
    x_fine = Vf_plot.tabulate_dof_coordinates()[:, 0]
    
    x_overkill = mesh_overkill_plot.geometry.x[:, 0]
    u_overkill_vals = u_overkill_plot.x.array
    sort_idx = np.argsort(x_overkill)
    u_overkill_interp = np.interp(x_fine, x_overkill[sort_idx], u_overkill_vals[sort_idx])
    
    x_coarse_fem = mesh_coarse_plot_fem.geometry.x[:, 0]
    u_coarse_vals_fem = u_coarse_plot_fem.x.array
    sort_idx_coarse_fem = np.argsort(x_coarse_fem)
    u_coarse_interp_fem = np.interp(x_fine, x_coarse_fem[sort_idx_coarse_fem], u_coarse_vals_fem[sort_idx_coarse_fem])
    
    x_coarse_fvm = mesh_coarse_plot_fvm.geometry.x[:, 0]
    u_coarse_vals_fvm = u_coarse_plot_fvm.x.array
    sort_idx_coarse_fvm = np.argsort(x_coarse_fvm)
    u_coarse_interp_fvm = np.interp(x_fine, x_coarse_fvm[sort_idx_coarse_fvm], u_coarse_vals_fvm[sort_idx_coarse_fvm])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, u_overkill_interp, 'b-', label=f'Overkill solution (N=1000)', linewidth=2, alpha=0.8)
    plt.plot(x_fine, u_coarse_interp_fem, 'r--', label=f'FEM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine, u_coarse_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_coarse_fem, u_coarse_vals_fem, 'o', markersize=6, label='FEM DOFs', alpha=0.6)
    plt.plot(x_coarse_fvm, u_coarse_vals_fvm, 'x', markersize=6, label='FVM DOFs', alpha=0.6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title('Comparison: Overkill vs Coarse Solution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
