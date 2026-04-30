# %%

"""
Fractional Dirichlet problem experiment with overkill method.
Source function: indicator function f(x) = 1_[0,1/2]

Problem:
    Find u such that:
        (-u''(x) + κu(x))^β u(x) = f(x)    in (0, 1)
        u(0) = u(1) = 0                    (Dirichlet boundary conditions)
    
    where:
        f(x) = 1_[0,1/2](x) = {1 if x ∈ [0, 1/2), 0 if x ∈ [1/2, 1]}
        κ = 1.0 (constant coefficient)
        β > 0 (fractional power)
    
    Reference solution: Computed using overkill mesh (N = max(n_list) * 16)
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
        # Use FEM-exact load (element-based integration)
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


def source_expr(x):
    return np.where(x[0] <= 0.5, 1.0, 0.0)
    
    # x_vals = x[0]
    # result = np.where(x_vals < 0.5, 1.0, 0.0)
    # result = np.where(np.abs(x_vals - 0.5) < 1e-10, 0.5, result)
    # return result
    
    # x_vals = x[0]
    # eps = 1e-12  # to avoid division by zero
    # safe_x = np.maximum(x_vals, eps)
    # return safe_x ** (-0.49)


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


# %%
# Run experiments

n_list_frac_indicator2 = [8, 16, 32, 64, 128]
beta_values_indicator2 = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

n_overkill_indicator2 = max(n_list_frac_indicator2) * 10

results_frac_indicator2_fem = {}
results_frac_indicator2_fvm = {}
results_frac_indicator2_fem_exact = {}
results_frac_indicator2_fvm_exact = {}

for beta in beta_values_indicator2:
    # FEM experiments
    mesh_overkill_indicator2_fem, _, u_overkill_indicator2_fem = solve_fractional_dirichlet_on_mesh(
        n_overkill_indicator2, beta=beta, f_source_func=source_expr, use_dual_fvm=False
    )
    
    meshf_indicator2 = dmesh.create_interval(MPI.COMM_WORLD, 3000, [0.0, 1.0])
    Vf_indicator2 = fem.functionspace(meshf_indicator2, ("CG", 1))
    
    errors_indicator2_fem = []
    errors_indicator2_fvm = []
    errors_indicator2_fem_exact = []
    errors_indicator2_fvm_exact = []
    mesh_sizes_indicator2 = []
    
    for n in n_list_frac_indicator2:
        mesh_local_indicator2_fem, _, u_h_local_indicator2_fem = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=source_expr, use_dual_fvm=False
        )
        mesh_local_indicator2_fvm, _, u_h_local_indicator2_fvm = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=source_expr, use_dual_fvm=True
        )
        mesh_local_indicator2_fem_exact, _, u_h_local_indicator2_fem_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=source_expr, use_exact_fvm_indicator=True
        )
        mesh_local_indicator2_fvm_exact, _, u_h_local_indicator2_fvm_exact = solve_fractional_dirichlet_on_mesh(
            n, beta=beta, f_source_func=source_expr, use_fvm_exact=True
        )
        
        h = 1.0 / n
        mesh_sizes_indicator2.append(h)
        norm_choice = "L2"
        
        error_fem = get_norm(u_h_local_indicator2_fem, u_overkill_indicator2_fem, mesh_local_indicator2_fem, meshf_indicator2, Vf_indicator2, norm_choice)
        errors_indicator2_fem.append(error_fem)
        
        error_fvm = get_norm(u_h_local_indicator2_fvm, u_overkill_indicator2_fem, mesh_local_indicator2_fvm, meshf_indicator2, Vf_indicator2, norm_choice)
        errors_indicator2_fvm.append(error_fvm)
        
        error_fem_exact = get_norm(u_h_local_indicator2_fem_exact, u_overkill_indicator2_fem, mesh_local_indicator2_fem_exact, meshf_indicator2, Vf_indicator2, norm_choice)
        errors_indicator2_fem_exact.append(error_fem_exact)
        
        error_fvm_exact = get_norm(u_h_local_indicator2_fvm_exact, u_overkill_indicator2_fem, mesh_local_indicator2_fvm_exact, meshf_indicator2, Vf_indicator2, norm_choice)
        errors_indicator2_fvm_exact.append(error_fvm_exact)
    
    results_frac_indicator2_fem[beta] = (np.array(mesh_sizes_indicator2), np.array(errors_indicator2_fem))
    results_frac_indicator2_fvm[beta] = (np.array(mesh_sizes_indicator2), np.array(errors_indicator2_fvm))
    results_frac_indicator2_fem_exact[beta] = (np.array(mesh_sizes_indicator2), np.array(errors_indicator2_fem_exact))
    results_frac_indicator2_fvm_exact[beta] = (np.array(mesh_sizes_indicator2), np.array(errors_indicator2_fvm_exact))
    
    if mesh_overkill_indicator2_fem.comm.rank == 0:
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Overkill method [0, 1/2] - FEM:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator2, mesh_sizes_indicator2, errors_indicator2_fem):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Overkill method [0, 1/2] - FVM:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator2, mesh_sizes_indicator2, errors_indicator2_fvm):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Overkill method [0, 1/2] - FEM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator2, mesh_sizes_indicator2, errors_indicator2_fem_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")
        
        print(f"\nFractional Dirichlet Indicator (β={beta}) - Overkill method [0, 1/2] - FVM-exact:")
        print(f"{'N':<6} {'h':<12} {f'{norm_choice} Error':<12}")
        print("-" * 30)
        for n, h, err in zip(n_list_frac_indicator2, mesh_sizes_indicator2, errors_indicator2_fvm_exact):
            print(f"{n:<6} {h:<12.6e} {err:<12.6e}")

dofs_list_frac_indicator2 = []
for n in n_list_frac_indicator2:
    mesh_temp_indicator2 = dmesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
    V_temp_indicator2 = fem.functionspace(mesh_temp_indicator2, ("CG", 1))
    dofs_list_frac_indicator2.append(V_temp_indicator2.dofmap.index_map.size_local)

dofs_array_frac_indicator2 = np.array(dofs_list_frac_indicator2)

if MPI.COMM_WORLD.rank == 0:
    colors_indicator2 = ['b', 'g', 'r', 'orange', 'purple', 'brown']
    markers_indicator2 = ['o', 's', '^', 'v', 'D', 'p']
    
    plt.figure(figsize=(10, 6))
    
    legend_handles = []
    legend_labels = []
    
    for i, beta in enumerate(beta_values_indicator2):
        h_values_indicator2, L2_errors_indicator2_fem = results_frac_indicator2_fem[beta]
        _, L2_errors_indicator2_fvm = results_frac_indicator2_fvm[beta]
        _, L2_errors_indicator2_fem_exact = results_frac_indicator2_fem_exact[beta]
        _, L2_errors_indicator2_fvm_exact = results_frac_indicator2_fvm_exact[beta]
        
        color = colors_indicator2[i % len(colors_indicator2)]
        marker = markers_indicator2[i % len(markers_indicator2)]
        
        log_dofs_indicator2 = np.log10(dofs_array_frac_indicator2)
        log_errors_fem = np.log10(L2_errors_indicator2_fem)
        log_errors_fvm = np.log10(L2_errors_indicator2_fvm)
        log_errors_fem_exact = np.log10(L2_errors_indicator2_fem_exact)
        log_errors_fvm_exact = np.log10(L2_errors_indicator2_fvm_exact)
        
        slope_fem, intercept_fem = np.polyfit(log_dofs_indicator2, log_errors_fem, 1)
        convergence_rate_fem = -slope_fem
        
        slope_fvm, intercept_fvm = np.polyfit(log_dofs_indicator2, log_errors_fvm, 1)
        convergence_rate_fvm = -slope_fvm
        
        slope_fem_exact, intercept_fem_exact = np.polyfit(log_dofs_indicator2, log_errors_fem_exact, 1)
        convergence_rate_fem_exact = -slope_fem_exact
        
        slope_fvm_exact, intercept_fvm_exact = np.polyfit(log_dofs_indicator2, log_errors_fvm_exact, 1)
        convergence_rate_fvm_exact = -slope_fvm_exact
        
        # Plot FEM (dashed), FVM (solid), FEM-exact (dotted), and FVM-exact (dashdot) with same color and marker
        plt.loglog(dofs_array_frac_indicator2, L2_errors_indicator2_fem, 
                  color=color, marker=marker, linestyle='--', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator2, L2_errors_indicator2_fvm, 
                  color=color, marker=marker, linestyle='-', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator2, L2_errors_indicator2_fem_exact, 
                  color=color, marker=marker, linestyle=':', 
                  linewidth=2, markersize=8, alpha=0.8)
        plt.loglog(dofs_array_frac_indicator2, L2_errors_indicator2_fvm_exact, 
                  color=color, marker=marker, linestyle='-.', 
                  linewidth=2, markersize=8, alpha=0.8)
        
        # Create legend entry with marker only
        marker_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=8)
        legend_handles.append(marker_handle)
        legend_labels.append(f'β={beta}, FEM-{convergence_rate_fem:.2f}, FVM-{convergence_rate_fvm:.2f}, FEM-exact-{convergence_rate_fem_exact:.2f}, FVM-exact-{convergence_rate_fvm_exact:.2f}')
    
    plt.xlabel('Number of DOFs', fontsize=12)
    plt.ylabel(f'L2 Error', fontsize=12)
    plt.title('Convergence: Error vs DOFs (log-log scale) - Fractional Dirichlet with Indicator f=1_[0,1/2]', fontsize=14)
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

mesh_indicator2_plot_fem, V_indicator2_plot_fem, u_h_indicator2_plot_fem = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=source_expr, use_dual_fvm=False)
mesh_indicator2_plot_fvm, V_indicator2_plot_fvm, u_h_indicator2_plot_fvm = solve_fractional_dirichlet_on_mesh(N_plot, beta=beta_plot, f_source_func=source_expr, use_dual_fvm=True)
mesh_overkill_indicator2_plot, V_overkill_indicator2_plot, u_overkill_indicator2_plot = solve_fractional_dirichlet_on_mesh(n_overkill_indicator2, beta=beta_plot, f_source_func=source_expr, use_dual_fvm=False)

meshf_indicator2_plot = dmesh.create_interval(MPI.COMM_WORLD, 1000, [0.0, 1.0])
Vf_indicator2_plot = fem.functionspace(meshf_indicator2_plot, ("CG", 1))

if mesh_indicator2_plot_fem.comm.rank == 0:
    x_fine_indicator2 = Vf_indicator2_plot.tabulate_dof_coordinates()[:, 0]
    
    x_coarse_indicator2_fem = mesh_indicator2_plot_fem.geometry.x[:, 0]
    u_h_indicator2_vals_fem = u_h_indicator2_plot_fem.x.array
    sort_idx_indicator2_fem = np.argsort(x_coarse_indicator2_fem)
    u_h_indicator2_interp_fem = np.interp(x_fine_indicator2, x_coarse_indicator2_fem[sort_idx_indicator2_fem], u_h_indicator2_vals_fem[sort_idx_indicator2_fem])
    
    x_coarse_indicator2_fvm = mesh_indicator2_plot_fvm.geometry.x[:, 0]
    u_h_indicator2_vals_fvm = u_h_indicator2_plot_fvm.x.array
    sort_idx_indicator2_fvm = np.argsort(x_coarse_indicator2_fvm)
    u_h_indicator2_interp_fvm = np.interp(x_fine_indicator2, x_coarse_indicator2_fvm[sort_idx_indicator2_fvm], u_h_indicator2_vals_fvm[sort_idx_indicator2_fvm])
    
    x_overkill_indicator2 = mesh_overkill_indicator2_plot.geometry.x[:, 0]
    u_overkill_indicator2_vals = u_overkill_indicator2_plot.x.array
    sort_idx_overkill_indicator2 = np.argsort(x_overkill_indicator2)
    u_overkill_indicator2_interp = np.interp(x_fine_indicator2, x_overkill_indicator2[sort_idx_overkill_indicator2], u_overkill_indicator2_vals[sort_idx_overkill_indicator2])
    
    f_indicator2_vals = np.array([source_expr(np.array([x])) for x in x_fine_indicator2])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine_indicator2, u_overkill_indicator2_interp, 'b-', label=f'Overkill solution (N={n_overkill_indicator2}, β={beta_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator2, u_h_indicator2_interp_fem, 'r--', label=f'FEM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator2, u_h_indicator2_interp_fvm, 'g-', label=f'FVM solution (N={N_plot})', linewidth=2, alpha=0.8)
    plt.plot(x_fine_indicator2, f_indicator2_vals, 'm:', label='Source f=1_[0,1/2]', linewidth=2, alpha=0.6)
    plt.plot(x_coarse_indicator2_fem, u_h_indicator2_vals_fem, 'o', markersize=4, label='FEM DOFs', alpha=0.6)
    plt.plot(x_coarse_indicator2_fvm, u_h_indicator2_vals_fvm, 'x', markersize=4, label='FVM DOFs', alpha=0.6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Comparison: Overkill vs Numerical Solution - Fractional Dirichlet with Indicator f=1_[0,1/2] (β={beta_plot})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# %%
