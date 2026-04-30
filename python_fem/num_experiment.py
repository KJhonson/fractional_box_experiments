# %%

import numpy as np
from mpmath import quad, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ufl

# FEM imports (reuse project utilities)
from dolfinx import fem
from mpi4py import MPI
from pathlib import Path
from domains import make_uniform_interval, tag_all_exterior_facets
from operators import assemble_K_M, build_operator_B
from utils.loads import assemble_rhs, assemble_rhs_dual_fvm, assemble_rhs_dual_fvm_exact
from utils.sinc_solver import sinc_solver
from playground import compute_errors_refined, dynamic_k
from dolfinx.fem import petsc
from petsc4py import PETSc

# --- Cálculo dos coeficientes f_k ---
def fk_continuo(k, alpha):
    integrand = lambda x: (x ** (-alpha)) * sin(k * pi * x)
    val = quad(integrand, [0, 1])
    return np.sqrt(2.0) * float(val)

# --- Solução espectral contínua ---
def solve_spectral_continua(alpha, beta, K, x_grid):
    uvals = np.zeros_like(x_grid, dtype=float)
    for k in range(1, K + 1):
        fk = fk_continuo(k, alpha)
        ak = fk / ((k * np.pi) ** (2 * beta))
        uvals += ak * np.sqrt(2.0) * np.sin(np.pi * k * x_grid)
    return uvals
# =====================
# Num. experiment (Dirichlet) for (-d^2/dx^2)^beta u = f with f=x^{-alpha}
# =====================

def run_dirichlet_fractional_experiment(alpha=0.49,
                                        beta=0.5,
                                        K=1000,
                                        dofs_list=(10, 20, 40, 80, 160, 240),
                                        N_ref=4000,
                                        sinc_k=0.15,
                                        only_check_N=40):
    # Choose exact/reference solution based on beta
    if abs(beta - 1.0) < 1e-12:
        # Closed-form exact solution for -u'' = x^(-alpha) with Dirichlet BC
        def u_exact_callable(x_arr):
            x = np.asarray(x_arr, dtype=float)
            mask = x > 1e-15
            result = np.zeros_like(x)
            denom = (1.0 - alpha) * (2.0 - alpha)
            result[mask] = (x[mask] - x[mask]**(2.0 - alpha)) / denom
            result[~mask] = 0.0
            return result
        x_ref = np.linspace(0, 1, N_ref+2)[1:-1]
        u_ref = u_exact_callable(x_ref)
        ref_label = 'Exact (closed-form)'
        ref_name_print = 'exact'
    else:
        # Fractional case: spectral reference
        x_ref = np.linspace(0, 1, N_ref+2)[1:-1]
        u_ref = solve_spectral_continua(alpha, beta, K, x_ref)
        x_ref_sorted = x_ref
        u_ref_sorted = u_ref
        def u_exact_callable(x_arr):
            return np.interp(x_arr, x_ref_sorted, u_ref_sorted)
        ref_label = f'Reference spectral (K={K})'
        ref_name_print = 'spectral'

    # Visual comparison on a chosen DOF (default N=40 for step-by-step validation)
    N_vis = only_check_N if only_check_N is not None else dofs_list[-1]
    mesh = make_uniform_interval(N_vis, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    facet_tags, _ = tag_all_exterior_facets(mesh)
    B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
    _, _, a_form, m_form = assemble_K_M(V)
    # Assemble RHS using UFL expression for f(x)=x^{-alpha}
    f_expr = lambda x: x[0]**(-alpha)
    b = assemble_rhs(V, f_expr=f_expr, bc=bc, combined_form_for_lifting=a_form)
    u_h_vis, _ = sinc_solver(B, M, b, V, bc=bc, beta=beta, k=sinc_k)

    # Plot comparison: exact/reference vs FEM solution (finest DOF)
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_ref, u_ref, 'k-', lw=2.5, label=ref_label)
    x_nodes = V.tabulate_dof_coordinates()[:, 0]
    ax1.plot(x_nodes, u_h_vis.x.array, 'ro-', lw=1.5, ms=4, label=f'FEM Sinc (−Δ)^β (DOF={N_vis+1})')
    ax1.set_title(f"Dirichlet frac. problem: α={alpha}, β={beta}")
    ax1.set_xlabel("x"); ax1.set_ylabel("u(x)")
    ax1.grid(True, alpha=0.4); ax1.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Quantitative check vs reference on fine grid
    # Interpolate u_h_vis to x_ref (linear nodal interpolation)
    xN_vis = mesh.geometry.x[:, 0]
    sort_idx_vis = np.argsort(xN_vis)
    xN_vis_sorted = xN_vis[sort_idx_vis]
    uN_vis_sorted = u_h_vis.x.array[sort_idx_vis]
    u_vis_on_ref = np.interp(x_ref, xN_vis_sorted, uN_vis_sorted)
    diff = u_vis_on_ref - u_ref
    l2_err_ref = float(np.sqrt(np.trapz(diff**2, x_ref)))
    linf_err_ref = float(np.max(np.abs(diff)))
    print(f"Check at N={N_vis}: L2 error vs {ref_name_print} = {l2_err_ref:.3e}, Linf = {linf_err_ref:.3e}")

    # If only_check_N is set, still proceed to DOF sweep but keep this printed check

    # Convergence: DOF vs L2 error (log-log)
    dof_values = []
    l2_errors_std = []
    l2_errors_fvm = []
    l2_errors_exact = []
    for N in dofs_list:
        meshN = make_uniform_interval(N, 0.0, 1.0)
        VN = fem.functionspace(meshN, ("Lagrange", 1))
        facet_tagsN, _ = tag_all_exterior_facets(meshN)
        BN, _, MN, _, bcN = build_operator_B(VN, bc_type="dirichlet", facet_tags=facet_tagsN, ids=(1,), kappa=0)
        _, _, aN, mN = assemble_K_M(VN)
        # Assemble RHS via UFL expression on VN
        f_expr_N = lambda x: x[0]**(-alpha)
        # Standard FEM RHS
        bN_std = assemble_rhs(VN, f_expr=f_expr_N, bc=bcN, combined_form_for_lifting=aN)
        uN_std, _ = sinc_solver(BN, MN, bN_std, VN, bc=bcN, beta=beta, k=sinc_k)
        # FVM RHS (dual control volumes) with lifting consistent with Dirichlet
        bN_fvm = assemble_rhs_dual_fvm(VN, f_expr=f_expr_N, quad_degree=3, bc=bcN, combined_form_for_lifting=aN)
        uN_fvm, _ = sinc_solver(BN, MN, bN_fvm, VN, bc=bcN, beta=beta, k=sinc_k)
        # Exact dual-FVM RHS for f=x^{-alpha}
        bN_exact = assemble_rhs_dual_fvm_exact(VN, alpha=alpha, bc=bcN, combined_form_for_lifting=aN)
        uN_exact, _ = sinc_solver(BN, MN, bN_exact, VN, bc=bcN, beta=beta, k=sinc_k)

        # If requested level N=40, plot approximate vs exact
        if N == 40:
            xN_nodes_plot = VN.tabulate_dof_coordinates()[:, 0]
            fig40, ax40 = plt.subplots(figsize=(8, 4.5))
            ax40.plot(x_ref, u_ref, 'k-', lw=2.0, label=ref_label)
            ax40.plot(xN_nodes_plot, uN_std.x.array, 'bo-', lw=1.5, ms=4, label='FEM Sinc STD (N=40)')
            ax40.plot(xN_nodes_plot, uN_fvm.x.array, 'gs--', lw=1.2, ms=4, label='FEM Sinc FVM (N=40)')
            ax40.plot(xN_nodes_plot, uN_exact.x.array, 'm^-.', lw=1.2, ms=4, label='FEM Sinc EXACT (N=40)')
            ax40.set_title(f"Comparison at N=40: α={alpha}, β={beta}")
            ax40.set_xlabel('x'); ax40.set_ylabel('u(x)')
            ax40.grid(True, alpha=0.4); ax40.legend(loc='best')
            plt.tight_layout(); plt.show()

        # Use refined-mesh error computation from existing workflow
        errs_std = compute_errors_refined(uN_std, u_exact_callable, VN, meshN, refine_factor=16)
        errs_fvm = compute_errors_refined(uN_fvm, u_exact_callable, VN, meshN, refine_factor=16)
        errs_exact = compute_errors_refined(uN_exact, u_exact_callable, VN, meshN, refine_factor=16)
        dof_values.append(len(uN_std.x.array))
        l2_errors_std.append(errs_std["l2_error"])
        l2_errors_fvm.append(errs_fvm["l2_error"])
        l2_errors_exact.append(errs_exact["l2_error"])

    dof_values = np.asarray(dof_values, dtype=float)
    l2_errors_std = np.asarray(l2_errors_std, dtype=float)
    l2_errors_fvm = np.asarray(l2_errors_fvm, dtype=float)
    l2_errors_exact = np.asarray(l2_errors_exact, dtype=float)
    # Fit slope in log-log (separately)
    valid_std = (dof_values > 0) & (l2_errors_std > 0)
    slope_std = float(np.polyfit(np.log(dof_values[valid_std]), np.log(l2_errors_std[valid_std]), 1)[0]) if np.count_nonzero(valid_std) >= 2 else np.nan
    valid_fvm = (dof_values > 0) & (l2_errors_fvm > 0)
    slope_fvm = float(np.polyfit(np.log(dof_values[valid_fvm]), np.log(l2_errors_fvm[valid_fvm]), 1)[0]) if np.count_nonzero(valid_fvm) >= 2 else np.nan
    valid_exact = (dof_values > 0) & (l2_errors_exact > 0)
    slope_exact = float(np.polyfit(np.log(dof_values[valid_exact]), np.log(l2_errors_exact[valid_exact]), 1)[0]) if np.count_nonzero(valid_exact) >= 2 else np.nan

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.loglog(dof_values, l2_errors_std, 'o-', lw=2, ms=6,
               label=f'STD RHS (order≈{abs(slope_std):.2f})')
    ax2.loglog(dof_values, l2_errors_fvm, 's--', lw=2, ms=6,
               label=f'FVM RHS (order≈{abs(slope_fvm):.2f})')
    ax2.loglog(dof_values, l2_errors_exact, '^-.', lw=2, ms=6,
               label=f'EXACT FVM RHS (order≈{abs(slope_exact):.2f})')
    ax2.set_xlabel('DOF')
    ax2.set_ylabel('L2 error')
    ax2.set_title(f'Convergence: α={alpha}, β={beta}')
    ax2.grid(True, which='both', alpha=0.4)
    ax2.legend(loc='best')
    plt.tight_layout(); plt.show()


# New: Adaptive experiment using standard FEM RHS (same load as STD RHS)
def run_dirichlet_fractional_experiment_adaptive_std(alpha=0.49,
                                                    beta=0.5,
                                                    K=1000,
                                                    N_base_values=(5, 10, 20, 40, 80),
                                                    sinc_k=0.15):
    """
    Convergence using standard FEM RHS on a smooth adaptive mesh.
    Uses closed-form exact solution when β=1, spectral reference when β≠1.
    Tracks ht and h_sing similarly to the EXACT FVM adaptive experiment.
    """
    if abs(beta - 1.0) < 1e-12:
        def u_exact_callable(x_arr):
            x = np.asarray(x_arr, dtype=float)
            mask = x > 1e-15
            result = np.zeros_like(x)
            denom = (1.0 - alpha) * (2.0 - alpha)
            result[mask] = (x[mask] - x[mask]**(2.0 - alpha)) / denom
            result[~mask] = 0.0
            return result
        x_ref = np.linspace(0, 1, 4000+2)[1:-1]
        u_ref = u_exact_callable(x_ref)
        dense_label = 'Exact (closed-form)'
    else:
        x_ref = np.linspace(0, 1, 4000+2)[1:-1]
        u_ref = solve_spectral_continua(alpha, beta, K, x_ref)
        x_ref_sorted = x_ref
        u_ref_sorted = u_ref
        def u_exact_callable(x_arr):
            return np.interp(x_arr, x_ref_sorted, u_ref_sorted)
        dense_label = 'Exact (spectral reference)'

    dof_values = []
    l2_errors_std = []
    ht_values = []
    hsing_values = []
    mesh_plot = None
    V_plot = None
    uN_std_plot = None

    for N_base in N_base_values:
        meshN = create_adaptive_mesh_1d_smooth(N_base, alpha=alpha)
        VN = fem.functionspace(meshN, ("Lagrange", 1))
        facet_tagsN, _ = tag_all_exterior_facets(meshN)
        BN, _, MN, _, bcN = build_operator_B(VN, bc_type="dirichlet", facet_tags=facet_tagsN, ids=(1,), kappa=0)
        _, _, aN, _ = assemble_K_M(VN)

        # Standard FEM RHS with f(x) = x^{-alpha}
        f_expr_N = lambda x: x[0]**(-alpha)
        bN_std = assemble_rhs(VN, f_expr=f_expr_N, bc=bcN, combined_form_for_lifting=aN)

        # Use dynamic Sinc step based on DOF
        dofN = VN.dofmap.index_map.size_global
        k_use = dynamic_k(beta, max(1, dofN - 1))
        uN_std, _ = sinc_solver(BN, MN, bN_std, VN, bc=bcN, beta=beta, k=k_use)

        # Error vs reference
        errs_std = compute_errors_refined(uN_std, u_exact_callable, VN, meshN, refine_factor=16)
        dof_values.append(len(uN_std.x.array))
        l2_errors_std.append(errs_std["l2_error"])

        # Track ht and h_sing
        ht_values.append(1.0/float(N_base))
        xnodes = np.sort(meshN.geometry.x[:,0])
        hs = np.diff(xnodes)
        eps_region = 0.05
        mask = xnodes[:-1] <= eps_region
        h_sing = hs[mask].min() if np.any(mask) else hs.min()
        hsing_values.append(float(h_sing))

        # Store finest for plotting
        mesh_plot = meshN
        V_plot = VN
        uN_std_plot = uN_std

    dof_values = np.asarray(dof_values, dtype=float)
    l2_errors_std = np.asarray(l2_errors_std, dtype=float)
    valid = (dof_values > 0) & (l2_errors_std > 0)
    slope_std = float(np.polyfit(np.log(dof_values[valid]), np.log(l2_errors_std[valid]), 1)[0]) if np.count_nonzero(valid) >= 2 else np.nan

    # Plot convergence vs DOF
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(dof_values, l2_errors_std, 'o-', lw=2, ms=6,
              label=f'STD RHS (adaptive) order≈{abs(slope_std):.2f}')
    ax.set_xlabel('DOF')
    ax.set_ylabel('L2 error')
    ax.set_title(f'Adaptive STD RHS load: α={alpha}, β={beta}')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Convergence vs ht
    ht_arr = np.asarray(ht_values, dtype=float)
    err_arr = np.asarray(l2_errors_std, dtype=float)
    valid_ht = (ht_arr > 0) & (err_arr > 0)
    slope_ht = float(np.polyfit(np.log(ht_arr[valid_ht]), np.log(err_arr[valid_ht]), 1)[0]) if np.count_nonzero(valid_ht) >= 2 else np.nan
    fig_ht, ax_ht = plt.subplots(figsize=(6.5, 4.5))
    ax_ht.loglog(ht_arr, err_arr, 'o-', lw=2, ms=6, label=f'Error vs ht (slope≈{slope_ht:.2f})')
    ax_ht.set_xlabel('ht = 1/N_base (parameter space)')
    ax_ht.set_ylabel('L2 error')
    ax_ht.set_title('Adaptive STD RHS: Error vs ht')
    ax_ht.grid(True, which='both', alpha=0.4)
    ax_ht.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Convergence vs h_sing
    hsing_arr = np.asarray(hsing_values, dtype=float)
    valid_hs = (hsing_arr > 0) & (err_arr > 0)
    slope_hs = float(np.polyfit(np.log(hsing_arr[valid_hs]), np.log(err_arr[valid_hs]), 1)[0]) if np.count_nonzero(valid_hs) >= 2 else np.nan
    fig_hs, ax_hs = plt.subplots(figsize=(6.5, 4.5))
    ax_hs.loglog(hsing_arr, err_arr, 's--', lw=2, ms=6, label=f'Error vs h_sing (slope≈{slope_hs:.2f})')
    ax_hs.set_xlabel('h_sing (min h near x=0)')
    ax_hs.set_ylabel('L2 error')
    ax_hs.set_title('Adaptive STD RHS: Error vs h_sing')
    ax_hs.grid(True, which='both', alpha=0.4)
    ax_hs.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Plot exact vs approximated on the finest adaptive mesh
    if V_plot is not None and uN_std_plot is not None:
        x_nodes = V_plot.tabulate_dof_coordinates()[:, 0]
        u_exact_on_nodes = u_exact_callable(x_nodes)
        u_approx_on_nodes = uN_std_plot.x.array
        x_dense = np.linspace(0.0, 1.0, 2000)
        u_exact_dense = u_exact_callable(x_dense)
        figc, axc = plt.subplots(figsize=(8, 4.5))
        axc.plot(x_dense, u_exact_dense, 'k-', lw=2.0, label=dense_label)
        axc.plot(x_nodes, u_approx_on_nodes, 'bo-', lw=1.5, ms=4, label='Adaptive STD RHS approx (nodes)')
        axc.set_title(f"Exact vs Approx (adaptive STD RHS): α={alpha}, β={beta}, DOF={len(u_approx_on_nodes)}")
        axc.set_xlabel('x'); axc.set_ylabel('u(x)')
        axc.grid(True, alpha=0.4); axc.legend(loc='best')
        plt.tight_layout(); plt.show()

def create_adaptive_mesh_1d_smooth(N_base, alpha=0.49):
    """
    Create a smooth adaptive 1D mesh on [0,1] with refinement near x=0
    using mapping x = t^{1/(1-α)} applied to a finer base partition.
    """
    from domains import make_uniform_interval
    # Use an oversampled uniform mesh and remap coordinates smoothly
    N_total = max(3 * int(N_base), 2)
    mesh_adapt = make_uniform_interval(N_total, 0.0, 1.0)
    # Build new monotonically increasing coordinate vector
    x_old = mesh_adapt.geometry.x[:, 0]
    order = np.argsort(x_old)
    t = np.linspace(0.0, 1.0, N_total + 1)
    p = 1.0 / (1.0 - float(alpha))
    x_new = t**p
    # Ensure endpoints exact
    x_new[0] = 0.0; x_new[-1] = 1.0
    # Assign in sorted-vertex order to preserve increasing coordinates along the line
    mesh_adapt.geometry.x[order, 0] = x_new
    return mesh_adapt


def run_dirichlet_fractional_experiment_adaptive_exact(alpha=0.49,
                                                      beta=0.5,
                                                      K=1000,
                                                      N_base_values=(5, 10, 20, 40, 80),
                                                      sinc_k=0.15):
    """
    Convergence using exact dual-FVM RHS on a smooth adaptive mesh.
    Uses closed-form exact solution when β=1, spectral reference when β≠1.
    """
    if abs(beta - 1.0) < 1e-12:
        def u_exact_callable(x_arr):
            x = np.asarray(x_arr, dtype=float)
            mask = x > 1e-15
            result = np.zeros_like(x)
            denom = (1.0 - alpha) * (2.0 - alpha)
            result[mask] = (x[mask] - x[mask]**(2.0 - alpha)) / denom
            result[~mask] = 0.0
            return result
        x_ref = np.linspace(0, 1, 4000+2)[1:-1]
        u_ref = u_exact_callable(x_ref)
        dense_label = 'Exact (closed-form)'
    else:
        x_ref = np.linspace(0, 1, 4000+2)[1:-1]
        u_ref = solve_spectral_continua(alpha, beta, K, x_ref)
        x_ref_sorted = x_ref
        u_ref_sorted = u_ref
        def u_exact_callable(x_arr):
            return np.interp(x_arr, x_ref_sorted, u_ref_sorted)
        dense_label = 'Exact (spectral reference)'

    dof_values = []
    l2_errors_exact = []
    ht_values = []  # parameter-space mesh size ht = 1/N_base
    hsing_values = []  # physical min h near singularity region
    # Keep finest-mesh data to plot exact vs approx
    mesh_plot = None
    V_plot = None
    uN_exact_plot = None

    for N_base in N_base_values:
        meshN = create_adaptive_mesh_1d_smooth(N_base, alpha=alpha)
        VN = fem.functionspace(meshN, ("Lagrange", 1))
        facet_tagsN, _ = tag_all_exterior_facets(meshN)
        BN, _, MN, _, bcN = build_operator_B(VN, bc_type="dirichlet", facet_tags=facet_tagsN, ids=(1,), kappa=0)
        _, _, aN, _ = assemble_K_M(VN)

        # Exact dual-FVM RHS and solve (use dynamic Sinc step based on DOF)
        bN_exact = assemble_rhs_dual_fvm_exact(VN, alpha=alpha, bc=bcN, combined_form_for_lifting=aN)
        dofN = VN.dofmap.index_map.size_global
        k_use = dynamic_k(beta, max(1, dofN - 1))
        uN_exact, _ = sinc_solver(BN, MN, bN_exact, VN, bc=bcN, beta=beta, k=k_use)

        # Error vs spectral reference using refined-mesh utility
        errs_exact = compute_errors_refined(uN_exact, u_exact_callable, VN, meshN, refine_factor=16)
        dof_values.append(len(uN_exact.x.array))
        l2_errors_exact.append(errs_exact["l2_error"])
        # Track ht and h_sing
        ht_values.append(1.0/float(N_base))
        # compute element sizes from sorted node coordinates
        xnodes = np.sort(meshN.geometry.x[:,0])
        hs = np.diff(xnodes)
        # region near singularity (x ~ 0)
        eps_region = 0.05
        mask = xnodes[:-1] <= eps_region
        h_sing = hs[mask].min() if np.any(mask) else hs.min()
        hsing_values.append(float(h_sing))
        # Store for plotting on the finest mesh (last iteration)
        mesh_plot = meshN
        V_plot = VN
        uN_exact_plot = uN_exact

    dof_values = np.asarray(dof_values, dtype=float)
    l2_errors_exact = np.asarray(l2_errors_exact, dtype=float)
    valid = (dof_values > 0) & (l2_errors_exact > 0)
    slope_exact = float(np.polyfit(np.log(dof_values[valid]), np.log(l2_errors_exact[valid]), 1)[0]) if np.count_nonzero(valid) >= 2 else np.nan

    # Plot convergence vs DOF
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(dof_values, l2_errors_exact, '^-.', lw=2, ms=6,
              label=f'EXACT FVM RHS (adaptive) order≈{abs(slope_exact):.2f}')
    ax.set_xlabel('DOF')
    ax.set_ylabel('L2 error')
    ax.set_title(f'Adaptive EXACT FVM load: α={alpha}, β={beta}')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Convergence vs ht (parameter-space size)
    ht_arr = np.asarray(ht_values, dtype=float)
    err_arr = np.asarray(l2_errors_exact, dtype=float)
    valid_ht = (ht_arr > 0) & (err_arr > 0)
    slope_ht = float(np.polyfit(np.log(ht_arr[valid_ht]), np.log(err_arr[valid_ht]), 1)[0]) if np.count_nonzero(valid_ht) >= 2 else np.nan
    fig_ht, ax_ht = plt.subplots(figsize=(6.5, 4.5))
    ax_ht.loglog(ht_arr, err_arr, 'o-', lw=2, ms=6, label=f'Error vs ht (slope≈{slope_ht:.2f})')
    ax_ht.set_xlabel('ht = 1/N_base (parameter space)')
    ax_ht.set_ylabel('L2 error')
    ax_ht.set_title('Adaptive EXACT FVM: Error vs ht')
    ax_ht.grid(True, which='both', alpha=0.4)
    ax_ht.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Convergence vs h_sing (min physical h near singularity zone)
    hsing_arr = np.asarray(hsing_values, dtype=float)
    valid_hs = (hsing_arr > 0) & (err_arr > 0)
    slope_hs = float(np.polyfit(np.log(hsing_arr[valid_hs]), np.log(err_arr[valid_hs]), 1)[0]) if np.count_nonzero(valid_hs) >= 2 else np.nan
    fig_hs, ax_hs = plt.subplots(figsize=(6.5, 4.5))
    ax_hs.loglog(hsing_arr, err_arr, 's--', lw=2, ms=6, label=f'Error vs h_sing (slope≈{slope_hs:.2f})')
    ax_hs.set_xlabel('h_sing (min h near x=0)')
    ax_hs.set_ylabel('L2 error')
    ax_hs.set_title('Adaptive EXACT FVM: Error vs h_sing')
    ax_hs.grid(True, which='both', alpha=0.4)
    ax_hs.legend(loc='best')
    plt.tight_layout(); plt.show()

    # Plot exact vs approximated on the finest adaptive mesh
    if V_plot is not None and uN_exact_plot is not None:
        x_nodes = V_plot.tabulate_dof_coordinates()[:, 0]
        u_exact_on_nodes = u_exact_callable(x_nodes)
        u_approx_on_nodes = uN_exact_plot.x.array
        # Build a dense curve for the exact/spectral reference to visualize curvature clearly
        x_dense = np.linspace(0.0, 1.0, 2000)
        u_exact_dense = u_exact_callable(x_dense)
        figc, axc = plt.subplots(figsize=(8, 4.5))
        axc.plot(x_dense, u_exact_dense, 'k-', lw=2.0, label=dense_label)
        axc.plot(x_nodes, u_approx_on_nodes, 'm^-.', lw=1.5, ms=4, label='Adaptive EXACT FVM approx (nodes)')
        axc.set_title(f"Exact vs Approx (adaptive EXACT FVM): α={alpha}, β={beta}, DOF={len(u_approx_on_nodes)}")
        axc.set_xlabel('x'); axc.set_ylabel('u(x)')
        axc.grid(True, alpha=0.4); axc.legend(loc='best')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    """
    Main execution block for numerical experiments.
    Uncomment the experiments you want to run.
    """
    
    # Example: Plot spectral solution
    # alpha = 0.499
    # beta = 0.5
    # K = 1000
    # N = 800
    # x = np.linspace(0, 1, N+2)[1:-1]
    # u_spec = solve_spectral_continua(alpha, beta, K, x)
    # fig, ax = plt.subplots(figsize=(8, 4.5))
    # ax.plot(x, u_spec, 'r-', lw=2, label=f'Solução espectral (α={alpha}, β={beta})')
    # ax.set_title(f"Solução espectral contínua – α={alpha}, β={beta}, K={K}")
    # ax.set_xlabel("x")
    # ax.set_ylabel("u(x)")
    # ax.legend(loc='best')
    # ax.grid(True)
    # plt.tight_layout()
    # plt.show()
    #%%
    # Run the numerical experiment specified
    run_dirichlet_fractional_experiment(alpha=0.499, beta=1, K=1000,
                                        dofs_list=(10, 20, 40, 80),
                                        sinc_k=0.15,
                                        only_check_N=100)
    #%%
    # Run adaptive exact FVM experiment
    run_dirichlet_fractional_experiment_adaptive_exact(alpha=0.5, beta=1, K=1000,
                                                       N_base_values=(5, 10, 20, 40, 80, 160),
                                                       sinc_k=0.15)
    #%%
    # Run adaptive STD RHS experiment
    run_dirichlet_fractional_experiment_adaptive_std(alpha=0.49999, beta=1, K=1000,
                                                     N_base_values=(5, 10, 20, 40, 80),
                                                     sinc_k=0.15)

# %%
