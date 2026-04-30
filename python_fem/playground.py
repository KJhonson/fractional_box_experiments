#%%


"""
main_refined_error.py — FEM examples with accurate error computation on a refined mesh

Key idea:
- Build a very fine 1D mesh for *evaluation only*.
- Interpolate the exact solution onto the fine mesh (u_exact_f).
- Treat u_h as a piecewise-linear function: interpolate it onto the fine mesh by sampling
  at fine nodes via a callable that does 1D linear interpolation (np.interp) over the coarse
  nodal values (u_h_f).
- Form the error e = u_h_f - u_exact_f on the fine mesh.
- Compute norms using fine-mesh matrices: sqrt(e^T M_f e) and sqrt(e^T K_f e).
"""

from __future__ import annotations
import numpy as np
import ufl
import matplotlib.pyplot as plt
from pathlib import Path
from dolfinx import fem
from mpi4py import MPI

# Local modules
from domains import make_uniform_interval, tag_all_exterior_facets
from operators import assemble_K_M, build_operator_B
from utils.loads import assemble_rhs
from utils.sinc_solver import solve_petsc, sinc_solver
from rationalv2 import rational_solve, rational_solve_unified
from plot_visualization import plot_1d_dirichlet_solution, plot_1d_neumann_solution

OUTPUT_DIR = Path("/home/dolfinx/shared/FEM_project")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXACT SOLUTIONS FOR ALL BETA VALUES
# ============================================================================

NEUMANN = "neumann"
DIRICHLET = "dirichlet"


# ----------------------------------------------------------------------------
# Unified selectors to avoid duplicating calls per boundary condition
# ----------------------------------------------------------------------------

def dynamic_k(beta: float, N: int) -> float:
    """
    Heuristic Sinc step size used in unified_comparison.py.
    k = - (pi^2) / (4 * beta * log(1/(N+1)))
    """
    if beta <= 0:
        raise ValueError("beta must be > 0 for dynamic_k")
    return - (np.pi ** 2) / (4 * beta * np.log(1.0 / (N + 1)))

def sol_exact(beta, bc_type=NEUMANN):
    """
    Unified exact solution selector for any beta>0 and boundary type.
    bc_type ∈ {NEUMANN, DIRICHLET}
    """
    if beta <= 0:
        raise ValueError(f"Invalid beta={beta}. Must be > 0")
    if bc_type == NEUMANN:
        return lambda x: np.cos(np.pi * x) / ((1 + np.pi**2)**beta)
    if bc_type == DIRICHLET:
        return lambda x: np.sin(np.pi * x) / ((1 + np.pi**2)**beta)
    raise ValueError(f"Unknown bc_type={bc_type}. Use '{NEUMANN}' or '{DIRICHLET}'.")


def source_exact(beta, bc_type=NEUMANN):
    """
    Unified source function selector for any beta>0 and boundary type.
    bc_type ∈ {NEUMANN, DIRICHLET}
    """
    if beta <= 0:
        raise ValueError(f"Invalid beta={beta}. Must be > 0")
    if bc_type == NEUMANN:
        return lambda x: ((1 + np.pi**2)**beta) * np.cos(np.pi * x)
    if bc_type == DIRICHLET:
        return lambda x: ((1 + np.pi**2)**beta) * np.sin(np.pi * x)
    raise ValueError(f"Unknown bc_type={bc_type}. Use '{NEUMANN}' or '{DIRICHLET}'.")


def solve_neumann_problem_for_beta(mesh, V, beta):
    """
    Solve Neumann problem for any beta value (standard or fractional).
    
    Parameters:
    -----------
    mesh : dolfinx.Mesh
        The mesh
    V : dolfinx.fem.FunctionSpace
        Function space
    beta : float
        Power of the operator (must be > 0)
        
    Returns:
    --------
    u_h : dolfinx.fem.Function
        Solution function
    """
    if beta <= 0:
        raise ValueError(f"Invalid beta={beta}. Must be > 0")
    
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr, bc=None)
    
    # Use sinc_solver for all beta > 0 (it handles beta=1.0, 0<beta<1, and beta>=1)
    u_h, _ = sinc_solver(B, M, b, V, bc=None, beta=beta)
    
    return u_h


def solve_dirichlet_problem_for_beta(mesh, V, beta):
    """
    Solve Dirichlet problem for any beta value (standard or fractional).
    
    Parameters:
    -----------
    mesh : dolfinx.Mesh
        The mesh
    V : dolfinx.fem.FunctionSpace
        Function space
    beta : float
        Power of the operator (must be > 0)
        
    Returns:
    --------
    u_h : dolfinx.fem.Function
        Solution function
    """
    if beta <= 0:
        raise ValueError(f"Invalid beta={beta}. Must be > 0")
    
    # Setup Dirichlet problem
    B, _, M, _, facet_tags = build_operator_B(V, bc_type="dirichlet", kappa=1)
    f_expr = lambda x: ufl.sin(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr, bc=None)
    bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, 1, facet_tags.find(1)), V)
    
    # Use sinc_solver for all beta > 0 (it handles beta=1.0, 0<beta<1, and beta>=1)
    u_h, _ = sinc_solver(B, M, b, V, bc=bc, beta=beta)
    
    return u_h


# ============================================================================
# FLEXIBLE COMPARISON PLOTTING FUNCTION
# ============================================================================

def plot_methods_comparison(u_exact_func, solutions_dict, errors_dict, V, mesh, 
                           title_prefix="Comparison", save_filename=None, 
                           show_mesh_points=True, mesh_info="N elements",
                           l2_convergence_data=None):
    """
    Create a comprehensive 4-subplot comparison visualization.
    
    Parameters:
    -----------
    u_exact_func : callable
        Exact solution function
    solutions_dict : dict
        Dictionary with method names as keys and dolfinx Functions as values
        e.g., {'Sinc': u_sinc, 'BRASIL': u_brasil}
    errors_dict : dict
        Dictionary with method names as keys and error dicts as values
        e.g., {'Sinc': {'l2_error': 1e-3, 'h1_error': 1e-2, 'linf_error': 1e-3}, ...}
    V : dolfinx FunctionSpace
        Function space for evaluation
    mesh : dolfinx Mesh
        Mesh for evaluation
    title_prefix : str
        Prefix for plot titles
    save_filename : str, optional
        Filename to save the plot
    show_mesh_points : bool
        Whether to show mesh points on plots
    mesh_info : str
        Information about mesh to display in titles
    """
    import matplotlib.pyplot as plt
    
    # Get coordinates and exact solution
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    u_exact_vals = u_exact_func(x_coords)
    
    # Get numerical solutions
    u_numerical = {}
    for method, u_h in solutions_dict.items():
        u_numerical[method] = u_h.x.array
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(solutions_dict.keys())}
    
    # Plot 1: Solutions comparison (1,1)
    ax1 = axes[0, 0]
    ax1.plot(x_coords, u_exact_vals, 'k-', linewidth=3, label='Exact', alpha=0.8)
    
    for method, u_vals in u_numerical.items():
        ax1.plot(x_coords, u_vals, color=method_colors[method], linestyle='--', 
                linewidth=2, label=method, alpha=0.8)
        
        if show_mesh_points:
            ax1.plot(x_coords, u_vals, 'o', color=method_colors[method], 
                    markersize=4, alpha=0.7)
    
    if show_mesh_points:
        ax1.plot(x_coords, u_exact_vals, 'ko', markersize=4, alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'{title_prefix} - Solutions ({mesh_info})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute errors - linear scale (1,2)
    ax2 = axes[0, 1]
    for method, u_vals in u_numerical.items():
        error_vals = np.abs(u_vals - u_exact_vals)
        ax2.plot(x_coords, error_vals, color=method_colors[method], 
                linewidth=2, label=f'{method} Error')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title(f'Absolute Errors - Linear Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error norms comparison - bar chart (2,1)
    ax3 = axes[1, 0]
    
    # Prepare data for grouped bar chart
    methods = list(solutions_dict.keys())
    norms = ['L²', 'H¹', 'L∞']
    norm_colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    # Set up bar positions
    x_pos = np.arange(len(methods))
    width = 0.25
    
    for i, norm in enumerate(norms):
        if norm == 'L²':
            norm_key = 'l2_error'
        elif norm == 'H¹':
            norm_key = 'h1_error'
        elif norm == 'L∞':
            norm_key = 'linf_error'
        else:
            norm_key = f'{norm.lower().replace("²", "2").replace("∞", "inf")}_error'
        errors = [errors_dict[method][norm_key] for method in methods]
        
        ax3.bar(x_pos + i * width, errors, width, 
               label=f'{norm} Error', color=norm_colors[i], alpha=0.8)
    
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Error')
    ax3.set_title(f'Error Norms Comparison')
    ax3.set_yscale('log')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: L² error analysis vs degrees of freedom (2,2)
    ax4 = axes[1, 1]
    
    # If provided, use full convergence series (multiple N) per method
    if l2_convergence_data is not None and len(l2_convergence_data) > 0:
        for method, series in l2_convergence_data.items():
            dofs, l2errs = series
            # Compute slope of log-log line via least squares
            dofs_arr = np.asarray(dofs, dtype=float)
            errs_arr = np.asarray(l2errs, dtype=float)
            valid = (dofs_arr > 0) & (errs_arr > 0)
            if np.count_nonzero(valid) >= 2:
                logx = np.log(dofs_arr[valid])
                logy = np.log(errs_arr[valid])
                slope = float(np.polyfit(logx, logy, 1)[0])
                label = f"{method} (slope: {slope:.2f})"
            else:
                label = method
            ax4.loglog(dofs, l2errs, 'o-', linewidth=2, markersize=8, label=label)
    else:
        # Fallback: compute single-point L2 vs DOF from current solutions
        dof_values = []
        l2_errors = []
        method_names = []
        for method, u_vals in u_numerical.items():
            error_vals = u_vals - u_exact_vals
            l2_error = np.sqrt(np.mean(error_vals**2))
            dof = len(u_vals)
            dof_values.append(dof)
            l2_errors.append(l2_error)
            method_names.append(method)
        ax4.loglog(dof_values, l2_errors, 'o-', linewidth=2, markersize=8)
        for i, (dof, error, method) in enumerate(zip(dof_values, l2_errors, method_names)):
            ax4.annotate(f'{method}', (dof, error), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, alpha=0.8)
    
    ax4.set_xlabel('Degrees of Freedom')
    ax4.set_ylabel('L² Error')
    ax4.set_title(f'L² Error vs DOF Analysis')
    ax4.grid(True, alpha=0.3)
    if l2_convergence_data is not None and len(l2_convergence_data) > 0:
        ax4.legend()
    
    # Add overall title
    fig.suptitle(f'{title_prefix} Analysis - {mesh_info}', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if save_filename:
        filepath = OUTPUT_DIR / save_filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {filepath}")
    
    plt.show()
    return fig

# ============================================================================
# ACCURATE ERROR COMPUTATION ON A VERY FINE MESH (INTERPOLATION-BASED)
# ============================================================================

def compute_errors_refined(u_h, u_exact_func, Vc, meshc, refine_factor=16, degree_fine=None, bc=None):
    """
    Compute L², H¹, and L∞ errors using a separate very fine mesh.

    Steps:
      1) Build fine evaluation mesh (meshf) and space (Vf).
      2) Interpolate exact solution onto Vf.
      3) Interpolate u_h (piecewise-linear) onto Vf by sampling u_h at fine nodes
         via a callable that uses 1D linear interpolation over the coarse nodes.
      4) Compute errors using UFL approach: sqrt(∫ e² dx) and sqrt(∫ |∇e|² dx)
      5) Return L2, H1, and L∞ errors and norms.
    """
    # ----------------------- 1) Fine mesh + space -----------------------
    # Coarse dofs ~ number of coarse nodes; use it to pick a much finer count
    N_coarse_dofs = u_h.x.array.size
    N_fine = max(refine_factor * N_coarse_dofs, 500)  # ensure "very fine"

    # Domain endpoints from the coarse mesh geometry
    x_min = float(meshc.geometry.x.min())
    x_max = float(meshc.geometry.x.max())

    meshf = make_uniform_interval(N_fine, p0=x_min, p1=x_max)

    # Choose a fine space degree (≥ coarse degree, but piecewise-linear is fine)
    p_coarse = Vc.ufl_element().degree
    degree_f = degree_fine or max(p_coarse, 1)
    Vf = fem.functionspace(meshf, ("Lagrange", degree_f))

    # Coordinates of fine/coarse mesh nodes (1D)
    x_fine = meshf.geometry.x[:, 0]
    x_coarse = meshc.geometry.x[:, 0]

    # ----------------------- 2) u_exact on Vf ---------------------------
    # Exact callable must accept dolfinx interpolate signature: x has shape (1, npts)
    def u_exact_callable(x):
        # x is shape (1, n), so use x[0]
        return u_exact_func(x[0])

    u_exact_f = fem.Function(Vf)
    u_exact_f.interpolate(u_exact_callable)

    # ----------------------- 3) u_h on Vf (interpolate) -----------------
    # Treat u_h as piecewise-linear over the coarse nodes and interpolate to fine nodes.
    # Build a callable that maps x -> u_h(x) via np.interp using coarse nodal data.
    u_h_coarse_vals = u_h.x.array

    # Ensure x_coarse is sorted (it should be for uniform interval meshes)
    sort_idx = np.argsort(x_coarse)
    x_coarse_sorted = x_coarse[sort_idx]
    u_h_sorted = u_h_coarse_vals[sort_idx]

    def u_h_callable(x):
        # linear interpolation of coarse nodal values at query points x[0]
        return np.interp(x[0], x_coarse_sorted, u_h_sorted)

    u_h_f = fem.Function(Vf)
    u_h_f.interpolate(u_h_callable)

    # ----------------------- 4) Error computation using UFL ---------------------
    u_vec = u_h_f.x.array
    uex_vec = u_exact_f.x.array
    e_vec = u_vec - uex_vec
    
    # Create error function
    e_f = fem.Function(Vf)
    e_f.x.array[:] = e_vec
    
    # Create exact solution function for norms
    uex_f = fem.Function(Vf)
    uex_f.x.array[:] = uex_vec
    
    # L² norm: sqrt(∫ u_exact² dx)
    l2_norm_form = ufl.inner(uex_f, uex_f) * ufl.dx
    l2_norm = float(np.sqrt(fem.assemble_scalar(fem.form(l2_norm_form))))
    
    # L² error: sqrt(∫ e² dx)
    l2_error_form = ufl.inner(e_f, e_f) * ufl.dx
    l2_error = float(np.sqrt(fem.assemble_scalar(fem.form(l2_error_form))))
    
    # H¹ norm: sqrt(∫ (u_exact² + |∇u_exact|²) dx)
    h1_norm_form = ufl.inner(uex_f, uex_f) * ufl.dx + ufl.inner(ufl.grad(uex_f), ufl.grad(uex_f)) * ufl.dx
    h1_norm = float(np.sqrt(fem.assemble_scalar(fem.form(h1_norm_form))))
    
    # H¹ error: sqrt(∫ (e² + |∇e|²) dx)
    h1_error_form = ufl.inner(e_f, e_f) * ufl.dx + ufl.inner(ufl.grad(e_f), ufl.grad(e_f)) * ufl.dx
    h1_error = float(np.sqrt(fem.assemble_scalar(fem.form(h1_error_form))))
    
    # L∞ error
    linf_error = float(np.max(np.abs(e_vec)))

    return {
        "l2_error": l2_error,
        "h1_error": h1_error,
        "linf_error": linf_error,
        "l2_norm": l2_norm,
        "h1_norm": h1_norm,
    }


# ============================================================================
# STANDARD FEM EXAMPLES
# ============================================================================


def run_neumann_example(N=20, return_solution=False, show_plot=True, save_plot=True):
    """Standard Neumann problem with flexible visualization."""
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr, bc=None)
    u_h, _ = solve_petsc(B, b, V)

    def u_exact(x):
        return np.cos(np.pi * x) / (1 + np.pi ** 2)

    if return_solution:
        return u_h

    errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
    
    if show_plot or save_plot:
        solutions = {'Standard FEM': u_h}
        errors_dict = {'Standard FEM': errors}
        filename = f"neumann_standard_N{N}.png" if save_plot else None
        
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors_dict,
            V=V, mesh=mesh,
            title_prefix="Standard Neumann",
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=f"N={N}"
    )

    print(f"Neumann BC Errors: L²={errors['l2_error']:.2e}, H¹={errors['h1_error']:.2e}, L∞={errors['linf_error']:.2e}")
    return errors

# ============================================================================
# FRACTIONAL PROBLEMS
# ============================================================================

def test_frac_neumann(N=20, beta=0.5, return_solution=False, show_plot=True, save_plot=True):
    """Fractional Neumann problem using Sinc method with flexible visualization."""
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr)
    u_h, _ = sinc_solver(B, M, b, V, bc=None, beta=beta)

    u_exact = sol_exact(beta, NEUMANN)

    if return_solution:
        return u_h

    errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
    
    if show_plot or save_plot:
        solutions = {'Sinc': u_h}
        errors_dict = {'Sinc': errors}
        filename = f"frac_neumann_sinc_N{N}_beta{beta}.png" if save_plot else None
        
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors_dict,
            V=V, mesh=mesh,
            title_prefix=f"Fractional Neumann Sinc (β={beta})",
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=f"N={N}"
        )

    print(f"Fractional Neumann Sinc (β={beta}): L²={errors['l2_error']:.2e}, H¹={errors['h1_error']:.2e}, L∞={errors['linf_error']:.2e}")
    return errors

def test_frac_neumann_rational(N=20, beta=0.5, m=2, interval=(1, 11), return_solution=False, show_plot=True, save_plot=True):
    """Fractional Neumann problem using BRASIL rational approximation with flexible visualization."""
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr)
    u_h, _ = rational_solve(B, M, b, V, beta=beta, m=m, interval=interval)

    def u_exact(x):
        return np.cos(np.pi * x) / ((1 + np.pi ** 2) ** beta)

    if return_solution:
        return u_h

    errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
    
    if show_plot or save_plot:
        solutions = {'BRASIL': u_h}
        errors_dict = {'BRASIL': errors}
        filename = f"frac_neumann_brasil_N{N}_beta{beta}_m{m}.png" if save_plot else None
        
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors_dict,
            V=V, mesh=mesh,
            title_prefix=f"Fractional Neumann BRASIL (β={beta}, m={m})",
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=f"N={N}"
        )

    print(f"Fractional Neumann Rational (β={beta}, m={m}): L²={errors['l2_error']:.2e}, H¹={errors['h1_error']:.2e}, L∞={errors['linf_error']:.2e}")
    return errors

def test_frac_neumann_comparison(N=100, beta=0.5, m=2, interval=(2, 11), 
                                methods_to_compare=['Sinc', 'BRASIL'], 
                                show_plot=True, save_plot=True,
                                convergence_N_list=(20, 40, 60, 80, 100, 240),
                                scale_factor=1.0,
                                k_option=None):
    """
    Compare different methods for fractional Neumann problem with flexible visualization.
    
    Parameters:
    -----------
    N : int
        Number of mesh elements
    beta : float
        Fractional power
    m : int
        BRASIL approximation order
    interval : tuple
        BRASIL approximation interval
    methods_to_compare : list
        List of methods to compare: ['Sinc', 'BRASIL']
    show_plot : bool
        Whether to display the plot
    save_plot : bool
        Whether to save the plot
    """
    print(f"🧪 Comparing methods: {methods_to_compare} (N={N}, β={beta})")
    
    # Setup problem
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr)
    
    def u_exact(x):
        return np.cos(np.pi * x) / ((1 + np.pi ** 2) ** beta)
    
    # Solve with different methods (single N for subplots 1,1 / 1,2 / 2,1)
    solutions = {}
    errors = {}
    
    if 'Sinc' in methods_to_compare:
        print("  Solving with Sinc...")
        # Decide Sinc step k based on k_option
        if k_option == "dynamic":
            k_use = dynamic_k(beta, N)
            u_sinc, _ = sinc_solver(B, M, b, V, bc=None, beta=beta, k=k_use)
        elif isinstance(k_option, (int, float)):
            u_sinc, _ = sinc_solver(B, M, b, V, bc=None, beta=beta, k=k_option)
        else:
            u_sinc, _ = sinc_solver(B, M, b, V, bc=None, beta=beta)
        solutions['Sinc'] = u_sinc
        errors['Sinc'] = compute_errors_refined(u_sinc, u_exact, V, mesh, refine_factor=16)
        print(f"    Sinc: L²={errors['Sinc']['l2_error']:.2e}, H¹={errors['Sinc']['h1_error']:.2e}, L∞={errors['Sinc']['linf_error']:.2e}")
    
    if 'BRASIL' in methods_to_compare:
        print("  Solving with BRASIL...")
        u_brasil, _ = rational_solve(B, M, b, V, beta=beta, m=m, interval=interval, scale_factor=scale_factor)
        solutions['BRASIL'] = u_brasil
        errors['BRASIL'] = compute_errors_refined(u_brasil, u_exact, V, mesh, refine_factor=16)
        print(f"    BRASIL: L²={errors['BRASIL']['l2_error']:.2e}, H¹={errors['BRASIL']['h1_error']:.2e}, L∞={errors['BRASIL']['linf_error']:.2e}")
    
    # Build L2 convergence series vs DOF (subplot 2,2) using varying N
    l2_convergence_data = {}
    if convergence_N_list:
        u_exact_series = sol_exact(beta, NEUMANN)
        for method in methods_to_compare:
            dofs_series = []
            l2_series = []
            for Nconv in convergence_N_list:
                mesh_c = make_uniform_interval(Nconv, 0.0, 1.0)
                V_c = fem.functionspace(mesh_c, ("Lagrange", 1))
                B_c, _, M_c, _, _ = build_operator_B(V_c, bc_type="neumann", kappa=1)
                f_expr_c = lambda x: ufl.cos(ufl.pi * x[0])
                b_c = assemble_rhs(V_c, f_expr=f_expr_c)
                if method == 'Sinc':
                    if k_option == "dynamic":
                        k_c = dynamic_k(beta, Nconv)
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=None, beta=beta, k=k_c)
                    elif isinstance(k_option, (int, float)):
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=None, beta=beta, k=k_option)
                    else:
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=None, beta=beta)
                elif method == 'BRASIL':
                    u_c, _ = rational_solve(B_c, M_c, b_c, V_c, beta=beta, m=m, interval=interval, scale_factor=scale_factor)
                else:
                    continue
                err_c = compute_errors_refined(u_c, u_exact_series, V_c, mesh_c, refine_factor=16)
                dofs_series.append(len(u_c.x.array))
                l2_series.append(err_c['l2_error'])
            if len(dofs_series) > 0:
                l2_convergence_data[method] = (dofs_series, l2_series)

    # Create comparison plot
    if show_plot or save_plot:
        mesh_info = f"N={N}"
        title_prefix = f"Fractional Neumann (β={beta})"
        filename = f"frac_neumann_comparison_N{N}_beta{beta}.png" if save_plot else None
        
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors,
            V=V,
            mesh=mesh,
            title_prefix=title_prefix,
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=mesh_info,
            l2_convergence_data=l2_convergence_data
        )
    
    return solutions, errors


def test_frac_dirichlet_comparison(N=100, beta=0.5, m=2, interval=(1, 11),
                                   methods_to_compare=['Sinc', 'BRASIL'],
                                   show_plot=True, save_plot=True,
                                   convergence_N_list=(20, 40, 60, 80, 100, 240),
                                   scale_factor=1.0,
                                   k_option=None):
    """
    Compare methods for fractional Dirichlet problem (focus on Sinc for now).
    """
    print(f"🧪 Comparing methods (Dirichlet): {methods_to_compare} (N={N}, β={beta})")

    # Setup problem (Dirichlet)
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    facet_tags, _ = tag_all_exterior_facets(mesh)
    B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=1)
    # Build combined form for lifting when assembling RHS
    _, _, a_form, m_form = assemble_K_M(V)
    f_expr = lambda x: ufl.sin(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr, bc=bc, combined_form_for_lifting=a_form + m_form)
    
    u_exact = sol_exact(beta, DIRICHLET)
    
    # Solve methods
    solutions = {}
    errors = {}
    
    if 'Sinc' in methods_to_compare:
        print("  Solving Dirichlet with Sinc...")
        if k_option == "dynamic":
            k_use = dynamic_k(beta, N)
            u_sinc, _ = sinc_solver(B, M, b, V, bc=bc, beta=beta, k=k_use)
        elif isinstance(k_option, (int, float)):
            u_sinc, _ = sinc_solver(B, M, b, V, bc=bc, beta=beta, k=k_option)
        else:
            u_sinc, _ = sinc_solver(B, M, b, V, bc=bc, beta=beta)
        solutions['Sinc'] = u_sinc
        errors['Sinc'] = compute_errors_refined(u_sinc, u_exact, V, mesh, refine_factor=16)
        print(f"    Sinc (Dirichlet): L²={errors['Sinc']['l2_error']:.2e}, H¹={errors['Sinc']['h1_error']:.2e}, L∞={errors['Sinc']['linf_error']:.2e}")
    
    if 'BRASIL' in methods_to_compare:
        print("  Solving Dirichlet with BRASIL...")
        u_brasil, _ = rational_solve_unified(B, M, b, V, bc=bc, beta=beta, m=m, interval=interval, scale_factor=scale_factor)
        solutions['BRASIL'] = u_brasil
        errors['BRASIL'] = compute_errors_refined(u_brasil, u_exact, V, mesh, refine_factor=16)
        print(f"    BRASIL (Dirichlet): L²={errors['BRASIL']['l2_error']:.2e}, H¹={errors['BRASIL']['h1_error']:.2e}, L∞={errors['BRASIL']['linf_error']:.2e}")
    
    # Convergence series (L2 vs DOF) for Dirichlet
    l2_convergence_data = {}
    if convergence_N_list:
        for method in methods_to_compare:
            dofs_series = []
            l2_series = []
            for Nconv in convergence_N_list:
                mesh_c = make_uniform_interval(Nconv, 0.0, 1.0)
                V_c = fem.functionspace(mesh_c, ("Lagrange", 1))
                facet_tags_c, _ = tag_all_exterior_facets(mesh_c)
                B_c, _, M_c, _, bc_c = build_operator_B(V_c, bc_type="dirichlet", facet_tags=facet_tags_c, ids=(1,), kappa=1)
                _, _, a_c, m_c = assemble_K_M(V_c)
                f_expr_c = lambda x: ufl.sin(ufl.pi * x[0])
                b_c = assemble_rhs(V_c, f_expr=f_expr_c, bc=bc_c, combined_form_for_lifting=a_c + m_c)
                if method == 'Sinc':
                    if k_option == "dynamic":
                        k_c = dynamic_k(beta, Nconv)
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=bc_c, beta=beta, k=k_c)
                    elif isinstance(k_option, (int, float)):
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=bc_c, beta=beta, k=k_option)
                    else:
                        u_c, _ = sinc_solver(B_c, M_c, b_c, V_c, bc=bc_c, beta=beta)
                elif method == 'BRASIL':
                    u_c, _ = rational_solve_unified(B_c, M_c, b_c, V_c, bc=bc_c, beta=beta, m=m, interval=interval, scale_factor=scale_factor)
                else:
                    continue
                err_c = compute_errors_refined(u_c, u_exact, V_c, mesh_c, refine_factor=16)
                dofs_series.append(len(u_c.x.array))
                l2_series.append(err_c['l2_error'])
            if len(dofs_series) > 0:
                l2_convergence_data[method] = (dofs_series, l2_series)
    
    # Plot
    if show_plot or save_plot:
        mesh_info = f"N={N}"
        title_prefix = f"Fractional Dirichlet (β={beta})"
        filename = f"frac_dirichlet_comparison_N{N}_beta{beta}.png" if save_plot else None
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors,
            V=V,
            mesh=mesh,
            title_prefix=title_prefix,
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=mesh_info,
            l2_convergence_data=l2_convergence_data
        )
    
    return solutions, errors


 
    
    # ============================================================================
# UNIFIED COMPARISON FUNCTIONS
    # ============================================================================

def compare_standard_methods(N=20, methods_to_compare=['Standard FEM', 'Sinc'], 
                            show_plot=True, save_plot=True):
    """
    Compare different methods for standard (non-fractional) problems.
    
    Parameters:
    -----------
    N : int
        Number of mesh elements
    bc_type : str
        Boundary condition type: "neumann" or "dirichlet"
    methods_to_compare : list
        List of methods to compare: ['Standard FEM', 'Sinc', 'BRASIL']
    show_plot : bool
        Whether to display the plot
    save_plot : bool
        Whether to save the plot
    """
    print(f"🧪 Comparing Standard Methods (NEUMANN BC, N={N})")
    # Setup Neumann problem
    mesh = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr=f_expr, bc=None)
    
    def u_exact(x):
        return np.cos(np.pi * x) / (1 + np.pi ** 2)
    
    # Solve with different methods
    solutions = {}
    errors = {}
    
    if 'Standard FEM' in methods_to_compare:
        print("  Solving with Standard FEM...")
        u_standard, _ = solve_petsc(B, b, V)
        solutions['Standard FEM'] = u_standard
        errors['Standard FEM'] = compute_errors_refined(u_standard, u_exact, V, mesh, refine_factor=16)
        print(f"    Standard FEM: L²={errors['Standard FEM']['l2_error']:.2e}, H¹={errors['Standard FEM']['h1_error']:.2e}, L∞={errors['Standard FEM']['linf_error']:.2e}")
    
    if 'Sinc' in methods_to_compare:
        print("  Solving with Sinc (β=1.0)...")
        u_sinc, _ = solve_petsc(B, b, V)
        solutions['Sinc'] = u_sinc
        errors['Sinc'] = compute_errors_refined(u_sinc, u_exact, V, mesh, refine_factor=16)
        print(f"    Sinc: L²={errors['Sinc']['l2_error']:.2e}, H¹={errors['Sinc']['h1_error']:.2e}, L∞={errors['Sinc']['linf_error']:.2e}")
    
    if 'BRASIL' in methods_to_compare:
        print("  Solving with BRASIL (β=1.0)...")
        u_brasil, _ = rational_solve_unified(B, M, b, V, bc=None, beta=1.0, m=2, interval=(1, 11))
        solutions['BRASIL'] = u_brasil
        errors['BRASIL'] = compute_errors_refined(u_brasil, u_exact, V, mesh, refine_factor=16)
        print(f"    BRASIL: L²={errors['BRASIL']['l2_error']:.2e}, H¹={errors['BRASIL']['h1_error']:.2e}, L∞={errors['BRASIL']['linf_error']:.2e}")
    
    title_prefix = "Standard Neumann Comparison"
    filename = f"standard_neumann_comparison_N{N}.png" if save_plot else None
        
    title_prefix = "Standard Neumann Comparison"
    filename = f"standard_neumann_comparison_N{N}.png" if save_plot else None
    
    # Create comparison plot
    if show_plot or save_plot and solutions:
        plot_methods_comparison(
            u_exact_func=u_exact,
            solutions_dict=solutions,
            errors_dict=errors,
            V=V, mesh=mesh,
            title_prefix=title_prefix,
            save_filename=filename,
            show_mesh_points=True,
            mesh_info=f"N={N}"
        )
    
    return solutions, errors

def compare_fractional_methods(N=20, beta=0.5, m=2, interval=(1, 11), 
                              methods_to_compare=['Sinc', 'BRASIL'], show_plot=True, save_plot=True):
    """
    Compare different methods for fractional problems.
    
    Parameters:
    -----------
    N : int
        Number of mesh elements
    beta : float
        Fractional power
    bc_type : str
        Boundary condition type: "neumann" or "dirichlet"
    m : int
        BRASIL approximation order
    interval : tuple
        BRASIL approximation interval
    methods_to_compare : list
        List of methods to compare: ['Sinc', 'BRASIL']
    show_plot : bool
        Whether to display the plot
    save_plot : bool
        Whether to save the plot
    """
    print(f"🧪 Comparing Fractional Methods (NEUMANN BC, N={N}, β={beta})")
    return test_frac_neumann_comparison(N, beta, m, interval, methods_to_compare, show_plot, save_plot)

    
    # ============================================================================
# MAIN
    # ============================================================================

 
 
# %%

if __name__ == "__main__":
    # Single example: Fractional Neumann comparison (Exact vs Sinc vs BRASIL)
    print("\nExecuting single example: Fractional Neumann (β=0.5), Sinc vs BRASIL")
    test_frac_neumann_comparison(
        N=80,
        beta=0.5,
        methods_to_compare=['Sinc', 'BRASIL'],
        show_plot=True,
        save_plot=False,
        convergence_N_list=(20, 40, 60, 80, 100, 240),
        m=2,
        interval=(0.2, 0.9),
        scale_factor=5
    )
    print("\n✅ Done. Plot saved in the output directory.")
    
# %%    
    # Dirichlet example: Fractional Dirichlet (Sinc) with dynamic k
    print("\nExecuting single example: Fractional Dirichlet (β=0.5), Sinc vs BRASIL")
    test_frac_dirichlet_comparison(
        N=240,
        beta=0.2,
        methods_to_compare=['Sinc'],
        show_plot=True,
        save_plot=False,
        convergence_N_list=(10, 20, 40, 80, 160, 240, 320),
        scale_factor=3,
        m=3,
        interval=(0.9,51),
        k_option=0.2,
    )
# %%
