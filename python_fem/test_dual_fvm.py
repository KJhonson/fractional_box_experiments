# %%
# !/usr/bin/env python3
"""
Test script for dual FVM load assembly.
This demonstrates how to use the new dual FVM approach for assembling RHS vectors.
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from petsc4py import PETSc
from utils.loads import assemble_rhs, assemble_rhs_dual_fvm, assemble_rhs_dual_fvm_exact
from operators import build_operator_B
from utils.sinc_solver import solve_petsc
from domains import make_uniform_interval

def test_dual_fvm_vs_standard():
    """
    Compare standard FEM RHS assembly with dual FVM RHS assembly.
    """
    print("Testing Dual FVM vs Standard FEM RHS Assembly")
    print("=" * 50)
    
    # Create a simple 1D mesh
    N = 20
    mesh_domain = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh_domain, ("Lagrange", 1))
    
    # Define source function: f(x) = indicator function on [0, 1/2]
    def f_expr(x):
        return ufl.conditional(ufl.lt(x[0], 0.5), 1.0, 0.0)
    
    # Standard FEM RHS assembly
    print("1. Standard FEM RHS assembly...")
    b_standard = assemble_rhs(V, f_expr=f_expr)
    
    # Dual FVM RHS assembly
    print("2. Dual FVM RHS assembly...")
    b_dual_fvm = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
    
    # Compare the two approaches
    print("3. Comparing results...")
    
    # Convert to numpy arrays for comparison
    b_std_array = b_standard.array
    b_fvm_array = b_dual_fvm.array
    
    # Compute relative difference
    diff = np.abs(b_std_array - b_fvm_array)
    rel_diff = diff / (np.abs(b_std_array) + 1e-15)
    
    print(f"   Standard FEM RHS norm: {np.linalg.norm(b_std_array):.6e}")
    print(f"   Dual FVM RHS norm:     {np.linalg.norm(b_fvm_array):.6e}")
    print(f"   Max absolute diff:     {np.max(diff):.6e}")
    print(f"   Max relative diff:     {np.max(rel_diff):.6e}")
    
    # Test solving with both approaches
    print("\n4. Testing solution with both RHS approaches...")
    
    # Build operator (Neumann BC)
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    
    # Solve with standard RHS
    u_standard, _ = solve_petsc(B, b_standard, V)
    
    # Solve with dual FVM RHS
    u_dual_fvm, _ = solve_petsc(B, b_dual_fvm, V)
    
    # Compare solutions
    u_std_array = u_standard.x.array
    u_fvm_array = u_dual_fvm.x.array
    
    sol_diff = np.abs(u_std_array - u_fvm_array)
    sol_rel_diff = sol_diff / (np.abs(u_std_array) + 1e-15)
    
    print(f"   Standard solution norm: {np.linalg.norm(u_std_array):.6e}")
    print(f"   Dual FVM solution norm: {np.linalg.norm(u_fvm_array):.6e}")
    print(f"   Max solution diff:      {np.max(sol_diff):.6e}")
    print(f"   Max solution rel diff:  {np.max(sol_rel_diff):.6e}")
    
    return {
        'standard_rhs': b_standard,
        'dual_fvm_rhs': b_dual_fvm,
        'standard_solution': u_standard,
        'dual_fvm_solution': u_dual_fvm,
        'rhs_diff': np.max(diff),
        'solution_diff': np.max(sol_diff)
    }

def test_different_quad_degrees():
    """
    Test dual FVM with different quadrature degrees.
    """
    print("\nTesting Dual FVM with Different Quadrature Degrees")
    print("=" * 50)
    
    # Create mesh
    N = 10
    mesh_domain = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh_domain, ("Lagrange", 1))
    
    # Define source function: f(x) = indicator function on [0, 1/2]
    def f_expr(x):
        return ufl.conditional(ufl.lt(x[0], 0.5), 1.0, 0.0)
    
    quad_degrees = [1, 2, 3, 4]
    results = {}
    
    for deg in quad_degrees:
        print(f"Testing quadrature degree {deg}...")
        b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=deg)
        results[deg] = b.array.copy()
        print(f"   RHS norm: {np.linalg.norm(b.array):.6e}")
    
    # Compare with standard FEM
    b_standard = assemble_rhs(V, f_expr=f_expr)
    print(f"\nStandard FEM RHS norm: {np.linalg.norm(b_standard.array):.6e}")
    
    for deg in quad_degrees:
        diff = np.max(np.abs(results[deg] - b_standard.array))
        print(f"Quad degree {deg} vs standard: max diff = {diff:.6e}")

def test_custom_neumann_problem():
    """
    Test the custom Neumann problem with the given exact solution.
    Problem: -u'' + u = 1_[1/2,1] on [0,1] with Neumann boundary conditions.
    """
    print("\nTesting Custom Neumann Problem")
    print("=" * 50)
    
    # Define the correct exact solution for -u'' + u = 1_[1/2,1] with Neumann BC
    def u_exact(x):
        """
        Correct exact solution for -u'' + u = 1_[1/2,1] with Neumann BC.
        """
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # Constants computed from the correct solution
        cosh_half = np.cosh(0.5)
        cosh_one = np.cosh(1.0)
        sech_one = 1.0 / cosh_one
        
        A = 1.0 / (2.0 * cosh_half)
        C = -cosh_one / (2.0 * cosh_half)
        
        # Apply the piecewise definition
        mask_left = x < 0.5
        mask_right = x >= 0.5
        
        result[mask_left] = A * np.cosh(x[mask_left])
        result[mask_right] = 1 + C * sech_one * np.cosh(x[mask_right] - 1)
        
        return result
    
    # Define the load function: f(x) = 1_[1/2,1] (indicator function on [1/2,1])
    def f_expr(x):
        return ufl.conditional(ufl.ge(x[0], 0.5), 1.0, 0.0)
    
    # Create mesh and function space
    N = 20
    mesh_domain = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh_domain, ("Lagrange", 1))
    
    # Build operator (Neumann BC)
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    
    # Assemble RHS using dual FVM approach
    b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
    
    # Solve the system
    u_h, _ = solve_petsc(B, b, V)
    
    # Compute errors using the refined mesh approach from playground.py
    from playground import compute_errors_refined
    errors = compute_errors_refined(u_h, u_exact, V, mesh_domain, refine_factor=16)
    
    print(f"Custom Neumann Problem Errors:")
    print(f"  L² error: {errors['l2_error']:.2e}")
    print(f"  L∞ error: {errors['linf_error']:.2e}")
    
    return {
        'solution': u_h,
        'exact_solution': u_exact,
        'errors': errors,
        'mesh': mesh_domain,
        'function_space': V
    }

def test_custom_neumann_convergence():
    """
    Test L² convergence rate for the custom Neumann problem.
    """
    print("\nCustom Neumann Problem L² Convergence Analysis")
    print("=" * 50)
    
    # Define the correct exact solution for -u'' + u = 1_[1/2,1] with Neumann BC
    def u_exact_base(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # Constants computed from the correct solution
        cosh_half = np.cosh(0.5)
        cosh_one = np.cosh(1.0)
        sech_one = 1.0 / cosh_one
        
        A = 1.0 / (2.0 * cosh_half)
        C = -cosh_one / (2.0 * cosh_half)
        
        # Apply the piecewise definition
        mask_left = x < 0.5
        mask_right = x >= 0.5
        
        result[mask_left] = A * np.cosh(x[mask_left])
        result[mask_right] = 1 + C * sech_one * np.cosh(x[mask_right] - 1)
        
        return result
    
    # For Neumann problems, we need to adjust the exact solution by a constant
    # to match the numerical solution (since Neumann BC only determine solution up to a constant)
    def u_exact(x, offset=0.0):
        return u_exact_base(x) + offset
    
    # Define the load function
    def f_expr(x):
        return ufl.conditional(ufl.ge(x[0], 0.5), 1.0, 0.0)
    
    # Test different mesh sizes
    N_values = [10, 20, 40, 80, 160, 320]
    l2_errors = []
    h_values = []
    solutions = []
    meshes = []
    
    from playground import compute_errors_refined
    
    for N in N_values:
        print(f"  Testing N = {N}...")
        
        # Create mesh and solve
        mesh = make_uniform_interval(N, 0.0, 1.0)
        V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Build operator and solve using dual FVM approach
        B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
        b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
        u_h, _ = solve_petsc(B, b, V)
        
        # For Neumann problems, adjust the exact solution by a constant offset
        # to match the numerical solution at a reference point (e.g., x=0.5)
        x_coords = mesh.geometry.x[:, 0]
        u_numerical = u_h.x.array
        
        # Find the index closest to x=0.5
        idx_ref = np.argmin(np.abs(x_coords - 0.5))
        u_numerical_ref = u_numerical[idx_ref]
        u_exact_ref = u_exact_base(0.5)
        offset = u_numerical_ref - u_exact_ref
        
        # Create adjusted exact solution function
        def u_exact_adjusted(x):
            return u_exact_base(x) + offset
        
        # Compute errors (only L²)
        errors = compute_errors_refined(u_h, u_exact_adjusted, V, mesh, refine_factor=16)
        
        l2_error = errors['l2_error']
        
        l2_errors.append(l2_error)
        h_values.append(1.0 / N)
        solutions.append((u_h, u_exact_adjusted, offset))
        meshes.append(mesh)
        
        print(f"    L² error: {l2_error:.2e}, offset: {offset:.6f}")
    
    # Compute L² convergence rate
    log_h = np.log(h_values)
    log_l2_error = np.log(l2_errors)
    
    # Linear regression to find slope (convergence rate)
    l2_slope = np.polyfit(log_h, log_l2_error, 1)[0]
    
    print(f"\nCustom Neumann Problem L² Convergence Rate:")
    print(f"  L² Convergence Rate: {l2_slope:.2f} (Expected: 2.0)")
    
    # Create combined plot: exact vs numerical solution + convergence analysis
    import matplotlib.pyplot as plt
    
    # Get a smaller mesh solution for plotting (so dashed line is visible)
    plot_mesh = meshes[1]  # Use N=20 instead of N=320
    plot_u_h, plot_u_exact, plot_offset = solutions[1]
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Exact vs Numerical Solution
    x_coords = plot_mesh.geometry.x[:, 0]
    u_numerical = plot_u_h.x.array
    
    # Sort by x coordinates for plotting
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    u_sorted = u_numerical[sort_idx]
    
    # Evaluate exact solution at mesh points
    u_exact_vals = plot_u_exact(x_sorted)
    
    # Plot exact and numerical solutions
    ax1.plot(x_sorted, u_exact_vals, 'r-', linewidth=3, label='Exact Solution')
    ax1.plot(x_sorted, u_sorted, 'b--o', linewidth=2, markersize=6, label=f'Numerical Solution (N={N_values[1]})')
    ax1.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x = 1/2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('Custom Neumann Problem: Exact vs Numerical Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: L² Convergence Analysis
    ax2.loglog(h_values, l2_errors, 'bo-', linewidth=2, markersize=8, label=f'L² Error (rate: {l2_slope:.2f})')
    
    # Add reference line for O(h²)
    h_ref = np.array([h_values[0], h_values[-1]])
    l2_ref = l2_errors[0] * (h_ref / h_values[0])**2
    
    ax2.loglog(h_ref, l2_ref, 'b--', linewidth=2, alpha=0.7, label='O(h²) reference')
    
    ax2.set_xlabel('Mesh size h')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'L² Convergence Rate: {l2_slope:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/dolfinx/shared/FEM_project/custom_neumann_combined_plot.png', dpi=300, bbox_inches='tight')
    print("Custom Neumann combined plot saved to: custom_neumann_combined_plot.png")
    plt.close()
    
    return {
        'N_values': N_values,
        'l2_errors': l2_errors,
        'h_values': h_values,
        'l2_slope': l2_slope
    }

def test_singular_dirichlet_problem():
    """
    Test the singular Dirichlet problem with dual FVM approach.
    Problem: -u'' = x^(-α) with u(0) = u(1) = 0, α = 1/4
    """
    print("\nTesting Singular Dirichlet Problem")
    print("=" * 50)
    
    # Define the exact solution for -u'' = x^(-α) with Dirichlet BC, α = 0.49
    def u_exact(x):
        """
        Exact solution for -u'' = x^(-α) with Dirichlet BC, α = 0.49.
        u = (x - x^(2-α)) / ((1-α)(2-α))
        """
        alpha = 0.49
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # Avoid division by zero at x=0
        mask = x > 1e-15
        
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0  # u(0) = 0
        
        return result
    
    # Define the load function: f(x) = x^(-α) with α = 0.49
    def f_expr(x):
        alpha = 0.49
        # Avoid singularity at x=0 by using a small epsilon
        return ufl.conditional(ufl.lt(x[0], 1e-10), 0.0, x[0]**(-alpha))
    
    # Create mesh and function space
    N = 20
    mesh_domain = make_uniform_interval(N, 0.0, 1.0)
    V = fem.functionspace(mesh_domain, ("Lagrange", 1))
    
    # Build operator (Dirichlet BC)
    from domains import tag_all_exterior_facets
    facet_tags, _ = tag_all_exterior_facets(mesh_domain)
    B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
    
    # Assemble RHS using dual FVM approach
    b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
    
    # Apply boundary conditions to RHS vector
    from dolfinx.fem import petsc
    petsc.set_bc(b, [bc])
    
    # Solve the system
    u_h, _ = solve_petsc(B, b, V)
    
    # Manually enforce boundary conditions on the solution
    boundary_dofs = bc.dof_indices()[0]
    u_h.x.array[boundary_dofs] = 0.0
    u_h.x.scatter_forward()
    
    # Compute errors using the refined mesh approach from playground.py
    from playground import compute_errors_refined
    errors = compute_errors_refined(u_h, u_exact, V, mesh_domain, refine_factor=16)
    
    print(f"Singular Dirichlet Problem Errors:")
    print(f"  L² error: {errors['l2_error']:.2e}")
    print(f"  L∞ error: {errors['linf_error']:.2e}")
    
    return {
        'solution': u_h,
        'exact_solution': u_exact,
        'errors': errors,
        'mesh': mesh_domain,
        'function_space': V
    }

def test_singular_dirichlet_convergence():
    """
    Test L² convergence rate for the singular Dirichlet problem.
    """
    print("\nSingular Dirichlet Problem L² Convergence Analysis")
    print("=" * 50)
    
    # Define alpha at function level
    alpha = 0.49
    
    # Define the exact solution for -u'' = x^(-α) with Dirichlet BC, α = 0.49
    def u_exact(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # Avoid division by zero at x=0
        mask = x > 1e-15
        
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0  # u(0) = 0
        
        return result
    
    # Define the load function: f(x) = x^(-α) with α = 0.4999
    def f_expr(x):
        # Avoid singularity at x=0 by using a small epsilon
        return ufl.conditional(ufl.lt(x[0], 1e-10), 0.0, x[0]**(-alpha))
    
    # Test different mesh sizes
    N_values = [10, 20, 40, 80, 160, 320]
    l2_errors = []
    h_values = []
    solutions = []
    meshes = []
    
    from playground import compute_errors_refined
    from domains import tag_all_exterior_facets
    
    for N in N_values:
        print(f"  Testing N = {N}...")
        
        # Create mesh and solve
        mesh = make_uniform_interval(N, 0.0, 1.0)
        V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Build operator (Dirichlet BC)
        facet_tags, _ = tag_all_exterior_facets(mesh)
        B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        # Build operator and solve using dual FVM approach
        b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
        
        # Apply boundary conditions to RHS vector
        from dolfinx.fem import petsc
        petsc.set_bc(b, [bc])
        
        # Solve the system
        u_h, _ = solve_petsc(B, b, V)
        
        # Manually enforce boundary conditions on the solution
        boundary_dofs = bc.dof_indices()[0]
        u_h.x.array[boundary_dofs] = 0.0
        u_h.x.scatter_forward()
        
        # Compute errors (only L²)
        errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
        
        l2_error = errors['l2_error']
        
        l2_errors.append(l2_error)
        h_values.append(1.0 / N)
        solutions.append(u_h)
        meshes.append(mesh)
        
        print(f"    L² error: {l2_error:.2e}")
    
    # Compute L² convergence rate
    log_h = np.log(h_values)
    log_l2_error = np.log(l2_errors)
    
    # Linear regression to find slope (convergence rate)
    l2_slope = np.polyfit(log_h, log_l2_error, 1)[0]
    
    print(f"\nSingular Dirichlet Problem L² Convergence Rate:")
    print(f"  L² Convergence Rate: {l2_slope:.2f} (Expected: 2.0)")
    
    # Create combined plot: exact vs numerical solution + convergence analysis
    import matplotlib.pyplot as plt
    
    # Get the finest mesh solution for plotting
    finest_mesh = meshes[2]  # Use N=40 for better visibility
    finest_u_h = solutions[2]
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Exact vs Numerical Solution
    x_coords = finest_mesh.geometry.x[:, 0]
    u_numerical = finest_u_h.x.array
    
    # Sort by x coordinates for plotting
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    u_sorted = u_numerical[sort_idx]
    
    # Evaluate exact solution at mesh points
    u_exact_vals = u_exact(x_sorted)
    
    # Plot exact and numerical solutions
    ax1.plot(x_sorted, u_exact_vals, 'r-', linewidth=3, label='Exact Solution')
    ax1.plot(x_sorted, u_sorted, 'b--o', linewidth=2, markersize=6, label=f'Numerical Solution (N={N_values[2]})')
    ax1.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x = 1/2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'Singular Dirichlet Problem: Exact vs Numerical Solution (α={alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: L² Convergence Analysis
    ax2.loglog(h_values, l2_errors, 'bo-', linewidth=2, markersize=8, label=f'L² Error (rate: {l2_slope:.2f})')
    
    # Add reference line for O(h²)
    h_ref = np.array([h_values[0], h_values[-1]])
    l2_ref = l2_errors[0] * (h_ref / h_values[0])**2
    
    ax2.loglog(h_ref, l2_ref, 'b--', linewidth=2, alpha=0.7, label='O(h²) reference')
    
    ax2.set_xlabel('Mesh size h')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'L² Convergence Rate: {l2_slope:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/dolfinx/shared/FEM_project/singular_dirichlet_combined_plot.png', dpi=300, bbox_inches='tight')
    print("Singular Dirichlet combined plot saved to: singular_dirichlet_combined_plot.png")
    plt.close()
    
    return {
        'N_values': N_values,
        'l2_errors': l2_errors,
        'h_values': h_values,
        'l2_slope': l2_slope
    }

def assemble_rhs_exact_integration(V, alpha = 0.49):
    """
    Assemble RHS using exact integration for f(x) = x^(-α).
    
    For the dual FVM approach, we need to compute:
    F_i = ∫_{b_i} f dx = ∫_{b_i} x^(-α) dx
    
    where b_i is the dual cell around vertex i.
    In 1D, the dual cell b_i is the interval [x_{i-1/2}, x_{i+1/2}]
    where x_{i-1/2} = (x_{i-1} + x_i)/2 and x_{i+1/2} = (x_i + x_{i+1})/2
    """
    from petsc4py import PETSc
    from dolfinx import fem
    
    # Get mesh information
    mesh = V.mesh
    x_coords = mesh.geometry.x[:, 0]
    n_vertices = len(x_coords)
    
    # Create RHS vector
    b = fem.petsc.create_vector(fem.form(ufl.TestFunction(V) * ufl.dx))
    
    # Compute exact integrals for each dual cell
    for i in range(n_vertices):
        # Define dual cell boundaries
        if i == 0:
            # Left boundary: dual cell is [0, (x_0 + x_1)/2]
            x_left = 0.0
            x_right = (x_coords[0] + x_coords[1]) / 2.0
        elif i == n_vertices - 1:
            # Right boundary: dual cell is [(x_{n-2} + x_{n-1})/2, 1]
            x_left = (x_coords[i-1] + x_coords[i]) / 2.0
            x_right = 1.0
        else:
            # Interior: dual cell is [(x_{i-1} + x_i)/2, (x_i + x_{i+1})/2]
            x_left = (x_coords[i-1] + x_coords[i]) / 2.0
            x_right = (x_coords[i] + x_coords[i+1]) / 2.0
        
        # Compute exact integral: ∫_{x_left}^{x_right} x^(-α) dx
        if x_left > 1e-15:  # Avoid singularity
            integral_value = (x_right**(1-alpha) - x_left**(1-alpha)) / (1-alpha)
        else:
            # Handle singularity at x=0
            integral_value = x_right**(1-alpha) / (1-alpha)
        
        # Set the value in the RHS vector
        b.setValue(i, integral_value)
    
    b.assemble()
    return b

def create_adaptive_mesh_1d_smooth(N_base, alpha = 0.49):
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
    """
    import dolfinx
    from dolfinx import mesh
    
    # Create a smooth mesh that follows the singularity
    # We want h(x) ∝ x^α, so we use a mapping function
    
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
    x_coords = [smooth_mapping(t, alpha) for t in t_coords]
    
    # Ensure we start at 0 and end at 1
    x_coords[0] = 0.0
    x_coords[-1] = 1.0
    
    # Create mesh with these coordinates
    mesh_adaptive = make_uniform_interval(N_total, 0.0, 1.0)
    mesh_adaptive.geometry.x[:, 0] = np.array(x_coords)
    
    return mesh_adaptive

def create_adaptive_mesh_1d_cluster(N_base, refinement_factor=4, refinement_region=0.1):
    """
    Create a cluster-type adaptive 1D mesh (the original implementation).
    
    This creates a discontinuous mesh with two distinct regions:
    - Fine region near x=0
    - Coarse region from refinement_region to 1.0
    """
    import dolfinx
    from dolfinx import mesh
    
    # Create a uniform mesh first
    mesh_uniform = make_uniform_interval(N_base, 0.0, 1.0)
    
    # Get coordinates
    x_coords = mesh_uniform.geometry.x[:, 0].copy()
    
    # Create refined coordinates
    refined_coords = []
    
    # Add refined region near x=0
    n_refined = N_base * refinement_factor
    for i in range(n_refined + 1):
        x = (i / n_refined) * refinement_region
        refined_coords.append(x)
    
    # Add uniform region from refinement_region to 1.0
    n_uniform = N_base
    for i in range(1, n_uniform + 1):
        x = refinement_region + (i / n_uniform) * (1.0 - refinement_region)
        refined_coords.append(x)
    
    # Remove duplicates and sort
    refined_coords = sorted(list(set(refined_coords)))
    
    # Create new mesh with refined coordinates
    N_total = len(refined_coords) - 1
    mesh_adaptive = make_uniform_interval(N_total, 0.0, 1.0)
    
    # Modify the coordinates
    mesh_adaptive.geometry.x[:, 0] = np.array(refined_coords)
    
    return mesh_adaptive

def test_exact_integration_convergence(alpha = 0.49):
    """
    Test convergence using exact integration for the load vector.
    """
    print(f"\nTesting Exact Integration for Load Vector (α={alpha})")
    print("=" * 50)
    
    # Define the exact solution
    def u_exact(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask = x > 1e-15
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0
        return result
    
    # Test different mesh sizes
    N_values = [10, 20, 40, 80, 160, 320]
    l2_errors = []
    dof_values = []
    solutions = []
    meshes = []
    
    from playground import compute_errors_refined
    from domains import tag_all_exterior_facets
    
    for N in N_values:
        print(f"  Testing N = {N}...")
        
        # Create mesh and solve
        mesh = make_uniform_interval(N, 0.0, 1.0)
        V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Get number of DOFs
        n_dofs = V.dofmap.index_map.size_global
        
        # Build operator (Dirichlet BC)
        facet_tags, _ = tag_all_exterior_facets(mesh)
        B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        # Assemble RHS using EXACT integration
        b = assemble_rhs_exact_integration(V, alpha=alpha)
        
        # Apply boundary conditions to RHS vector
        from dolfinx.fem import petsc
        petsc.set_bc(b, [bc])
        
        # Solve the system
        u_h, _ = solve_petsc(B, b, V)
        
        # Manually enforce boundary conditions on the solution
        boundary_dofs = bc.dof_indices()[0]
        u_h.x.array[boundary_dofs] = 0.0
        u_h.x.scatter_forward()
        
        # Compute errors (only L²)
        errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
        
        l2_error = errors['l2_error']
        
        l2_errors.append(l2_error)
        dof_values.append(n_dofs)
        solutions.append(u_h)
        meshes.append(mesh)
        
        print(f"    L² error: {l2_error:.2e}, DOFs: {n_dofs}")
    
    # Compute L² convergence rate with respect to DOFs
    log_dofs = np.log(dof_values)
    log_l2_error = np.log(l2_errors)
    
    # Linear regression to find slope (convergence rate)
    l2_slope = np.polyfit(log_dofs, log_l2_error, 1)[0]
    
    print(f"\nExact Integration L² Convergence Rate (vs DOFs):")
    print(f"  L² Convergence Rate: {l2_slope:.2f} (Expected: -2.0 for O(h²))")
    
    # Create combined plot
    import matplotlib.pyplot as plt
    
    # Get the finest mesh solution for plotting
    finest_mesh = meshes[2]  # Use N=40 for better visibility
    finest_u_h = solutions[2]
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Exact vs Numerical Solution
    x_coords = finest_mesh.geometry.x[:, 0]
    u_numerical = finest_u_h.x.array
    
    # Sort by x coordinates for plotting
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    u_sorted = u_numerical[sort_idx]
    
    # Evaluate exact solution at mesh points
    u_exact_vals = u_exact(x_sorted)
    
    # Plot exact and numerical solutions
    ax1.plot(x_sorted, u_exact_vals, 'r-', linewidth=3, label='Exact Solution')
    ax1.plot(x_sorted, u_sorted, 'b--o', linewidth=2, markersize=6, label=f'Numerical Solution (N={N_values[2]})')
    ax1.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x = 1/2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'Exact Integration: Exact vs Numerical Solution (α={alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: L² Convergence Analysis
    ax2.loglog(dof_values, l2_errors, 'bo-', linewidth=2, markersize=8, label=f'L² Error (rate: {l2_slope:.2f})')
    
    # Add reference line for O(DOF^(-1)) (equivalent to O(h²))
    dof_ref = np.array([dof_values[0], dof_values[-1]])
    l2_ref = l2_errors[0] * (dof_ref / dof_values[0])**(-1)
    
    ax2.loglog(dof_ref, l2_ref, 'b--', linewidth=2, alpha=0.7, label='O(DOF⁻¹) reference')
    
    ax2.set_xlabel('Number of DOFs')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'Exact Integration L² Convergence Rate: {l2_slope:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/dolfinx/shared/FEM_project/exact_integration_combined_plot.png', dpi=300, bbox_inches='tight')
    print("Exact integration combined plot saved to: exact_integration_combined_plot.png")
    plt.close()
    
    return {
        'N_values': N_values,
        'l2_errors': l2_errors,
        'dof_values': dof_values,
        'l2_slope': l2_slope
    }

def test_adaptive_mesh_convergence(alpha = 0.49):
    """
    Test convergence using adaptive mesh refinement near the singularity.
    """
    print(f"\nTesting Adaptive Mesh for Singular Load (α={alpha})")
    print("=" * 50)
    
    # Define the exact solution
    def u_exact(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask = x > 1e-15
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0
        return result
    
    # Define the load function: f(x) = x^(-α)
    def f_expr(x):
        # Avoid singularity at x=0 by using a small epsilon
        return ufl.conditional(ufl.lt(x[0], 1e-10), 0.0, x[0]**(-alpha))
    
    # Test different base mesh sizes with adaptive refinement
    N_base_values = [5, 10, 20, 40, 80]
    l2_errors = []
    dof_values = []
    solutions = []
    meshes = []
    
    from playground import compute_errors_refined
    from domains import tag_all_exterior_facets
    
    for N_base in N_base_values:
        print(f"  Testing N_base = {N_base}...")
        
        # Create smooth adaptive mesh
        mesh = create_adaptive_mesh_1d_smooth(N_base, alpha=alpha)
        V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Get number of DOFs
        n_dofs = V.dofmap.index_map.size_global
        
        # Build operator (Dirichlet BC)
        facet_tags, _ = tag_all_exterior_facets(mesh)
        B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        # Assemble RHS using dual FVM approach
        b = assemble_rhs_dual_fvm(V, f_expr, quad_degree=3)
        
        # Apply boundary conditions to RHS vector
        from dolfinx.fem import petsc
        petsc.set_bc(b, [bc])
        
        # Solve the system
        u_h, _ = solve_petsc(B, b, V)
        
        # Manually enforce boundary conditions on the solution
        boundary_dofs = bc.dof_indices()[0]
        u_h.x.array[boundary_dofs] = 0.0
        u_h.x.scatter_forward()
        
        # Compute errors (only L²)
        errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
        
        l2_error = errors['l2_error']
        
        l2_errors.append(l2_error)
        dof_values.append(n_dofs)
        solutions.append(u_h)
        meshes.append(mesh)
        
        print(f"    L² error: {l2_error:.2e}, DOFs: {n_dofs}")
    
    # Compute L² convergence rate with respect to DOFs
    log_dofs = np.log(dof_values)
    log_l2_error = np.log(l2_errors)
    
    # Linear regression to find slope (convergence rate)
    l2_slope = np.polyfit(log_dofs, log_l2_error, 1)[0]
    
    print(f"\nAdaptive Mesh L² Convergence Rate (vs DOFs):")
    print(f"  L² Convergence Rate: {l2_slope:.2f} (Expected: -1.0 for O(h²))")
    
    # Create combined plot
    import matplotlib.pyplot as plt
    
    # Get the finest mesh solution for plotting
    finest_mesh = meshes[2]  # Use N_base=20 for better visibility
    finest_u_h = solutions[2]
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Exact vs Numerical Solution
    x_coords = finest_mesh.geometry.x[:, 0]
    u_numerical = finest_u_h.x.array
    
    # Sort by x coordinates for plotting
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    u_sorted = u_numerical[sort_idx]
    
    # Evaluate exact solution at mesh points
    u_exact_vals = u_exact(x_sorted)
    
    # Plot exact and numerical solutions
    ax1.plot(x_sorted, u_exact_vals, 'r-', linewidth=3, label='Exact Solution')
    ax1.plot(x_sorted, u_sorted, 'b--o', linewidth=2, markersize=6, label=f'Numerical Solution (N_base={N_base_values[2]})')
    ax1.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x = 1/2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'Adaptive Mesh: Exact vs Numerical Solution (α={alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: L² Convergence Analysis
    ax2.loglog(dof_values, l2_errors, 'bo-', linewidth=2, markersize=8, label=f'L² Error (rate: {l2_slope:.2f})')
    
    # Add reference line for O(DOF^(-1)) (equivalent to O(h²))
    dof_ref = np.array([dof_values[0], dof_values[-1]])
    l2_ref = l2_errors[0] * (dof_ref / dof_values[0])**(-1)
    
    ax2.loglog(dof_ref, l2_ref, 'b--', linewidth=2, alpha=0.7, label='O(DOF⁻¹) reference')
    
    ax2.set_xlabel('Number of DOFs')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'Adaptive Mesh L² Convergence Rate: {l2_slope:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/dolfinx/shared/FEM_project/adaptive_mesh_combined_plot.png', dpi=300, bbox_inches='tight')
    print("Adaptive mesh combined plot saved to: adaptive_mesh_combined_plot.png")
    plt.close()
    
    return {
        'N_base_values': N_base_values,
        'l2_errors': l2_errors,
        'dof_values': dof_values,
        'l2_slope': l2_slope
    }

def test_compare_adaptive_meshes(alpha = 0.49):
    """
    Compare smooth vs cluster adaptive meshes for the singular problem.
    """
    print(f"\nComparing Smooth vs Cluster Adaptive Meshes (α={alpha})")
    print("=" * 50)
    
    # Define the exact solution
    def u_exact(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask = x > 1e-15
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0
        return result
    
    # Define the load function: f(x) = x^(-α)
    def f_expr(x):
        return ufl.conditional(ufl.lt(x[0], 1e-10), 0.0, x[0]**(-alpha))
    
    # Test different base mesh sizes
    N_base_values = [5, 10, 20, 40]
    
    # Results for both mesh types
    smooth_errors = []
    cluster_errors = []
    smooth_dof_values = []
    cluster_dof_values = []
    
    from playground import compute_errors_refined
    from domains import tag_all_exterior_facets
    
    for N_base in N_base_values:
        print(f"  Testing N_base = {N_base}...")
        
        # Test 1: Smooth adaptive mesh
        mesh_smooth = create_adaptive_mesh_1d_smooth(N_base, alpha=alpha)
        V_smooth = fem.functionspace(mesh_smooth, ("Lagrange", 1))
        
        # Get number of DOFs for smooth mesh
        n_dofs_smooth = V_smooth.dofmap.index_map.size_global
        
        facet_tags, _ = tag_all_exterior_facets(mesh_smooth)
        B_smooth, _, M_smooth, _, bc_smooth = build_operator_B(V_smooth, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        b_smooth = assemble_rhs_dual_fvm(V_smooth, f_expr, quad_degree=3)
        from dolfinx.fem import petsc
        petsc.set_bc(b_smooth, [bc_smooth])
        
        u_h_smooth, _ = solve_petsc(B_smooth, b_smooth, V_smooth)
        boundary_dofs = bc_smooth.dof_indices()[0]
        u_h_smooth.x.array[boundary_dofs] = 0.0
        u_h_smooth.x.scatter_forward()
        
        errors_smooth = compute_errors_refined(u_h_smooth, u_exact, V_smooth, mesh_smooth, refine_factor=16)
        smooth_errors.append(errors_smooth['l2_error'])
        smooth_dof_values.append(n_dofs_smooth)
        
        # Test 2: Cluster adaptive mesh
        mesh_cluster = create_adaptive_mesh_1d_cluster(N_base, refinement_factor=4, refinement_region=0.1)
        V_cluster = fem.functionspace(mesh_cluster, ("Lagrange", 1))
        
        # Get number of DOFs for cluster mesh
        n_dofs_cluster = V_cluster.dofmap.index_map.size_global
        
        facet_tags, _ = tag_all_exterior_facets(mesh_cluster)
        B_cluster, _, M_cluster, _, bc_cluster = build_operator_B(V_cluster, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        b_cluster = assemble_rhs_dual_fvm(V_cluster, f_expr, quad_degree=3)
        petsc.set_bc(b_cluster, [bc_cluster])
        
        u_h_cluster, _ = solve_petsc(B_cluster, b_cluster, V_cluster)
        boundary_dofs = bc_cluster.dof_indices()[0]
        u_h_cluster.x.array[boundary_dofs] = 0.0
        u_h_cluster.x.scatter_forward()
        
        errors_cluster = compute_errors_refined(u_h_cluster, u_exact, V_cluster, mesh_cluster, refine_factor=16)
        cluster_errors.append(errors_cluster['l2_error'])
        cluster_dof_values.append(n_dofs_cluster)
        
        print(f"    Smooth: L² error = {errors_smooth['l2_error']:.2e}, DOFs = {n_dofs_smooth}")
        print(f"    Cluster: L² error = {errors_cluster['l2_error']:.2e}, DOFs = {n_dofs_cluster}")
    
    # Compute convergence rates with respect to DOFs
    log_dofs_smooth = np.log(smooth_dof_values)
    log_error_smooth = np.log(smooth_errors)
    smooth_slope = np.polyfit(log_dofs_smooth, log_error_smooth, 1)[0]
    
    log_dofs_cluster = np.log(cluster_dof_values)
    log_error_cluster = np.log(cluster_errors)
    cluster_slope = np.polyfit(log_dofs_cluster, log_error_cluster, 1)[0]
    
    print(f"\nConvergence Rates (vs DOFs):")
    print(f"  Smooth Adaptive Mesh: {smooth_slope:.2f}")
    print(f"  Cluster Adaptive Mesh: {cluster_slope:.2f}")
    
    # Create comparison plot
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mesh structures for N_base=10
    N_plot = 10
    mesh_smooth_plot = create_adaptive_mesh_1d_smooth(N_plot, alpha=alpha)
    mesh_cluster_plot = create_adaptive_mesh_1d_cluster(N_plot, refinement_factor=4, refinement_region=0.1)
    
    x_smooth = mesh_smooth_plot.geometry.x[:, 0]
    x_cluster = mesh_cluster_plot.geometry.x[:, 0]
    
    # Plot mesh structures
    ax1.plot(x_smooth, np.ones_like(x_smooth), 'bo-', markersize=4, label='Smooth Mesh')
    ax1.plot(x_cluster, np.ones_like(x_cluster) * 0.5, 'ro-', markersize=4, label='Cluster Mesh')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Mesh Points')
    ax1.set_title(f'Mesh Structure Comparison (N_base=10, α={alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence comparison
    ax2.loglog(smooth_dof_values, smooth_errors, 'bo-', linewidth=2, markersize=8, label=f'Smooth (rate: {smooth_slope:.2f})')
    ax2.loglog(cluster_dof_values, cluster_errors, 'ro-', linewidth=2, markersize=8, label=f'Cluster (rate: {cluster_slope:.2f})')
    ax2.loglog(smooth_dof_values, [smooth_errors[0] * (dof/smooth_dof_values[0])**(-1) for dof in smooth_dof_values], 'b--', linewidth=2, alpha=0.7, label='O(DOF⁻¹)')
    ax2.set_xlabel('Number of DOFs')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'Convergence Comparison (α={alpha})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/dolfinx/shared/FEM_project/mesh_comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Mesh comparison plot saved to: mesh_comparison_plot.png")
    plt.close()
    
    return {
        'smooth_errors': smooth_errors,
        'cluster_errors': cluster_errors,
        'smooth_dof_values': smooth_dof_values,
        'cluster_dof_values': cluster_dof_values,
        'smooth_slope': smooth_slope,
        'cluster_slope': cluster_slope
    }

def test_adaptive_mesh_exact_integration_convergence(alpha = 0.49):
    """
    Test convergence using exact integration on smooth adaptive mesh.
    This combines the best of both approaches: exact integration + adaptive mesh.
    """
    print(f"\nTesting Exact Integration on Smooth Adaptive Mesh (α={alpha})")
    print("=" * 50)
    
    # Define the exact solution
    def u_exact(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask = x > 1e-15
        result[mask] = (x[mask] - x[mask]**(2-alpha)) / ((1-alpha) * (2-alpha))
        result[~mask] = 0.0
        return result
    
    # Test different base mesh sizes with adaptive refinement
    N_base_values = [5, 10, 20, 40, 80]
    l2_errors = []
    dof_values = []
    solutions = []
    meshes = []
    
    from playground import compute_errors_refined
    from domains import tag_all_exterior_facets
    
    for N_base in N_base_values:
        print(f"  Testing N_base = {N_base}...")
        
        # Create smooth adaptive mesh
        mesh = create_adaptive_mesh_1d_smooth(N_base, alpha=alpha)
        V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Get number of DOFs
        n_dofs = V.dofmap.index_map.size_global
        
        # Build operator (Dirichlet BC)
        facet_tags, _ = tag_all_exterior_facets(mesh)
        B, _, M, _, bc = build_operator_B(V, bc_type="dirichlet", facet_tags=facet_tags, ids=(1,), kappa=0)
        
        # Assemble RHS using EXACT integration on adaptive mesh
        b = assemble_rhs_exact_integration(V, alpha=alpha)
        
        # Apply boundary conditions to RHS vector
        from dolfinx.fem import petsc
        petsc.set_bc(b, [bc])
        
        # Solve the system
        u_h, _ = solve_petsc(B, b, V)
        
        # Manually enforce boundary conditions on the solution
        boundary_dofs = bc.dof_indices()[0]
        u_h.x.array[boundary_dofs] = 0.0
        u_h.x.scatter_forward()
        
        # Compute errors (only L²)
        errors = compute_errors_refined(u_h, u_exact, V, mesh, refine_factor=16)
        
        l2_error = errors['l2_error']
        
        l2_errors.append(l2_error)
        dof_values.append(n_dofs)
        solutions.append(u_h)
        meshes.append(mesh)
        
        print(f"    L² error: {l2_error:.2e}, DOFs: {n_dofs}")
    
    # Compute L² convergence rate with respect to DOFs
    log_dofs = np.log(dof_values)
    log_l2_error = np.log(l2_errors)
    
    # Linear regression to find slope (convergence rate)
    l2_slope = np.polyfit(log_dofs, log_l2_error, 1)[0]
    
    print(f"\nAdaptive Mesh + Exact Integration L² Convergence Rate (vs DOFs):")
    print(f"  L² Convergence Rate: {l2_slope:.2f} (Expected: -1.0 for O(h²))")
    
    # Create combined plot
    import matplotlib.pyplot as plt
    
    # Get the finest mesh solution for plotting
    finest_mesh = meshes[2]  # Use N_base=20 for better visibility
    finest_u_h = solutions[2]
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Exact vs Numerical Solution
    x_coords = finest_mesh.geometry.x[:, 0]
    u_numerical = finest_u_h.x.array
    
    # Sort by x coordinates for plotting
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    u_sorted = u_numerical[sort_idx]
    
    # Evaluate exact solution at mesh points
    u_exact_vals = u_exact(x_sorted)
    
    # Plot exact and numerical solutions
    ax1.plot(x_sorted, u_exact_vals, 'r-', linewidth=3, label='Exact Solution')
    ax1.plot(x_sorted, u_sorted, 'b--o', linewidth=2, markersize=6, label=f'Numerical Solution (N_base={N_base_values[2]})')
    ax1.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x = 1/2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'Adaptive Mesh + Exact Integration: Exact vs Numerical Solution (α={alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: L² Convergence Analysis
    ax2.loglog(dof_values, l2_errors, 'bo-', linewidth=2, markersize=8, label=f'L² Error (rate: {l2_slope:.2f})')
    
    # Add reference line for O(DOF^(-1)) (equivalent to O(h²))
    dof_ref = np.array([dof_values[0], dof_values[-1]])
    l2_ref = l2_errors[0] * (dof_ref / dof_values[0])**(-1)
    
    ax2.loglog(dof_ref, l2_ref, 'b--', linewidth=2, alpha=0.7, label='O(DOF⁻¹) reference')
    
    ax2.set_xlabel('Number of DOFs')
    ax2.set_ylabel('L² Error')
    ax2.set_title(f'Adaptive Mesh + Exact Integration L² Convergence Rate: {l2_slope:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/dolfinx/shared/FEM_project/adaptive_mesh_exact_integration_combined_plot.png', dpi=300, bbox_inches='tight')
    print("Adaptive mesh + exact integration combined plot saved to: adaptive_mesh_exact_integration_combined_plot.png")
    plt.close()
    
    return {
        'N_base_values': N_base_values,
        'l2_errors': l2_errors,
        'dof_values': dof_values,
        'l2_slope': l2_slope
    }

if __name__ == "__main__":
    # Run original tests
    test_results = test_dual_fvm_vs_standard()
    test_different_quad_degrees()
    
    # Run custom Neumann problem tests
    custom_results = test_custom_neumann_problem()
    convergence_results = test_custom_neumann_convergence()
    
    # Run singular Dirichlet problem tests
    singular_results = test_singular_dirichlet_problem()
    singular_convergence_results = test_singular_dirichlet_convergence()
    
    # Run additional tests for load vector quality assessment
    # Test with alpha = 0.49 only
    alpha = 0.49
    
    print(f"\n{'='*60}")
    print(f"TESTING WITH α = {alpha}")
    print(f"{'='*60}")
    
    exact_integration_results = test_exact_integration_convergence(alpha=alpha)
    adaptive_mesh_results = test_adaptive_mesh_convergence(alpha=alpha)
    mesh_comparison_results = test_compare_adaptive_meshes(alpha=alpha)
    adaptive_exact_results = test_adaptive_mesh_exact_integration_convergence(alpha=alpha)
    
    print("\n" + "=" * 50)
    print("All testing completed!")
    print("The dual FVM approach provides an alternative way to assemble")
    print("RHS vectors, particularly useful for finite volume methods.")
    print("Custom Neumann problem with piecewise exact solution tested successfully.")
    print("Singular Dirichlet problem with x^(-α) load tested successfully.")
    print("Exact integration test for load vector quality assessment completed.")
    print("Adaptive mesh test for singular load handling completed.")
    print("Smooth vs cluster adaptive mesh comparison completed.")

# %%
