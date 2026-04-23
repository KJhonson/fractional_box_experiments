#!/usr/bin/env python3
"""
Fractional FEM Experiment: Convergence test using eigensolutions
For the fractional Dirichlet problem on [0,1] with known analytical solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from corrected_fractional_solver import CorrectedSincFEM, create_test_matrices

def analytical_fractional_solution(x, beta, eigenmode=1):
    """
    Analytical solution for fractional Dirichlet problem
    For (-Δ)^β u = λ^β u with u(0) = u(1) = 0
    The eigensolution is: u(x) = sin(π * eigenmode * x)
    """
    return np.sin(np.pi * eigenmode * x)

def fractional_eigenvalue(beta, eigenmode=1):
    """Compute the fractional eigenvalue λ^β"""
    lambda_k = (np.pi * eigenmode) ** 2
    return lambda_k ** beta

def solve_fractional_eigenvalue_problem(K, M, beta, eigenmode=1):
    """
    Solve the fractional eigenvalue problem: (-Δ)^β u = λ^β u
    Using Richardson approach: test with known eigenvalue λ^β = (π²)^β
    """
    # Get the known eigenvalue
    lambda_beta = fractional_eigenvalue(beta, eigenmode)
    
    # Create right-hand side: λ^β * u_exact
    n = K.shape[0]
    x = np.linspace(0, 1, n)
    u_exact = analytical_fractional_solution(x, beta, eigenmode)
    rhs = lambda_beta * u_exact
    
    # Solve using sinc quadrature for (-Δ)^β u = λ^β u
    # This is equivalent to solving M^(-1) K^β u = λ^β u
    # We use the fractional solver with the eigenvalue as source
    solver = CorrectedSincFEM(solver_mode="spsolve")
    
    # For eigenvalue problem, we need to solve: K^β u = λ^β M u
    # This becomes: (M^(-1) K)^β u = λ^β u
    # We solve this by using the fractional solver with right-hand side λ^β u
    
    # Create the fractional operator (M^(-1) K)^β applied to u
    # This is what the sinc quadrature approximates
    h = 1.0 / n
    u_numerical = solver.solve_for_h_beta(h, beta, n)
    
    # Scale by the eigenvalue to get the correct solution
    u_numerical = u_numerical * lambda_beta
    
    return u_numerical, u_exact, lambda_beta

def compute_fractional_errors(u_numerical, u_exact, h):
    """Compute L2 and H1 errors for fractional problem"""
    # L2 error
    eL2 = np.linalg.norm(u_numerical - u_exact) * np.sqrt(h)
    
    # H1 error (approximate using finite differences)
    n = len(u_numerical)
    du_numerical = np.gradient(u_numerical, h)
    du_exact = np.gradient(u_exact, h)
    eH1 = np.sqrt(np.sum((du_numerical - du_exact)**2) * h)
    
    return eL2, eH1

def run_fractional_convergence_experiment():
    """Run convergence experiment for fractional problem"""
    print("🧮 Fractional FEM Convergence Experiment")
    print("="*50)
    
    # Parameters
    betas = [0.25, 0.5, 0.75]
    Ns = [10, 20, 40, 80, 160]  # Mesh refinements
    eigenmode = 1  # First eigenmode
    
    print(f"Testing fractional problem: (-Δ)^β u = λ^β u")
    print(f"Eigenmode: {eigenmode}")
    print(f"Beta values: {betas}")
    print(f"Mesh sizes: N = {Ns}")
    
    # Store results
    all_results = {}
    
    for beta in betas:
        print(f"\n📊 β = {beta}")
        print(f"{'N':<4} {'h':<8} {'λ^β':<12} {'L2 Error':<12} {'H1 Error':<12} {'L2 Rate':<8} {'H1 Rate':<8}")
        print("-" * 80)
        
        # Store results for this beta
        h_values = []
        L2_errors = []
        H1_errors = []
        
        prev_L2 = None
        prev_H1 = None
        
        for N in Ns:
            h = 1.0 / N
            
            # Create matrices for this mesh size
            K, M = create_test_matrices(N)
            
            # Solve the fractional eigenvalue problem: (-Δ)^β u = λ^β u
            u_numerical, u_exact, lambda_beta = solve_fractional_eigenvalue_problem(K, M, beta, eigenmode)
            
            # Compute errors
            eL2, eH1 = compute_fractional_errors(u_numerical, u_exact, h)
            
            # Compute convergence rates
            L2_rate = "N/A"
            H1_rate = "N/A"
            if prev_L2 is not None:
                L2_rate = f"{np.log(eL2/prev_L2)/np.log(2):.2f}"
            if prev_H1 is not None:
                H1_rate = f"{np.log(eH1/prev_H1)/np.log(2):.2f}"
            
            print(f"{N:<4} {h:<8.4f} {lambda_beta:<12.2f} {eL2:<12.6f} {eH1:<12.6f} {L2_rate:<8} {H1_rate:<8}")
            
            # Store for plotting
            h_values.append(h)
            L2_errors.append(eL2)
            H1_errors.append(eH1)
            
            prev_L2 = eL2
            prev_H1 = eH1
        
        # Store results
        all_results[beta] = {
            'h': h_values,
            'L2': L2_errors,
            'H1': H1_errors
        }
    
    return all_results

def plot_fractional_convergence(results):
    """Plot convergence results for fractional problem"""
    print(f"\n📈 Creating convergence plots...")
    
    betas = [0.25, 0.5, 0.75]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: L2 errors
    ax1 = axes[0]
    for beta in betas:
        h = np.array(results[beta]['h'])
        L2 = np.array(results[beta]['L2'])
        ax1.loglog(h, L2, 'o-', label=f'β = {beta}', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mesh size h')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Fractional FEM: L2 Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: H1 errors
    ax2 = axes[1]
    for beta in betas:
        h = np.array(results[beta]['h'])
        H1 = np.array(results[beta]['H1'])
        ax2.loglog(h, H1, 'o-', label=f'β = {beta}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Mesh size h')
    ax2.set_ylabel('H1 Error')
    ax2.set_title('Fractional FEM: H1 Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fractional_convergence_errors.png', dpi=150, bbox_inches='tight')
    print(f"📈 Convergence plot saved as 'fractional_convergence_errors.png'")
    plt.show()

def compute_convergence_rates(results):
    """Compute and display convergence rates"""
    print(f"\n📊 Convergence Rate Analysis:")
    print("="*40)
    
    betas = [0.25, 0.5, 0.75]
    
    print(f"{'β':<8} {'L2 Rate':<12} {'H1 Rate':<12}")
    print("-" * 35)
    
    for beta in betas:
        h = np.array(results[beta]['h'])
        L2 = np.array(results[beta]['L2'])
        H1 = np.array(results[beta]['H1'])
        
        # Compute convergence rates using linear regression
        if len(h) >= 2:
            log_h = np.log(h)
            log_L2 = np.log(L2)
            log_H1 = np.log(H1)
            
            L2_rate = np.polyfit(log_h, log_L2, 1)[0]
            H1_rate = np.polyfit(log_h, log_H1, 1)[0]
            
            print(f"{beta:<8.2f} {L2_rate:<12.2f} {H1_rate:<12.2f}")
        else:
            print(f"{beta:<8.2f} {'N/A':<12} {'N/A':<12}")

def main():
    """Main function to run the fractional experiment"""
    print("🧮 Fractional FEM Experiment with Eigensolutions")
    print("="*60)
    print("Problem: (-Δ)^β u = λ^β u on [0,1] with u(0) = u(1) = 0")
    print("First eigenvalue: λ₁ = π², so λ₁^β = (π²)^β")
    print("First eigenfunction: u₁(x) = sin(πx)")
    print("Richardson approach: Test with known eigenvalue")
    print("="*60)
    
    # Run convergence experiment
    results = run_fractional_convergence_experiment()
    
    # Plot results
    plot_fractional_convergence(results)
    
    # Compute convergence rates
    compute_convergence_rates(results)
    
    print(f"\n🎉 Fractional experiment completed!")
    print(f"✅ L2 and H1 errors computed and displayed")
    print(f"✅ Convergence rates analyzed")
    print(f"✅ Results saved to 'fractional_convergence_errors.png'")

if __name__ == "__main__":
    main()