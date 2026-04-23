from mpi4py import MPI
import numpy as np
import ufl
import matplotlib.pyplot as plt
from solver_utils import solver_1d
from err_metrics import error_metrics

comm = MPI.COMM_WORLD
Ns = [10, 20, 40, 80, 160]
p = 1

# Define the source function for sin(pi*x)
def sin_source(x):
    return ufl.sin(ufl.pi * x[0])

# Define exact solution: u = sin(pi*x) / pi^2
def exact_solution(x):
    return ufl.sin(ufl.pi * x[0]) / (ufl.pi**2)

print("Running convergence test for -u'' = sin(pi*x) with u(0) = u(1) = 0")
print("N\t\th\t\tL2 Error\tH1 Error")
print("-" * 60)

# Store results for plotting
h_values = []
L2_errors = []
H1_errors = []

for N in Ns:
    # Solve the problem
    domain, V, uh = solver_1d(N=N, p=p, bdr_values=[0.0, 0.0], f_expr=sin_source)
    
    # Create exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_solution(x)
    
    # Compute error metrics
    h, eL2, eH1 = error_metrics(uh, u_exact, domain, V, p)
    print(f"{N}\t\t{h:.3e}\t\t{eL2:.3e}\t\t{eH1:.3e}")
    
    # Store for plotting
    h_values.append(h)
    L2_errors.append(eL2)
    H1_errors.append(eH1)

# Convert to numpy arrays
h_values = np.array(h_values)
L2_errors = np.array(L2_errors)
H1_errors = np.array(H1_errors)

# Create log-log plot
plt.figure(figsize=(10, 6))

# Plot L2 errors
plt.loglog(h_values, L2_errors, 'bo-', label='L2 Error', linewidth=2, markersize=8)

# Plot H1 errors
plt.loglog(h_values, H1_errors, 'ro-', label='H1 Error', linewidth=2, markersize=8)

# Add reference lines for convergence rates
# For linear elements: L2 should be O(h^2), H1 should be O(h^1)
h_ref = np.linspace(h_values.min(), h_values.max(), 100)
L2_ref = h_ref**2 * L2_errors[0] / h_values[0]**2
H1_ref = h_ref**1 * H1_errors[0] / h_values[0]**1

plt.loglog(h_ref, L2_ref, 'b--', alpha=0.7, label='O(h²) reference')
plt.loglog(h_ref, H1_ref, 'r--', alpha=0.7, label='O(h¹) reference')

plt.xlabel('Mesh size h', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('FEM Convergence: -u\'\' = sin(πx)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
print(f"\nConvergence plot saved as 'convergence_plot.png'")

# Compute convergence rates using different methods
if len(h_values) >= 2:
    print("\n" + "="*60)
    print("CONVERGENCE RATE ANALYSIS")
    print("="*60)
    
    # Method 1: Simple polyfit (current method)
    L2_rate_simple = np.polyfit(np.log(h_values), np.log(L2_errors), 1)[0]
    H1_rate_simple = np.polyfit(np.log(h_values), np.log(H1_errors), 1)[0]
    
    print(f"Method 1 (polyfit):")
    print(f"  L2 convergence rate: {L2_rate_simple:.3f}")
    print(f"  H1 convergence rate: {H1_rate_simple:.3f}")
    
    # Method 2: Full linear regression with statistics
    from scipy import stats
    
    # L2 error regression
    L2_slope, L2_intercept, L2_r_value, L2_p_value, L2_std_err = stats.linregress(np.log(h_values), np.log(L2_errors))
    
    # H1 error regression  
    H1_slope, H1_intercept, H1_r_value, H1_p_value, H1_std_err = stats.linregress(np.log(h_values), np.log(H1_errors))
    
    print(f"\nMethod 2 (full linear regression):")
    print(f"  L2 convergence rate: {L2_slope:.3f} ± {L2_std_err:.3f}")
    print(f"    R² = {L2_r_value**2:.4f}, p-value = {L2_p_value:.2e}")
    print(f"  H1 convergence rate: {H1_slope:.3f} ± {H1_std_err:.3f}")
    print(f"    R² = {H1_r_value**2:.4f}, p-value = {H1_p_value:.2e}")
    
    # Method 3: Manual calculation for verification
    log_h = np.log(h_values)
    log_L2 = np.log(L2_errors)
    log_H1 = np.log(H1_errors)
    
    n = len(h_values)
    L2_slope_manual = (n * np.sum(log_h * log_L2) - np.sum(log_h) * np.sum(log_L2)) / (n * np.sum(log_h**2) - np.sum(log_h)**2)
    H1_slope_manual = (n * np.sum(log_h * log_H1) - np.sum(log_h) * np.sum(log_H1)) / (n * np.sum(log_h**2) - np.sum(log_h)**2)
    
    print(f"\nMethod 3 (manual calculation):")
    print(f"  L2 convergence rate: {L2_slope_manual:.3f}")
    print(f"  H1 convergence rate: {H1_slope_manual:.3f}")
    
    # Create a detailed regression plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # L2 error regression plot
    ax1.loglog(h_values, L2_errors, 'bo-', label='L2 Error', linewidth=2, markersize=8)
    ax1.loglog(h_values, np.exp(L2_intercept) * h_values**L2_slope, 'b--', 
               label=f'Regression (slope={L2_slope:.2f})', alpha=0.7)
    ax1.set_xlabel('Mesh size h')
    ax1.set_ylabel('L2 Error')
    ax1.set_title(f'L2 Error Convergence\nSlope = {L2_slope:.2f} ± {L2_std_err:.2f}, R² = {L2_r_value**2:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # H1 error regression plot
    ax2.loglog(h_values, H1_errors, 'ro-', label='H1 Error', linewidth=2, markersize=8)
    ax2.loglog(h_values, np.exp(H1_intercept) * h_values**H1_slope, 'r--', 
               label=f'Regression (slope={H1_slope:.2f})', alpha=0.7)
    ax2.set_xlabel('Mesh size h')
    ax2.set_ylabel('H1 Error')
    ax2.set_title(f'H1 Error Convergence\nSlope = {H1_slope:.2f} ± {H1_std_err:.2f}, R² = {H1_r_value**2:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nDetailed regression analysis saved as 'regression_analysis.png'")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Expected rates: L2 = 2.0, H1 = 1.0")
    print(f"  Achieved rates: L2 = {L2_slope:.2f}, H1 = {H1_slope:.2f}")
    print(f"  Quality: R² = {L2_r_value**2:.3f} (L2), R² = {H1_r_value**2:.3f} (H1)")

plt.show()