#%%

import numpy as np
import matplotlib.pyplot as plt
import time
from mpmath import gammainc, power, pi, im
from numpy.polynomial.legendre import leggauss
from scipy.fftpack import dst
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.sinc_solver import sinc_solver  # Just to ensure path is correct

# ---------- Individual methods ----------

def I_n_gammainc(alpha, n):
    z = -1j * n * pi
    return im(power(-1j * n * pi, alpha - 1) * gammainc(1 - alpha, 0, z))

def reference_solution_gammainc(alpha, beta, N):
    n_vals = np.arange(1, N + 1)
    I_n_vec = np.array([float(I_n_gammainc(alpha, n)) for n in n_vals])  # Convert to float
    f_n = np.sqrt(2) * I_n_vec
    u_n = (n_vals * np.pi) ** (-2 * beta) * f_n
    x = np.linspace(0, 1, 300)[1:-1]
    u = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, n_vals)) @ u_n
    return x, u

def reference_solution_GLQ(alpha, beta, N, Q):
    t, w = leggauss(Q)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    power = 1.0 / (1.0 - alpha)
    x_quad = t ** power
    C_alpha = 1.0 / (1.0 - alpha)
    n_vals = np.arange(1, N + 1)
    S = np.sin(np.pi * np.outer(n_vals, x_quad))
    I_n_vec = C_alpha * (S @ w)
    f_n = np.sqrt(2.0) * I_n_vec
    u_n = (n_vals * np.pi) ** (-2.0 * beta) * f_n
    x = np.linspace(0, 1, 300)[1:-1]
    u = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, n_vals)) @ u_n
    return x, u

def reference_solution_DST(alpha, beta, M=8000, N=None, x_eval=None):
    """
    Uses DST-I to approximate the coefficients f_n of f(x)=x^{-alpha}
    and then builds u(x) = sum u_n * sqrt(2) * sin(n*pi*x).

    M: number of interior mesh points (x_j = j/(M+1)).
    N: number of modes used in the solution (<= M). If None, uses N = M.
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 300)[1:-1]
    x_eval = np.asarray(x_eval)

    if N is None or N > M:
        N = M

    # mesh points
    j = np.arange(1, M + 1)
    x = j / (M + 1)
    dx = 1.0 / (M + 1)

    # f(x) = x^{-alpha}
    f_vals = x**(-alpha)

    # unnormalized DST-I
    y = dst(f_vals, type=1)   # len M

    # continuous coefficients f_n ≈ sqrt(2)/(2(M+1)) * y_{n-1}
    n_vals = np.arange(1, N + 1)
    f_n = np.sqrt(2.0) * dx * 0.5 * y[:N]   # = sqrt(2)/(2(M+1)) * y

    # u coefficients
    u_n = (n_vals * np.pi)**(-2.0 * beta) * f_n

    # evaluate u(x) at x_eval
    Sx = np.sin(np.pi * np.outer(x_eval, n_vals))
    u_vals = np.sqrt(2.0) * (Sx @ u_n)

    return x_eval, u_vals

# Reference method used in f_ex_xalpha.py
def reference_solution_f_ex_xalpha(alpha, beta, N=400, Q=60, x_eval=None):
    """
    High-accuracy spectral solution para (-d^2/dx^2)^beta u = x^{-alpha}, u(0)=u(1)=0.
    Sine basis with Dirichlet boundary conditions.
    This is the method used in f_ex_xalpha.py.
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 200)[1:-1]  # avoid endpoints 0 and 1, where u=0
    
    # --- Gauss-Legendre quadrature on [0,1]
    t, w = np.polynomial.legendre.leggauss(Q)
    t = 0.5 * (t + 1.0)   # maps from [-1,1] -> [0,1]
    w = 0.5 * w
    
    # --- Change of variables that removes the singularity
    power = 1.0 / (1.0 - alpha)
    x_quad = t ** power
    C_alpha = 1.0 / (1.0 - alpha)
    
    # --- Compute u_n coefficients
    u_n = np.zeros(N)
    for n in range(1, N + 1):
        sin_vals = np.sin(n * np.pi * x_quad)
        I_n = C_alpha * np.sum(w * sin_vals)
        f_n = np.sqrt(2.0) * I_n
        u_n[n - 1] = (n * np.pi) ** (-2.0 * beta) * f_n
    
    # --- Evaluate the solution at x_eval
    u_vals = np.zeros_like(x_eval)
    for n in range(1, N + 1):
        u_vals += np.sqrt(2.0) * u_n[n - 1] * np.sin(n * np.pi * x_eval)
    
    return x_eval, u_vals

# ---------- Benchmark ----------
# ============================================================================
# HYPERPARAMETERS - Adjust these values to change experiment settings
# ============================================================================

alpha = 0.499
beta = 0.3

# Parameters for the reference solution from f_ex_xalpha.py
N_ref = 1000  # Number of Fourier modes for the reference
Q_ref = 2000  # Quadrature points for the reference

# Parameters for visual comparison
x_plot = np.linspace(0.01, 0.99, 500)  # Plot points, avoiding endpoints

# Parameters for the Gammainc test; increase N for better quality
N_gammainc_list = [50, 100, 200, 400, 800, 1600]  # Number of Fourier modes for Gammainc

# Parameters for the GLQ test; increase N and Q for better quality
N_glq_list = [50, 100, 200, 400, 800, 1600]  # Number of Fourier modes for GLQ
Q_glq_list = [100, 200, 400, 800, 1600]  # Quadrature points for GLQ

# Parameters for the DST test; increase M for better quality
M_dst_list = [500, 1000, 2000, 4000, 8000]  # Number of interior mesh points for DST
N_dst_list = [None, 200, 400, 800]  # Number of modes for DST (None = uses M)

# Reference solution using the method from f_ex_xalpha.py
print("="*70)
print("Computing reference solution with f_ex_xalpha.py method...")
print(f"Parameters: α={alpha}, β={beta}, N={N_ref}, Q={Q_ref}")
t0_ref = time.time()
x_ref, u_ref = reference_solution_f_ex_xalpha(alpha, beta, N=N_ref, Q=Q_ref, x_eval=x_plot)
time_ref = time.time() - t0_ref
print(f"Reference time: {time_ref:.4f}s")
print("="*70)
print()

# Function to measure relative L2 error
# Compare solutions by interpolating both to the same points
def rel_L2(x_ref, u_ref, x, u):
    # x_ref and u_ref are already on the correct points
    # x and u are the points and values of the tested method
    # Interpolate u to the x_ref points
    u_interp = np.interp(x_ref, x, u)
    # Compute relative L2 error
    return np.sqrt(np.mean((u_interp - u_ref)**2)) / np.sqrt(np.mean(u_ref**2))

results = []

print("="*70)
print("BENCHMARK: Comparison of Fourier solution methods")
print(f"Parameters: α={alpha}, β={beta}")
print("="*70)
print()

# DST test
print("Testing DST method...")
for M in M_dst_list:
    for N_dst in N_dst_list:
        if N_dst is not None and N_dst > M:
            continue
        t0 = time.time()
        x, u = reference_solution_DST(alpha, beta, M=M, N=N_dst, x_eval=x_plot)
        elapsed = time.time() - t0
        # Interpolate to the same reference points for comparison
        u_interp = np.interp(x_plot, x, u)
        err = rel_L2(x_ref, u_ref, x_plot, u_interp)
        params_str = f"M={M}" if N_dst is None else f"M={M},N={N_dst}"
        results.append(("DST", params_str, elapsed, err, x, u))
        print(f"  {params_str:20s}: time={elapsed:.4f}s, error={err:.2e}")

print()

# Gammainc test
print("Testing Gammainc method...")
for N in N_gammainc_list:
    t0 = time.time()
    x, u = reference_solution_gammainc(alpha, beta, N)
    elapsed = time.time() - t0
    # Interpolate to the same reference points for comparison
    u_interp = np.interp(x_plot, x, u)
    err = rel_L2(x_ref, u_ref, x_plot, u_interp)
    results.append(("Gammainc", N, elapsed, err, x, u))
    print(f"  N={N:5d}: time={elapsed:.4f}s, error={err:.2e}")

print()

# GLQ test
print("Testing GLQ method...")
for N in N_glq_list:
    for Q in Q_glq_list:
        t0 = time.time()
        x, u = reference_solution_GLQ(alpha, beta, N, Q)
        elapsed = time.time() - t0
        # Interpolate to the same reference points for comparison
        u_interp = np.interp(x_plot, x, u)
        err = rel_L2(x_ref, u_ref, x_plot, u_interp)
        results.append(("GLQ", f"N={N},Q={Q}", elapsed, err, x, u))
        print(f"  N={N:3d}, Q={Q:3d}: time={elapsed:.4f}s, error={err:.2e}")

print()
print("="*70)
print("RESULT ANALYSIS")
print("="*70)

# Analysis by method
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    if not data:
        continue
    
    times = [r[2] for r in data]
    errs = [r[3] for r in data]
    
    # Find the best time-error tradeoff: lowest time for error < 1e-2
    best_idx = None
    best_time = float('inf')
    for i, (t, e) in enumerate(zip(times, errs)):
        if e < 1e-2 and t < best_time:
            best_time = t
            best_idx = i
    
    print(f"\n{method}:")
    print(f"  Number of tests: {len(data)}")
    print(f"  Minimum time: {min(times):.4f}s")
    print(f"  Maximum time: {max(times):.4f}s")
    print(f"  Mean time: {np.mean(times):.4f}s")
    print(f"  Minimum error: {min(errs):.2e}")
    print(f"  Maximum error: {max(errs):.2e}")
    if best_idx is not None:
        print(f"  Best (error < 1e-2, minimum time): {data[best_idx][1]}, time={best_time:.4f}s, error={errs[best_idx]:.2e}")
    else:
        print(f"  No test reached error < 1e-2")
    
    # Efficiency: error/time (lower is better)
    efficiencies = [e/t for e, t in zip(errs, times)]
    print(f"  Mean efficiency (error/time): {np.mean(efficiencies):.2e}")
    
    # Best absolute result (smallest error)
    best_err_idx = np.argmin(errs)
    print(f"  Smallest absolute error: {data[best_err_idx][1]}, time={times[best_err_idx]:.4f}s, error={errs[best_err_idx]:.2e}")

print()
print("="*70)
print("SPEED RANKING (for error < 1e-2)")
print("="*70)

# Ranking
ranking = []
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    for r in data:
        if r[3] < 1e-2:  # error < 1e-2
            ranking.append((r[2], method, r[1], r[3]))  # (time, method, parameters, error)

ranking.sort()  # Sort by time

if ranking:
    print("\nTop 10 fastest methods (error < 1e-2):")
    for i, (t, method, params, err) in enumerate(ranking[:10], 1):  # Top 10
        print(f"{i:2d}. {method:10s} | Parameters: {str(params):15s} | Time: {t:.4f}s | Error: {err:.2e}")
else:
    print("No method reached error < 1e-2")

print()
print("="*70)
print("ACCURACY RANKING (smallest error)")
print("="*70)

# Ranking by accuracy
ranking_precision = [(r[3], r[2], r[0], r[1]) for r in results]  # (error, time, method, parameters)
ranking_precision.sort()  # Sort by error

print("\nTop 10 most accurate methods:")
for i, (err, t, method, params) in enumerate(ranking_precision[:10], 1):
    print(f"{i:2d}. {method:10s} | Parameters: {str(params):15s} | Error: {err:.2e} | Time: {t:.4f}s")

print()
print("="*70)
print("CONCLUSIONS")
print("="*70)

# Final analysis
dst_data = [r for r in results if r[0]=="DST"]
gammainc_data = [r for r in results if r[0]=="Gammainc"]
glq_data = [r for r in results if r[0]=="GLQ"]

if dst_data:
    dst_min_err = min([r[3] for r in dst_data])
    dst_best = min(dst_data, key=lambda x: x[3])
    print(f"\nDST: Minimum error = {dst_min_err:.2e}")
    print(f"  ✓ Best configuration: {dst_best[1]}, time={dst_best[2]:.4f}s")
    if dst_min_err < 1e-2:
        print("  ✓ Recommended method for speed with good accuracy.")

if gammainc_data:
    gammainc_min_err = min([r[3] for r in gammainc_data])
    gammainc_best = min(gammainc_data, key=lambda x: x[3])
    print(f"\nGammainc: Minimum error = {gammainc_min_err:.2e}")
    print(f"  ✓ Best configuration: N={gammainc_best[1]}, time={gammainc_best[2]:.4f}s")
    if gammainc_min_err < 1e-2:
        print("  ✓ Recommended method for high accuracy.")

if glq_data:
    glq_min_err = min([r[3] for r in glq_data])
    glq_best = min(glq_data, key=lambda x: x[3])
    print(f"\nGLQ: Minimum error = {glq_min_err:.2e}")
    print(f"  ✓ Best configuration: {glq_best[1]}, time={glq_best[2]:.4f}s")
    if glq_min_err < 1e-2:
        print("  ✓ Recommended method for good accuracy with parameter flexibility.")

# Final recommendation
print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)

best_overall = min(results, key=lambda x: x[3])  # Smallest error
fastest_good = None
for r in sorted(results, key=lambda x: x[2]):  # Sort by time
    if r[3] < 1e-2:
        fastest_good = r
        break

if fastest_good:
    print(f"\nFor speed (error < 1e-2):")
    print(f"  Method: {fastest_good[0]}")
    print(f"  Parameters: {fastest_good[1]}")
    print(f"  Time: {fastest_good[2]:.4f}s")
    print(f"  Error: {fastest_good[3]:.2e}")

print(f"\nFor accuracy:")
print(f"  Method: {best_overall[0]}")
print(f"  Parameters: {best_overall[1]}")
print(f"  Time: {best_overall[2]:.4f}s")
print(f"  Error: {best_overall[3]:.2e}")

print()
print("="*70)

# ---------- Plot 1: Performance (time vs error) ----------
fig, ax = plt.subplots(figsize=(7,5))
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    times = [r[2] for r in data]
    errs = [r[3] for r in data]
    ax.loglog(times, errs, 'o-', label=method)

ax.loglog([time_ref], [1e-10], 'r*', markersize=15, label='Reference (f_ex_xalpha.py)')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Error L2 relativo")
ax.set_title(f"Performance comparison — α={alpha}, β={beta}")
ax.legend()
ax.grid(True, which="both")
plt.tight_layout()
plt.show()

# ---------- Plot 2: Visual comparison of solutions ----------
print()
print("="*70)
print("VISUAL COMPARISON OF SOLUTIONS")
print("="*70)
print("Plotting the solutions to check whether they approximate the same function...")
print()

# Select the best results from each method for visual comparison
best_dst = min([r for r in results if r[0]=="DST"], key=lambda x: x[3])
best_gammainc = min([r for r in results if r[0]=="Gammainc"], key=lambda x: x[3])
best_glq = min([r for r in results if r[0]=="GLQ"], key=lambda x: x[3])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Full plot
ax1.plot(x_ref, u_ref, 'k-', linewidth=3, label=f'Reference (f_ex_xalpha.py, N={N_ref}, Q={Q_ref}, t={time_ref:.4f}s)', alpha=0.9)
ax1.plot(best_dst[4], best_dst[5], 'b--', linewidth=2, label=f'DST ({best_dst[1]}, error={best_dst[3]:.2e}, t={best_dst[2]:.4f}s)', alpha=0.7)
ax1.plot(best_gammainc[4], best_gammainc[5], 'r:', linewidth=2, label=f'Gammainc (N={best_gammainc[1]}, error={best_gammainc[3]:.2e}, t={best_gammainc[2]:.4f}s)', alpha=0.7)
ax1.plot(best_glq[4], best_glq[5], 'g-.', linewidth=2, label=f'GLQ ({best_glq[1]}, error={best_glq[3]:.2e}, t={best_glq[2]:.4f}s)', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('u(x)', fontsize=12)
ax1.set_title(f'Visual comparison: all solutions - alpha={alpha}, beta={beta}', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Error plot (difference from the reference)
ax2.plot(x_ref, u_ref - u_ref, 'k-', linewidth=3, label='Reference (zero)', alpha=0.9)
# Interpolate to the same points
u_dst_interp = np.interp(x_ref, best_dst[4], best_dst[5])
u_gammainc_interp = np.interp(x_ref, best_gammainc[4], best_gammainc[5])
u_glq_interp = np.interp(x_ref, best_glq[4], best_glq[5])
ax2.plot(x_ref, u_ref - u_dst_interp, 'b--', linewidth=2, label=f'Error DST (max={np.max(np.abs(u_ref - u_dst_interp)):.2e})', alpha=0.7)
ax2.plot(x_ref, u_ref - u_gammainc_interp, 'r:', linewidth=2, label=f'Error Gammainc (max={np.max(np.abs(u_ref - u_gammainc_interp)):.2e})', alpha=0.7)
ax2.plot(x_ref, u_ref - u_glq_interp, 'g-.', linewidth=2, label=f'Error GLQ (max={np.max(np.abs(u_ref - u_glq_interp)):.2e})', alpha=0.7)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Error: u_ref - u_method', fontsize=12)
ax2.set_title('Difference from the reference solution', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('symlog', linthresh=1e-6)  # Symmetric log scale for positive and negative errors

plt.tight_layout()
plt.show()

print("="*70)
print("TIME SUMMARY")
print("="*70)
print(f"Reference (f_ex_xalpha.py): {time_ref:.4f}s")
print(f"DST best: {best_dst[2]:.4f}s ({best_dst[1]}, error={best_dst[3]:.2e})")
print(f"Gammainc best: {best_gammainc[2]:.4f}s (N={best_gammainc[1]}, error={best_gammainc[3]:.2e})")
print(f"GLQ best: {best_glq[2]:.4f}s ({best_glq[1]}, error={best_glq[3]:.2e})")
print("="*70)

# %%
