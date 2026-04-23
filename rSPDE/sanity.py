# %%

import numpy as np
import matplotlib.pyplot as plt
from baryrat import brasil

# Define target function: fractional power
beta = 2.2
f = lambda x: x ** beta

# Approximation interval
interval = (0.1, 1)

# Degree of rational approximation
m = 2

# Compute rational approximation using BRASIL
r = brasil(f, interval, m)

# Evaluate true function and approximation on a log scale
x_vals = np.logspace(np.log10(interval[0]), np.log10(interval[1]), 500)
f_vals = f(x_vals)
r_vals = r(x_vals)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_vals, f_vals, label=f"$x^{{{beta}}}$", linewidth=2)
plt.plot(x_vals, r_vals, '--', label=f"BRASIL approximation (m={m})", linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("x")
plt.ylabel("y")
plt.title("BRASIL Rational Approximation vs $x^{%.2f}$" % beta)
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()

# Print some numerical results
print(f"BRASIL approximation for x^{beta}:")
print(f"Poles: {r.poles()}")
print(f"Zeros: {r.zeros()}")
print(f"Gain: {r.gain():.6f}")

# Check at a few points
test_points = [1.5, 3.0, 5.0, 8.0, 10.0]
print(f"\nApproximation quality:")
print("x    | True    | BRASIL  | Error")
print("-" * 35)
for x in test_points:
    true_val = f(x)
    approx_val = r(x)
    error = abs(true_val - approx_val) / true_val
    print(f"{x:5.1f}  | {true_val:6.3f}  | {approx_val:6.3f}  | {error:.2e}")


# Now construct the polynomials Pl and Pr and plot the resulting r(x)
print(f"\n=== Constructing Polynomials Pl and Pr ===")

# CORRECTED CONSTRUCTION: Use the direct BRASIL approximation
# For direct approximation r(x) ≈ x^β:
# r(x) = gain * Π(1 - zero/x) / Π(1 - pole/x)

poles = r.poles()
zeros = r.zeros()
gain = r.gain()

print(f"Using direct BRASIL approximation:")
print(f"Poles: {poles}")
print(f"Zeros: {zeros}")
print(f"Gain: {gain:.6f}")

# Construct Pr (numerator polynomial): gain * Π(1 - zero/x)
Pr_vals = gain * np.ones_like(x_vals, dtype=complex)
for zero in zeros:
    Pr_vals *= (1 - zero / x_vals)

# Construct Pl (denominator polynomial): Π(1 - pole/x)
Pl_vals = np.ones_like(x_vals, dtype=complex)
for pole in poles:
    Pl_vals *= (1 - pole / x_vals)

# The rational function r(x) = Pr(x) / Pl(x)
r_constructed = (Pr_vals / Pl_vals).real

# ============================================================================
# THIRD APPROACH: Fractional part approximation
# ============================================================================
print(f"\n=== THIRD APPROACH: Fractional Part Approximation ===")

# Decompose beta for fractional part approach
beta_floor = int(np.floor(beta))
beta_frac = beta - beta_floor
print(f"Decomposition: β = {beta_floor} + {beta_frac}")
print(f"x^{beta} = x^{beta_floor} * x^{beta_frac}")

# Approximate only the fractional part: r_frac(x) ≈ x^{β_frac}
if abs(beta_frac) < 1e-14:
    print("Fractional part is zero; skipping rational approximation for x^{β_frac}.")
    r_frac_vals = np.ones_like(x_vals)
    r_fractional_vals = (x_vals ** beta_floor)
else:
    r_frac = brasil(lambda x: x ** beta_frac, interval, m)
    poles_frac = r_frac.poles()
    zeros_frac = r_frac.zeros()
    gain_frac = r_frac.gain()

    print(f"Fractional part approximation (x^{beta_frac}):")
    print(f"Poles: {poles_frac}")
    print(f"Zeros: {zeros_frac}")
    print(f"Gain: {gain_frac:.6f}")

    # Construct r_frac(x) from poles/zeros
    Pr_frac_vals = gain_frac * np.ones_like(x_vals)
    for zero in zeros_frac:
        Pr_frac_vals *= (1 - zero / x_vals)

    Pl_frac_vals = np.ones_like(x_vals)
    for pole in poles_frac:
        Pl_frac_vals *= (1 - pole / x_vals)

    r_frac_vals = Pr_frac_vals / Pl_frac_vals

    # Construct complete approximation: r_full(x) = x^[β] * r_frac(x)
    r_fractional_vals = (x_vals ** beta_floor) * r_frac_vals

# Plot da terceira ideia
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Comparação das três abordagens
axes[0,0].plot(x_vals, f_vals, 'b-', label=f'$x^{{{beta}}}$ (true)', linewidth=3)
axes[0,0].plot(x_vals, r_vals, 'r--', label='BRASIL direto', linewidth=2)
axes[0,0].plot(x_vals, r_constructed, 'g:', label='Construído direto', linewidth=2)
axes[0,0].plot(x_vals, r_fractional_vals, 'm-.', label='Parte fracionária', linewidth=2)
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('y')
axes[0,0].set_title('Comparação das Três Abordagens')
axes[0,0].legend()
axes[0,0].grid(True, linestyle='--')

# Plot 2: Análise de erro
error_brasil = np.abs(f_vals - r_vals) / f_vals
error_constructed = np.abs(f_vals - r_constructed) / f_vals
error_fractional = np.abs(f_vals - r_fractional_vals) / f_vals
axes[0,1].plot(x_vals, error_brasil, 'r--', label='Erro BRASIL direto', linewidth=2)
axes[0,1].plot(x_vals, error_constructed, 'g:', label='Erro construído direto', linewidth=2)
axes[0,1].plot(x_vals, error_fractional, 'm-.', label='Erro parte fracionária', linewidth=2)
axes[0,1].set_xlabel('x')
axes[0,1].set_ylabel('Erro Relativo')
axes[0,1].set_title('Análise de Erro')
axes[0,1].legend()
axes[0,1].grid(True, linestyle='--')

# Plot 3: Parte fracionária r_frac(x)
axes[1,0].plot(x_vals, x_vals ** beta_frac, 'b-', label=f'$x^{{{beta_frac}}}$ (true)', linewidth=2)
axes[1,0].plot(x_vals, r_frac_vals, 'r--', label='r_frac(x)', linewidth=2)
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('y')
axes[1,0].set_title(f'Aproximação da Parte Fracionária (x^{beta_frac})')
axes[1,0].legend()
axes[1,0].grid(True, linestyle='--')

# Plot 4: Verificação: x^[β] * r_frac(x) vs x^β
axes[1,1].plot(x_vals, f_vals, 'b-', label=f'$x^{{{beta}}}$ (true)', linewidth=2)
axes[1,1].plot(x_vals, r_fractional_vals, 'm--', label=f'$x^{{{beta_floor}}} \\cdot r_{{frac}}(x)$', linewidth=2)
axes[1,1].set_xlabel('x')
axes[1,1].set_ylabel('y')
axes[1,1].set_title('Verificação: Parte Inteira × Parte Fracionária')
axes[1,1].legend()
axes[1,1].grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# Comparação numérica das três abordagens
print(f"\nComparação das três abordagens:")
print("x    | True    | BRASIL  | Construído | Parte Frac. | Erro BRASIL | Erro Const. | Erro Frac.")
print("-" * 90)
for x in test_points:
    idx = np.argmin(np.abs(x_vals - x))
    true_val = f_vals[idx]
    brasil_val = r_vals[idx]
    constructed_val = r_constructed[idx]
    full_val = r_fractional_vals[idx]
    error_brasil = abs(true_val - brasil_val) / true_val
    error_constructed = abs(true_val - constructed_val) / true_val
    error_full = abs(true_val - full_val) / true_val
    print(f"{x:5.1f}  | {true_val:6.3f}  | {brasil_val:6.3f}  | {constructed_val:8.3f}  | {full_val:10.3f}  | {error_brasil:10.2e} | {error_constructed:10.2e} | {error_full:10.2e}")

# Verificar se a terceira ideia funciona
max_diff_full = np.max(np.abs(f_vals - r_fractional_vals))
print(f"\nErro máximo da abordagem parte fracionária: {max_diff_full:.2e}")

if max_diff_full < 1e-10:
    print("✅ SUCCESS: Abordagem parte fracionária funciona perfeitamente!")
else:
    print("⚠️  A abordagem parte fracionária tem erro, mas pode ser aceitável dependendo da aplicação")

# %%
