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

# ---------- Métodos individuais ----------

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
    Usa DST-I para aproximar os coeficientes f_n de f(x)=x^{-alpha}
    e depois constrói u(x) = sum u_n * sqrt(2) * sin(n*pi*x).

    M: número de pontos internos da malha (x_j = j/(M+1)).
    N: número de modos usados na solução (<= M). Se None, usa N = M.
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 300)[1:-1]
    x_eval = np.asarray(x_eval)

    if N is None or N > M:
        N = M

    # pontos da malha
    j = np.arange(1, M + 1)
    x = j / (M + 1)
    dx = 1.0 / (M + 1)

    # f(x) = x^{-alpha}
    f_vals = x**(-alpha)

    # DST-I não normalizada
    y = dst(f_vals, type=1)   # len M

    # coeficientes contínuos f_n ≈ sqrt(2)/(2(M+1)) * y_{n-1}
    n_vals = np.arange(1, N + 1)
    f_n = np.sqrt(2.0) * dx * 0.5 * y[:N]   # = sqrt(2)/(2(M+1)) * y

    # coeficientes de u
    u_n = (n_vals * np.pi)**(-2.0 * beta) * f_n

    # avalia u(x) em x_eval
    Sx = np.sin(np.pi * np.outer(x_eval, n_vals))
    u_vals = np.sqrt(2.0) * (Sx @ u_n)

    return x_eval, u_vals

# Método de referência usado em f_ex_xalpha.py (confiável)
def reference_solution_f_ex_xalpha(alpha, beta, N=400, Q=60, x_eval=None):
    """
    Solução espectral de alta precisão para (-d^2/dx^2)^beta u = x^{-alpha}, u(0)=u(1)=0.
    Base senoidal com condições de Dirichlet.
    Este é o método usado em f_ex_xalpha.py (confiável).
    """
    if x_eval is None:
        x_eval = np.linspace(0, 1, 200)[1:-1]  # evita os extremos 0 e 1 (onde u=0)
    
    # --- Quadratura de Gauss-Legendre em [0,1]
    t, w = np.polynomial.legendre.leggauss(Q)
    t = 0.5 * (t + 1.0)   # mapeia de [-1,1] -> [0,1]
    w = 0.5 * w
    
    # --- Mudança de variável que remove a singularidade
    power = 1.0 / (1.0 - alpha)
    x_quad = t ** power
    C_alpha = 1.0 / (1.0 - alpha)
    
    # --- Calcula coeficientes u_n
    u_n = np.zeros(N)
    for n in range(1, N + 1):
        sin_vals = np.sin(n * np.pi * x_quad)
        I_n = C_alpha * np.sum(w * sin_vals)
        f_n = np.sqrt(2.0) * I_n
        u_n[n - 1] = (n * np.pi) ** (-2.0 * beta) * f_n
    
    # --- Avalia solução em x_eval
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

# Parâmetros para solução de referência (método confiável de f_ex_xalpha.py)
N_ref = 1000  # Número de modos de Fourier para referência
Q_ref = 2000  # Pontos de quadratura para referência

# Parâmetros para comparação visual
x_plot = np.linspace(0.01, 0.99, 500)  # Pontos para plot (evita extremos)

# Parâmetros para teste Gammainc (aumente N para melhor qualidade)
N_gammainc_list = [50, 100, 200, 400, 800, 1600]  # Número de modos de Fourier para Gammainc

# Parâmetros para teste GLQ (aumente N e Q para melhor qualidade)
N_glq_list = [50, 100, 200, 400, 800, 1600]  # Número de modos de Fourier para GLQ
Q_glq_list = [100, 200, 400, 800, 1600]  # Pontos de quadratura para GLQ

# Parâmetros para teste DST (aumente M para melhor qualidade)
M_dst_list = [500, 1000, 2000, 4000, 8000]  # Número de pontos internos da malha para DST
N_dst_list = [None, 200, 400, 800]  # Número de modos para DST (None = usa M)

# Solução de referência usando método confiável de f_ex_xalpha.py
print("="*70)
print("Calculando solução de referência (método confiável de f_ex_xalpha.py)...")
print(f"Parâmetros: α={alpha}, β={beta}, N={N_ref}, Q={Q_ref}")
t0_ref = time.time()
x_ref, u_ref = reference_solution_f_ex_xalpha(alpha, beta, N=N_ref, Q=Q_ref, x_eval=x_plot)
time_ref = time.time() - t0_ref
print(f"Tempo de referência: {time_ref:.4f}s")
print("="*70)
print()

# Função para medir erro L2 relativo
# Compara soluções interpolando ambas para os mesmos pontos
def rel_L2(x_ref, u_ref, x, u):
    # x_ref e u_ref já estão nos pontos corretos
    # x e u são os pontos e valores do método testado
    # Interpola u para os pontos x_ref
    u_interp = np.interp(x_ref, x, u)
    # Calcula erro L2 relativo
    return np.sqrt(np.mean((u_interp - u_ref)**2)) / np.sqrt(np.mean(u_ref**2))

results = []

print("="*70)
print("BENCHMARK: Comparação de Métodos para Solução de Fourier")
print(f"Parâmetros: α={alpha}, β={beta}")
print("="*70)
print()

# Teste DST
print("Testando método DST...")
for M in M_dst_list:
    for N_dst in N_dst_list:
        if N_dst is not None and N_dst > M:
            continue
        t0 = time.time()
        x, u = reference_solution_DST(alpha, beta, M=M, N=N_dst, x_eval=x_plot)
        elapsed = time.time() - t0
        # Interpola para os mesmos pontos da referência para comparação
        u_interp = np.interp(x_plot, x, u)
        err = rel_L2(x_ref, u_ref, x_plot, u_interp)
        params_str = f"M={M}" if N_dst is None else f"M={M},N={N_dst}"
        results.append(("DST", params_str, elapsed, err, x, u))
        print(f"  {params_str:20s}: tempo={elapsed:.4f}s, erro={err:.2e}")

print()

# Teste Gammainc
print("Testando método Gammainc...")
for N in N_gammainc_list:
    t0 = time.time()
    x, u = reference_solution_gammainc(alpha, beta, N)
    elapsed = time.time() - t0
    # Interpola para os mesmos pontos da referência para comparação
    u_interp = np.interp(x_plot, x, u)
    err = rel_L2(x_ref, u_ref, x_plot, u_interp)
    results.append(("Gammainc", N, elapsed, err, x, u))
    print(f"  N={N:5d}: tempo={elapsed:.4f}s, erro={err:.2e}")

print()

# Teste GLQ
print("Testando método GLQ...")
for N in N_glq_list:
    for Q in Q_glq_list:
        t0 = time.time()
        x, u = reference_solution_GLQ(alpha, beta, N, Q)
        elapsed = time.time() - t0
        # Interpola para os mesmos pontos da referência para comparação
        u_interp = np.interp(x_plot, x, u)
        err = rel_L2(x_ref, u_ref, x_plot, u_interp)
        results.append(("GLQ", f"N={N},Q={Q}", elapsed, err, x, u))
        print(f"  N={N:3d}, Q={Q:3d}: tempo={elapsed:.4f}s, erro={err:.2e}")

print()
print("="*70)
print("ANÁLISE DE RESULTADOS")
print("="*70)

# Análise por método
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    if not data:
        continue
    
    times = [r[2] for r in data]
    errs = [r[3] for r in data]
    
    # Encontra melhor compromisso tempo-erro (menor tempo para erro < 1e-2)
    best_idx = None
    best_time = float('inf')
    for i, (t, e) in enumerate(zip(times, errs)):
        if e < 1e-2 and t < best_time:
            best_time = t
            best_idx = i
    
    print(f"\n{method}:")
    print(f"  Número de testes: {len(data)}")
    print(f"  Tempo mínimo: {min(times):.4f}s")
    print(f"  Tempo máximo: {max(times):.4f}s")
    print(f"  Tempo médio: {np.mean(times):.4f}s")
    print(f"  Erro mínimo: {min(errs):.2e}")
    print(f"  Erro máximo: {max(errs):.2e}")
    if best_idx is not None:
        print(f"  Melhor (erro<1e-2, tempo mínimo): {data[best_idx][1]}, tempo={best_time:.4f}s, erro={errs[best_idx]:.2e}")
    else:
        print(f"  Nenhum teste atingiu erro < 1e-2")
    
    # Eficiência: erro/tempo (menor é melhor)
    efficiencies = [e/t for e, t in zip(errs, times)]
    print(f"  Eficiência média (erro/tempo): {np.mean(efficiencies):.2e}")
    
    # Melhor resultado absoluto (menor erro)
    best_err_idx = np.argmin(errs)
    print(f"  Menor erro absoluto: {data[best_err_idx][1]}, tempo={times[best_err_idx]:.4f}s, erro={errs[best_err_idx]:.2e}")

print()
print("="*70)
print("RANKING POR VELOCIDADE (para erro < 1e-2)")
print("="*70)

# Ranking
ranking = []
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    for r in data:
        if r[3] < 1e-2:  # erro < 1e-2
            ranking.append((r[2], method, r[1], r[3]))  # (tempo, método, parâmetros, erro)

ranking.sort()  # Ordena por tempo

if ranking:
    print("\nTop 10 métodos mais rápidos (erro < 1e-2):")
    for i, (t, method, params, err) in enumerate(ranking[:10], 1):  # Top 10
        print(f"{i:2d}. {method:10s} | Parâmetros: {str(params):15s} | Tempo: {t:.4f}s | Erro: {err:.2e}")
else:
    print("Nenhum método atingiu erro < 1e-2")

print()
print("="*70)
print("RANKING POR PRECISÃO (menor erro)")
print("="*70)

# Ranking por precisão
ranking_precision = [(r[3], r[2], r[0], r[1]) for r in results]  # (erro, tempo, método, parâmetros)
ranking_precision.sort()  # Ordena por erro

print("\nTop 10 métodos mais precisos:")
for i, (err, t, method, params) in enumerate(ranking_precision[:10], 1):
    print(f"{i:2d}. {method:10s} | Parâmetros: {str(params):15s} | Erro: {err:.2e} | Tempo: {t:.4f}s")

print()
print("="*70)
print("CONCLUSÕES")
print("="*70)

# Análise final
dst_data = [r for r in results if r[0]=="DST"]
gammainc_data = [r for r in results if r[0]=="Gammainc"]
glq_data = [r for r in results if r[0]=="GLQ"]

if dst_data:
    dst_min_err = min([r[3] for r in dst_data])
    dst_best = min(dst_data, key=lambda x: x[3])
    print(f"\nDST: Erro mínimo = {dst_min_err:.2e}")
    print(f"  ✓ Melhor configuração: {dst_best[1]}, tempo={dst_best[2]:.4f}s")
    if dst_min_err < 1e-2:
        print("  ✓ Método recomendado para velocidade com boa precisão.")

if gammainc_data:
    gammainc_min_err = min([r[3] for r in gammainc_data])
    gammainc_best = min(gammainc_data, key=lambda x: x[3])
    print(f"\nGammainc: Erro mínimo = {gammainc_min_err:.2e}")
    print(f"  ✓ Melhor configuração: N={gammainc_best[1]}, tempo={gammainc_best[2]:.4f}s")
    if gammainc_min_err < 1e-2:
        print("  ✓ Método recomendado para alta precisão.")

if glq_data:
    glq_min_err = min([r[3] for r in glq_data])
    glq_best = min(glq_data, key=lambda x: x[3])
    print(f"\nGLQ: Erro mínimo = {glq_min_err:.2e}")
    print(f"  ✓ Melhor configuração: {glq_best[1]}, tempo={glq_best[2]:.4f}s")
    if glq_min_err < 1e-2:
        print("  ✓ Método recomendado para boa precisão com flexibilidade de parâmetros.")

# Recomendação final
print("\n" + "="*70)
print("RECOMENDAÇÃO FINAL")
print("="*70)

best_overall = min(results, key=lambda x: x[3])  # Menor erro
fastest_good = None
for r in sorted(results, key=lambda x: x[2]):  # Ordena por tempo
    if r[3] < 1e-2:
        fastest_good = r
        break

if fastest_good:
    print(f"\nPara velocidade (erro < 1e-2):")
    print(f"  Método: {fastest_good[0]}")
    print(f"  Parâmetros: {fastest_good[1]}")
    print(f"  Tempo: {fastest_good[2]:.4f}s")
    print(f"  Erro: {fastest_good[3]:.2e}")

print(f"\nPara precisão:")
print(f"  Método: {best_overall[0]}")
print(f"  Parâmetros: {best_overall[1]}")
print(f"  Tempo: {best_overall[2]:.4f}s")
print(f"  Erro: {best_overall[3]:.2e}")

print()
print("="*70)

# ---------- Plot 1: Performance (tempo vs erro) ----------
fig, ax = plt.subplots(figsize=(7,5))
for method in ["DST", "Gammainc", "GLQ"]:
    data = [r for r in results if r[0]==method]
    times = [r[2] for r in data]
    errs = [r[3] for r in data]
    ax.loglog(times, errs, 'o-', label=method)

ax.loglog([time_ref], [1e-10], 'r*', markersize=15, label='Referência (f_ex_xalpha.py)')
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Erro L2 relativo")
ax.set_title(f"Comparação de performance — α={alpha}, β={beta}")
ax.legend()
ax.grid(True, which="both")
plt.tight_layout()
plt.show()

# ---------- Plot 2: Comparação visual das soluções ----------
print()
print("="*70)
print("COMPARAÇÃO VISUAL DAS SOLUÇÕES")
print("="*70)
print("Plotando as soluções para verificar se aproximam a mesma função...")
print()

# Seleciona os melhores resultados de cada método para comparação visual
best_dst = min([r for r in results if r[0]=="DST"], key=lambda x: x[3])
best_gammainc = min([r for r in results if r[0]=="Gammainc"], key=lambda x: x[3])
best_glq = min([r for r in results if r[0]=="GLQ"], key=lambda x: x[3])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot completo
ax1.plot(x_ref, u_ref, 'k-', linewidth=3, label=f'Referência (f_ex_xalpha.py, N={N_ref}, Q={Q_ref}, t={time_ref:.4f}s)', alpha=0.9)
ax1.plot(best_dst[4], best_dst[5], 'b--', linewidth=2, label=f'DST ({best_dst[1]}, erro={best_dst[3]:.2e}, t={best_dst[2]:.4f}s)', alpha=0.7)
ax1.plot(best_gammainc[4], best_gammainc[5], 'r:', linewidth=2, label=f'Gammainc (N={best_gammainc[1]}, erro={best_gammainc[3]:.2e}, t={best_gammainc[2]:.4f}s)', alpha=0.7)
ax1.plot(best_glq[4], best_glq[5], 'g-.', linewidth=2, label=f'GLQ ({best_glq[1]}, erro={best_glq[3]:.2e}, t={best_glq[2]:.4f}s)', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('u(x)', fontsize=12)
ax1.set_title(f'Comparação Visual: Todas as Soluções — α={alpha}, β={beta}', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot do erro (diferença em relação à referência)
ax2.plot(x_ref, u_ref - u_ref, 'k-', linewidth=3, label='Referência (zero)', alpha=0.9)
# Interpola para os mesmos pontos
u_dst_interp = np.interp(x_ref, best_dst[4], best_dst[5])
u_gammainc_interp = np.interp(x_ref, best_gammainc[4], best_gammainc[5])
u_glq_interp = np.interp(x_ref, best_glq[4], best_glq[5])
ax2.plot(x_ref, u_ref - u_dst_interp, 'b--', linewidth=2, label=f'Erro DST (max={np.max(np.abs(u_ref - u_dst_interp)):.2e})', alpha=0.7)
ax2.plot(x_ref, u_ref - u_gammainc_interp, 'r:', linewidth=2, label=f'Erro Gammainc (max={np.max(np.abs(u_ref - u_gammainc_interp)):.2e})', alpha=0.7)
ax2.plot(x_ref, u_ref - u_glq_interp, 'g-.', linewidth=2, label=f'Erro GLQ (max={np.max(np.abs(u_ref - u_glq_interp)):.2e})', alpha=0.7)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Erro: u_ref - u_método', fontsize=12)
ax2.set_title('Diferença em relação à solução de referência', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('symlog', linthresh=1e-6)  # Escala log simétrica para mostrar erros positivos e negativos

plt.tight_layout()
plt.show()

print("="*70)
print("RESUMO DOS TEMPOS")
print("="*70)
print(f"Referência (f_ex_xalpha.py): {time_ref:.4f}s")
print(f"DST melhor: {best_dst[2]:.4f}s ({best_dst[1]}, erro={best_dst[3]:.2e})")
print(f"Gammainc melhor: {best_gammainc[2]:.4f}s (N={best_gammainc[1]}, erro={best_gammainc[3]:.2e})")
print(f"GLQ melhor: {best_glq[2]:.4f}s ({best_glq[1]}, erro={best_glq[3]:.2e})")
print("="*70)

# %%
