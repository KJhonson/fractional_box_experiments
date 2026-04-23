#!/usr/bin/env python3
"""
Comparação simples entre BRASIL e Sinc para problemas fracionários de Neumann.
"""



import sys
sys.path.append('/home/dolfinx/shared/FEM_project')
sys.path.append('/home/dolfinx/shared/rSPDE')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import ufl
from dolfinx import fem
from petsc4py import PETSc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import importlib.util

# Importações do FEM_project
from domains import make_uniform_interval
from operators import build_operator_B, lumped_from_rowsum
from loads import assemble_rhs
from solver import sinc_solve

# Load fractional operators
spec = importlib.util.spec_from_file_location('fractional_operators', '/home/dolfinx/shared/rSPDE/fractional.operators.py')
fractional_operators_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fractional_operators_module)
fractional_operators = fractional_operators_module.fractional_operators

# Output directory
OUTPUT_DIR = Path("/home/dolfinx/shared/rSPDE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GLOBAL PARAMETERS - MODIFY THESE AS NEEDED
# ============================================================================
BETA = 0.2  # Fractional power (0 < beta < 1)
M_ORDER = 2 # BRASIL approximation order (m >= 1) - increased for better accuracy
N = 320      # Number of mesh elements
INTERVAL = (0.2, 1)  # BRASIL approximation interval (a, b) - wider for better coverage
K_STEP = -np.pi**2/(4* BETA * np.log(1/(N+1)))  # Sinc step size (default: 0.25)
FACTOR = 10 # <<<<===== Parâmetro escala global que você pode mudar!
# ============================================================================

def solve_neumann_brasil(B, M, b, V, beta=BETA, m=M_ORDER, interval=INTERVAL, factor=FACTOR):
    """
    Resolve problema de Neumann usando operador BRASIL com massa lumped.
    """
    print(f"=== Resolvendo com BRASIL (β={beta}, m={m}, factor={factor}) ===")
    
    start_time = time.time()
    
    # Converter matrizes PETSc para scipy.sparse
    B_size = B.getSize()
    M_size = M.getSize()
    
    B_np = B.getValues(range(B_size[0]), range(B_size[1]))
    M_np = M.getValues(range(M_size[0]), range(M_size[1]))
    
    B_scipy = csr_matrix(B_np)
    M_scipy = csr_matrix(M_np)
    
    # Converter vetor de carga para numpy
    b_np = b.array.copy()
    
    # Permite escolher o fator de escala
    scale_factor = factor
    
    # Aplicar operador fracionário BRASIL
    Pl, Pr = fractional_operators(B_scipy, beta, M_scipy, scale_factor, m=m, interval=interval)
    
    # Resolver: Pl * v = b, depois u = Pr * v
    v = spsolve(Pl, b_np)
    u_np = Pr @ v
    
    # Converter de volta para dolfinx Function
    u_h = fem.Function(V)
    u_h.x.array[:] = u_np
    u_h.x.scatter_forward()
    
    solve_time = time.time() - start_time
    print(f"BRASIL solver completed in {solve_time:.4f}s")
    
    return u_h, solve_time

def solve_neumann_sinc(B, M, b, V, beta=BETA, k_step=K_STEP):
    """
    Resolve problema de Neumann usando método sinc com massa lumped (para comparação justa).
    """
    print(f"=== Resolvendo com Sinc (β={beta}) ===")
    
    start_time = time.time()
    
    # Criar massa lumped para comparação justa com BRASIL
    M_lumped_petsc, M_lumped_diag = lumped_from_rowsum(M)
    
    # Usar o solver sinc com massa lumped
    u_h, ksp = sinc_solve(B, M_lumped_petsc, b, V, beta=beta, k=k_step)
    
    solve_time = time.time() - start_time
    print(f"Sinc solver completed in {solve_time:.4f}s")
    
    return u_h, solve_time

def compute_l2_error(u_h, u_exact_func, V):
    """
    Calcula erro L2 entre solução aproximada e exata.
    """
    # Pontos de avaliação
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    u_exact_vals = u_exact_func(x_coords)
    u_h_vals = u_h.x.array
    
    # Calcular norma L2
    diff = u_h_vals - u_exact_vals
    l2_error = np.sqrt(np.sum(diff**2) / len(diff))
    
    return l2_error

def unified_comparison(N_mesh=N, beta=BETA, m=M_ORDER, interval=INTERVAL, k_step=K_STEP, create_plots=True, factor=FACTOR):
    """
    Comparação simples entre BRASIL e Sinc.
    """
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO SIMPLES BRASIL vs SINC")
    print(f"Problema: -u'' + u = cos(πx) com Neumann BC")
    print(f"Malha: {N_mesh} elementos, β={beta}, m={m}, factor={factor}")
    print(f"{'='*60}")
    
    # Criar malha e espaço de funções
    mesh = make_uniform_interval(N_mesh, 0.0, 1.0)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    
    # Montar operadores
    B, _, M, _, _ = build_operator_B(V, bc_type="neumann", kappa=1)
    f_expr = lambda x: ufl.cos(ufl.pi * x[0])
    b = assemble_rhs(V, f_expr)
    
    # Solução exata para comparação (como no main.py original)
    u_exact_func = lambda x: np.cos(np.pi * x) / ((1 + np.pi**2) ** beta)
    
    print("\n1. Resolvendo com método BRASIL...")
    try:
        u_brasil, time_brasil = solve_neumann_brasil(B, M, b, V, beta, m, interval, factor)
        brasil_success = True
        print("✅ Método BRASIL funcionou!")
    except Exception as e:
        print(f"❌ Erro no método BRASIL: {e}")
        brasil_success = False
        time_brasil = 0
    
    print("\n2. Resolvendo com método Sinc...")
    try:
        u_sinc, time_sinc = solve_neumann_sinc(B, M, b, V, beta, k_step)
        sinc_success = True
        print("✅ Método Sinc funcionou!")
    except Exception as e:
        print(f"❌ Erro no método Sinc: {e}")
        sinc_success = False
        time_sinc = 0
    
    # Calcular erros
    if brasil_success:
        error_brasil = compute_l2_error(u_brasil, u_exact_func, V)
    else:
        error_brasil = float('inf')
    
    if sinc_success:
        error_sinc = compute_l2_error(u_sinc, u_exact_func, V)
    else:
        error_sinc = float('inf')
    
    # Criar visualizações
    if create_plots and brasil_success and sinc_success:
        print("\n📊 Criando visualizações...")
        
        # Pontos para plot
        x_coords = V.tabulate_dof_coordinates()[:, 0]
        u_exact_vals = u_exact_func(x_coords)
        u_brasil_vals = u_brasil.x.array
        u_sinc_vals = u_sinc.x.array
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(x_coords, u_exact_vals, 'g-', linewidth=2, label='Exata')
        plt.plot(x_coords, u_sinc_vals, 'b--', linewidth=2, label='Sinc')
        plt.plot(x_coords, u_brasil_vals, 'r:', linewidth=2, label='BRASIL')
        # Adicionar pontos nos locais de interpolação
        plt.plot(x_coords, u_exact_vals, 'go', markersize=4, alpha=0.7)
        plt.plot(x_coords, u_sinc_vals, 'bo', markersize=4, alpha=0.7)
        plt.plot(x_coords, u_brasil_vals, 'ro', markersize=4, alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Soluções Comparadas (pontos = nós de interpolação)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(x_coords, np.abs(u_sinc_vals - u_exact_vals), 'b-', linewidth=2, label='Erro Sinc')
        plt.plot(x_coords, np.abs(u_brasil_vals - u_exact_vals), 'r-', linewidth=2, label='Erro BRASIL')
        plt.xlabel('x')
        plt.ylabel('|Erro|')
        plt.title('Erro Absoluto')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.semilogy(x_coords, np.abs(u_sinc_vals - u_exact_vals), 'b-', linewidth=2, label='Erro Sinc')
        plt.semilogy(x_coords, np.abs(u_brasil_vals - u_exact_vals), 'r-', linewidth=2, label='Erro BRASIL')
        plt.xlabel('x')
        plt.ylabel('|Erro| (log)')
        plt.title('Erro Absoluto (Escala Log)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        methods = ['Sinc', 'BRASIL']
        errors = [error_sinc, error_brasil]
        times = [time_sinc, time_brasil]
        
        plt.bar(methods, errors, color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Erro L2')
        plt.title('Comparação de Erros')
        plt.yscale('log')
        
        plt.tight_layout()
        
        # Salvar plot
        filename = f"unified_comparison_beta{beta}_N{N_mesh}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico salvo em: {filepath}")
        plt.close()
    
    # Métricas de comparação
    print(f"\n📊 MÉTRICAS DE COMPARAÇÃO:")
    print(f"  Erro L2 Sinc:    {error_sinc:.2e}")
    print(f"  Erro L2 BRASIL:  {error_brasil:.2e}")
    if brasil_success and sinc_success:
        print(f"  Diferença máxima: {np.max(np.abs(u_brasil_vals - u_sinc_vals)):.2e}")
    print(f"  Tempo Sinc:      {time_sinc:.4f}s")
    print(f"  Tempo BRASIL:    {time_brasil:.4f}s")
    if time_sinc > 0:
        print(f"  Speedup:         {time_brasil/time_sinc:.2f}x")
    
    return {
        'N': N_mesh,
        'beta': beta,
        'm': m,
        'factor': factor,
        'error_sinc': error_sinc,
        'error_brasil': error_brasil,
        'time_sinc': time_sinc,
        'time_brasil': time_brasil,
        'brasil_success': brasil_success,
        'sinc_success': sinc_success
    }

def main():
    """
    Função principal que executa a comparação simples.
    """
    print("COMPARAÇÃO SIMPLES BRASIL vs SINC")
    print("Problema de Neumann Fracionário")
    print("="*60)
    
    # Teste único com parâmetros globais
    print(f"\nTeste com β={BETA}, m={M_ORDER}, N={N}, intervalo={INTERVAL}, k={K_STEP}, factor={FACTOR}")
    print("-" * 40)
    result = unified_comparison(N_mesh=N, beta=BETA, m=M_ORDER, interval=INTERVAL, k_step=K_STEP, create_plots=True, factor=FACTOR)
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    
    if result['sinc_success'] and result['brasil_success']:
        speedup = result['time_brasil'] / result['time_sinc']
        print(f"Speedup BRASIL/Sinc: {speedup:.2f}x")
        print(f"Erro Sinc: {result['error_sinc']:.2e}")
        print(f"Erro BRASIL: {result['error_brasil']:.2e}")
        print(f"Melhoria Sinc vs BRASIL: {result['error_brasil']/result['error_sinc']:.0f}x")
        print("✅ Ambos os métodos funcionaram!")
    else:
        print("❌ Algum método falhou")
    
    print("\n✅ Comparação simples concluída!")
    print(f"📁 Gráfico salvo em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

# %%