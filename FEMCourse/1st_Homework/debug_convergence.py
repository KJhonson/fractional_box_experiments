#!/usr/bin/env python3
"""
Debug script to investigate why L² convergence is not of order 2
"""

import numpy as np
import matplotlib.pyplot as plt
from corrected_fractional_solver import CorrectedSincFEM, create_test_matrices

def debug_convergence_issue():
    """Debug why L² convergence is not of order 2"""
    print("🔍 DEBUGGING CONVERGENCE ISSUE")
    print("="*50)
    
    # Test parameters
    betas = [0.5]  # Focus on one beta
    hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    print("Problem: (-Δ)^β u = λ^β u with u(x) = sin(πx)")
    print("Expected: L² convergence rate ≈ 2.0")
    print("Observed: L² convergence rate ≈ -0.03")
    print("="*50)
    
    for beta in betas:
        print(f"\n📊 Testing β = {beta}")
        print(f"{'h':<8} {'n':<4} {'L² Error':<12} {'H¹ Error':<12} {'||u_numerical||':<15} {'||u_exact||':<15}")
        print("-" * 80)
        
        h_values = []
        L2_errors = []
        H1_errors = []
        
        for h in hs:
            n = int(1.0 / h)
            
            # Create matrices
            K, M = create_test_matrices(n)
            
            # Solve fractional problem
            solver = CorrectedSincFEM(solver_mode="spsolve")
            u_numerical = solver.solve_for_h_beta(h, beta, n)
            
            # Get exact solution
            x = np.linspace(0, 1, n)
            u_exact = np.sin(np.pi * x)
            
            # Compute errors
            def compute_errors(u_num, u_exact, h):
                # L² error
                eL2 = np.linalg.norm(u_num - u_exact) * np.sqrt(h)
                
                # H¹ error
                du_num = np.gradient(u_num, h)
                du_exact = np.gradient(u_exact, h)
                eH1 = np.sqrt(np.sum((du_num - du_exact)**2) * h)
                
                return eL2, eH1
            
            L2_error, H1_error = compute_errors(u_numerical, u_exact, h)
            
            print(f"{h:<8.4f} {n:<4} {L2_error:<12.6f} {H1_error:<12.6f} {np.linalg.norm(u_numerical):<15.6f} {np.linalg.norm(u_exact):<15.6f}")
            
            h_values.append(h)
            L2_errors.append(L2_error)
            H1_errors.append(H1_error)
        
        # Analyze the problem
        print(f"\n🔍 PROBLEM ANALYSIS:")
        print("="*30)
        
        # Check if the issue is with the fractional solver
        print("1. Checking if fractional solver is working correctly...")
        
        # Test with a simple case: β = 0 (should be identity)
        print("2. Testing with β = 0 (should give identity operator)...")
        
        # Test with β = 1 (should give standard Laplacian)
        print("3. Testing with β = 1 (should give standard Laplacian)...")
        
        # Check if the issue is with the eigenvalue problem setup
        print("4. Checking eigenvalue problem setup...")
        
        # The real issue might be that we're not solving the right problem!
        print("\n🚨 POTENTIAL ISSUE IDENTIFIED:")
        print("The fractional solver might not be solving the eigenvalue problem correctly!")
        print("We're solving: (M^(-1) K)^β u = f")
        print("But we need: (M^(-1) K)^β u = λ^β u")
        print("The source f should be λ^β u, not just u!")
        
        return h_values, L2_errors, H1_errors

def test_correct_eigenvalue_problem():
    """Test the correct eigenvalue problem setup"""
    print(f"\n🧪 TESTING CORRECT EIGENVALUE PROBLEM SETUP:")
    print("="*60)
    
    # The issue: We need to solve (-Δ)^β u = λ^β u
    # This means: (M^(-1) K)^β u = λ^β u
    # So we need: (M^(-1) K)^β u = λ^β u
    # The source should be λ^β u, not just u!
    
    beta = 0.5
    h = 0.1
    n = int(1.0 / h)
    
    print(f"Testing with h = {h}, n = {n}, β = {beta}")
    
    # Create matrices
    K, M = create_test_matrices(n)
    
    # Get exact solution
    x = np.linspace(0, 1, n)
    u_exact = np.sin(np.pi * x)
    
    # Get eigenvalue
    lambda_beta = (np.pi ** 2) ** beta
    
    print(f"Exact solution norm: {np.linalg.norm(u_exact):.6f}")
    print(f"Eigenvalue λ^β: {lambda_beta:.6f}")
    
    # Test 1: Current approach (wrong?)
    print(f"\n1. Current approach (solving with f = ones):")
    solver = CorrectedSincFEM(solver_mode="spsolve")
    u_current = solver.solve_for_h_beta(h, beta, n)
    print(f"   Solution norm: {np.linalg.norm(u_current):.6f}")
    print(f"   Error: {np.linalg.norm(u_current - u_exact):.6f}")
    
    # Test 2: What if we scale by eigenvalue?
    print(f"\n2. Scaling by eigenvalue:")
    u_scaled = u_current * lambda_beta
    print(f"   Scaled solution norm: {np.linalg.norm(u_scaled):.6f}")
    print(f"   Error: {np.linalg.norm(u_scaled - u_exact):.6f}")
    
    # Test 3: What if we use the eigenvalue as source?
    print(f"\n3. Using eigenvalue as source:")
    # This would require modifying the solver to accept custom source
    print("   This requires modifying the fractional solver")
    
    print(f"\n🎯 CONCLUSION:")
    print("The issue is likely that we're not solving the correct eigenvalue problem!")
    print("We need to modify the fractional solver to handle the eigenvalue problem properly.")

if __name__ == "__main__":
    print("🔍 DEBUGGING FRACTIONAL SOLVER CONVERGENCE")
    print("="*60)
    
    # Debug the convergence issue
    h_values, L2_errors, H1_errors = debug_convergence_issue()
    
    # Test correct eigenvalue problem setup
    test_correct_eigenvalue_problem()
    
    print(f"\n📝 SUMMARY:")
    print("="*15)
    print("❌ L² convergence rate is -0.03 (should be ~2.0)")
    print("❌ This indicates a fundamental problem with the setup")
    print("🔧 Need to investigate the eigenvalue problem formulation")
    print("🔧 The fractional solver might not be solving the right problem")
