# %%
"""
1D stochastic fractional FVM experiment (Neumann, fully lumped mass).

Problem (matches the LaTeX notation):
    L = -d^2/dx^2 + I  on (0, 1) with Neumann BC u'(0) = u'(1) = 0.

    Source SPDE:
        L^alpha f = W,   W ~ N(0, \tilde C_ok),    \tilde C_ok = diag of dual lengths
                                                   (i.e. lumped mass on overkill mesh)
    Main problem:
        L^beta u = f.

Discrete pipeline (FVM):
    1) Sample W on the overkill mesh once (single realization, fixed seed).
    2) Solve L_ok^alpha f_ok = W  via `sinc_solver`.
    3) Build the overkill load:    F_ok = f_ok * diag(\tilde C_ok).
    4) Solve L_ok^beta U_ok = F_ok  via `sinc_solver`.
    5) For each coarse mesh:
         - Build A_{a->o} (1D CG1 evaluation matrix; in 1D it equals linear interpolation).
         - Build B from A_{a->o} by the 1D rule:
              0 in (0, 0.5), 0.5 at 0.5, 1 in (0.5, +inf).
         - Coarse load: F_a = B^T F_ok.
         - Solve L_a^beta U_a = F_a.
    6) Compute lumped-mass L^2 error on the overkill mesh:
         ||u_ok - u_a||^2 ≈ (U_ok - A U_a)^T \tilde C_ok (U_ok - A U_a).

We use an explicit vertex-centered 1D FVM operator (no FEM assembly), consistent
with an FVM convergence theory.

Hyperparameter `n_ok` is chosen as `OK_FACTOR * max(n_list)` so that every coarse
overkill ratio is an exact integer; this avoids ambiguities in the threshold rule.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem

# Reuse the user's sinc solver (only support function we need)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.sinc_solver import sinc_solver  # noqa: E402

# =============================================================================
# HYPERPARAMETERS  (edit here)
# =============================================================================

# Coarse meshes: number of intervals (powers of 2 keep ratios with n_ok integer)
n_list = [16, 32, 64, 128, 256, 512]

# Fractional powers β to test in L^β u = F
beta_values = [0.2, 0.3, 0.5, 0.6, 0.7]

# Source power α in L^α f = W (non-integer to use sinc; close to 0.6)
alpha_source = 0.5001

# Overkill mesh: integer multiple of max(n_list) so that ratios are clean
OK_FACTOR = 6                       # ok mesh = OK_FACTOR * max(n_list)
n_ok = OK_FACTOR * max(n_list)      # = 4096 with default settings

# Single stochastic realization
random_seed = 1

# Solution comparison plot (overkill u_ok vs projected coarse A u_a)
PLOT_SOLUTION_COMPARISON = True
beta_plot = 0.5
N_plot = max(n_list)

# Save outputs (errors and slopes) into a sub-folder named like this script
SAVE_OUTPUTS = True

# Theoretical reference rate, for annotation purposes only
def expected_rate(beta: float) -> float:
    return float(min(2.0 * beta + 0.5, 1.5))

# =============================================================================
# Sinc step parameter (kept identical across all solves and meshes)
# =============================================================================

def dynamic_k(beta: float, dofs: int) -> float:
    """k = -π^2 / (6 log h),    h = 1/(dofs+1)."""
    h = 1.0 / float(dofs + 1)
    return float(-np.pi ** 2 / (4 * np.log(h)))

# =============================================================================
# 1D vertex-centered FVM operator and lumped mass for L = -u'' + u (Neumann)
# =============================================================================
#  Vertices x_j = j h,  j = 0..n,  h = 1/n.
#  Dual cells:  vol_0 = h/2,  vol_j = h (1<=j<=n-1),  vol_n = h/2.
#  Two-point flux:  u'(face) ≈ (u_{j+1}-u_j)/h.
#
#  Tridiagonal operator rows (κ = 1):
#     j=0:        ( 1/h + h/2) u_0 - (1/h) u_1
#     1..n-1:    -(1/h) u_{j-1} + (2/h + h) u_j - (1/h) u_{j+1}
#     j=n:       -(1/h) u_{n-1} + ( 1/h + h/2) u_n

def assemble_fvm_operator_1d(n_elements: int, comm) -> tuple[PETSc.Mat, PETSc.Mat, np.ndarray]:
    """
    Assemble the FVM operator L = -u'' + u and the lumped mass M (diagonal of dual volumes)
    on a uniform mesh of [0,1] with `n_elements` cells. Neumann boundaries.
    Returns: (L, M, vol)  with vol = diagonal of M (numpy array).
    """
    n = int(n_elements) + 1                    # number of vertices/dofs
    h = 1.0 / float(n_elements)

    # ---- L (tridiagonal) ----
    L = PETSc.Mat().createAIJ(size=(n, n), nnz=3, comm=comm)
    L.setUp()
    inv_h = 1.0 / h

    L.setValue(0, 0, inv_h + 0.5 * h)
    L.setValue(0, 1, -inv_h)
    for j in range(1, n - 1):
        L.setValue(j, j - 1, -inv_h)
        L.setValue(j, j, 2.0 * inv_h + h)
        L.setValue(j, j + 1, -inv_h)
    L.setValue(n - 1, n - 2, -inv_h)
    L.setValue(n - 1, n - 1, inv_h + 0.5 * h)
    L.assemble()

    # ---- M (diagonal lumped mass = dual volumes) ----
    vol = np.empty(n, dtype=np.float64)
    vol[0] = 0.5 * h
    if n > 2:
        vol[1:-1] = h
    vol[-1] = 0.5 * h

    vol_vec = PETSc.Vec().createWithArray(vol, comm=comm)
    M = PETSc.Mat().createAIJ(size=(n, n), nnz=1, comm=comm)
    M.setUp()
    M.setDiagonal(vol_vec)
    M.assemble()
    return L, M, vol


def build_env(n_elements: int) -> dict:
    """Build a 1D Neumann FVM environment for a given mesh resolution."""
    mesh_ = dmesh.create_interval(MPI.COMM_WORLD, int(n_elements), [0.0, 1.0])
    V_ = fem.functionspace(mesh_, ("CG", 1))
    L_, M_, vol_ = assemble_fvm_operator_1d(n_elements, mesh_.comm)
    return {
        "n": int(n_elements),
        "h": 1.0 / float(n_elements),
        "mesh": mesh_,
        "V": V_,
        "L": L_,
        "M": M_,
        "vol": vol_,                        # diag(M)
    }


# =============================================================================
# Coarse <-> overkill mappings  (1D)
# =============================================================================

def build_A_a_to_o_1d(n_a: int, n_ok: int) -> np.ndarray:
    """
    A_{a->o}[w, z] = phi_z^a(w),  with hat basis on the coarse mesh evaluated at
    overkill nodes. In 1D this is just linear interpolation.
    Rows / columns are in *sorted* (left-to-right) ordering, which for a uniform
    mesh on [0,1] is also the natural CG1 dof ordering.
    """
    if n_ok % n_a != 0:
        raise ValueError(f"n_ok={n_ok} must be a multiple of n_a={n_a} for clean threshold-B.")
    x_a = np.linspace(0.0, 1.0, n_a + 1)
    x_o = np.linspace(0.0, 1.0, n_ok + 1)
    n_a1 = x_a.size
    n_o1 = x_o.size

    A = np.zeros((n_o1, n_a1), dtype=np.float64)
    idx = np.searchsorted(x_a, x_o, side="right") - 1
    idx = np.clip(idx, 0, n_a1 - 2)

    xL = x_a[idx]
    xR = x_a[idx + 1]
    t = (x_o - xL) / (xR - xL)
    rows = np.arange(n_o1)
    A[rows, idx] = 1.0 - t
    A[rows, idx + 1] = t
    return A


def threshold_B_from_A_1d(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """1D threshold rule (LaTeX): 0 on (0,0.5), 0.5 at 0.5, 1 on (0.5,∞)."""
    B = np.zeros_like(A, dtype=np.float64)
    half = 0.5
    B[np.abs(A - half) <= tol] = half
    B[A > half + tol] = 1.0
    return B


# =============================================================================
# Errors
# =============================================================================

def lumped_l2_error_overkill(U_ok: np.ndarray, A: np.ndarray, U_a: np.ndarray, vol_ok: np.ndarray) -> float:
    """sqrt((U_ok - A U_a)^T diag(vol_ok) (U_ok - A U_a))."""
    diff = U_ok - A @ U_a
    return float(np.sqrt(np.sum(vol_ok * diff * diff)))


# =============================================================================
# Build environments
# =============================================================================

if MPI.COMM_WORLD.rank == 0:
    print("=" * 72)
    print("1D Neumann fractional SPDE  ---  FVM overkill experiment")
    print("=" * 72)
    print(f"  n_ok        = {n_ok}    (= {OK_FACTOR} * max(n_list))")
    print(f"  n_list      = {n_list}")
    print(f"  alpha       = {alpha_source}")
    print(f"  beta_list   = {beta_values}")
    print(f"  random_seed = {random_seed}")
    print("=" * 72)

ok_env = build_env(n_ok)
V_ok = ok_env["V"]
L_ok = ok_env["L"]
M_ok = ok_env["M"]
vol_ok = ok_env["vol"]
n_ok_dofs = vol_ok.size

# Coarse environments + projection / load-transfer matrices (precomputed once)
approx = []
for n in n_list:
    env = build_env(n)
    A_ao = build_A_a_to_o_1d(n, n_ok)            # shape (n_ok+1, n+1)
    Bmat = threshold_B_from_A_1d(A_ao)            # shape (n_ok+1, n+1)
    env["A"] = A_ao
    env["B"] = Bmat
    approx.append(env)

# =============================================================================
# Sample W ONCE on overkill, build f_ok and F_ok ONCE
# =============================================================================

if random_seed is not None:
    np.random.seed(random_seed)
Z = np.random.normal(0.0, 1.0, size=n_ok_dofs)
W_arr = np.sqrt(vol_ok) * Z                  # W ~ N(0, diag(vol_ok))
W_vec = PETSc.Vec().createWithArray(W_arr, comm=ok_env["mesh"].comm)

# Solve L_ok^alpha f_ok = W
k_alpha = dynamic_k(alpha_source, n_ok_dofs)
f_ok_func, _ = sinc_solver(L_ok, M_ok, W_vec, V_ok, bc=None, beta=alpha_source, k=k_alpha)
f_ok = f_ok_func.x.array.copy()

# FVM load on overkill: F_ok ≈ ∫_{b_i} f dx ≈ f_ok * diag(\tilde C_ok)
F_ok = f_ok * vol_ok
F_ok_vec = PETSc.Vec().createWithArray(F_ok, comm=ok_env["mesh"].comm)

# =============================================================================
# Main β-loop  (single realization throughout)
# =============================================================================

results = {}            # beta -> (h_array, error_array)
U_ok_cache = {}         # beta -> U_ok array (used by the comparison plot)

for beta in beta_values:
    # Overkill solution
    k_ok = dynamic_k(beta, n_ok_dofs)
    U_ok_func, _ = sinc_solver(L_ok, M_ok, F_ok_vec, V_ok, bc=None, beta=beta, k=k_ok)
    U_ok = U_ok_func.x.array.copy()
    U_ok_cache[beta] = U_ok

    errs = np.zeros(len(approx), dtype=np.float64)
    hs = np.array([env["h"] for env in approx], dtype=np.float64)

    for j, env in enumerate(approx):
        # Coarse load: F_a = B^T F_ok
        F_a = env["B"].T @ F_ok
        F_a_vec = PETSc.Vec().createWithArray(F_a, comm=env["mesh"].comm)

        k_a = dynamic_k(beta, env["vol"].size)
        U_a_func, _ = sinc_solver(env["L"], env["M"], F_a_vec, env["V"], bc=None, beta=beta, k=k_a)
        U_a = U_a_func.x.array.copy()

        errs[j] = lumped_l2_error_overkill(U_ok, env["A"], U_a, vol_ok)

    results[beta] = (hs, errs)

    if MPI.COMM_WORLD.rank == 0:
        slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
        print(f"\nβ = {beta}  (expected slope ≤ {expected_rate(beta):.2f}, fitted: {slope:.2f})")
        print(f"  {'N':>6}  {'h':>10}  {'L2 error':>14}")
        for n, h, e in zip(n_list, hs, errs):
            print(f"  {n:>6d}  {h:>10.4e}  {e:>14.6e}")

# =============================================================================
# Plot 1:  L^2 error vs h, with theoretical-rate guides
# =============================================================================

if MPI.COMM_WORLD.rank == 0:
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(beta_values)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    plt.figure(figsize=(9, 6))
    hs = np.array([env["h"] for env in approx], dtype=np.float64)

    for i, beta in enumerate(beta_values):
        _, errs = results[beta]
        slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
        plt.loglog(
            hs, errs,
            color=colors[i], marker=markers[i % len(markers)],
            linestyle="-", linewidth=1.8, markersize=7,
            label=f"β={beta} (fit≈{slope:.2f}, theory≤{expected_rate(beta):.2f})",
        )

    # reference triangles
    h_ref = hs[-2]
    e_ref = max(errs[-2], 1e-12)
    for p, lbl in [(1.3, "h^1.3"), (1.7, "h^1.7")]:
        c = e_ref / h_ref ** p * 4.0
        plt.loglog(hs, c * hs ** p, "k--", linewidth=0.8, alpha=0.5)
        plt.text(hs[0], c * hs[0] ** p, lbl, fontsize=8, color="k", alpha=0.7)

    plt.xlabel("mesh size h", fontsize=12)
    plt.ylabel("L^2 error (lumped mass on overkill)", fontsize=12)
    plt.title(
        f"FVM Neumann overkill: 1D fractional SPDE  (α={alpha_source}, n_ok={n_ok})",
        fontsize=12,
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Plot 2:  solution comparison u_ok vs A u_a on the overkill grid
# =============================================================================

if PLOT_SOLUTION_COMPARISON and MPI.COMM_WORLD.rank == 0:
    if beta_plot not in U_ok_cache:
        # Compute on demand if user picked a non-listed beta
        k_ok = dynamic_k(beta_plot, n_ok_dofs)
        U_ok_func, _ = sinc_solver(L_ok, M_ok, F_ok_vec, V_ok, bc=None, beta=beta_plot, k=k_ok)
        U_ok = U_ok_func.x.array.copy()
    else:
        U_ok = U_ok_cache[beta_plot]

    # find env for N_plot
    env_plot = next((env for env in approx if env["n"] == int(N_plot)), approx[-1])
    F_a = env_plot["B"].T @ F_ok
    F_a_vec = PETSc.Vec().createWithArray(F_a, comm=env_plot["mesh"].comm)
    k_a = dynamic_k(beta_plot, env_plot["vol"].size)
    U_a_func, _ = sinc_solver(env_plot["L"], env_plot["M"], F_a_vec, env_plot["V"], bc=None, beta=beta_plot, k=k_a)
    U_a = U_a_func.x.array.copy()
    U_a_on_ok = env_plot["A"] @ U_a

    x_ok = np.linspace(0.0, 1.0, n_ok + 1)
    plt.figure(figsize=(11, 5))
    plt.plot(x_ok, U_ok, "k-", linewidth=2.0, label=f"u_ok (overkill, N={n_ok})")
    plt.plot(x_ok, U_a_on_ok, "r--", linewidth=1.8, label=f"A u_h (coarse, N={env_plot['n']})")
    plt.title(f"Neumann FVM comparison  (α={alpha_source}, β={beta_plot}, seed={random_seed})", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# Save tables (errors + slopes vs β)
# =============================================================================

if SAVE_OUTPUTS and MPI.COMM_WORLD.rank == 0:
    out_dir = Path(__file__).parent / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # errors_1dfvm.dat: rows = N, columns = beta values
    errors_path = out_dir / "errors_1dfvm.dat"
    with errors_path.open("w") as fh:
        header = ["h"] + [str(b) for b in beta_values]
        fh.write("\t".join(header) + "\n")
        for j, n in enumerate(n_list):
            row = [str(n)]
            for beta in beta_values:
                row.append(f"{results[beta][1][j]:.10e}")
            fh.write("\t".join(row) + "\n")

    # betaxslope_1dfvm.dat: beta -> fitted slope vs h, plus theoretical rate
    slopes_path = out_dir / "betaxslope_1dfvm.dat"
    with slopes_path.open("w") as fh:
        fh.write("beta\tfitted_slope_vs_h\ttheoretical_rate\n")
        hs = np.array([env["h"] for env in approx], dtype=np.float64)
        for beta in beta_values:
            errs = results[beta][1]
            slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
            fh.write(f"{beta}\t{slope:.6f}\t{expected_rate(beta):.6f}\n")

    print(f"\nSaved errors to: {errors_path}")
    print(f"Saved beta-vs-slope to: {slopes_path}")

# %%
