# %%
import numpy as np
import matplotlib.pyplot as plt

# Gauss-Legendre quadrature
def gauss_legendre(a, b, n):
    xp, wp = np.polynomial.legendre.leggauss(n)
    xm = 0.5 * (b - a)
    xc = 0.5 * (b + a)
    x = xc + xm * xp
    w = xm * wp
    return x, w

# Exact solution and forcing term
def u_exact(x, lam):
    return x**lam * (1 - x)

def f_rhs(x, lam):
    return -(lam*(lam-1)*x**(lam-2) - (lam+1)*lam*x**(lam-1))

# Assembly
def assemble(lam, h, quad_n=8):
    num_elements = int(round(1.0/h))
    h = 1.0 / num_elements
    x_nodes = np.linspace(0, 1, num_elements + 1)
    N = num_elements - 1
    A = (2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h

    F_fem = np.zeros(N)
    F_fvm = np.zeros(N)

    # FEM load
    for i in range(1, num_elements):
        xi_minus = x_nodes[i-1]
        xi = x_nodes[i]
        xi_plus = x_nodes[i+1]

        # left
        xl, wl = gauss_legendre(xi_minus, xi, quad_n)
        phi_left = (xl - xi_minus) / h
        F_fem[i-1] += np.sum(f_rhs(xl, lam) * phi_left * wl)

        # right
        xr, wr = gauss_legendre(xi, xi_plus, quad_n)
        phi_right = (xi_plus - xr) / h
        F_fem[i-1] += np.sum(f_rhs(xr, lam) * phi_right * wr)

    # FVM load
    for i in range(1, num_elements):
        xi = x_nodes[i]
        a = max(0.0, xi - 0.5*h)
        b = min(1.0, xi + 0.5*h)
        xq, wq = gauss_legendre(a, b, quad_n)
        F_fvm[i-1] = np.sum(f_rhs(xq, lam) * wq)

    return A, F_fem, F_fvm, x_nodes

# Continuous L2 error
def L2_error(u_vec, lam, x_nodes, quad_n=8):
    num_elements = len(x_nodes)-1
    h = x_nodes[1]-x_nodes[0]
    err_sq = 0.0
    for e in range(num_elements):
        xL = x_nodes[e]
        xR = x_nodes[e+1]
        if e == 0:
            uL = 0.0
        else:
            uL = u_vec[e-1]
        if e == num_elements-1:
            uR = 0.0
        else:
            uR = u_vec[e]
        xq, wq = gauss_legendre(xL, xR, quad_n)
        uhq = uL*(xR-xq)/h + uR*(xq-xL)/h
        uexq = u_exact(xq, lam)
        err_sq += np.sum((uexq - uhq)**2 * wq)
    return np.sqrt(err_sq)

# Main experiment
lams = [1.1, 1.5, 2.0]
hs = np.array([2**(-k) for k in range(4, 9)])
errors_FEM = {lam: [] for lam in lams}
errors_FVM = {lam: [] for lam in lams}

for lam in lams:
    for h in hs:
        A, F_fem, F_fvm, x_nodes = assemble(lam, h)
        u_fem = np.linalg.solve(A, F_fem)
        u_fvm = np.linalg.solve(A, F_fvm)
        err_fem = L2_error(u_fem, lam, x_nodes)
        err_fvm = L2_error(u_fvm, lam, x_nodes)
        errors_FEM[lam].append(err_fem)
        errors_FVM[lam].append(err_fvm)

# Plot convergence
plt.figure(figsize=(8,6))
for lam in lams:
    plt.loglog(hs, errors_FEM[lam], 'o--', label=f'FEM λ={lam}')
    plt.loglog(hs, errors_FVM[lam], 's-', label=f'FVM λ={lam}')
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("‖u - u_h‖_{L2}")
plt.legend()
plt.title("Convergence of FEM and FVM for uλ(x)=x^λ(1-x)")
plt.grid(True, which="both")
plt.show()

# Compute slopes
for lam in lams:
    slope_fem = np.polyfit(np.log(hs), np.log(errors_FEM[lam]), 1)[0]
    slope_fvm = np.polyfit(np.log(hs), np.log(errors_FVM[lam]), 1)[0]
    print(f"λ={lam}: FEM slope={slope_fem:.3f}, FVM slope={slope_fvm:.3f}")

# %%

# ============ Experiment: Smooth concentrated load ============
# Problem: -u'' = f(x) = (x^2 + a^2)^(-p/2)  on (0,1), u(0)=u(1)=0
# Exact solution via Green's function:
# u(x) = ∫_0^x (x-t) f(t) dt - x ∫_0^1 (1-t) f(t) dt

# -------- Problem setup --------
def f_rhs(x, a, p):
    return (x**2 + a**2)**(-0.5*p)

# Exact solution via Green's function
def u_exact_green(x_eval, a, p):
    x_eval = np.asarray(x_eval)
    # precompute the global integral C = ∫_0^1 (1-t) f(t) dt
    t_global = np.linspace(0, 1, 20000)
    f_global = f_rhs(t_global, a, p)
    C = np.trapz((1 - t_global) * f_global, t_global)
    
    u_val = np.zeros_like(x_eval)
    for k, x in enumerate(x_eval):
        t_loc = np.linspace(0, x, 2000)
        f_loc = f_rhs(t_loc, a, p)
        first_term = np.trapz((x - t_loc) * f_loc, t_loc)
        u_val[k] = first_term - x * C
    return u_val

# -------- Assembly of FEM / FVM systems on uniform mesh --------
def assemble_system(h, a, p, quad_n=8):
    # snap N
    N = int(round(1.0/h)) - 1
    h = 1.0/(N+1)
    x_nodes = np.linspace(0, 1, N+2)  # includes boundaries
    # stiffness matrix
    A = (2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h
    
    F_fem = np.zeros(N)
    F_fvm = np.zeros(N)
    
    # FEM load: ∑ elements f * hat_i
    for i in range(N):
        xL, xM, xR = x_nodes[i], x_nodes[i+1], x_nodes[i+2]
        
        # left piece [xL,xM], hat on node i+1 is (x - xL)/h
        xl, wl = gauss_legendre(xL, xM, quad_n)
        phi_left = (xl - xL)/h
        F_fem[i] += np.sum(f_rhs(xl, a, p) * phi_left * wl)
        
        # right piece [xM,xR], hat on node i+1 is (xR - x)/h
        xr, wr = gauss_legendre(xM, xR, quad_n)
        phi_right = (xR - xr)/h
        F_fem[i] += np.sum(f_rhs(xr, a, p) * phi_right * wr)
    
    # FVM load: control volume around each interior node
    for i in range(N):
        xC = x_nodes[i+1]
        a_cv = max(0.0, xC - 0.5*h)
        b_cv = min(1.0, xC + 0.5*h)
        xq, wq = gauss_legendre(a_cv, b_cv, quad_n)
        F_fvm[i] = np.sum(f_rhs(xq, a, p) * wq)
    
    return A, F_fem, F_fvm, x_nodes

# -------- Compute continuous L2 error --------
def L2_error(u_vec, a, p, x_nodes, quad_n=8):
    N = len(u_vec)
    h = x_nodes[1]-x_nodes[0]
    err_sq = 0.0
    for e in range(N+1):
        xL = x_nodes[e]
        xR = x_nodes[e+1]
        
        # nodal values on this elem for uh
        if e == 0:
            uL = 0.0
        else:
            uL = u_vec[e-1]
        if e == N:
            uR = 0.0
        else:
            uR = u_vec[e]
        
        xq, wq = gauss_legendre(xL, xR, quad_n)
        uhq = uL*(xR-xq)/h + uR*(xq-xL)/h
        uexq = u_exact_green(xq, a, p)
        err_sq += np.sum((uexq - uhq)**2 * wq)
    return np.sqrt(err_sq)

# -------- Run convergence study --------
a = 1e-3
p = 1.1
hs = np.array([2**(-k) for k in range(4, 10)])  # h = 1/16 ... 1/512
errors_FEM = []
errors_FVM = []
hs_effective = []

for h in hs:
    # assemble and solve
    A, F_fem, F_fvm, x_nodes = assemble_system(h, a, p)
    u_fem = np.linalg.solve(A, F_fem)
    u_fvm = np.linalg.solve(A, F_fvm)
    
    # true error in continuous L2
    err_fem = L2_error(u_fem, a, p, x_nodes)
    err_fvm = L2_error(u_fvm, a, p, x_nodes)
    
    hs_effective.append(x_nodes[1]-x_nodes[0])
    errors_FEM.append(err_fem)
    errors_FVM.append(err_fvm)

hs_effective = np.array(hs_effective)
errors_FEM = np.array(errors_FEM)
errors_FVM = np.array(errors_FVM)

# slopes (least squares fit on log-log)
slope_fem = np.polyfit(np.log(hs_effective), np.log(errors_FEM), 1)[0]
slope_fvm = np.polyfit(np.log(hs_effective), np.log(errors_FVM), 1)[0]

# plot
plt.figure(figsize=(8,6))
plt.loglog(hs_effective, errors_FEM, 'o--', label=f'FEM (a={a}, p={p})')
plt.loglog(hs_effective, errors_FVM, 's-', label=f'FVM (a={a}, p={p})')
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("||u - u_h||_{L2(0,1)}")
plt.title(f"Smooth concentrated load f(x)=(x^2+a^2)^(-p/2)\nslopes: FEM={slope_fem:.2f}, FVM={slope_fvm:.2f}")
plt.grid(True, which="both")
plt.legend()
plt.show()

(slope_fem, slope_fvm)

# %%
