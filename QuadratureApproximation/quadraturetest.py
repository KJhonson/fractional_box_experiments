# ✅ FEM quadrature tests for the dual control-volume operator Q
# Computes both: 
#  (1) ‖Qv_h − v_h‖_{L²} ≈ O(h)
#  (2) |(f,Qv_h) − (f,v_h)| ≈ O(h²)
# Works directly in Google Colab

import numpy as np, math, pandas as pd, matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Smooth test function and mesh generation
# ------------------------------------------------------------------

def u_fun(x, y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

def generate_uniform_tri_mesh(N):
    xs = np.linspace(0,1,N+1)
    ys = np.linspace(0,1,N+1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    def idx(i,j): return i*(N+1) + j
    tris = []
    for i in range(N):
        for j in range(N):
            v00 = idx(i,j); v10 = idx(i+1,j)
            v01 = idx(i,j+1); v11 = idx(i+1,j+1)
            tris.append([v00,v10,v11])
            tris.append([v00,v11,v01])
    return np.array(coords), np.array(tris,int), 1.0/N

def tri_area(a,b,c):
    # Convert 2D vectors to 3D for np.cross (NumPy 2.0 compatibility)
    vec1 = np.append(b-a, 0)  # Add z=0 component
    vec2 = np.append(c-a, 0)  # Add z=0 component
    return 0.5*abs(np.cross(vec1, vec2)[2])  # Take z-component of cross product

# ------------------------------------------------------------------
# 2. Local geometry for Q
# ------------------------------------------------------------------

# Degree-2 (exact for P₂) symmetric quadrature on reference triangle
quad_bary = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
quad_w_ref = np.array([1/6,1/6,1/6])

# Six sub-triangles per macro triangle (vertex regions)
def build_small_triangles_for_Q(tri_coords):
    v0,v1,v2 = tri_coords
    m01 = 0.5*(v0+v1); m12 = 0.5*(v1+v2); m20 = 0.5*(v2+v0)
    c = (v0+v1+v2)/3.0
    return [
        (np.array([v0,m01,c]),0), (np.array([v0,c,m20]),0),
        (np.array([v1,m12,c]),1), (np.array([v1,c,m01]),1),
        (np.array([v2,m20,c]),2), (np.array([v2,c,m12]),2)
    ]

# ------------------------------------------------------------------
# 3. Error (a)  ‖Qv_h − v_h‖_{L²}
# ------------------------------------------------------------------

def l2_error_Q_minus_v(N):
    coords,tris,h = generate_uniform_tri_mesh(N)
    vnod = u_fun(coords[:,0], coords[:,1])
    err2 = 0.0
    for t in tris:
        tri_coords = coords[t]; v_vals = vnod[t]
        for small, which_vertex in build_small_triangles_for_Q(tri_coords):
            A = tri_area(*small)
            for bary,w_ref in zip(quad_bary,quad_w_ref):
                l1,l2,l3 = bary
                x = l1*small[0]+l2*small[1]+l3*small[2]
                # barycentric coordinates wrt macro triangle
                T = np.column_stack((tri_coords[1]-tri_coords[0], tri_coords[2]-tri_coords[0]))
                rhs = x - tri_coords[0]
                ab = np.linalg.solve(T, rhs)
                lam = np.array([1-ab[0]-ab[1], ab[0], ab[1]])
                v_val = np.dot(lam, v_vals)
                Qv_val = v_vals[which_vertex]
                weight = 2*A*w_ref
                err2 += weight * (Qv_val - v_val)**2
    return h, math.sqrt(err2)

# ------------------------------------------------------------------
# 4. Error (b)  |(f,Qv_h) − (f,v_h)|
# ------------------------------------------------------------------

def rhs_error_Q_minus_v(N):
    coords,tris,h = generate_uniform_tri_mesh(N)
    vnod = u_fun(coords[:,0], coords[:,1])
    err_int = 0.0
    for t in tris:
        tri_coords = coords[t]; v_vals = vnod[t]
        for small, which_vertex in build_small_triangles_for_Q(tri_coords):
            A = tri_area(*small)
            for bary,w_ref in zip(quad_bary,quad_w_ref):
                l1,l2,l3 = bary
                x = l1*small[0]+l2*small[1]+l3*small[2]
                T = np.column_stack((tri_coords[1]-tri_coords[0], tri_coords[2]-tri_coords[0]))
                rhs = x - tri_coords[0]
                ab = np.linalg.solve(T, rhs)
                lam = np.array([1-ab[0]-ab[1], ab[0], ab[1]])
                v_val = np.dot(lam, v_vals)
                Qv_val = v_vals[which_vertex]
                f_val = u_fun(*x)
                weight = 2*A*w_ref
                err_int += weight * f_val * (Qv_val - v_val)
    return h, abs(err_int)

# ------------------------------------------------------------------
# 5. Convergence studies
# ------------------------------------------------------------------

Ns = [8,12,16,24,32,48,64]
records = []
for N in Ns:
    h, errL2 = l2_error_Q_minus_v(N)
    _, errRHS = rhs_error_Q_minus_v(N)
    records.append({"N": N, "h": h, "L2_error": errL2, "RHS_error": errRHS})
df = pd.DataFrame(records)

# Compute slopes
slope_L2, _ = np.polyfit(np.log(df.h), np.log(df.L2_error), 1)
slope_RHS, _ = np.polyfit(np.log(df.h), np.log(df.RHS_error), 1)

# ------------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------------

print(df)
print(f"\nObserved slope for ‖Qv−v‖ ≈ {slope_L2:.3f} (expected ≈ 1)")
print(f"Observed slope for |(f,Qv)-(f,v)| ≈ {slope_RHS:.3f} (expected ≈ 2)")

plt.figure(figsize=(7,5))
plt.loglog(df.h, df.L2_error, 'o-', label=r"$\|Qv_h - v_h\|_{L^2}$  (≈1)")
plt.loglog(df.h, df.RHS_error, 's--', label=r"$|(f,Qv_h)-(f,v_h)|$  (≈2)")
plt.xlabel("h"); plt.ylabel("Error")
plt.legend(); plt.title("Convergence of dual control-volume quadrature Q")
plt.grid(True, which='both', ls=':')
plt.savefig('quadrature_convergence.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("Plot saved as 'quadrature_convergence.png'")