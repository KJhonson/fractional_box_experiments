# %%

import numpy as np
import dolfinx
from dolfinx import mesh, fem
from dolfinx.mesh import create_mesh
from mpi4py import MPI
import ufl
import basix.ufl
import matplotlib.pyplot as plt
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Warning: shapely not available. Using matplotlib for region visualization.")

def create_dual_region(degree, dof_index=0, nx=300, ny=300):
    """
    Create dual region for a given polynomial degree.
    
    Parameters:
    -----------
    degree : int
        Polynomial degree (1, 2, 3, ...)
    dof_index : int
        Which DOF to visualize (default: 0, first vertex)
    nx, ny : int
        Grid resolution for sampling
    """
    ##########################################
    # 1. Criar malha e espaço P{degree}
    ##########################################
    
    # Triângulo de referência
    coords = np.array([[0., 0.],
                       [1., 0.],
                       [0., 1.]], dtype=np.float64)
    
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    
    # Create coordinate element for triangle (3D coordinates, z=0)
    coords_3d = np.column_stack([coords, np.zeros(len(coords))])
    coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    domain = ufl.Mesh(coord_element)
    domain = create_mesh(MPI.COMM_WORLD, cells, coords_3d, domain)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    ##########################################
    # 2. Construir a base nodal ℓ_i
    ##########################################
    
    phi = fem.Function(V)
    phi.x.array[:] = 0
    
    # Print DOF information
    dof_coords_all = V.tabulate_dof_coordinates()
    n_dofs = len(dof_coords_all)
    
    # Use specified DOF index (clamp to valid range)
    i = min(dof_index, n_dofs - 1)
    phi.x.array[i] = 1.0
    phi.x.scatter_forward()
    
    ##########################################
    # 3. Calcular peso w_i = ∫ ℓ_i
    ##########################################
    
    w_i = fem.assemble_scalar(fem.form(phi * ufl.dx(domain)))
    
    ##########################################
    # 4. Amostrar ℓ_i em uma malha cartesiana
    ##########################################
    
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    vals = np.zeros((nx, ny))
    
    # Create points inside triangle
    points_2d = []
    point_indices = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if x >= 0 and y >= 0 and x + y <= 1:
                points_2d.append([x, y])
                point_indices.append((ix, iy))
    
    if len(points_2d) > 0:
        points_2d = np.array(points_2d, dtype=np.float64)
        dof_coords = V.tabulate_dof_coordinates()
        phi_vals = phi.x.array
        
        # Use scipy's griddata for interpolation
        from scipy.interpolate import griddata
        phi_interp = griddata(dof_coords[:, :2], phi_vals, points_2d, 
                             method='linear', fill_value=np.nan)
        
        for idx, (ix, iy) in enumerate(point_indices):
            vals[ix, iy] = phi_interp[idx] if not np.isnan(phi_interp[idx]) else np.nan
    
    ##########################################
    # 5. Encontrar c_i tal que área({ℓ_i ≥ c_i}) = w_i
    ##########################################
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    area_pixel = dx * dy
    
    vals_flat = vals[~np.isnan(vals)].flatten()
    unique_vals = np.unique(vals_flat)
    unique_vals_sorted = np.sort(unique_vals)[::-1]
    
    cum_areas = []
    for threshold in unique_vals_sorted:
        mask = vals >= threshold
        area_above_threshold = np.sum(mask) * area_pixel
        cum_areas.append(area_above_threshold)
    
    cum_areas = np.array(cum_areas)
    w_i_abs = abs(w_i)
    idx = np.argmin(np.abs(cum_areas - w_i_abs))
    c_i = unique_vals_sorted[idx]
    
    ##########################################
    # 6. Construir a região B_i = {ℓ_i ≥ c_i}
    ##########################################
    
    mask = vals >= c_i
    
    if HAS_SHAPELY:
        polys = []
        for ix in range(nx-1):
            for iy in range(ny-1):
                cell_vals = vals[ix:ix+2, iy:iy+2]
                if np.all(cell_vals >= c_i):
                    x0, x1 = xs[ix], xs[ix+1]
                    y0, y1 = ys[iy], ys[iy+1]
                    p = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
                    polys.append(p)
        B_i_region = unary_union(polys) if polys else None
    else:
        B_i_region = mask
    
    return xs, ys, vals, c_i, w_i, B_i_region, dof_coords_all[i, :2], i

##########################################
# Compare different degrees
##########################################

degrees = [1, 2, 3, 4, 5]  # You can add more: 6, 7, 8, 9, 10
dof_index = 0  # First vertex DOF

fig, axes = plt.subplots(2, len(degrees), figsize=(5*len(degrees), 10))

for col, degree in enumerate(degrees):
    print(f"\nProcessing degree {degree}...")
    try:
        xs, ys, vals, c_i, w_i, B_i_region, dof_pos, dof_idx = create_dual_region(
            degree, dof_index=dof_index, nx=200, ny=200
        )
        
        # Top row: Function values
        ax1 = axes[0, col]
        im1 = ax1.imshow(vals.T, origin='lower', extent=[0, 1, 0, 1], 
                         aspect='equal', cmap='viridis', interpolation='bilinear')
        triangle_x = [0, 1, 0, 0]
        triangle_y = [0, 0, 1, 0]
        ax1.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)
        ax1.set_title(f'P{degree} - Function φ_{dof_idx}\nDOF at ({dof_pos[0]:.2f}, {dof_pos[1]:.2f})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # Bottom row: Dual region
        ax2 = axes[1, col]
        ax2.set_aspect('equal')
        
        # Background function
        im2 = ax2.imshow(vals.T, origin='lower', extent=[0, 1, 0, 1], 
                         aspect='equal', cmap='gray', alpha=0.2, interpolation='bilinear')
        
        # Triangle boundary
        ax2.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)
        
        # Contour
        max_val = np.nanmax(vals)
        min_val = np.nanmin(vals)
        if max_val > min_val and not np.isnan(c_i) and min_val < c_i < max_val:
            try:
                cs = ax2.contour(xs, ys, vals.T, levels=[c_i], colors="red", linewidths=1.5)
            except:
                pass
        
        # Dual region
        if HAS_SHAPELY and B_i_region is not None:
            if isinstance(B_i_region, Polygon):
                x, y = B_i_region.exterior.xy
                ax2.fill(x, y, color="blue", alpha=0.4)
            elif isinstance(B_i_region, MultiPolygon):
                for p in B_i_region:
                    x, y = p.exterior.xy
                    ax2.fill(x, y, color="blue", alpha=0.4)
        else:
            max_val = np.nanmax(vals)
            if max_val > c_i:
                try:
                    ax2.contourf(xs, ys, vals.T, levels=[c_i, max_val], 
                               colors=['blue'], alpha=0.4)
                except:
                    mask_plot = np.ma.masked_where(vals.T < c_i, vals.T)
                    ax2.imshow(mask_plot, origin='lower', extent=[0, 1, 0, 1], 
                              aspect='equal', cmap='Blues', alpha=0.4, interpolation='bilinear')
        
        ax2.set_title(f'P{degree} - Dual Region B_{dof_idx}\nc_i={c_i:.3f}, w_i={w_i:.4e}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
    except Exception as e:
        print(f"Error processing degree {degree}: {e}")
        axes[0, col].text(0.5, 0.5, f'Error\nP{degree}', 
                         ha='center', va='center', transform=axes[0, col].transAxes)
        axes[1, col].text(0.5, 0.5, f'Error\nP{degree}', 
                         ha='center', va='center', transform=axes[1, col].transAxes)

plt.suptitle(f'Comparison of Dual Regions for Different Polynomial Degrees\n(DOF {dof_index} at first vertex)', 
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n" + "="*70)
print("Note: As the polynomial degree increases:")
print("  - The basis functions become more oscillatory")
print("  - The dual cells become more curved and complex")
print("  - Higher degrees show more intricate boundary shapes")
print("="*70)

# %%

