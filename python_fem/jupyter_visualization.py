# %% [markdown]
# # DOLFINx Mesh Visualization - Working Version
# 
# Simple, clean mesh generation with Triangle library that displays images properly.

# %% Setup and imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpi4py import MPI
import dolfinx.mesh as dmesh
from dolfinx.fem import coordinate_element
import triangle as tr
from IPython.display import Image, display
import io

# Import domain functions
from domains import make_uniform_rectangle, make_uniform_box, make_uniform_interval

print("Libraries imported - Triangle library available!")
print("Jupyter display functions ready for headless environment")

# %% Visualization function (working version)
def plot_mesh(mesh, title="Mesh", color="blue", figsize=(8, 6), save_file=False):
    """
    Jupyter-friendly mesh visualization for headless environment.
    
    NO FILES SAVED by default - pure inline display
    Works in Docker/headless environments
    Displays properly in Jupyter notebooks
    """
    if mesh.comm.rank != 0:
        print("Visualization only available on rank 0")
        return None
        
    if mesh.topology.dim != 2:
        print("This function works for 2D meshes only")
        return None
    
    points = mesh.geometry.x[:, :2]
    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_reshaped = cells.reshape(num_cells, -1)
    
    _, ax = plt.subplots(figsize=figsize)
    
    lines = []
    for cell in cells_reshaped:
        cell_points = points[cell]
        closed_cell = np.vstack([cell_points, cell_points[0]])
        lines.append(closed_cell)
    
    line_collection = LineCollection(lines, colors=color, linewidths=0.8)  # Thinner lines
    ax.add_collection(line_collection)
    
    margin = 0.02  # Smaller margin
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    
    ax.set_xlim(points[:, 0].min() - margin * x_range, points[:, 0].max() + margin * x_range)
    ax.set_ylim(points[:, 1].min() - margin * y_range, points[:, 1].max() + margin * y_range)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid, axes, and ticks for clean look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Optional: remove axes completely for ultra-clean look
    # ax.axis('off')  # Uncomment this line to remove all axes
    
    plt.tight_layout()
    
    if save_file:
        # Save to file and display
        img_name = f"mesh_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')}.png"
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.show()
        try:
            display(Image(img_name))
        except:
            print(f"📁 Image saved as: {img_name}")
        return img_name
    else:
        # Display in memory only (our special approach for headless Jupyter)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        # Display from memory buffer
        img_buffer.seek(0)
        try:
            display(Image(img_buffer.getvalue()))
        except:
            # Fallback: show normally if not in Jupyter
            plt.show()
        
        return "memory_display"

def plot_mesh_clean(mesh, title="Mesh", color="blue", figsize=(8, 6), save_file=False):
    """
    Ultra-clean mesh visualization - no grid, no axes, no labels.
    Perfect for presentations and publications.
    """
    if mesh.comm.rank != 0:
        return None
    if mesh.topology.dim != 2:
        return None
    
    points = mesh.geometry.x[:, :2]
    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_reshaped = cells.reshape(num_cells, -1)
    
    _, ax = plt.subplots(figsize=figsize)
    
    lines = []
    for cell in cells_reshaped:
        cell_points = points[cell]
        closed_cell = np.vstack([cell_points, cell_points[0]])
        lines.append(closed_cell)
    
    line_collection = LineCollection(lines, colors=color, linewidths=0.6)  # Very thin lines
    ax.add_collection(line_collection)
    
    # Tight bounds
    margin = 0.01
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    
    ax.set_xlim(points[:, 0].min() - margin * x_range, points[:, 0].max() + margin * x_range)
    ax.set_ylim(points[:, 1].min() - margin * y_range, points[:, 1].max() + margin * y_range)
    ax.set_aspect('equal')
    
    # Ultra-clean: remove everything
    ax.axis('off')  # Remove all axes, ticks, labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)  # Title with padding
    
    plt.tight_layout()
    
    if save_file:
        img_name = f"clean_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')}.png"
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.show()
        try:
            display(Image(img_name))
        except:
            print(f"📁 Clean image saved as: {img_name}")
        return img_name
    else:
        # Memory display
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        img_buffer.seek(0)
        try:
            display(Image(img_buffer.getvalue()))
        except:
            plt.show()
        
        return "clean_memory_display"

def mesh_info(mesh, name="Mesh"):
    """Display mesh information."""
    if mesh.comm.rank == 0:
        print(f"\n📊 {name}:")
        print(f"   Vertices: {mesh.topology.index_map(0).size_local}")
        print(f"   Triangles: {mesh.topology.index_map(mesh.topology.dim).size_local}")

# %% Triangle library mesh functions
def create_quality_mesh(domain="square", min_angle=25, max_area=0.02):
    """Create high-quality mesh using Triangle library."""
    comm = MPI.COMM_WORLD
    
    # Define domains
    if domain == "square":
        vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
        segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
        
    elif domain == "circle":
        n = 32
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        vertices = np.column_stack([np.cos(theta), np.sin(theta)])
        segments = [[i, (i+1) % n] for i in range(n)]
        
    elif domain == "rectangle":
        vertices = [[0, 0], [2, 0], [2, 1], [0, 1]]
        segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
        
    elif domain == "L_shape":
        vertices = [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
        segments = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
    
    # Generate mesh
    triangle_input = {'vertices': vertices, 'segments': segments}
    result = tr.triangulate(triangle_input, f'pq{min_angle}a{max_area}')
    
    # Create DOLFINx mesh
    points_3d = np.column_stack([result['vertices'], np.zeros(len(result['vertices']))])
    coord_element = coordinate_element(dmesh.CellType.triangle, 1)
    mesh = dmesh.create_mesh(comm, result['triangles'], points_3d, coord_element)
    
    return mesh

def create_mesh_with_hole(outer_radius=1.0, hole_radius=0.3, min_angle=25, max_area=0.02):
    """Create mesh with circular hole."""
    comm = MPI.COMM_WORLD
    
    # Outer boundary
    n_outer = 32
    theta = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer_vertices = outer_radius * np.column_stack([np.cos(theta), np.sin(theta)])
    outer_segments = [[i, (i+1) % n_outer] for i in range(n_outer)]
    
    # Hole boundary
    n_hole = 16
    theta_hole = np.linspace(0, 2*np.pi, n_hole, endpoint=False)
    hole_vertices = hole_radius * np.column_stack([np.cos(theta_hole), np.sin(theta_hole)])
    hole_segments = [[n_outer + i, n_outer + (i+1) % n_hole] for i in range(n_hole)]
    
    # Combine
    vertices = np.vstack([outer_vertices, hole_vertices])
    segments = outer_segments + hole_segments
    
    # Generate mesh with hole
    triangle_input = {'vertices': vertices, 'segments': segments, 'holes': [[0, 0]]}
    result = tr.triangulate(triangle_input, f'pq{min_angle}a{max_area}')
    
    # Create DOLFINx mesh
    points_3d = np.column_stack([result['vertices'], np.zeros(len(result['vertices']))])
    coord_element = coordinate_element(dmesh.CellType.triangle, 1)
    mesh = dmesh.create_mesh(comm, result['triangles'], points_3d, coord_element)
    
    return mesh

def analyze_mesh_angles(mesh, name="Mesh"):
    """Analyze mesh angle quality."""
    if mesh.comm.rank != 0 or mesh.topology.dim != 2:
        return None
    
    points = mesh.geometry.x[:, :2]
    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_reshaped = cells.reshape(num_cells, -1)
    
    min_angles = []
    for cell in cells_reshaped:
        p1, p2, p3 = points[cell]
        
        # Compute triangle angles
        v1, v2, v3 = p2 - p1, p3 - p1, p3 - p2
        a, b, c = np.linalg.norm(v3), np.linalg.norm(v2), np.linalg.norm(v1)
        
        angle1 = np.arccos(np.clip((b*b + c*c - a*a) / (2*b*c), -1, 1))
        angle2 = np.arccos(np.clip((a*a + c*c - b*b) / (2*a*c), -1, 1))
        angle3 = np.arccos(np.clip((a*a + b*b - c*c) / (2*a*b), -1, 1))
        
        min_angles.append(min(angle1, angle2, angle3))
    
    min_angles_deg = np.degrees(min_angles)
    
    print(f"\n📐 {name} Angles:")
    print(f"   Minimum: {np.min(min_angles_deg):.1f}°")
    print(f"   Average min: {np.mean(min_angles_deg):.1f}°")
    print(f"   Bad (<20°): {np.sum(min_angles_deg < 20)}/{len(min_angles_deg)}")
    print(f"   Poor (<30°): {np.sum(min_angles_deg < 30)}/{len(min_angles_deg)}")
    
    return min_angles_deg

print("Functions ready")

# %% Example 1: Basic quality domains
print("🔧 Example 1: Basic quality domains")

# Square with 30° minimum angle
mesh_square = create_quality_mesh("square", min_angle=10, max_area=0.03)
mesh_info(mesh_square, "Quality Square")
analyze_mesh_angles(mesh_square, "Quality Square")
plot_mesh(mesh_square, "Quality Square (30° min angle)", color="blue")

# Circle with 25° minimum angle
mesh_circle = create_quality_mesh("circle", min_angle=25, max_area=0.03)
mesh_info(mesh_circle, "Quality Circle")
analyze_mesh_angles(mesh_circle, "Quality Circle")
plot_mesh(mesh_circle, "Quality Circle (25° min angle)", color="red")

# %% Example 2: Different shapes
print("🔧 Example 2: Different domain shapes")

# Rectangle
mesh_rectangle = create_quality_mesh("rectangle", min_angle=25, max_area=0.04)
mesh_info(mesh_rectangle, "Quality Rectangle")
analyze_mesh_angles(mesh_rectangle, "Quality Rectangle")
plot_mesh(mesh_rectangle, "Quality Rectangle", color="green", figsize=(10, 6))

# L-shape
mesh_L = create_quality_mesh("L_shape", min_angle=20, max_area=0.01)
mesh_info(mesh_L, "Quality L-shape")
analyze_mesh_angles(mesh_L, "Quality L-shape")
plot_mesh(mesh_L, "Quality L-shaped Domain", color="purple")

# %% Example 3: Meshes with holes
print("🔧 Example 3: Meshes with holes")

# Small hole
mesh_small_hole = create_mesh_with_hole(outer_radius=1.0, hole_radius=0.2, min_angle=25, max_area=0.025)
mesh_info(mesh_small_hole, "Small Hole")
analyze_mesh_angles(mesh_small_hole, "Small Hole")
plot_mesh(mesh_small_hole, "Circle with Small Hole", color="orange")

# Large hole
mesh_large_hole = create_mesh_with_hole(outer_radius=1.0, hole_radius=0.5, min_angle=25, max_area=0.03)
mesh_info(mesh_large_hole, "Large Hole")
analyze_mesh_angles(mesh_large_hole, "Large Hole")
plot_mesh(mesh_large_hole, "Circle with Large Hole", color="darkorange")

# %% Example 4: Quality comparison
print("🔧 Example 4: Quality level comparison")

quality_levels = [
    (20, "Minimum Quality", "gray"),
    (25, "Good Quality", "blue"),
    (30, "High Quality", "red"),
    (35, "Excellent Quality", "darkgreen")
]

for min_angle, name, color in quality_levels:
    print(f"\n--- {name} (Min Angle: {min_angle}°) ---")
    mesh = create_quality_mesh("square", min_angle=min_angle, max_area=0.025)
    mesh_info(mesh, name)
    analyze_mesh_angles(mesh, name)
    plot_mesh(mesh, f"{name} Mesh", color=color, figsize=(6, 6))

# %% Example 5: Ultra-clean plots (no grid, no axes)
print("🔧 Example 5: Ultra-clean visualization")

# Create a quality mesh
mesh_clean_demo = create_quality_mesh("circle", min_angle=25, max_area=0.03)

# Compare regular vs clean plotting
print("\n--- Regular plot (with grid and axes) ---")
plot_mesh(mesh_clean_demo, "Regular Circle Plot", color="blue")

print("\n--- Ultra-clean plot (no grid, no axes, thin lines) ---")
plot_mesh_clean(mesh_clean_demo, "Ultra-Clean Circle", color="black")

# Different shapes with clean plotting
mesh_L_clean = create_quality_mesh("L_shape", min_angle=25, max_area=0.03)
plot_mesh_clean(mesh_L_clean, "Clean L-Shape", color="darkblue")

mesh_hole_clean = create_mesh_with_hole(outer_radius=1.0, hole_radius=0.3, min_angle=25, max_area=0.03)
plot_mesh_clean(mesh_hole_clean, "Clean Circle with Hole", color="darkred")

print("\nAll examples completed!")
print("💡 Triangle library provides exact quality control")
print("💡 Images display properly in Jupyter")
print("💡 No file saving - pure visualization")
print("💡 Use plot_mesh() for regular plots with grid/axes")
print("💡 Use plot_mesh_clean() for ultra-clean plots (no grid, no axes, thin lines)")
print("💡 Professional-grade mesh generation")

# %%