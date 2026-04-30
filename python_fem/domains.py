# %%
# domains.py
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.io import gmshio

# ---------- Mesh builders ----------

def make_uniform_interval(nx=64, p0=0.0, p1=1.0, comm=MPI.COMM_WORLD):
    """
    Uniform mesh on an interval [p0, p1] in 1D.
    """
    return dmesh.create_interval(comm, nx, [p0, p1])

def make_uniform_box(nx=8, ny=8, nz=8, 
                     p0=(0.0, 0.0, 0.0), p1=(1.0, 1.0, 1.0),
                     cell_type="tetrahedron", comm=MPI.COMM_WORLD):
    """
    Uniform mesh on a box [p0, p1] in 3D.
    cell_type: 'tetrahedron' or 'hexahedron'
    """
    ct = dmesh.CellType.tetrahedron if cell_type == "tetrahedron" else dmesh.CellType.hexahedron
    return dmesh.create_box(comm, [np.array(p0), np.array(p1)], [nx, ny, nz], cell_type=ct)
# %%
def make_uniform_rectangle(nx=64, ny=64,
                           p0=(0.0, 0.0), p1=(1.0, 1.0),
                           cell_type="triangle",
                           comm=MPI.COMM_WORLD):
    """
    Uniform mesh on a rectangle [p0, p1] in 2D.
    cell_type: 'triangle' or 'quadrilateral'
    """
    ct = dmesh.CellType.triangle if cell_type == "triangle" else dmesh.CellType.quadrilateral
    mesh = dmesh.create_rectangle(comm, [np.array(p0), np.array(p1)], [nx, ny], cell_type=ct)
    return mesh


def load_gmsh_mesh(msh_path, comm=MPI.COMM_WORLD, gdim=None):
    """
    Load any (1D/2D/3D) volume/surface mesh created in Gmsh.
    Returns: mesh, cell_tags (may be None), facet_tags (may be None)
    """
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(msh_path, comm, gdim=gdim)
    return mesh, cell_tags, facet_tags

# (Optional) If you have meshes in other formats, convert to .msh or .xdmf with meshio externally.
# Example conversion (run once in a utility script/notebook):
#
# import meshio
# m = meshio.read("polygon.geojson")   # or .stl/.vtk
# meshio.write("polygon.msh", m)       # then use load_gmsh_mesh("polygon.msh", ...)
#

# ---------- Boundary tagging utilities ----------

def tag_all_exterior_facets(mesh):
    """
    Return MeshTags that mark ALL exterior facets with id=1 (works in 1D/2D/3D).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    facet_tags = dmesh.meshtags(mesh, fdim, facets.astype(np.int32),
                                np.ones(len(facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    return facet_tags, ds

def dirichlet_bc_from_tags(V, facet_tags, ids=(1,)):
    """
    Homogeneous Dirichlet u=0 on the facets with given 'ids'.
    """
    fdim = V.mesh.topology.dim - 1
    sel = np.concatenate([facet_tags.find(i) for i in np.atleast_1d(ids)])
    dofs = fem.locate_dofs_topological(V, fdim, sel)
    uD = fem.Function(V)  # zero
    return fem.dirichletbc(uD, dofs)

# ---------- Visualization utilities ----------

def visualize_mesh(mesh, filename="mesh_plot.png", title="Mesh Visualization", 
                   show_edges=True, style="wireframe", color="black"):
    """
    Visualize DOLFINx mesh using PyVista (works in headless environments).
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh to visualize
    filename : str
        Output filename for the image
    title : str
        Plot title
    show_edges : bool
        Whether to show mesh edges
    style : str
        Visualization style ("wireframe", "surface", "points")
    color : str
        Mesh color
    
    Returns:
    --------
    str : Path to the saved image file
    """
    import pyvista as pv
    
    if mesh.comm.rank == 0:
        # Extract mesh data
        points = mesh.geometry.x
        cells = mesh.topology.connectivity(mesh.topology.dim, 0).array
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cells_pv = cells.reshape(num_cells, -1)
        
        # Determine cell type
        if mesh.topology.dim == 2:
            cell_type = pv.CellType.TRIANGLE if cells_pv.shape[1] == 3 else pv.CellType.QUAD
        else:
            cell_type = pv.CellType.TETRA if cells_pv.shape[1] == 4 else pv.CellType.HEXAHEDRON
            
        grid = pv.UnstructuredGrid({cell_type: cells_pv}, points)
        
        # Configure PyVista for headless rendering
        pv.set_plot_theme("document")
        
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(grid, show_edges=show_edges, style=style, color=color, line_width=2)
        pl.add_title(title, font_size=16)
        
        # Set appropriate camera position
        if mesh.topology.dim == 2:
            pl.camera_position = 'xy'  # Top-down for 2D
        else:
            pl.camera_position = 'iso'  # Isometric for 3D
            
        pl.background_color = 'white'
        
        # Save image
        pl.screenshot(filename, window_size=[800, 600])
        pl.close()
        
        print(f"✅ Mesh visualization saved to: {filename}")
        return filename
    else:
        print("ℹ️  Visualization only available on rank 0")
        return None

# %%
