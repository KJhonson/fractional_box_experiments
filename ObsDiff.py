"""
FEM Torus Solver with Anisotropic Diffusion
=========================================

This script implements a finite element solver for anisotropic diffusion on a torus.
It compares two different approaches:
1. Standard FEM Laplace operator
2. Anisotropic FEM diffusion

The code demonstrates:
- Mesh generation with Gmsh
- Anisotropic diffusion tensor definition
- FEM implementation
- Parallel computation with MPI
- Solution visualization with varying angle multipliers
- Video generation script for ParaView
"""

# =============================================================================
# Import Libraries
# =============================================================================
# Standard libraries
import os
import numpy as np
import gmsh
import meshio

# Parallel computing
from mpi4py import MPI
from petsc4py import PETSc

# Finite element libraries
from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_matrix
from ufl import TrialFunction, TestFunction, inner, grad, dx, dot, SpatialCoordinate, sqrt, as_matrix

# =============================================================================
# Configuration Parameters
# =============================================================================
# Rotation parameters for anisotropic diffusion
axis_u = 1.0  # x-component of rotation axis
axis_v = 1.0  # y-component of rotation axis
axis_w = 1.0  # z-component of rotation axis

# Angle multiplier values to test
angle_multipliers = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# Output directory
output_dir = "torus_diff"
os.makedirs(output_dir, exist_ok=True)

# Video parameters
frames_per_solution = 30  # Number of frames for each solution
video_fps = 30           # Frames per second for the output video

# =============================================================================
# Mesh Generation with Gmsh
# =============================================================================
# Initialize Gmsh
gmsh.initialize()

# Define torus parameters
R, r, mesh_size = 1.0, 0.3, 0.04  # Major radius, minor radius, mesh size
gmsh.model.add("torus")
_ = gmsh.model.occ.addTorus(0,0,0, R, r)
gmsh.model.occ.synchronize()

# Set mesh size and generate mesh
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
gmsh.model.addPhysicalGroup(2, [1], name="Surface")
gmsh.model.mesh.generate(2)

# Save mesh to file
msh_file = f"{output_dir}/torus.msh"
gmsh.write(msh_file)
gmsh.finalize()

# =============================================================================
# Import Mesh into DOLFINx
# =============================================================================
# Set up MPI communicator
comm = MPI.COMM_WORLD

# Read mesh from Gmsh file
domain, cell_markers, facet_markers = gmshio.read_from_msh(msh_file, comm)

# =============================================================================
# Function Spaces and Variational Forms
# =============================================================================
# Create function space for solution
element = ("Lagrange", 1)
V = fem.functionspace(domain, element)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define standard Laplace and mass forms
a_form = inner(grad(u), grad(v)) * dx  # Laplace operator
m_form = inner(u, v) * dx              # Mass matrix

# =============================================================================
# Define Anisotropic Diffusion Tensor
# =============================================================================
# Create function space for diffusion tensor
gdim = domain.geometry.dim
V_D = fem.functionspace(domain, ("Lagrange", 1, (gdim, gdim)))
D = Function(V_D)

# Get mesh coordinates
coords = domain.geometry.x
nvert = coords.shape[0]
D_local = np.empty((nvert, 3, 3), dtype=np.float64)

# Define diffusion tensor at each vertex
for i, (x, y, z) in enumerate(coords):
    # Define eigenvalues (strictly positive)
    lam1 = 1.0 + x**2
    lam2 = 2.0 + y**2
    lam3 = 3.0 + z**2
    Lambda = np.diag((lam1, lam2, lam3))
    
    # Define rotation angle based on position
    theta = x + y
    cth, sth = np.cos(theta), np.sin(theta)
    
    # Create rotation matrix
    Rmat = np.array([[ cth, -sth, 0.0],
                     [ sth,  cth, 0.0],
                     [0.0,   0.0, 1.0]])
    
    # Compute diffusion tensor: D = R Λ R^T
    D_local[i] = Rmat @ Lambda @ Rmat.T

# Assign values to DOLFINx Function
D.x.array[:] = D_local.ravel()
D.x.scatter_forward()

# =============================================================================
# Create random right-hand side vector (same for all angle multipliers)
# =============================================================================
n_dofs = V.dofmap.index_map.size_local
# Set a fixed seed for reproducibility
np.random.seed(42)
# Create random right-hand side vector
b = PETSc.Vec().createWithArray(np.random.randn(n_dofs), comm=comm)

# =============================================================================
# Solve standard Laplace system
# =============================================================================
# Assemble standard Laplace and mass matrices
A = assemble_matrix(fem.form(a_form)); A.assemble()
M = assemble_matrix(fem.form(m_form)); M.assemble()

# Solve standard FEM Laplace + mass
ksp1 = PETSc.KSP().create(comm=comm)
L1 = A.copy(); L1.axpy(1.0, M)
ksp1.setOperators(L1); ksp1.setType(PETSc.KSP.Type.CG)
ksp1.getPC().setType(PETSc.PC.Type.JACOBI)
x1 = b.duplicate(); ksp1.solve(b, x1)

# Create standard solution function
u_fem = Function(V)
u_fem.x.array[:] = x1.getArray()
u_fem.x.scatter_forward()
u_fem.name = "u_fem"

# =============================================================================
# Solve anisotropic systems with different angle multipliers
# =============================================================================
# Create a list to store all anisotropic solutions
u_diff_solutions = []

# Import UFL functions for symbolic computation
from ufl import cos, sin
x = SpatialCoordinate(domain)

# Define rotation axis
norm_axis = sqrt(axis_u**2 + axis_v**2 + axis_w**2)
ax = axis_u / norm_axis
ay = axis_v / norm_axis
az = axis_w / norm_axis

# Loop through different angle multipliers
for angle_multiplier in angle_multipliers:
    print(f"Solving with angle_multiplier = {angle_multiplier}")
    
    # Define rotation angle based on position and current multiplier
    theta = angle_multiplier * (x[0] + x[1] + x[2])
    cost = cos(theta)
    sint = sin(theta)
    
    # Create rotation matrix using Rodrigues' formula
    Rmat = as_matrix([
      [cost + (1-cost)*ax*ax,     (1-cost)*ax*ay - az*sint,  (1-cost)*ax*az + ay*sint],
      [(1-cost)*ay*ax + az*sint,   cost + (1-cost)*ay*ay,    (1-cost)*ay*az - ax*sint],
      [(1-cost)*az*ax - ay*sint,   (1-cost)*az*ay + ax*sint,  cost + (1-cost)*az*az]
    ])
    
    # Define anisotropic diffusion tensor D(x) = R Λ R^T
    lam1 = 1.0 + x[0]*x[0]
    lam2 = 2.0 + x[1]*x[1]
    lam3 = 3.0 + x[2]*x[2]
    Lambda = as_matrix([[lam1, 0, 0], [0, lam2, 0], [0, 0, lam3]])
    D_ufl = Rmat * Lambda * Rmat.T
    
    # Define anisotropic diffusion form
    aD_form = inner(dot(D_ufl, grad(u)), grad(v)) * dx
    
    # Assemble anisotropic diffusion matrix
    AD = assemble_matrix(fem.form(aD_form)); AD.assemble()
    
    # Solve anisotropic FEM + mass
    ksp2 = PETSc.KSP().create(comm=comm)
    L2 = AD.copy(); L2.axpy(1.0, M)
    ksp2.setOperators(L2); ksp2.setType(PETSc.KSP.Type.CG)
    ksp2.getPC().setType(PETSc.PC.Type.JACOBI)
    x2 = b.duplicate(); ksp2.solve(b, x2)
    
    # Create anisotropic solution function
    u_diff = Function(V)
    u_diff.x.array[:] = x2.getArray()
    u_diff.x.scatter_forward()
    u_diff.name = f"u_diff_{angle_multiplier}"
    
    # Add to list of solutions
    u_diff_solutions.append(u_diff)

# =============================================================================
# Export Results
# =============================================================================
# Export standard solution
with io.VTKFile(comm, f"{output_dir}/standard_solution.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_fem)

# Export all anisotropic solutions to a single PVD file
with io.VTKFile(comm, f"{output_dir}/anisotropic_solutions.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    for u_diff in u_diff_solutions:
        vtk.write_function(u_diff)

# =============================================================================
# Create ParaView Python Script for Video Generation
# =============================================================================
# Create ParaView Python script
pv_script = f"""
from paraview.simple import *
import os

# Load the solutions
standard = PVDReader(FileName="{output_dir}/standard_solution.pvd")
anisotropic = PVDReader(FileName="{output_dir}/anisotropic_solutions.pvd")

# Create a layout
layout = GetLayout()
layout.SetSize(1920, 1080)

# Create views
view1 = CreateRenderView()
view1.ViewSize = [960, 1080]
view1.CameraPosition = [0, 0, 5]
view1.CameraFocalPoint = [0, 0, 0]
view1.CameraViewUp = [0, 1, 0]

view2 = CreateRenderView()
view2.ViewSize = [960, 1080]
view2.CameraPosition = [0, 0, 5]
view2.CameraFocalPoint = [0, 0, 0]
view2.CameraViewUp = [0, 1, 0]

# Add views to layout
layout.AddView(view1)
layout.AddView(view2)

# Show data in views
standardDisplay = Show(standard, view1)
anisotropicDisplay = Show(anisotropic, view2)

# Set color map
standardDisplay.SetScalarBarVisibility(view1, True)
anisotropicDisplay.SetScalarBarVisibility(view2, True)

# Create animation
animation = GetAnimationScene()
animation.NumberOfFrames = {len(angle_multipliers) * frames_per_solution}
animation.StartTime = 0
animation.EndTime = {len(angle_multipliers)}

# Save animation
SaveAnimation("{output_dir}/solution_animation.mp4", view1, view2, 
             ImageResolution=[1920, 1080],
             FrameRate={video_fps},
             Compression=True)
"""

# Write ParaView script to file
pv_script_path = f"{output_dir}/create_video.py"
with open(pv_script_path, "w") as f:
    f.write(pv_script)

print("Done! Outputs available in:", output_dir)
print("\nTo create the video:")
print("1. Open ParaView")
print("2. Go to Tools > Python Shell")
print("3. Click on 'Run Script'")
print("4. Select the file:", pv_script_path)
print("5. The video will be saved as:", f"{output_dir}/solution_animation.mp4")
