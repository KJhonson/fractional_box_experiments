"""
fem

fem
FEM Torus Solver with Anisotropic Diffusion
=========================================

This script implements a finite element solver for anisotropic diffusion on a torus.
It compares three different approaches:
1. Standard FEM Laplace operator
2. Anisotropic FEM diffusion
3. Asymmetric Finite Volume Method (FVM)

The code demonstrates:
- Mesh generation with Gmsh
- Anisotropic diffusion tensor definition
- FEM and FVM implementations
- Parallel computation with MPI
- Solution visualization
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
angle_multiplier = 1.0  # Multiplier for rotation angle

# Output directory
output_dir = "torusFVM_v3"
os.makedirs(output_dir, exist_ok=True)

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
# Define Anisotropic Diffusion Form in UFL
# =============================================================================
# Import UFL functions for symbolic computation
from ufl import cos, sin
x = SpatialCoordinate(domain)

# Define rotation axis and angle with user parameters
norm_axis = sqrt(axis_u**2 + axis_v**2 + axis_w**2)
ax = axis_u / norm_axis
ay = axis_v / norm_axis
az = axis_w / norm_axis

# Define rotation angle based on position
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

# =============================================================================
# Assemble FEM Matrices
# =============================================================================
# Assemble standard Laplace, mass, and anisotropic diffusion matrices
A = assemble_matrix(fem.form(a_form));   A.assemble()
M = assemble_matrix(fem.form(m_form));   M.assemble()
AD = assemble_matrix(fem.form(aD_form)); AD.assemble()

# =============================================================================
# Asymmetric FVM Assembly
# =============================================================================
# Create PETSc matrix for FVM
n_dofs = V.dofmap.index_map.size_local
FVM = PETSc.Mat().create(comm=comm)
FVM.setSizes([n_dofs, n_dofs])
FVM.setType(PETSc.Mat.Type.AIJ)
FVM.setFromOptions()
FVM.setUp()

# Set up connectivity for triangles
dim = domain.topology.dim
domain.topology.create_connectivity(dim, 0)
cell_to_vertex = domain.topology.connectivity(dim, 0)
coords3 = domain.geometry.x

# Assemble FVM matrix cell by cell
for cell in range(domain.topology.index_map(dim).size_local):
    # Get vertices of the cell
    verts = cell_to_vertex.links(cell)
    if len(verts) != dim + 1:
        continue
    i, j, k = verts
    v0, v1, v2 = coords3[[i, j, k], :]

    # Calculate triangle normal and area
    e_ij = v1 - v0
    e_ik = v2 - v0
    nT_vec = np.cross(e_ij, e_ik)
    area2 = np.linalg.norm(nT_vec)
    if area2 < 1e-12:
        continue
    nT = nT_vec / area2
    area = 0.5 * area2

    # Calculate gradients of local basis functions
    grad_phi = [
        np.cross(nT, (v2 - v1)) / area2,
        np.cross(nT, (v0 - v2)) / area2,
        np.cross(nT, (v1 - v0)) / area2
    ]

    # Calculate cell centroid
    c = (v0 + v1 + v2) / 3.0
    
    # Calculate cell-averaged diffusion tensor
    D_cell = (D_local[i] + D_local[j] + D_local[k]) / 3.0

    # Calculate rotation parameters at centroid
    xc, yc, zc = c
    axis = np.array([float(axis_u), float(axis_v), float(axis_w)])
    axis = axis / np.linalg.norm(axis)
    ax_c, ay_c, az_c = axis
    theta_c = float(angle_multiplier)*(xc + yc + zc)
    cost_c = np.cos(theta_c)
    sint_c = np.sin(theta_c)

    # Create rotation matrix at centroid
    Rmat_c = np.array([
      [cost_c + (1-cost_c)*ax_c*ax_c,     (1-cost_c)*ax_c*ay_c - az_c*sint_c,  (1-cost_c)*ax_c*az_c + ay_c*sint_c],
      [(1-cost_c)*ay_c*ax_c + az_c*sint_c, cost_c + (1-cost_c)*ay_c*ay_c,      (1-cost_c)*ay_c*az_c - ax_c*sint_c],
      [(1-cost_c)*az_c*ax_c - ay_c*sint_c, (1-cost_c)*az_c*ay_c + ax_c*sint_c,  cost_c + (1-cost_c)*az_c*az_c]
    ])

    # Assemble fluxes for each vertex
    to_idx = [i, j, k]
    for local_i, vi in enumerate((v0, v1, v2)):
        row = to_idx[local_i]
        
        # Calculate midpoints of adjacent edges
        if local_i == 0:
            mids = [0.5*(v0 + v1), 0.5*(v0 + v2)]
        elif local_i == 1:
            mids = [0.5*(v1 + v2), 0.5*(v1 + v0)]
        else:
            mids = [0.5*(v2 + v0), 0.5*(v2 + v1)]

        # Calculate fluxes for each midpoint
        for mid in mids:
            d = c - mid
            L = np.linalg.norm(d)
            if L < 1e-12:
                continue
                
            # Calculate normal vector
            eta = np.cross(nT, d)
            eta /= np.linalg.norm(eta)
            
            # Ensure outward normal
            if np.dot(eta, vi - mid) > 0:
                eta = -eta

            # Calculate contributions for each trial basis
            for local_j in range(3):
                col = to_idx[local_j]
                flux = float((D_cell.dot(grad_phi[local_j])) @ eta) * L
                FVM.setValue(row, col, -flux, PETSc.InsertMode.ADD)

# Finalize FVM matrix assembly
FVM.assemble()

# =============================================================================
# Solve Systems and Export Results
# =============================================================================
# Create random right-hand side vector
b = PETSc.Vec().createWithArray(np.random.randn(n_dofs), comm=comm)

# Solve system 1: Standard FEM Laplace + mass
ksp1 = PETSc.KSP().create(comm=comm)
L1 = A.copy(); L1.axpy(1.0, M)
ksp1.setOperators(L1); ksp1.setType(PETSc.KSP.Type.CG)
ksp1.getPC().setType(PETSc.PC.Type.JACOBI)
x1 = b.duplicate(); ksp1.solve(b, x1)

# Solve system 2: Anisotropic FEM + mass
ksp2 = PETSc.KSP().create(comm=comm)
L2 = AD.copy(); L2.axpy(1.0, M)
ksp2.setOperators(L2); ksp2.setType(PETSc.KSP.Type.CG)
ksp2.getPC().setType(PETSc.PC.Type.JACOBI)
x2 = b.duplicate(); ksp2.solve(b, x2)

# Solve system 3: Asymmetric FVM + mass
ksp3 = PETSc.KSP().create(comm=comm)
L3 = FVM.copy(); L3.axpy(1.0, M)
ksp3.setOperators(L3); ksp3.setType(PETSc.KSP.Type.CG)
ksp3.getPC().setType(PETSc.PC.Type.JACOBI)
x3 = b.duplicate(); ksp3.solve(b, x3)

# =============================================================================
# Create Solution Functions and Export
# =============================================================================
# Create DOLFINx Functions for each solution
u_fem = Function(V);   u_fem.x.array[:] = x1.getArray(); u_fem.x.scatter_forward(); u_fem.name = "u_fem"
u_diff= Function(V);  u_diff.x.array[:] = x2.getArray(); u_diff.x.scatter_forward(); u_diff.name= "u_diff"
u_asym= Function(V);  u_asym.x.array[:] = x3.getArray(); u_asym.x.scatter_forward(); u_asym.name= "u_fvm_asym"

# Compute all possible differences between the solutions
u_fem_minus_u_diff = Function(V)
u_fem_minus_u_diff.x.array[:] = u_fem.x.array - u_diff.x.array
u_fem_minus_u_diff.x.scatter_forward()
u_fem_minus_u_diff.name = "u_fem_minus_u_diff"

u_fem_minus_u_asym = Function(V)
u_fem_minus_u_asym.x.array[:] = u_fem.x.array - u_asym.x.array
u_fem_minus_u_asym.x.scatter_forward()
u_fem_minus_u_asym.name = "u_fem_minus_u_asym"

u_diff_minus_u_fem = Function(V)
u_diff_minus_u_fem.x.array[:] = u_diff.x.array - u_fem.x.array
u_diff_minus_u_fem.x.scatter_forward()
u_diff_minus_u_fem.name = "u_diff_minus_u_fem"

u_diff_minus_u_asym = Function(V)
u_diff_minus_u_asym.x.array[:] = u_diff.x.array - u_asym.x.array
u_diff_minus_u_asym.x.scatter_forward()
u_diff_minus_u_asym.name = "u_diff_minus_u_asym"

u_asym_minus_u_fem = Function(V)
u_asym_minus_u_fem.x.array[:] = u_asym.x.array - u_fem.x.array
u_asym_minus_u_fem.x.scatter_forward()
u_asym_minus_u_fem.name = "u_asym_minus_u_fem"

u_asym_minus_u_diff = Function(V)
u_asym_minus_u_diff.x.array[:] = u_asym.x.array - u_diff.x.array
u_asym_minus_u_diff.x.scatter_forward()
u_asym_minus_u_diff.name = "u_asym_minus_u_diff"

# Export all solutions and all differences to a single PVD file for ParaView
with io.VTKFile(comm, f"{output_dir}/solutions.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_fem)
    vtk.write_function(u_diff)
    vtk.write_function(u_asym)
    vtk.write_function(u_fem_minus_u_diff)
    vtk.write_function(u_fem_minus_u_asym)
    vtk.write_function(u_diff_minus_u_fem)
    vtk.write_function(u_diff_minus_u_asym)
    vtk.write_function(u_asym_minus_u_fem)
    vtk.write_function(u_asym_minus_u_diff)

print("Done! Outputs available in:", output_dir)
