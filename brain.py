# import os
# import numpy as np
# import meshio
# import ufl
# import basix.ufl
# from mpi4py import MPI
# from petsc4py import PETSc
# import dolfinx.fem.petsc as petsc
# from dolfinx import fem, io
# from dolfinx.mesh import create_mesh, CellType
# from dolfinx.io import XDMFFile

import os
import numpy as np
import meshio
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc as petsc
from dolfinx import fem, io
from dolfinx.mesh import create_mesh

# =============================================================================
# Setup and Configuration
# =============================================================================
# Define paths for input mesh and output directory
obj_path = "/home/dolfinx/shared/brain.obj"
output_dir = "/home/dolfinx/shared/brain"
os.makedirs(output_dir, exist_ok=True)  # Creates output directory if it doesn't exist

# =============================================================================
# Mesh Processing
# =============================================================================
# Read the .obj file and extract geometric data
mesh_data = meshio.read(obj_path)  # Reads 3D mesh from .obj file
points = mesh_data.points.astype(np.float64)  # Convert vertices to double precision
cells = None
# Extract triangle elements from the mesh
for cb in mesh_data.cells:
    if cb.type == "triangle":
        cells = cb.data.astype(np.int32)  # Convert cell indices to 32-bit integers
        break
if cells is None:
    raise RuntimeError("No triangle cells found in the .obj file!")

# =============================================================================
# DOLFINx Mesh Creation
# =============================================================================
# Create a first-order Lagrange element for 3D coordinates
# This is necessary because the surface is embedded in 3D space
coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))  # Creates finite element for 3D coordinates
domain = ufl.Mesh(coord_element)  # Defines the computational domain
mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)  # Creates parallel mesh for finite element analysis

# =============================================================================
# Finite Element Setup
# =============================================================================
# Create a function space using first-order Lagrange elements
V = fem.functionspace(mesh, ("Lagrange", 1))  # Creates function space for solution

# Define trial and test functions for the variational formulation
u = ufl.TrialFunction(V)  # Unknown function to be solved for
v = ufl.TestFunction(V)   # Test function for variational formulation

# =============================================================================
# Variational Formulation
# =============================================================================
# Define the bilinear forms:
# - Laplace-Beltrami operator (diffusion term)
# - Mass matrix (reaction term)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Stiffness matrix: represents diffusion
m = ufl.inner(u, v) * ufl.dx                      # Mass matrix: represents reaction term

# Define diffusion tensor D (positive definite) and corresponding bilinear form
# Example: diagonal SPD tensor for anisotropic diffusion
# Define diffusion tensor D (positive definite) using dolfinx Constant
D = fem.Constant(mesh, ((1.0, 0.0, 0.0),
                        (0.0, 2.0, 0.0),
                        (0.0, 0.0, 1.5)))
aD = ufl.inner(ufl.dot(D, ufl.grad(u)), ufl.grad(v)) * ufl.dx


# =============================================================================
# Matrix Assembly
# =============================================================================
# Compile the variational forms for efficient assembly
a_form = fem.form(a)  # Compiles stiffness matrix form
m_form = fem.form(m)  # Compiles mass matrix form

# Assemble the stiffness matrix (Laplace-Beltrami operator)
A = petsc.assemble_matrix(a_form)  # Creates sparse matrix for stiffness term
A.assemble()  # Finalize assembly by accumulating ghost contributions

# Assemble the mass matrix
M = petsc.assemble_matrix(m_form)  # Creates sparse matrix for mass term
M.assemble()  # Finalizes mass matrix assembly

# Compile and assemble diffusion-based stiffness matrix
aD_form = fem.form(aD)
AD = petsc.assemble_matrix(aD_form)
AD.assemble()

# Print matrix dimensions for verification
print("Stiffness matrix size:", A.getSize())
print("Mass matrix size:",     M.getSize())

# =============================================================================
# Right-hand Side and Solution
# =============================================================================
# Create a random right-hand side vector using white noise
local_size = V.dofmap.index_map.size_local  # Gets number of local degrees of freedom
g = np.random.normal(0.0, 1.0, size=local_size)  # Generates random forcing term
b = PETSc.Vec().createWithArray(g, comm=MPI.COMM_WORLD)  # Creates parallel vector

# Solve the system (A + M) x = b using conjugate gradients
L = A.copy()  # Creates copy of stiffness matrix
L.axpy(1.0, M)  # L = A + M: combines stiffness and mass matrices
ksp = PETSc.KSP().create(comm=MPI.COMM_WORLD)  # Creates Krylov subspace solver
ksp.setOperators(L)  # Sets the system matrix
ksp.setType("cg")  # Use conjugate gradient method
ksp.getPC().setType("jacobi")  # Use Jacobi preconditioner
ksp.setTolerances(rtol=1e-5)  # Set relative tolerance
x = b.duplicate()  # Creates solution vector
ksp.solve(b, x)  # Solves the linear system

# Solve the diffusion-based system (A_D + M) x_D = b
LD = AD.copy()
LD.axpy(1.0, M)  # LD = AD + M
ksp2 = PETSc.KSP().create(comm=MPI.COMM_WORLD)
ksp2.setOperators(LD)
ksp2.setType("cg")
ksp2.getPC().setType("jacobi")
ksp2.setTolerances(rtol=1e-5)
xD = b.duplicate()
ksp2.solve(b, xD)

# =============================================================================
# Solution Processing
# =============================================================================
# Create a DOLFINx function to store the solution
u_sol = fem.Function(V)  # Creates function to store solution
u_sol.x.array[:] = x.array  # Copies solution values
u_sol.x.scatter_forward()  # Updates ghost values for parallel computation
u_sol.name = "u"  # Names the solution field

# Store the diffusion solution in a Function
uD = fem.Function(V)
uD.x.array[:] = xD.array
uD.x.scatter_forward()
uD.name = "u_diffusion"

# =============================================================================
# Output and Visualization
# =============================================================================
# Set up MPI communicator for parallel I/O
comm = MPI.COMM_WORLD  # Gets MPI communicator for parallel operations

# Save the solution in VTK format for visualization
with io.VTKFile(comm, f"{output_dir}/brain_solution.pvd", "w") as vtk:  # Opens VTK file for writing
    vtk.write_mesh(mesh)      # Writes mesh geometry
    vtk.write_function(u_sol) # Writes solution field

# Save diffusion-based solution
with io.VTKFile(comm, f"{output_dir}/brain_solution_diffusion.pvd", "w") as vtk2:
    vtk2.write_mesh(mesh)
    vtk2.write_function(uD)

print("✅ Simulation completed. Solution saved at:", output_dir)
