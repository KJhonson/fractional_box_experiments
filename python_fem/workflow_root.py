# %%
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl

# %%
mesh = dmesh.create_interval(MPI.COMM_WORLD, 64, [0.0, 1.0])
# mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [32, 32], cell_type=dmesh.CellType.triangle)
V = fem.functionspace(mesh, ("CG", 1))
# print(f"Domain: 1D interval [0, 1]")
# # print(f"Domain: 2D rectangle [0,1] x [0,1]")
# print(f"Number of elements: {mesh.topology.index_map(1).size_local}")
# # print(f"Number of elements: {mesh.topology.index_map(2).size_local}")
# print(f"Number of vertices: {mesh.topology.index_map(0).size_local}")
# print(f"Boundary facets tagged: {facet_tags.find(1).size}")

# %%
def kappa(x=None, value=1.0):
    if x is None:
        return ufl.Constant(mesh, value)
    else:
        return ufl.cos(x[0])
        # For 2D: return ufl.cos(x[0]) * ufl.sin(x[1]) or ufl.cos(x[0]) + x[1]
x = ufl.SpatialCoordinate(mesh)
kappa_val = kappa(x)
# For constant kappa: kappa_val = kappa() or kappa_val = kappa(value=2.0)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + kappa_val * ufl.inner(u, v) * ufl.dx
# For scalars: ufl.inner(u, v) is equivalent to u * v, so also: kappa_val * u * v * ufl.dx
from petsc4py import PETSc
from dolfinx.fem import petsc

K = petsc.assemble_matrix(fem.form(a))
K.assemble()
print(f"dim(K): {K.getSize()}")
# %%
def f_source(x):
    return np.sin(np.pi * x[0])
# def f_source(x):
#     return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
v = ufl.TestFunction(V)
f_func = fem.Function(V) #  func in V, 0 on the nodes. (print(function.x.array) to see values)
f_func.interpolate(f_source) #Interpolate f_source according to V

f = f_func
L = f * v * ufl.dx #form for the load vector

from petsc4py import PETSc
from dolfinx.fem import petsc

b = petsc.assemble_vector(fem.form(L))  # b.getArray() to see values
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"dim(b): {b.getSize()}")
# Alternative method: Using UFL expression directly (same result as above)
# x = ufl.SpatialCoordinate(mesh)
# f_ufl = ufl.sin(np.pi * x[0])
# # f_ufl = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
# L2 = f_ufl * v * ufl.dx
# b2 = petsc.assemble_vector(fem.form(L2))
# b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
# print(f"Load vector b (UFL expression): {b2.getSize()}")

# %%
ksp = PETSc.KSP().create(K.comm) #Creates a Krylov Subspace Solver (KSP) object on the same MPI communicator (K)
ksp.setOperators(K) #tells the solver that K is the system matrix (the operator to invert)
ksp.setType("cg") #Chooses the Conjugate Gradient (CG) iterative solver — used for symmetric positive definite matrices
ksp.getPC().setType("hypre") #Sets the Preconditioner (PC) to HYPRE
ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=10000) #Sets the Tolerances for the solver

u_vec = K.createVecRight() #Creates a PETSc (empty) vector to store the solution
ksp.solve(b, u_vec) #Solves the system Ku = b using KSP, setted up above (u_vec object turn on the solution vector)

u_h = fem.Function(V) #Creates a DOLFINx function to store the solution
u_h.x.array[:] = u_vec.array
u_h.x.scatter_forward() #Scatters the solution from the PETSc vector to the DOLFINx function

print(f"Solution computed: dim(u_h) = {len(u_h.x.array)}")

# %%

import matplotlib.pyplot as plt

if mesh.comm.rank == 0:
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    u_values = u_h.x.array
    f_values = f_func.x.array
#Plotting process:
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, u_values, 'o-', label='Numerical solution u_h', linewidth=2, markersize=6)
    plt.plot(x_coords, f_values, 'o--', label='Source function f', linewidth=2, markersize=4, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('Solution and Source Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# For 2D plotting:
# import pyvista as pv
# import dolfinx.plot
# topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
# grid = pv.UnstructuredGrid(topology, cell_types, geometry)
# grid.point_data["u"] = u_h.x.array
# grid.set_active_scalars("u")
# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# plotter.show()


