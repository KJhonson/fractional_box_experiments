import numpy as np  #we use this for manipulations with arrays...
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

import dolfinx.plot
import ufl


##############################
######CREATING THE MESH#######
##############################


# Create 1D mesh
N = 5
mesh = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, 1.0])

######PLOTTING THE MESH#######

# Get VTK (Visualization ToolKit) data (out is a tuple) and plot with matplotlib
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, dim=mesh.topology.dim)

plt.plot(geometry[:, 0], np.zeros_like(geometry[:, 0]), "o-")

#---SETTINGS OF THE PLOT---#

# Optional: Set x and y axis labels & title
plt.title("1D mesh")
# plt.xlabel("x")
# plt.ylabel("y")

# Customize ticks on the axes
plt.xticks(np.linspace(0, 1, 3)) #Ex: ticks from 0 to 1 using 3 points
plt.yticks([0])  # Only show tick at y=0

#Showgrid (based ony and x ticks)
# plt.grid(True)  

#modify bottom and top margins to "squash" axes vertically:
plt.subplots_adjust(top=0.7, bottom=0.3)
#adjusting overall dimensions of the plot (not the subplot, is whole plot).
plt.gcf().set_size_inches(10, 2)  # modify 10=width and 2=height

#---SETTINGS OF THE PLOT---#

#save finalplot and plot
plt.savefig("mesh.png")
# plt.show()

######PLOTTING THE MESH#######

V = fem.functionspace(mesh, ("Lagrange", 1)) #1 means piecewise polinomial of degree 1 over the elements.

#boundary conditions

ends = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)) #out is check if : is it close to 0 or 1.

uD = fem.Function(V) #empty function (0,0,...,0)
# uD.x.array[:] = 0.5 #set all values to 0.5
uD.interpolate(lambda x: 0.5+(1-0.5)*x[0]) #interpolated version of f(x)=a+(1-b)*x (a,b are the)

bc = fem.dirichletbc(uD, ends) #evaluation of uD where ends are TRUE.

w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(V.mesh)
a = ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx
f = ufl.sin(ufl.pi*(x[0])) #why not f = np.sin(ufl.pi*(x[0])) : (consistence) because ufl is a symbolic math library. np. is for arrays. 
L = f*v*ufl.dx


d_options = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}
problem = LinearProblem(a, L, bcs=[bc],
                        petsc_options=d_options,
                        form_compiler_options={"optimize": True}
                        )
u_h = problem.solve()

##############################
######PLOTTING THE SOLUTION###
##############################

# Get the solution values at mesh vertices
x_coords = mesh.geometry.x[:, 0]
u_values = u_h.x.array

# Create a plot of the solution
plt.figure(figsize=(6, 6))
plt.plot(x_coords, u_values, 'bo-', linewidth=1, markersize=3, label='FEM Solution')
plt.title('1D FEM Solution: -u\'\' = sin(πx)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the solution plot
plt.savefig("solution.png", dpi=150, bbox_inches='tight')
# plt.show()




