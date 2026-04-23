import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh as dolfinx_mesh, fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc


def fem_utils(mesh, bdr_condition = "dirichlet", bdr_values=[0.0, 0.0], p=1):
    # Function space
    V = fem.functionspace(mesh, ("Lagrange", p))

    # Boundary conditions
    v1, v2 = bdr_values[0], bdr_values[1]
    ends = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
    uD = fem.Function(V)
    uD.interpolate(lambda x: v1 + (v2 - v1) * x[0])
    bc = fem.dirichletbc(uD, ends)

    # Variational formulation
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)  #Fixed: should be mesh, not V
    g = ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx
    c1 = ufl.inner(w, v) * ufl.dx
    G = fem.petsc.assemble_matrix(fem.form(g), bcs=[bc]); G.assemble()
    C1 = fem.petsc.assemble_matrix(fem.form(c1), bcs=[bc]); C1.assemble()
    # Lumped mass matrix C0 is obtained by summing each row of C1 and placing the sum on the diagonal (row-sum lumping)
    rowsum = C1.getRowSum()
    C0 = C1.duplicate(copy=PETSc.Mat.DuplicateOption.DO_NOT_COPY_VALUES) #Duplicate the matrix without copying the values
    C0.zeroEntries() #Zero out all entries
    C0.setDiagonal(rowsum); C0.assemble() #Set the diagonal of the lumped matrix
    return V, G, C1, C0, bc

#Ex 

mesh = dolfinx_mesh.create_interval(MPI.COMM_WORLD, 5, [0.0, 1.0]) #Create a 1D mesh
V, G, C1, C0, bc = fem_utils(mesh, bdr_condition="dirichlet", bdr_values=[0.0, 0.0], p=1)







# def load_utils(mesh, f_expr=None):
#     V = fem.functionspace(mesh, ("Lagrange", p))
#     x = ufl.SpatialCoordinate(mesh)  #Fixed: should be mesh, not V
#     v = ufl.TestFunction(V)
#     # Handle source term
#     if f_expr is None:
#         # Default: zero load
#         f_zero = fem.Function(V)
#         f_zero.x.array[:] = 0.0
#         L = f_zero * v * ufl.dx #L comes from "LOAD" term
#     else:
#         # Custom UFL expression
#         f = f_expr(x)
#         L = f * v * ufl.dx
#     # Problem and solving
#     return L

# ksp_opts = None
#  d_options = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}
#     if ksp_opts:
#         d_options.update(ksp_opts)

#     problem = LinearProblem(a, L, bcs=[bc],
#                             petsc_options=d_options,
#                             form_compiler_options={"optimize": True})
#     u_h = problem.solve()







