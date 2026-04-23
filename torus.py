import os
import numpy as np
import gmsh
import meshio

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem import FunctionSpace, Function
from dolfinx.fem.petsc import assemble_matrix  # ✅ Corrected import
from ufl import TrialFunction, TestFunction, inner, grad, dx



# === ETAPA 1: Configuração ===
output_dir = "torus"
os.makedirs(output_dir, exist_ok=True)

# === ETAPA 2: Geração da malha do toro com GMSH ===
gmsh.initialize()
R = 1.0
r = 0.3
mesh_size = 0.04
gmsh.model.add("torus")
torus = gmsh.model.occ.addTorus(0, 0, 0, R, r)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
gmsh.model.addPhysicalGroup(2, [torus], tag=1, name="Surface")
gmsh.model.mesh.generate(2)
mesh_path = f"{output_dir}/torus.msh"
gmsh.write(mesh_path)
gmsh.finalize()
print(f"✅ Malha salva em: {mesh_path}")

# === ETAPA 4: Leitura da malha com dolfinx ===
comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, comm)

# === ETAPA 5: Espaço funcional e formas ===
element = ("Lagrange", 1)
V = fem.functionspace(domain, element)
u = TrialFunction(V)
v = TestFunction(V)

print(f"\n=== Informações da Malha ===")
n_dofs = V.dofmap.index_map.size_local
print(f"Número de DOFs: {n_dofs}")

# === ETAPA 6: Montagem das matrizes ===
a = inner(grad(u), grad(v)) * dx
m = inner(u, v) * dx
a_form = fem.form(a)
m_form = fem.form(m)

# ✅ Use updated assemble_matrix import
A = assemble_matrix(a_form)
A.assemble()

M = assemble_matrix(m_form)
M.assemble()

print(f"\n=== Matrix Info ===")
print(f"Tamanho das matrizes: {n_dofs} x {n_dofs}")

# === ETAPA 7: Criar vetor de ruído branco ===
g = np.random.normal(0, 1, size=n_dofs)
b = PETSc.Vec().createWithArray(g, comm=comm)

# === ETAPA 8: Resolver o sistema ===
ksp = PETSc.KSP().create(comm=comm)
L = A.copy()
L.axpy(1.0, M)
ksp.setOperators(L)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.JACOBI)
ksp.setTolerances(rtol=1e-5)
x = b.duplicate()
ksp.solve(b, x)

# === ETAPA 9: Exportar a solução ===
u_sol = Function(V)
u_sol.x.array[:] = x.getArray()
u_sol.x.scatter_forward()

# Nome da função para visualização
u_sol.name = "u"

# Exportar para visualização
with io.VTKFile(comm, f"{output_dir}/torus_solution.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_sol)

print("\n✅ Simulação finalizada. Arquivos salvos em 'output/'")
print("Para visualizar, abra o arquivo 'output/torus_solution.pvd' no ParaView")
