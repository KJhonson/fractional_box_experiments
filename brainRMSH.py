#!/usr/bin/env python3
import os                              # para manipulação de caminhos e diretórios
import numpy as np                     # para operações numéricas e geração de vetores
import meshio                          # para ler e escrever arquivos de malha (OBJ, XDMF, etc.)
import pyvista as pv                   # para decimação da malha com algoritmos de visualização
import ufl                             # UFL: descrição de formas variacionais
from mpi4py import MPI                 # MPI para paralelismo distribuído
from petsc4py import PETSc             # interface PETSc para resolução de sistemas lineares
import dolfinx.fem.petsc as petsc      # montagem de matrizes em PETSc a partir de formulários variacionais
from dolfinx import fem, io            # definição de espaços de função, funções e I/O VTK/PVD
from dolfinx.mesh import create_mesh   # construção de malha DOLFINx a partir de pontos e células
import basix                           # para criar elementos de base (coordinate elements)

# --- Configuração de caminhos e parâmetros de decimação ---
BASE_DIR         = "/home/dolfinx/shared/brainRMSH"                     # pasta de saída
INPUT_OBJ        = "/home/dolfinx/shared/brain.obj"                     # malha original em OBJ
REMESH_OBJ       = os.path.join(BASE_DIR, "brain_remesh.obj")           # arquivo OBJ decimado de saída
REMESH_XDMF      = os.path.join(BASE_DIR, "brain_remesh.xdmf")          # arquivo XDMF/HDF5 decimado
RESULT_PVD       = os.path.join(BASE_DIR, "brain_remesh_solution.pvd")  # PVD com malha + solução
REMESH_GEOM_PVD  = os.path.join(BASE_DIR, "brain_remesh_geometry.pvd")  # PVD apenas geometria remesh
TARGET_REDUCTION = 0.95                                                 # fração de faces a remover

# garante que a pasta de saída exista
os.makedirs(BASE_DIR, exist_ok=True)

# 1) Leitura da malha original (OBJ) com meshio
mesh0 = meshio.read(INPUT_OBJ)                      # carrega vértices e células do OBJ
points = mesh0.points.astype(np.float64)            # array de coordenadas (N×3)
cells  = mesh0.cells_dict.get("triangle")           # conectividade de triângulos
if cells is None:
    raise RuntimeError("Nenhuma célula triangular encontrada no OBJ original")
cells = cells.astype(np.int64)
print(f"▶️  Original mesh: {points.shape[0]} vértices, {cells.shape[0]} triângulos")

# 2) Decimação da malha com PyVista
faces   = np.hstack([
    np.full((cells.shape[0], 1), 3, dtype=np.int64),   # número de nós por face (3)
    cells                                              # índices dos nós por face
])
pv_mesh = pv.PolyData(points, faces)                  # constrói PolyData para decimação
dec_mesh = pv_mesh.decimate(TARGET_REDUCTION)         # remove TARGET_REDUCTION% das faces
points_dec = dec_mesh.points.astype(np.float64)       # novos vértices
faces_dec  = dec_mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.int64)  # novas faces
print(f"▶️  Decimated mesh: {points_dec.shape[0]} vértices, {faces_dec.shape[0]} triângulos")

# 3) Salva a malha decimada como XDMF/HDF5 e OBJ
mesh_rem = meshio.Mesh(points=points_dec, cells=[("triangle", faces_dec)])
meshio.write(REMESH_XDMF, mesh_rem)                   # gera .xdmf + .h5
meshio.write(REMESH_OBJ, mesh_rem, file_format="obj") # gera .obj para visualização
print(f"✅ Salvou remesh OBJ: {REMESH_OBJ}")
print(f"✅ Salvou remesh XDMF: {REMESH_XDMF}")

# 4) Carrega novamente a malha decimada no DOLFINx
mesh1      = meshio.read(REMESH_XDMF)                 # lê .xdmf recém-gravado
points_dec = mesh1.points.astype(np.float64)          # coordenadas decimadas
cells_dec  = mesh1.cells_dict.get("triangle")         # conectividade triangular
if cells_dec is None:
    raise RuntimeError("Nenhuma célula triangular no XDMF decimado")
cells_dec = cells_dec.astype(np.int64)
print(f"▶️ Remesh reload: {points_dec.shape[0]} vértices, {cells_dec.shape[0]} triângulos")

# cria domínio UFL para geometria
coord_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
domain   = ufl.Mesh(coord_el)
# monta o objeto dolfinx.mesh.Mesh distribuído
msh      = create_mesh(MPI.COMM_WORLD, cells_dec, points_dec, domain)

# exporta apenas a geometria remesh para PVD (visualização no ParaView)
with io.VTKFile(MPI.COMM_WORLD, REMESH_GEOM_PVD, "w") as vtkgeo:
    vtkgeo.write_mesh(msh)
print(f"✅ Salvou remesh geometry PVD: {REMESH_GEOM_PVD}")

# 5) Define espaço de funções e formas variacionais
V = fem.functionspace(msh, ("Lagrange", 1))            # espaço contínuo P1
u = ufl.TrialFunction(V)                               # função de ensaio
v = ufl.TestFunction(V)                                # função de teste
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx       # forma bilinear A
m = ufl.inner(u, v) * ufl.dx                           # forma de massa M

# 6) Monta as matrizes A e M em PETSc
A = petsc.assemble_matrix(fem.form(a)); A.assemble()   # matriz de rigidez
M = petsc.assemble_matrix(fem.form(m)); M.assemble()   # matriz de massa
print(f"[Dolfin-X] A size {A.getSize()}, M size {M.getSize()}")

# 7) Constrói RHS W = diag(M_lumped) * ruído gaussiano
diag_vec = M.getDiagonal()                             # extrai diagonal lumped de M
M0       = diag_vec.getArray()                         # converte para numpy array
nloc     = V.dofmap.index_map.size_local               # nº local de graus de liberdade
g        = np.random.normal(size=nloc)                 # vetor de ruído N(0,1)
W_arr    = M0 * g                                      # escala linha-a-linha
b        = PETSc.Vec().createWithArray(W_arr, comm=MPI.COMM_WORLD)

# 8) Resolve (A + M) x = W com KSP
L   = A.copy(); L.axpy(1.0, M); L.assemble()           # L = A + M
ksp = PETSc.KSP().create(comm=MPI.COMM_WORLD)
ksp.setOperators(L)
ksp.setType(PETSc.KSP.Type.CG)                         # método conjugado
ksp.getPC().setType(PETSc.PC.Type.JACOBI)              # pré-condicionador Jacobi
ksp.setTolerances(rtol=1e-5)                           # tolerância relativa
x   = b.duplicate()                                    # vetor solução
ksp.solve(b, x)                                        # resolve L x = b

# 9) Exporta solução para PVD (malha + campo)
u_sol = fem.Function(V)                                # wrapper para x
u_sol.x.array[:] = x.array                             # copia valores
u_sol.x.scatter_forward()                              # sincroniza MPI
u_sol.name = "white_noise"

with io.VTKFile(MPI.COMM_WORLD, RESULT_PVD, "w") as vtk:
    vtk.write_mesh(msh)
    vtk.write_function(u_sol)
print(f"✅ Simulation complete. Output in: {RESULT_PVD}")
