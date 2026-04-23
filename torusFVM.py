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
from ufl import TrialFunction, TestFunction, inner, grad, dx, dot



# === ETAPA 1: Configuração ===
output_dir = "torusFVM"
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

# === ANISOTROPIC DIFFUSION TENSOR PER VERTEX ===
# Create tensor function space for 3x3 tensor fields (Lagrange elements, degree 1)
gdim = domain.geometry.dim
V_D = fem.functionspace(domain, ("Lagrange", 1, (gdim, gdim)))
D = Function(V_D)

# Example: assign a tensor at each vertex based on its coordinates
coords = domain.geometry.x  # vertex coordinates array of shape (n_vertices, 3)
num_vertices = coords.shape[0]
# Initialize an array for the local tensor values (n_vertices x 3 x 3)
D_local = np.zeros((num_vertices, 3, 3), dtype=float)
for i, (x, y, z) in enumerate(coords):
    # Replace this formula with your anisotropy specification per vertex
    D_local[i] = np.diag([1.0 + x, 2.0 + y, 1.5 + z])

# Assign to the Function D and scatter
D.x.array[:] = D_local.flatten()
D.x.scatter_forward()

# Bilinear form with spatially varying diffusion tensor
aD = inner(dot(D, grad(u)), grad(v)) * dx

a_form = fem.form(a)
m_form = fem.form(m)

# ✅ Use updated assemble_matrix import
A = assemble_matrix(a_form)
A.assemble()

M = assemble_matrix(m_form)
M.assemble()

#
# Assemble diffusion-based stiffness matrix
aD_form = fem.form(aD)
AD = assemble_matrix(aD_form)
AD.assemble()

# === ETAPA FVM (revised): Montagem do operador de difusão via FVM baseado em triângulos ===
from petsc4py import PETSc
import numpy as _np

# Cria matriz PETSc para FVM
FVM = PETSc.Mat().create(comm=comm)
FVM.setSizes([n_dofs, n_dofs])
FVM.setType(PETSc.Mat.Type.AIJ)
FVM.setFromOptions()
FVM.setUp()

# Pré-computar conectividade: triângulos -> vértices
domain.topology.create_connectivity(domain.topology.dim, 0)
cell_to_vertex = domain.topology.connectivity(domain.topology.dim, 0)
coords = domain.geometry.x

# Montagem via contribuição de cada triângulo
for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
    verts = cell_to_vertex.links(cell)
    if len(verts) != domain.topology.dim + 1:
        continue
    i, j, k = verts
    Xi, Xj, Xk = coords[i], coords[j], coords[k]
    # Calcula normal e área do triângulo
    e_ij = Xj - Xi
    e_ik = Xk - Xi
    normal = _np.cross(e_ij, e_ik)
    area = 0.5 * _np.linalg.norm(normal)
    if area < 1e-15:
        continue
    normal = normal / _np.linalg.norm(normal)
    # Base tangente local (t1 ao longo i->j, t2 ortogonal no plano)
    t1 = e_ij / _np.linalg.norm(e_ij)
    t2 = _np.cross(normal, t1)
    # Média do tensor de difusão nos três vértices
    D_t = (D_local[i] + D_local[j] + D_local[k]) / 3.0
    # Projeção do tensor no plano tangente (2×2)
    D_loc2 = _np.array([
        [t1.dot(D_t @ t1), t1.dot(D_t @ t2)],
        [t2.dot(D_t @ t1), t2.dot(D_t @ t2)]
    ])
    # Coordenadas locais 2D dos vértices
    xi2 = _np.array([0.0, 0.0])
    xj2 = _np.array([_np.linalg.norm(e_ij), 0.0])
    xk2 = _np.array([e_ik.dot(t1), e_ik.dot(t2)])
    # Função para computar grad φ em 2D
    def grad_phi(xp, xq, xr):
        return _np.array([xq[1] - xr[1], xr[0] - xq[0]]) / (2 * area)
    grad_i = grad_phi(xj2, xk2, xi2)
    grad_j = grad_phi(xk2, xi2, xj2)
    grad_k = grad_phi(xi2, xj2, xk2)
    grads = [grad_i, grad_j, grad_k]
    verts_list = [i, j, k]
    # Montagem dos coeficientes no PETSc Mat
    for p_idx, p in enumerate(verts_list):
        for q_idx, q in enumerate(verts_list):
            w = float(grads[p_idx] @ (D_loc2 @ grads[q_idx]) * area)
            if p == q:
                FVM.setValue(p, q, w, addv=True)
            else:
                FVM.setValue(p, q, -w, addv=True)

FVM.assemble()

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

# Solve the diffusion-based system (A_D + M) x_D = b
LD = AD.copy()
LD.axpy(1.0, M)
ksp2 = PETSc.KSP().create(comm=comm)
ksp2.setOperators(LD)
ksp2.setType(PETSc.KSP.Type.CG)
ksp2.getPC().setType(PETSc.PC.Type.JACOBI)
ksp2.setTolerances(rtol=1e-5)
xD = b.duplicate()
ksp2.solve(b, xD)

# === ETAPA 8b: Resolver o sistema FVM (FVM + massa lumped) ===
ksp3 = PETSc.KSP().create(comm=comm)
LD_fvm = FVM.copy()
LD_fvm.axpy(1.0, M)
ksp3.setOperators(LD_fvm)
ksp3.setType(PETSc.KSP.Type.CG)
ksp3.getPC().setType(PETSc.PC.Type.JACOBI)
ksp3.setTolerances(rtol=1e-5)
x_fvm = b.duplicate()
ksp3.solve(b, x_fvm)


# === ETAPA 9: Exportar a solução ===
u_sol = Function(V)
u_sol.x.array[:] = x.getArray()
u_sol.x.scatter_forward()

# Nome da função para visualização
u_sol.name = "u"

# Store the diffusion solution in a Function
uD = Function(V)
uD.x.array[:] = xD.getArray()
uD.x.scatter_forward()
uD.name = "u_diffusion"

# === ETAPA 9b: Exportar a solução FVM ===
uFVM = Function(V)
uFVM.x.array[:] = x_fvm.getArray()
uFVM.x.scatter_forward()
uFVM.name = "u_fvm"

# Exportar para visualização
with io.VTKFile(comm, f"{output_dir}/torus_solution.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_sol)

# Save diffusion-based solution
with io.VTKFile(comm, f"{output_dir}/torus_solution_diffusion.pvd", "w") as vtk2:
    vtk2.write_mesh(domain)
    vtk2.write_function(uD)

with io.VTKFile(comm, f"{output_dir}/torus_solution_fvm.pvd", "w") as vtk3:
    vtk3.write_mesh(domain)
    vtk3.write_function(uFVM)

print("\n✅ Simulação finalizada. Arquivos salvos em 'output/'")
print("Para visualizar, abra o arquivo 'output/torus_solution.pvd' no ParaView")