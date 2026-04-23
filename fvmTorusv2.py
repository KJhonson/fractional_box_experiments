# torusFVM_asymmetric.py
import os
import numpy as np
import gmsh
import meshio

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem import FunctionSpace, Function
from dolfinx.fem.petsc import assemble_matrix
from ufl import TrialFunction, TestFunction, inner, grad, dx, dot

# --- (1) Mesh generation and reading, identical to your torusFVM.py ---
output_dir = "torusFVM_asymm"
os.makedirs(output_dir, exist_ok=True)

gmsh.initialize()
R, r, mesh_size = 1.0, 0.3, 0.04
gmsh.model.add("torus")
_ = gmsh.model.occ.addTorus(0,0,0, R, r)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
gmsh.model.addPhysicalGroup(2, [1], name="Surface")
gmsh.model.mesh.generate(2)
msh_file = f"{output_dir}/torus.msh"
gmsh.write(msh_file)
gmsh.finalize()

comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.read_from_msh(msh_file, comm)

# --- (2) Function spaces and forms ---
element = ("Lagrange", 1)
V = fem.functionspace(domain, element)
u = TrialFunction(V)
v = TestFunction(V)

# mass + diffusion for comparison
a = inner(grad(u), grad(v))*dx
m = inner(u, v)*dx

gdim = domain.geometry.dim
V_D = fem.functionspace(domain, ("Lagrange", 1, (gdim, gdim)))
D = Function(V_D)
coords = domain.geometry.x
D_local = np.zeros((coords.shape[0], 3, 3))
for i, (x,y,z) in enumerate(coords):
    D_local[i] = np.diag([1.0+x, 2.0+y, 1.5+z])
D.x.array[:] = D_local.flatten()
D.x.scatter_forward()

aD = inner(dot(D, grad(u)), grad(v))*dx

A = assemble_matrix(fem.form(a)); A.assemble()
M = assemble_matrix(fem.form(m)); M.assemble()
AD = assemble_matrix(fem.form(aD)); AD.assemble()

# --- (3) ASYMMETRIC FVM assembly ---
# Create PETSc matrix for non-symmetric assembly
n_dofs = V.dofmap.index_map.size_local
FVM = PETSc.Mat().create(comm=comm)
FVM.setSizes([n_dofs, n_dofs])
FVM.setType(PETSc.Mat.Type.AIJ)
FVM.setFromOptions()
FVM.setUp()

# connectivity: triangles → vertices
dim = domain.topology.dim
domain.topology.create_connectivity(dim, 0)
cell_to_vertex = domain.topology.connectivity(dim, 0)
coords3 = domain.geometry.x

# loop over each triangle
for cell in range(domain.topology.index_map(dim).size_local):
    verts = cell_to_vertex.links(cell)
    if len(verts) != dim+1:
        continue
    i, j, k = verts
    Vtx = coords3[[i,j,k],:]
    v0, v1, v2 = Vtx
    # triangle normal & area
    nT_vec = np.cross(v1-v0, v2-v0)
    area2 = np.linalg.norm(nT_vec)
    if area2 < 1e-12:
        continue
    nT = nT_vec / area2
    area = area2 * 0.5

    # gradients of local basis φ0,φ1,φ2 (constant on triangle)
    grad_phi = [
        np.cross(nT, (v2-v1)) / area2,   # ∇φ0
        np.cross(nT, (v0-v2)) / area2,   # ∇φ1
        np.cross(nT, (v1-v0)) / area2    # ∇φ2
    ]

    # centroid
    c = 0.3333333*(v0+v1+v2)

    # choose diffusion tensor for this cell (average of vertices)
    D_cell =  (D_local[i] + D_local[j] + D_local[k]) / 3.0

    # for each vertex of triangle, build its dual-cell boundary ∂b
    for local_i, vi in enumerate((v0, v1, v2)):
        row = [i, j, k][local_i]  # global dof index
        # midpoints of edges adjacent to vi
        if local_i == 0:
            mids = [0.5*(v0+v1), 0.5*(v0+v2)]
        elif local_i == 1:
            mids = [0.5*(v1+v2), 0.5*(v1+v0)]
        else:
            mids = [0.5*(v2+v0), 0.5*(v2+v1)]

        # accumulate flux integrals over the two dual edges
        for mid in mids:
            d = c - mid
            L = np.linalg.norm(d)
            if L < 1e-12:
                continue
            η = np.cross(nT, d)
            η /= np.linalg.norm(η)
            # ensure outward w.r.t. vi
            if np.dot(η, vi-mid) > 0:
                η = -η

            # loop over trial bases φ_j
            for local_j, _vj in enumerate((v0, v1, v2)):
                col = [i,j,k][local_j]
                flux = float((D_cell.dot(grad_phi[local_j])) @ η) * L
                # asymmetric bilinear form: -∫ (A ∇φ_j · η) φ_i(ds)
                FVM.setValue(row, col, -flux, PETSc.InsertMode.ADD)

# finalize FVM

FVM.assemble()

# === Matrix equality checks ===
# Compute difference matrices and their Frobenius norms
diff_AD = A.copy()
diff_AD.axpy(-1.0, AD)
norm_A_AD = diff_AD.norm(PETSc.NormType.FROBENIUS)

diff_AB = A.copy()
diff_AB.axpy(-1.0, FVM)
norm_A_B = diff_AB.norm(PETSc.NormType.FROBENIUS)

diff_ADB = AD.copy()
diff_ADB.axpy(-1.0, FVM)
norm_AD_B = diff_ADB.norm(PETSc.NormType.FROBENIUS)

# Print results on rank 0
if comm.rank == 0:
    print(f"||A - A_D||_F = {norm_A_AD:.6e}")
    print(f"||A - B||_F   = {norm_A_B:.6e}")
    print(f"||A_D - B||_F = {norm_AD_B:.6e}")

# --- (4) Solve (FVM + lumped mass) ---
b = PETSc.Vec().createWithArray(np.random.randn(n_dofs), comm=comm)

ksp1 = PETSc.KSP().create(comm=comm)
L1 = A.copy(); L1.axpy(1.0, M)
ksp1.setOperators(L1); ksp1.setType(PETSc.KSP.Type.CG)
ksp1.getPC().setType(PETSc.PC.Type.JACOBI)
x1 = b.duplicate(); ksp1.solve(b, x1)

ksp2 = PETSc.KSP().create(comm=comm)
L2 = AD.copy(); L2.axpy(1.0, M)
ksp2.setOperators(L2); ksp2.setType(PETSc.KSP.Type.CG)
ksp2.getPC().setType(PETSc.PC.Type.JACOBI)
x2 = b.duplicate(); ksp2.solve(b, x2)

ksp3 = PETSc.KSP().create(comm=comm)
L3 = FVM.copy(); L3.axpy(1.0, M)
ksp3.setOperators(L3); ksp3.setType(PETSc.KSP.Type.CG)
ksp3.getPC().setType(PETSc.PC.Type.JACOBI)
x3 = b.duplicate(); ksp3.solve(b, x3)

# --- (5) Export solutions ---
u_fem   = Function(V); u_fem.x.array[:]   = x1.getArray(); u_fem.x.scatter_forward(); u_fem.name   = "u_fem"
u_diff  = Function(V); u_diff.x.array[:]  = x2.getArray(); u_diff.x.scatter_forward(); u_diff.name  = "u_diff"
u_asym  = Function(V); u_asym.x.array[:]  = x3.getArray(); u_asym.x.scatter_forward(); u_asym.name  = "u_fvm_asym"

with io.VTKFile(comm, f"{output_dir}/solutions.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u_fem)
    vtk.write_function(u_diff)
    vtk.write_function(u_asym)

print("Done! Outputs in", output_dir)