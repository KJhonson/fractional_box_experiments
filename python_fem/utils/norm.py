"""
Norm computation utilities for error analysis in finite element methods.

This module provides functions to compute various norms (L2, H1, SH1, LI) 
between numerical solutions and exact solutions or reference solutions.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl
from typing import Union, Callable


def get_norm(u_h: fem.Function, u_exact_func: Union[Callable, fem.Function], meshc: dmesh.Mesh, meshf: dmesh.Mesh, Vf, norm_type: str = "L2") -> float:
    """
    Compute the norm of the error between a numerical solution and an exact/reference solution.
    
    Parameters
    ----------
    u_h : fem.Function
        Numerical solution function
    u_exact_func : Union[Callable, fem.Function]
        Exact solution (can be a callable function or a fem.Function)
    meshc : dmesh.Mesh
        Mesh on which the numerical solution u_h is defined
    meshf : dmesh.Mesh
        Fine mesh used for error integration
    Vf : fem.FunctionSpace
        Function space on the fine mesh meshf
    norm_type : str, optional
        Type of norm to compute. Options: "L2", "H1", "SH1", "LI" (default: "L2")
        - "L2": L2 norm
        - "H1": H1 norm (L2 + H1 seminorm)
        - "SH1": H1 seminorm (gradient norm)
        - "LI": L-infinity norm (maximum norm)
    
    Returns
    -------
    float
        The computed error norm
    """
    dof_coords = Vf.tabulate_dof_coordinates()
    dim = meshf.topology.dim
    
    def evaluate_function(u, mesh_src):
        if isinstance(u, fem.Function):
            if dim == 1:
                x_eval = dof_coords[:, 0]
                x_src = mesh_src.geometry.x[:, 0]
                vals = u.x.array
                sort_idx = np.argsort(x_src)
                return np.interp(x_eval, x_src[sort_idx], vals[sort_idx])
            else:
                mesh_src.topology.create_entities(dim)
                mesh_src.topology.create_connectivity(dim, dim)
                from dolfinx import geometry
                bb_tree = geometry.bb_tree(mesh_src, dim)
                cell_candidates = geometry.compute_collisions_points(bb_tree, dof_coords)
                colliding_cells = geometry.compute_colliding_cells(mesh_src, cell_candidates, dof_coords)
                cells = colliding_cells.array.astype(np.int32)
                values = np.zeros((1, len(dof_coords)), dtype=np.float64)
                u.eval(dof_coords.T, cells, values)
                return values[0]
        else:
            return np.array([u(p) for p in dof_coords])
    
    u_h_vals = evaluate_function(u_h, meshc)
    if isinstance(u_exact_func, fem.Function):
        u_exact_mesh = u_exact_func.function_space.mesh
        u_exact_vals = evaluate_function(u_exact_func, u_exact_mesh)
    else:
        u_exact_vals = evaluate_function(u_exact_func, None)
    
    u_h_f = fem.Function(Vf)
    u_exact_f = fem.Function(Vf)
    u_h_f.x.array[:] = u_h_vals
    u_exact_f.x.array[:] = u_exact_vals
    
    e_f = fem.Function(Vf)
    e_f.x.array[:] = u_h_f.x.array - u_exact_f.x.array
    
    if norm_type == "L2":
        error_form = ufl.inner(e_f, e_f) * ufl.dx
        error_norm = float(np.sqrt(fem.assemble_scalar(fem.form(error_form))))
    elif norm_type == "H1":
        error_form = ufl.inner(e_f, e_f) * ufl.dx + ufl.inner(ufl.grad(e_f), ufl.grad(e_f)) * ufl.dx
        error_norm = float(np.sqrt(fem.assemble_scalar(fem.form(error_form))))
    elif norm_type == "SH1":
        error_norm = ufl.inner(ufl.grad(e_f), ufl.grad(e_f)) * ufl.dx
        error_norm = float(np.sqrt(fem.assemble_scalar(fem.form(error_norm))))
    elif norm_type == "LI":
        error_norm = float(np.max(np.abs(e_f.x.array)))
    else:
        raise ValueError(f"norm_type must be 'L2', 'H1', 'SH1', or 'LI', got '{norm_type}'")
    
    return error_norm


# %%
# Example 

if __name__ == "__main__":
 
    mesh = dmesh.create_interval(MPI.COMM_WORLD, 64, [0.0, 1.0])
    V = fem.functionspace(mesh, ("CG", 1))
    
    # artificial numerical solution u_h
    u_h = fem.Function(V)
    u_h.interpolate(lambda x: np.sin(2* np.pi * x[0]))
    
    # artificial source function
    def f_source(x):
        return np.sin(np.pi * x[0])
    f_func = fem.Function(V)
    f_func.interpolate(f_source)
    
    # Example 1: Compute H1 norm between u_h and f_func
    meshf = dmesh.create_interval(MPI.COMM_WORLD, 400, [0.0, 1.0])
    Vf = fem.functionspace(meshf, ("CG", 1))
    
    H1_error = get_norm(u_h, f_func, mesh, meshf, Vf, "H1")
    if mesh.comm.rank == 0:
        print(f"H1 error: {H1_error:.6e}")
    
    # Example 2: Compute L2 norm between u_h and another function on a different mesh
    mesh_test = dmesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 1.0])
    V_test = fem.functionspace(mesh_test, ("CG", 1))
    uh_test = fem.Function(V_test)
    uh_test.interpolate(lambda x: np.sin(np.pi * x[0]))
    
    L2_exmp = get_norm(u_h, uh_test, mesh_test, mesh, Vf, "L2")
    if mesh.comm.rank == 0:
        print(f"L2 error: {L2_exmp:.6e}")


# %%
