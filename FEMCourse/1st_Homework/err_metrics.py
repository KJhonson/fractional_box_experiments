import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
import ufl

####EXPLICIT APPROACH####

def error_metrics(uh, u_exact, domain, V, p):
    # 1) h = approximate mesh size (1/N for uniform mesh)
    # For 1D interval [0,1] with N elements: h ≈ 1/N
    N = domain.topology.index_map(domain.topology.dim).size_global
    h = 1.0 / N

    # 2) choose a reasonable quadrature degree
    qdeg = max(2*p + 1, 4)
    dx = ufl.Measure("dx", domain=domain)             # be explicit about the domain
    meta = {"quadrature_degree": qdeg}

    # 3) L2 error = sqrt(∫ (uh - u)^2 dx)
    e = uh - u_exact 
    L2_sq_local = fem.assemble_scalar(
        fem.form(ufl.inner(e, e) * dx, form_compiler_options=meta)) #convert ufl -> dolfinx using FFCx. It attaches all components for eff. comp.
    L2_sq = domain.comm.allreduce(L2_sq_local, op=MPI.SUM) #sum all the local contributions
    eL2 = np.sqrt(L2_sq) #sqrt of the sum of the local contributions

    # 4) H1-seminorm error = sqrt(∫ |∇(uh - u)|^2 dx)
    H1_sq_local = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx, form_compiler_options=meta)) #convert ufl -> dolfinx using FFCx. It attaches all components for eff. comp.
    H1_sq = domain.comm.allreduce(H1_sq_local, op=MPI.SUM) #sum all the local contributions
    eH1 = np.sqrt(H1_sq) #sqrt of the sum of the local contributions

    return h, eL2, eH1



# ####IMPLICIT APPROACH####
# import numpy as np
# from dolfinx import fem
# import ufl

# def error_metrics(domain, V, uh, u_exact, p):

#     h = domain.hmax()
#     qdeg = max(2 * p + 1, 4)
#     meta = {"quadrature_degree": qdeg}
#     eL2 = np.sqrt(
#         fem.assemble_scalar(
#             fem.form(
#                 ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx,
#                 metadata=meta
#             )
#         )
#     )
#     eH1 = np.sqrt(
#         fem.assemble_scalar(
#             fem.form(
#                 ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx,
#                 metadata=meta
#             )
#         )
#     )
#     return h, eL2, eH1