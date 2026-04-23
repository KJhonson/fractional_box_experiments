import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from baryrat import brasil

# def fractional_operators(L, beta, C, scale_factor=1.0, m=1, tau=1.0, interval=(1.0, 100.0)):
#     """
#     Fractional operator approximation using BRASIL, returning only Pl and Pr.

#     Parameters:
#         L: stiffness matrix (scipy sparse)
#         beta: fractional power (float)
#         C: mass matrix (scipy sparse)
#         scale_factor: normalization factor (float)
#         m: rational approximation order (int)
#         tau: scalar or array
#         interval: tuple, approximation interval

#     Returns:
#         Pl, Pr: left and right rational approximation operators (scipy sparse)
#     """
#     n = C.shape[0]

#     if np.min(tau) <= 0:
#         raise ValueError("tau must be positive")
#     if not isinstance(m, int) or m < 0:
#         raise ValueError("m must be a positive integer")
#     if scale_factor <= 0:
#         raise ValueError("scale_factor must be positive")

#     # Diagonal mass matrix approximation
#     C_diag = np.array(C.sum(axis=1)).flatten()
#     C = sp.diags(C_diag)
#     Ci = sp.diags(1.0 / C_diag)
#     I = sp.identity(n)

#     # Normalize L and define CiL
#     L = L / scale_factor
#     CiL = Ci @ L

#     # Integer beta case
#     if beta % 1 == 0:
#         Pr = I.copy()
#         Pl = L.copy()
#         for _ in range(int(beta) - 1):
#             Pl = Pl @ CiL
#         Pl = Pl * (scale_factor ** beta)
#     else:
#         beta_floor = int(np.floor(beta))
#         beta_frac = beta - beta_floor

#         r = brasil(lambda x: x ** beta_frac, interval, m)
#         rb = r.poles()
#         rc = r.zeros()
#         gain = r.gain()

#         # Construct Pl
#         Pl = I - rb[0] * CiL
#         for root in rb[1:]:
#             Pl = Pl @ (I - root * CiL)
#         for _ in range(max(1, beta_floor) - 1):
#             Pl = C @ CiL @ Pl     ##Looks like we can remove the C @ here
#         Pl = C @ Pl
#         Pl = Pl * (scale_factor ** beta / gain)

#         # Construct Pr
#         Pr = I - rc[0] * CiL
#         for root in rc[1:]:
#             Pr = Pr @ (I - root * CiL)

#     # Tau scaling on Pr
#     if np.isscalar(tau):
#         Phi = sp.diags([1.0 / tau] * n)
#     else:
#         Phi = sp.diags(1.0 / np.asarray(tau))
#     Pr = Phi @ Pr

#     return Pl, Pr



def fractional_operators(L, beta, C, scale_factor=1.0, m=1, tau=1.0, interval=(1.0, 100.0)):
    """
    Construct Pl and Pr according to:
        Pl = g * C * (C^{-1}L)^[β] * ∏ [ C^{-1}(L - zero_i C) ]
        Pr = ∏ [ C^{-1}(L - pole_j C) ]
    
    Parameters
    ----------
    L : scipy.sparse matrix
        Stiffness matrix.
    beta : float
        Fractional exponent.
    C : scipy.sparse matrix
        Mass matrix.
    scale_factor : float, optional
        Normalization factor (default 1.0).
    m : int, optional
        Rational approximation order.
    tau : float or array-like, optional
        Temporal scaling parameter.
    interval : tuple(float, float), optional
        Approximation interval for BRASIL.
    
    Returns
    -------
    Pl, Pr : scipy.sparse.csr_matrix
        Left and right fractional operators.
    """

    n = C.shape[0]

    # --- basic checks ---
    if np.min(np.atleast_1d(tau)) <= 0:
        raise ValueError("tau must be positive")
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a positive integer")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    # --- normalize L by scale_factor ---
    L = (L / scale_factor).tocsr()

    # --- lumped mass (diagonal C) ---
    C_diag = np.array(C.sum(axis=1)).flatten()
    if np.any(C_diag <= 0):
        raise ValueError("C must be positive definite.")
    Ci = sp.diags(1.0 / C_diag)
    C = sp.diags(C_diag)
    I = sp.identity(n, format="csr")

    # --- decompose beta ---
    beta_floor = int(np.floor(beta))
    beta_frac = beta - beta_floor

    # --- get BRASIL rational approximation ---
    if abs(beta_frac) > 1e-14:
        r = brasil(lambda x: x ** beta_frac, interval, m)
        zeros = np.array(r.zeros())
        poles = np.array(r.poles())
        gain = float(r.gain())
    else:
        zeros, poles, gain = [], [], 1.0

    # --- build Pr and Pl with optimized approach ---
    # Compute CiL once for better stability and efficiency
    CiL = Ci @ L
    
    # Build Pr (product over poles) - highly optimized
    if len(poles) > 0:
        # Use polynomial evaluation approach for better stability
        Pr = I.copy()
        for p in poles:
            Pr = Pr @ (CiL - p * I)
    else:
        Pr = I.copy()
    Pr = Pr.tocsr()

    # Build Pl (full product with gain and integer part) - highly optimized
    # Start with integer component: (C^{-1}L)^[β]
    if beta_floor > 0:
        # More efficient: compute C @ CiL^beta_floor directly
        A_int = I.copy()
        for _ in range(beta_floor):
            A_int = A_int @ CiL
        Pl = gain * (C @ A_int)
    else:
        # For pure fractional case (0 < beta < 1) - start with C
        Pl = gain * C
    
    # Apply zeros - highly optimized
    if len(zeros) > 0:
        # Direct multiplication for better numerical stability
        for z in zeros:
            Pl = Pl @ (CiL - z * I)
    
    # Add extra C @ multiplication for stability (only for beta >= 1)
    # This is equivalent to multiplying by (C @ CiL)^(beta_floor-1)
    extra_iterations = max(1, beta_floor) - 1
    if extra_iterations > 0:
        # More efficient: compute (C @ CiL)^extra_iterations
        C_CiL = C @ CiL
        for _ in range(extra_iterations):
            Pl = C_CiL @ Pl
    
    # Apply final scaling
    Pl = (scale_factor ** beta) * Pl.tocsr()

    # --- tau scaling on Pr (same as original FEM version) ---
    if np.isscalar(tau):
        Phi = sp.diags([1.0 / tau] * n)
    else:
        Phi = sp.diags(1.0 / np.asarray(tau))
    Pr = Phi @ Pr

    return Pl, Pr