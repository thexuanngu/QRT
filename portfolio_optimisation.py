import numpy as np


# Minimum Variance Portfolio (MVP) - Method 1
def MVP1(exp_ret, cov_matrix):
    """
    Minimum-Variance Portfolio (MVP) under the full-investment constraint u w^T = 1.
    
    Inputs
    ------
    exp_ret : (n,) ndarray      # expected returns
    cov_matrix  : (n,n) ndarray     # covariance matrix (symmetric PD)

    Returns
    -------
    w      : (n,) ndarray  # MVP weights (sum to 1)
    mu_p   : float         # expected return of the MVP
    std_p  : float         # standard deviation of the MVP
    """

    n = exp_ret.shape[0]
    u = np.ones(n)

    L = np.linalg.cholesky(cov_matrix)  # C = L L^T
    y = np.linalg.solve(L.T, np.linalg.solve(L, u))

    denom = u @ y                    # u^T C^{-1} u
    w = y / denom                    # w_MVP
    mu_p = exp_ret @ w                    # μ_MVP
    var_p = 1.0 / denom              # σ²_MVP
    std_p = np.sqrt(var_p)           # σ_MVP

    return w, mu_p, std_p



# Minimum Variance Portfolio (MVP) Given a Fixed Expected Return - Method 2
def MVP2(exp_ret, cov_matrix, target_ret):
    """
    Minimum-variance portfolio achieving expected return mu_target.

    Inputs
    ------
    exp_ret : (n,) ndarray      # expected returns
    cov_matrix  : (n,n) ndarray     # covariance matrix (symmetric PD)
    target_ret : float      # target expected return

    Returns
    -------
    w      : (n,) ndarray  weights (sum to 1), achieving mu_target
    mu_p   : float         expected return (~= mu_target, up to roundoff)
    std_p  : float         minimal standard deviation at mu_target
    """

    n = exp_ret.shape[0]
    u = np.ones(n, dtype=cov_matrix.dtype)

    # One Cholesky factorization; solve for C^{-1}u and C^{-1}μ in a single pass
    L = np.linalg.cholesky(cov_matrix)
    X = np.linalg.solve(L, np.column_stack((u, exp_ret)))   # solve L Z = [u, μ]
    Y = np.linalg.solve(L.T, X)                        # solve L^T Y = Z
    y_u, y_mu = Y[:, 0], Y[:, 1]                       # C^{-1} u, C^{-1} μ

    # Scalars A, B, Cc and determinant Δ
    A  = float(u @ y_u)            # u^T C^{-1} u
    B  = float(u @ y_mu)           # u^T C^{-1} μ = μ^T C^{-1} u
    Cc = float(exp_ret @ y_mu)          # μ^T C^{-1} μ
    Delta = A * Cc - B * B

    # Lagrange multipliers and weights
    lam = (Cc - B * target_ret) / Delta
    gam = (A * target_ret - B) / Delta
    w = lam * y_u + gam * y_mu

    # Portfolio stats
    mu_p  = float(exp_ret @ w)                                  # should ≈ mu_target
    var_p = (A * target_ret**2 - 2 * B * target_ret+ Cc) / Delta  # closed form
    std_p = np.sqrt(var_p)

    return w, mu_p, std_p



# Portfolio with Maximum Portfolio Performance Criteria(MPP)
def MPP(exp_ret, cov_matrix, gamma):
    """
    Maximize w^T mu - (gamma/2) w^T C w  subject to 1^T w = 1.

    Inputs
    ------
    exp_ret : (n,) ndarray      # expected returns
    cov_matrix  : (n,n) ndarray     # covariance matrix (symmetric PD)
    gamma : float               # risk aversion parameter

    Returns
    -------
    w      : (n,) ndarray  optimal weights
    mu_p   : float         expected return of the optimal portfolio
    std_p  : float         standard deviation of the optimal portfolio
    """

    n = exp_ret.shape[0]
    u = np.ones(n, dtype=cov_matrix.dtype)

    # One Cholesky factorization; solve for C^{-1}u and C^{-1}mu simultaneously
    L = np.linalg.cholesky(cov_matrix)
    Z = np.linalg.solve(L, np.column_stack((u, exp_ret)))
    Y = np.linalg.solve(L.T, Z)
    y_u, y_mu = Y[:, 0], Y[:, 1]           # C^{-1}u, C^{-1}mu

    # Scalars
    A  = float(u @ y_u)                     # u^T C^{-1} u
    B  = float(u @ y_mu)                    # u^T C^{-1} mu
    Cc = float(exp_ret @ y_mu)                   # mu^T C^{-1} mu
    psi = A * Cc - B * B                    # same ψ as in Theorem 2.5

    # MVP stats
    mu_mvp  = B / A
    var_mvp = 1.0 / A

    # Optimal weights (projection of C^{-1}mu onto the budget hyperplane)
    w = (y_u / A) + (1.0 / gamma) * (y_mu - (B / A) * y_u)

    # Portfolio stats (closed forms; equivalent to dot-products but cheaper)
    mu_p  = mu_mvp + (psi * var_mvp) / gamma
    var_p = var_mvp + (psi * var_mvp) / (gamma ** 2)
    std_p = np.sqrt(var_p)

    return w, mu_p, std_p
    