"""Generalizacion n-D del solver explicito (Daudin-Delarue Thm 1.4).

Diferencias respecto a explicit_solver.py (d1=1):
  - x_i ∈ R^{d1}        en lugar de R
  - a = (a0, a1, a2) con a0 ∈ R^{d1}, a1 ∈ R^{d1}, a2 ∈ R   (dim_a = 2 d1 + 1)
  - J_i(t) ∈ R^{d1 x d1}   (Jacobiano del flujo, NO escalar)
  - b: R^{d1} x R^{dim_a} -> R^{d1}
        b(x, a) = tanh(a1 . x + a2) · a0
  - dx b: R^{d1 x d1}
        (∂b_k / ∂x_j)(x, a) = sech²(a1·x + a2) · a1_j · a0_k
  - Variacional matricial: dJ/dt = (∂x bbar)|_{X(t)} · J,   J(0) = I
  - Gradiente del coste terminal por caracteristicas:
        ∇_x u(t, X_i(t), y_i) = (J_i(T) · J_i(t)^{-1})^T · (X_i(T) - y_i)
    (cocycle del Jacobiano del flujo backward)
  - Gibbs identico: log nu(a) = log nu_inf(a) - (1/eps) (1/N) Σ_i b·∇_x u
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


# =============================================================================
# Vector field y derivada
# =============================================================================
def b_nd(x, A, d1):
    """b(x, a) = tanh(a1 . x + a2) * a0.

    x : (N, d1)
    A : (M, 2 d1 + 1) con columnas [a0 (d1), a1 (d1), a2 (1)]
    -> (N, M, d1)
    """
    a0 = A[:, :d1]                  # (M, d1)
    a1 = A[:, d1:2 * d1]            # (M, d1)
    a2 = A[:, 2 * d1]               # (M,)
    pre = x @ a1.T + a2[None, :]    # (N, M)
    s = np.tanh(pre)                # (N, M)
    return s[:, :, None] * a0[None, :, :]   # (N, M, d1)


def db_dx_nd(x, A, d1):
    """(∂b_k/∂x_j)(x, a) = sech²(a1·x + a2) · a1_j · a0_k.

    -> (N, M, d1, d1) con ultimo eje (k, j)
    """
    a0 = A[:, :d1]
    a1 = A[:, d1:2 * d1]
    a2 = A[:, 2 * d1]
    pre = x @ a1.T + a2[None, :]                # (N, M)
    sech2 = 1.0 - np.tanh(pre) ** 2             # (N, M)
    # outer a0_k * a1_j por particula => (M, d1, d1)
    out = a0[:, :, None] * a1[:, None, :]       # (M, d1, d1)
    return sech2[:, :, None, None] * out[None, :, :, :]   # (N, M, d1, d1)


def ell_nd(A, c1=0.05, c2=0.5):
    """Potencial ell(a) = c1 |a|^4 + c2 |a|^2."""
    n2 = (A ** 2).sum(axis=1)
    return c1 * n2 ** 2 + c2 * n2


# =============================================================================
# Grid sobre A = R^{2 d1 + 1}
# =============================================================================
def build_grid_nd(d1, R=2.0, M_per_dim=4):
    """Grid uniforme en [-R, R]^{2 d1 + 1}.  Devuelve (A, dim_a)."""
    dim_a = 2 * d1 + 1
    g = np.linspace(-R, R, M_per_dim)
    grids = np.meshgrid(*([g] * dim_a), indexing="ij")
    A = np.stack([g_.ravel() for g_ in grids], axis=1)   # (M_per_dim^dim_a, dim_a)
    return A, dim_a


# =============================================================================
# Forward: integra (X, J) por RK4
# =============================================================================
def forward_pass_nd(nu, x0, A, d1, dt, K):
    """Integra (X_i, J_i) en [0, T] con bbar(X, nu_t) y dJ/dt = (dxbar) · J.

    nu : (K+1, M)
    x0 : (N, d1)
    -> X : (K+1, N, d1)
       J : (K+1, N, d1, d1)
    """
    N = x0.shape[0]
    X = np.zeros((K + 1, N, d1))
    J = np.zeros((K + 1, N, d1, d1))
    X[0] = x0
    J[0] = np.broadcast_to(np.eye(d1), (N, d1, d1)).copy()

    def deriv(x, j, nu_k):
        bX = b_nd(x, A, d1)                              # (N, M, d1)
        dbX = db_dx_nd(x, A, d1)                          # (N, M, d1, d1)
        bbar = np.einsum('nmd,m->nd', bX, nu_k)           # (N, d1)
        dbar = np.einsum('nmkj,m->nkj', dbX, nu_k)        # (N, d1, d1)
        dj = np.einsum('nkj,nji->nki', dbar, j)           # (N, d1, d1)
        return bbar, dj

    for k in range(K):
        nu_k, nu_mid, nu_end = nu[k], 0.5 * (nu[k] + nu[k + 1]), nu[k + 1]
        x, j = X[k], J[k]

        kx1, kj1 = deriv(x, j, nu_k)
        kx2, kj2 = deriv(x + 0.5 * dt * kx1, j + 0.5 * dt * kj1, nu_mid)
        kx3, kj3 = deriv(x + 0.5 * dt * kx2, j + 0.5 * dt * kj2, nu_mid)
        kx4, kj4 = deriv(x + dt * kx3, j + dt * kj3, nu_end)

        X[k + 1] = x + dt / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
        J[k + 1] = j + dt / 6 * (kj1 + 2 * kj2 + 2 * kj3 + kj4)
    return X, J


# =============================================================================
# Update de nu via Gibbs
# =============================================================================
def update_nu_nd(X, J, y, A, d1, eps, log_nu_inf):
    """Aplica la formula de Gibbs (Thm 1.4) en cada t_k.

    log nu_new[k, l] = log nu_inf - (1/eps) (1/N) Σ_i b(X_k_i, a_l) . ∇x u(t_k, X_k_i)
    con  ∇x u(t_k, X_k_i) = (J(T)_i · J(t_k)_i^{-1})^T · (X(T)_i - y_i)
    """
    K1, N, _ = X.shape
    M = A.shape[0]
    new_nu = np.zeros((K1, M))
    dxu_T = X[-1] - y                          # (N, d1)
    JT = J[-1]                                  # (N, d1, d1)

    for k in range(K1):
        # M_t = J(T) · J(t_k)^{-1}, gradiente = M_t^T · dxu_T
        Jk_inv = np.linalg.inv(J[k])           # (N, d1, d1)
        M_t = np.einsum('nij,njk->nik', JT, Jk_inv)   # (N, d1, d1)
        dxu_k = np.einsum('nij,ni->nj', M_t, dxu_T)   # (N, d1)
        bXk = b_nd(X[k], A, d1)                # (N, M, d1)
        # term[l] = (1/N) Σ_i Σ_d bXk[i, l, d] * dxu_k[i, d]
        term = np.einsum('nmd,nd->m', bXk, dxu_k) / N
        log_w = log_nu_inf - term / eps
        new_nu[k] = np.exp(log_w - logsumexp(log_w))
    return new_nu


# =============================================================================
# Coste discreto
# =============================================================================
def cost_nd(X, y, nu, log_nu_inf, eps, dt):
    terminal = 0.5 * ((X[-1] - y) ** 2).sum(axis=1).mean()
    log_nu = np.log(np.clip(nu, 1e-300, None))
    kl = (nu * (log_nu - log_nu_inf[None, :])).sum(axis=1)
    entropic = eps * dt * kl.sum()
    return terminal + entropic, terminal, entropic


# =============================================================================
# Picard fixed-point
# =============================================================================
@dataclass
class PicardResultND:
    nu: np.ndarray
    X: np.ndarray
    J: np.ndarray
    cost_history: list
    diff_history: list
    iters: int
    converged: bool


def picard_nd(x0, y, A, d1, *, T=1.0, K=10, eps=0.5,
              c1=0.05, c2=0.5, omega=0.2, max_iter=200, tol=1e-4,
              verbose=True, nu_init=None, log_nu_inf=None):
    """log_nu_inf opcional: si se pasa, se usa en lugar de calcular -ell(A) sobre
    cuadratura uniforme.  Necesario para QMC + importance sampling, donde el
    soporte ya no es un grid uniforme y log_nu_inf debe incluir -log q(a)."""
    M = A.shape[0]
    dt = T / K
    if log_nu_inf is None:
        log_w_inf = -ell_nd(A, c1, c2)
        log_nu_inf = log_w_inf - logsumexp(log_w_inf)
    nu_inf = np.exp(log_nu_inf)

    nu = nu_init.copy() if nu_init is not None else np.tile(nu_inf, (K + 1, 1))

    cost_hist, diff_hist = [], []
    converged = False
    for it in range(max_iter):
        X, J = forward_pass_nd(nu, x0, A, d1, dt, K)
        nu_new = update_nu_nd(X, J, y, A, d1, eps, log_nu_inf)
        nu_next = (1 - omega) * nu + omega * nu_new
        nu_next /= nu_next.sum(axis=1, keepdims=True)
        diff = np.abs(nu_next - nu).sum(axis=1).max() / 2
        Jval, term, ent = cost_nd(X, y, nu_next, log_nu_inf, eps, dt)
        cost_hist.append(Jval)
        diff_hist.append(diff)
        if verbose and (it < 5 or it % 10 == 0):
            print(f"  it {it:3d}  J={Jval:.5f}  term={term:.5f}  ent={ent:.5f}  "
                  f"diffTV={diff:.2e}")
        nu = nu_next
        if diff < tol:
            converged = True
            if verbose:
                print(f"  converged at iter {it} (diff={diff:.2e})")
            break

    X, J = forward_pass_nd(nu, x0, A, d1, dt, K)
    return PicardResultND(nu=nu, X=X, J=J,
                          cost_history=cost_hist, diff_history=diff_hist,
                          iters=it + 1, converged=converged)


def picard_continuation_nd(x0, y, A, d1, eps_schedule, *, T=1.0, K=10,
                            c1=0.05, c2=0.5, omega=0.2, max_iter=200, tol=1e-4,
                            verbose=True, log_nu_inf=None):
    results, nu_warm = [], None
    for i, eps in enumerate(eps_schedule):
        if verbose:
            print(f"\n[continuation {i+1}/{len(eps_schedule)}] eps={eps}")
        res = picard_nd(x0, y, A, d1, T=T, K=K, eps=eps, c1=c1, c2=c2,
                        omega=omega, max_iter=max_iter, tol=tol,
                        verbose=verbose, nu_init=nu_warm,
                        log_nu_inf=log_nu_inf)
        results.append((eps, res))
        nu_warm = res.nu
    return results
