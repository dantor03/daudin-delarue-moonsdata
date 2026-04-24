"""Solver de la solucion explicita del Teorema 1.4 de Daudin-Delarue (2025).

Resuelve el sistema forward-backward acoplado por Picard fixed-point:

    Forward (continuidad):
        d/dt gamma_t + div_x ( b(.,nu_t) gamma_t ) = 0,   gamma_{t0}=gamma_0

    Backward (transporte adjunto):
        -d/dt u_t - b(.,nu_t) . grad_x u_t = 0,           u_T = L

    Optimalidad (Gibbs explicito, ec. del Teorema 1.4):
        nu*_t(a) = (1/Z_t) exp( -ell(a) - (1/eps) E_{gamma_t}[ b(x,a) . grad_x u(t,x,y) ] )

Setup (maxima fidelidad al paper, Ejemplo 1.3):
    - d_1 = d_2 = 1
    - L(x,y) = 0.5 (x-y)^2  (cuadratica)
    - b(x,a) = tanh(a1 x + a2) a0,   a = (a0,a1,a2) en R^3
    - ell(a) = c1 |a|^4 + c2 |a|^2
    - A discretizado en grid uniforme [-R,R]^3
    - gamma_t representada por N particulas (x_i(t), y_i)
    - Backward via caracteristicas: u(t,x,y) = L(X^{t,x}(T), y),
      grad_x u(t, X_i(t), y_i) = (X_i(T) - y_i) * J_i(T) / J_i(t)
      donde J_i(t) = dX_i(t)/dx_i(0) integra la ec. variacional
        dJ/dt = (d/dx) bbar(X_i(t), nu_t) * J,  J(0)=1
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


# =============================================================================
# Vector field y potencial
# =============================================================================
def b(x, A):
    """b(x, a) = tanh(a1 x + a2) a0.

    x : (N,)
    A : (M, 3) grid de parametros, columnas (a0, a1, a2)
    -> (N, M) matriz con b(x_i, a_l)
    """
    pre = A[:, 1][None, :] * x[:, None] + A[:, 2][None, :]   # (N, M)
    return np.tanh(pre) * A[:, 0][None, :]


def db_dx(x, A):
    """d/dx b(x,a) = sech^2(a1 x + a2) * a1 * a0."""
    pre = A[:, 1][None, :] * x[:, None] + A[:, 2][None, :]
    sech2 = 1.0 - np.tanh(pre) ** 2
    return sech2 * A[:, 1][None, :] * A[:, 0][None, :]


def ell(A, c1=0.05, c2=0.5):
    """Potencial ell(a) = c1 |a|^4 + c2 |a|^2.  A : (M, 3) -> (M,)."""
    norm2 = (A ** 2).sum(axis=1)
    return c1 * norm2 ** 2 + c2 * norm2


# =============================================================================
# Grid sobre A = R^3
# =============================================================================
def build_grid(R=2.5, M_per_dim=10):
    """Grid uniforme en [-R,R]^3.  Devuelve (A, da) con A:(M,3), da: volumen de celda."""
    g = np.linspace(-R, R, M_per_dim)
    A0, A1, A2 = np.meshgrid(g, g, g, indexing="ij")
    A = np.stack([A0.ravel(), A1.ravel(), A2.ravel()], axis=1)   # (M_per_dim^3, 3)
    da = (2 * R / (M_per_dim - 1)) ** 3
    return A, da


# =============================================================================
# Forward: integra particulas y sensibilidad J
# =============================================================================
def forward_pass(nu, x0, A, dt, K, scheme="rk4"):
    """Integra (X_i, J_i) en [0,T] con dinamica bbar(x, nu_t).

    nu : (K+1, M) probabilidades discretas en cada t_k
    x0 : (N,)
    -> X : (K+1, N), J : (K+1, N)
    """
    N = x0.shape[0]
    M = A.shape[0]
    X = np.zeros((K + 1, N))
    J = np.zeros((K + 1, N))
    X[0] = x0
    J[0] = 1.0

    for k in range(K):
        if scheme == "euler":
            bX = b(X[k], A)              # (N, M)
            dbX = db_dx(X[k], A)         # (N, M)
            bbar = bX @ nu[k]            # (N,)
            dxbar = dbX @ nu[k]
            X[k + 1] = X[k] + dt * bbar
            J[k + 1] = J[k] + dt * dxbar * J[k]
        else:  # rk4 (interpolando nu linealmente entre t_k y t_{k+1})
            nu_mid = 0.5 * (nu[k] + nu[k + 1])
            nu_end = nu[k + 1]
            x = X[k]; j = J[k]

            bx = b(x, A);   dbx = db_dx(x, A)
            kx1 = bx @ nu[k];     kj1 = (dbx @ nu[k]) * j

            x2 = x + 0.5 * dt * kx1;   j2 = j + 0.5 * dt * kj1
            bx = b(x2, A);  dbx = db_dx(x2, A)
            kx2 = bx @ nu_mid;    kj2 = (dbx @ nu_mid) * j2

            x3 = x + 0.5 * dt * kx2;   j3 = j + 0.5 * dt * kj2
            bx = b(x3, A);  dbx = db_dx(x3, A)
            kx3 = bx @ nu_mid;    kj3 = (dbx @ nu_mid) * j3

            x4 = x + dt * kx3;         j4 = j + dt * kj3
            bx = b(x4, A);  dbx = db_dx(x4, A)
            kx4 = bx @ nu_end;    kj4 = (dbx @ nu_end) * j4

            X[k + 1] = x + dt / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
            J[k + 1] = j + dt / 6 * (kj1 + 2 * kj2 + 2 * kj3 + kj4)
    return X, J


# =============================================================================
# Update Gibbs explicito
# =============================================================================
def update_nu(X, J, y, A, eps, log_nu_inf):
    """Aplica la formula de Gibbs (Teorema 1.4) en cada t_k.

    nu_new[k, l] propto exp( -ell(a_l) - (1/eps) (1/N) sum_i b(X[k,i],a_l) * dxu_i_k )
    donde dxu_i_k = (X[K,i] - y_i) * J[K,i] / J[k,i].
    """
    K_plus_1, N = X.shape
    M = A.shape[0]
    new_nu = np.zeros((K_plus_1, M))
    dxu_T = X[-1] - y               # (N,)
    JT = J[-1]                      # (N,)

    for k in range(K_plus_1):
        # grad_x u en (t_k, X_i(t_k))
        dxu_k = dxu_T * JT / J[k]   # (N,)
        bXk = b(X[k], A)            # (N, M)
        # term[l] = (1/N) sum_i b(X_i(t_k), a_l) * dxu_i(t_k)
        term = (bXk * dxu_k[:, None]).mean(axis=0)   # (M,)
        # log nu propto -ell(a) - (1/eps) term  =>  log_nu_inf = -ell(a) (modulo cte)
        log_w = log_nu_inf - term / eps
        new_nu[k] = np.exp(log_w - logsumexp(log_w))
    return new_nu


# =============================================================================
# Coste J = terminal + eps * integral entropica
# =============================================================================
def cost(X, y, nu, log_nu_inf, eps, dt):
    """J = (1/N) sum_i 0.5 (X_i(T)-y_i)^2  +  eps * sum_k dt * KL(nu_k || nu_inf)."""
    N = X.shape[1]
    terminal = 0.5 * ((X[-1] - y) ** 2).mean()
    # KL discreta: sum_l nu_l * (log nu_l - log nu_inf_l)
    log_nu = np.log(np.clip(nu, 1e-300, None))
    kl = (nu * (log_nu - log_nu_inf[None, :])).sum(axis=1)   # (K+1,)
    # nu_inf_l aqui ya esta normalizado en el grid (log_nu_inf incluye la cte)
    entropic = eps * dt * kl.sum()
    return terminal + entropic, terminal, entropic


# =============================================================================
# Picard fixed-point
# =============================================================================
@dataclass
class PicardResult:
    nu: np.ndarray            # (K+1, M)
    X: np.ndarray             # (K+1, N)
    J: np.ndarray             # (K+1, N)
    cost_history: list        # lista de J por iteracion
    diff_history: list        # ||nu_new - nu_old||_TV por iteracion
    iters: int
    converged: bool


def picard(x0, y, A, da,
           T=1.0, K=20, eps=0.5,
           c1=0.05, c2=0.5,
           omega=0.3, max_iter=200, tol=1e-4,
           scheme="rk4", verbose=True, nu_init=None):
    """Picard fixed-point sobre (gamma, u, nu).

    Por defecto inicializa nu_t = nu_inf (prior) para todo t.
    Si se pasa nu_init (shape (K+1, M)), lo usa como warm-start (continuacion).
    """
    M = A.shape[0]
    dt = T / K

    # log nu_inf normalizado en el grid (es nuestra version discreta de nu^infty)
    log_w_inf = -ell(A, c1, c2)
    log_nu_inf = log_w_inf - logsumexp(log_w_inf)
    nu_inf = np.exp(log_nu_inf)

    # Inicializacion: warm-start si nu_init dado, si no nu_inf
    if nu_init is not None:
        assert nu_init.shape == (K + 1, M), f"nu_init shape {nu_init.shape} != ({K+1}, {M})"
        nu = nu_init.copy()
    else:
        nu = np.tile(nu_inf, (K + 1, 1))

    cost_hist, diff_hist = [], []
    converged = False
    for it in range(max_iter):
        X, J = forward_pass(nu, x0, A, dt, K, scheme=scheme)
        nu_new = update_nu(X, J, y, A, eps, log_nu_inf)
        # damping
        nu_next = (1 - omega) * nu + omega * nu_new
        # renormalizar (por si numeria)
        nu_next /= nu_next.sum(axis=1, keepdims=True)
        diff = np.abs(nu_next - nu).sum(axis=1).max() / 2   # TV maximo en t
        Jval, term, ent = cost(X, y, nu_next, log_nu_inf, eps, dt)
        cost_hist.append(Jval)
        diff_hist.append(diff)
        if verbose and (it < 5 or it % 10 == 0):
            print(f"  iter {it:3d}  J={Jval:.6f}  term={term:.6f}  ent={ent:.6f}  diffTV={diff:.2e}")
        nu = nu_next
        if diff < tol:
            converged = True
            if verbose:
                print(f"  converged at iter {it} (diff={diff:.2e} < tol={tol:.2e})")
            break

    X, J = forward_pass(nu, x0, A, dt, K, scheme=scheme)
    return PicardResult(nu=nu, X=X, J=J,
                        cost_history=cost_hist, diff_history=diff_hist,
                        iters=it + 1, converged=converged)


def picard_continuation(x0, y, A, da, eps_schedule,
                        T=1.0, K=20, c1=0.05, c2=0.5,
                        omega=0.2, max_iter=300, tol=1e-5,
                        scheme="rk4", verbose=True):
    """Continuacion en eps: resuelve secuencialmente con warm-start.

    eps_schedule: lista decreciente de eps, e.g. [0.5, 0.2, 0.1, 0.05, 0.02, 0.01].
    Cada paso usa la solucion del eps anterior como nu_init.

    Retorna lista de (eps, PicardResult).
    """
    results = []
    nu_warm = None
    for i, eps in enumerate(eps_schedule):
        if verbose:
            print(f"\n[continuation {i+1}/{len(eps_schedule)}] eps={eps}")
        res = picard(x0, y, A, da, T=T, K=K, eps=eps,
                     c1=c1, c2=c2, omega=omega, max_iter=max_iter, tol=tol,
                     scheme=scheme, verbose=verbose, nu_init=nu_warm)
        results.append((eps, res))
        nu_warm = res.nu   # warm-start para el siguiente eps
    return results
