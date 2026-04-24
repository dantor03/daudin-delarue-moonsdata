"""Ejecuta la solucion explicita en Old Faithful y produce figuras.

Uso:  python -m paper.src.01_solucion_explicita.run
o    cd paper/src/01_solucion_explicita && python run.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_faithful
from explicit_solver import build_grid, picard

FIG_DIR = HERE.parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def linear_baseline(x, y):
    """OLS y = a x + b en datos estandarizados.  Retorna MSE/2."""
    a = (x * y).sum() / (x ** 2).sum()
    yhat = a * x
    return 0.5 * ((yhat - y) ** 2).mean()


def main(eps=0.05, R=2.5, M_per_dim=10, K=20, T=1.0,
         omega=0.3, max_iter=200, tol=1e-5, seed=0):
    np.random.seed(seed)
    x, y = load_faithful()
    A, da = build_grid(R=R, M_per_dim=M_per_dim)
    print(f"Old Faithful: N={len(x)}")
    print(f"Grid A: {A.shape[0]} pts en R^3, R=±{R}")
    print(f"eps={eps}, K={K}, T={T}, omega={omega}\n")

    res = picard(x, y, A, da, T=T, K=K, eps=eps,
                 omega=omega, max_iter=max_iter, tol=tol, scheme="rk4")
    print(f"\nPicard: iters={res.iters}, converged={res.converged}")
    print(f"J final = {res.cost_history[-1]:.6f}")

    mse_lin = linear_baseline(x, y)
    print(f"Baseline OLS lineal (MSE/2) = {mse_lin:.6f}")

    # =============== FIGURA ===============
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (1) Convergencia Picard
    ax = axes[0, 0]
    ax.plot(res.cost_history, "o-", ms=3, label="J(itera)")
    ax.axhline(mse_lin, color="r", ls="--", label="OLS lineal (MSE/2)")
    ax.set_xlabel("iteracion Picard")
    ax.set_ylabel("J")
    ax.set_title(f"Convergencia (eps={eps})")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) Diferencia TV entre iteraciones
    ax = axes[0, 1]
    ax.semilogy(res.diff_history, "o-", ms=3)
    ax.axhline(tol, color="r", ls="--", label=f"tol={tol:.0e}")
    ax.set_xlabel("iteracion Picard")
    ax.set_ylabel("max_t ||nu_new - nu_old||_TV")
    ax.set_title("Convergencia (norma TV)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) Predicciones X(T) vs y
    ax = axes[0, 2]
    ax.scatter(y, res.X[-1], s=14, alpha=0.5, label="X_i(T) vs y_i")
    lim = max(abs(y).max(), abs(res.X[-1]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
    ax.set_xlabel("y_i (label estandarizado)")
    ax.set_ylabel("X_i(T)")
    ax.set_title("Prediccion final")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) Trayectorias de particulas
    ax = axes[1, 0]
    t_grid = np.linspace(0, T, K + 1)
    n_show = min(40, res.X.shape[1])
    idx_show = np.random.choice(res.X.shape[1], n_show, replace=False)
    for i in idx_show:
        ax.plot(t_grid, res.X[:, i], color="C0", alpha=0.25, lw=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("X_i(t)")
    ax.set_title(f"Trayectorias de {n_show} particulas")
    ax.grid(alpha=0.3)

    # (5) Marginales de nu*_t en a0 (tres tiempos)
    ax = axes[1, 1]
    M = M_per_dim
    nu_3d = res.nu.reshape(K + 1, M, M, M)   # (t, a0, a1, a2)
    g = np.linspace(-R, R, M)
    for k_show, color in zip([0, K // 2, K], ["C0", "C1", "C2"]):
        marg_a0 = nu_3d[k_show].sum(axis=(1, 2))     # marginal en a0
        ax.plot(g, marg_a0, "o-", color=color, ms=4, label=f"t={t_grid[k_show]:.2f}")
    ax.set_xlabel("a0")
    ax.set_ylabel("nu_t (marginal a0)")
    ax.set_title("Marginal de nu*_t en a0")
    ax.legend()
    ax.grid(alpha=0.3)

    # (6) Marginales en a1
    ax = axes[1, 2]
    for k_show, color in zip([0, K // 2, K], ["C0", "C1", "C2"]):
        marg_a1 = nu_3d[k_show].sum(axis=(0, 2))
        ax.plot(g, marg_a1, "o-", color=color, ms=4, label=f"t={t_grid[k_show]:.2f}")
    ax.set_xlabel("a1")
    ax.set_ylabel("nu_t (marginal a1)")
    ax.set_title("Marginal de nu*_t en a1")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Solucion explicita (Daudin-Delarue Thm 1.4)  -  Old Faithful  "
        f"(eps={eps}, grid {M}^3, K={K}, RK4)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / f"01_explicit_old_faithful_eps{eps}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigura guardada en: {out}")
    return res


if __name__ == "__main__":
    main()
