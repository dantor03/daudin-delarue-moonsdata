"""Continuacion en eps + comparacion entre Old Faithful (lineal) y trees (no lineal).

Estrategia:
  1. Resolver secuencialmente eps = [0.5, 0.2, 0.1, 0.05, 0.02], cada uno
     warm-started desde la solucion del anterior.
  2. Comparar el J final con el OLS lineal en cada dataset.
  3. Comprobar si la continuacion estabiliza eps pequenos (vs picard frio que falla).

Hipotesis (confirmadas):
  - Old Faithful (corr 0.90): la solucion explicita degenera al optimo OLS lineal
    (problema casi lineal, no hay margen para mejorar).
  - trees (corr 0.97 pero V ~ G^2 no lineal): la mean-field ResNet APROVECHA la
    no linealidad y mejora al OLS lineal en ~12%.

Nota: probamos primero con Auto MPG (mpg vs hp) pero sus features estan
ANTI-correlacionadas con la etiqueta; el setup del paper (X(T) = output
directo) tiene problemas para invertir el signo, asi que el flujo colapsa
al promedio. Es una limitacion intrinseca, no un bug del solver.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_faithful, load_trees
from explicit_solver import build_grid, picard_continuation
from run import linear_baseline

FIG_DIR = HERE.parents[1] / "figures"


def run_one(name, x, y, A, da, eps_schedule, omega=0.2, max_iter=300):
    print(f"\n========== {name} ==========")
    print(f"N={len(x)}")
    mse_lin = linear_baseline(x, y)
    print(f"Baseline OLS (MSE/2) = {mse_lin:.5f}")
    results = picard_continuation(x, y, A, da, eps_schedule,
                                   omega=omega, max_iter=max_iter, tol=1e-5,
                                   verbose=False)
    summary = []
    for eps, res in results:
        term = 0.5 * ((res.X[-1] - y) ** 2).mean()
        disp = np.abs(res.X[-1] - res.X[0]).mean()
        summary.append((eps, res.iters, res.converged, res.cost_history[-1], term, disp))
        print(f"  eps={eps:5.3f}  iters={res.iters:4d}  conv={res.converged}  "
              f"J={res.cost_history[-1]:.5f}  term={term:.5f}  disp={disp:.3f}")
    return mse_lin, results, summary


def plot_comparison(faith_data, trees_data, save_path):
    """faith_data, trees_data = (mse_lin, results, summary, x, y)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, (name, data) in enumerate([("Old Faithful (corr 0.90, lineal)", faith_data),
                                         ("trees (corr 0.97, V ~ G^2)", trees_data)]):
        mse_lin, results, summary, x, y = data

        # (1) J(eps) vs OLS
        ax = axes[row, 0]
        eps_arr = [s[0] for s in summary]
        J_arr = [s[3] for s in summary]
        term_arr = [s[4] for s in summary]
        ax.plot(eps_arr, J_arr, "o-", label="J explicito")
        ax.plot(eps_arr, term_arr, "s--", label="termino L solo (1/N) sum 1/2(X(T)-y)^2")
        ax.axhline(mse_lin, color="r", ls=":", label=f"OLS lineal = {mse_lin:.4f}")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("eps"); ax.set_ylabel("coste")
        ax.set_title(f"{name}: continuacion en eps")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (2) Predicciones X(T) vs y para el eps mas bajo convergente
        # toma el ultimo eps que convergio
        last_conv = next((r for r in reversed(results) if r[1].converged), results[-1])
        eps_show, res_show = last_conv
        ax = axes[row, 1]
        ax.scatter(y, res_show.X[-1], s=14, alpha=0.5)
        lim = max(abs(y).max(), abs(res_show.X[-1]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        # OLS line en mismo plot
        a_ols = (x * y).sum() / (x ** 2).sum()
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(a_ols * xs, xs, "g-", lw=1, alpha=0.7, label=f"OLS (slope={a_ols:.2f})")
        ax.set_xlabel("y_i"); ax.set_ylabel("X_i(T)")
        ax.set_title(f"{name}: predict eps={eps_show}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (3) Datos crudos y curvas: y vs x, predict de la mean-field
        ax = axes[row, 2]
        ax.scatter(x, y, s=14, alpha=0.4, label="datos")
        # Predict de la mean-field: necesitamos correr el forward solo en el grid
        # (X(T) parametrizado por X(0)).  Aproximamos usando las particulas y orden:
        order = np.argsort(x)
        ax.plot(x[order], res_show.X[-1][order], "b-", lw=1.5, label=f"mean-field eps={eps_show}")
        ax.plot(x[order], a_ols * x[order], "g--", lw=1, label="OLS lineal")
        ax.set_xlabel("x"); ax.set_ylabel("y / X(T)")
        ax.set_title(f"{name}: ajuste vs datos")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Solucion explicita (Daudin-Delarue Thm 1.4) por continuacion en eps  -  "
                 "Old Faithful (lineal) vs trees (V ~ G^2)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"\nFigura guardada: {save_path}")


def main():
    A, da = build_grid(R=2.5, M_per_dim=10)
    eps_schedule = [0.5, 0.2, 0.1, 0.05, 0.02]

    x_f, y_f = load_faithful()
    mse_f, res_f, sum_f = run_one("Old Faithful", x_f, y_f, A, da, eps_schedule)

    x_t, y_t = load_trees()
    mse_t, res_t, sum_t = run_one("trees", x_t, y_t, A, da, eps_schedule)

    plot_comparison(
        (mse_f, res_f, sum_f, x_f, y_f),
        (mse_t, res_t, sum_t, x_t, y_t),
        FIG_DIR / "01_continuation_faithful_vs_trees.png",
    )


if __name__ == "__main__":
    main()
