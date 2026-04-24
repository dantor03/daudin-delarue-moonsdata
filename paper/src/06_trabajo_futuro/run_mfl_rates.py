"""Seccion 6.1 -- Estudio cuantitativo de la propagacion del caos.

Propuesta del paper (\ref{ssec:tf-rates}):
  Variar M in {8, 16, 32, 64, 128, 256, 512, 1024} y, para cada uno, medir
    MMD^2(nu_T^{proxy}, nu^*_T)
  por metodo (pSGLD, MMD^2, Sinkhorn). Ajustar MMD^2 ~ M^{-alpha}.
  Prediccion Monte Carlo: alpha = 1 para pSGLD (rate M^{-1/2} en MMD).

Setup:
  make_moons N=400, d_1 = d_2 = 2, eps = 0.05, T = 1, n_steps = 10.
  nu^*_T se calcula una sola vez por Picard con continuacion en eps
  [0.5, 0.2, 0.1, 0.05] sobre grid 4^5 = 1024 puntos en A = R^5.
  El kernel Gaussiano del MMD usa un sigma FIJO (mediana heuristica sobre
  S=2000 muestras de nu^*_T) para que los valores sean comparables entre
  distintos M. Se usa el estimador UNBIASED para no confundir la tasa
  asintotica con el sesgo 1/M del estimador biased.

Outputs:
  paper/src/06_trabajo_futuro/rates_results.csv  (seed, M, metodo, MMD^2)
  paper/figures/06_mfl_rates.png                  (log-log + ajustes)
  Tabla por stdout con alpha +/- SE por metodo.
"""
import argparse
import sys
import time
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "src" / "02_regularizadores_moons"))
sys.path.insert(0, str(ROOT / "paper" / "src" / "04_metodos_alternativos"))

from codigo.config import SEED, DEVICE  # noqa: E402
from codigo.data import get_moons  # noqa: E402
from codigo.model import MeanFieldResNet  # noqa: E402
from codigo.train import train  # noqa: E402
from codigo.metrics import sample_prior_langevin  # noqa: E402

from explicit_nd import build_grid_nd, picard_continuation_nd  # noqa: E402
from bridge_distributions import (  # noqa: E402
    extract_neurons, sample_from_nu, embed_labels,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)
CSV_OUT = Path(__file__).resolve().parent / "rates_results.csv"

METHOD_COLORS = {
    "pSGLD":    "#f39c12",
    "MMD$^2$":  "#2ecc71",
    "Sinkhorn": "#3498db",
}


# ---------------------------------------------------------------------------
# MMD^2 unbiased con kernel Gaussiano y sigma fijo
# ---------------------------------------------------------------------------
def median_heuristic_sigma(samples, subsample=500, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(samples), subsample)
    idx = rng.choice(len(samples), n, replace=False)
    sub = samples[idx]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    d2 = d2[d2 > 0]
    return float(np.sqrt(0.5 * np.median(d2)))


def mmd2_unbiased(X, Y, sigma):
    """Estimador MMD^2 UNBIASED, kernel k(x,y) = exp(-|x-y|^2 / (2 sigma^2)).

    E[MMD^2_u] = MMD^2(P, Q); varianza -> 0 con n_X, n_Y -> infinito.
    Puede ser negativo por fluctuacion muestral cuando P ~= Q.
    """
    def Kmat(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))

    n, m = len(X), len(Y)
    Kxx = Kmat(X, X)
    Kyy = Kmat(Y, Y)
    Kxy = Kmat(X, Y)
    # Quitar diagonales para el estimador unbiased
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.sum() / (n * m)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


# ---------------------------------------------------------------------------
# Entrenamiento de un proxy para un valor de M y seed
# ---------------------------------------------------------------------------
def train_one(method, M, X, y, eps, n_epochs, prior, seed):
    torch.manual_seed(seed)
    model = MeanFieldResNet(d1=2, M=M, T=1.0, n_steps=10).to(DEVICE)
    if method == "pSGLD":
        train(model, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
              verbose=False, use_sgld=True)
    elif method == "MMD$^2$":
        train(model, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
              verbose=False, use_mmd=True, prior_samples=prior)
    elif method == "Sinkhorn":
        train(model, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
              verbose=False, use_sinkhorn=True, prior_samples=prior,
              sinkhorn_blur=0.05)
    else:
        raise ValueError(method)
    return model


# ---------------------------------------------------------------------------
# Ajuste de ley de potencias  log(MMD^2) = beta - alpha * log(M)
# ---------------------------------------------------------------------------
def fit_power_law(Ms, mmd2s):
    """Regresion lineal log-log con MMD^2 > 0. Devuelve (alpha, SE, intercept).

    Puntos con MMD^2 <= 0 se descartan (ruido muestral: el proxy ya esta en
    el ruido del estimador, la ley de potencias ya no se puede estimar).
    """
    Ms = np.asarray(Ms, dtype=float)
    mmd2s = np.asarray(mmd2s, dtype=float)
    mask = mmd2s > 0
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, mask
    x = np.log10(Ms[mask])
    y = np.log10(mmd2s[mask])
    X = np.vstack([np.ones_like(x), -x]).T  # y = beta + (-alpha)*(-x)
    coef, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta, alpha = coef
    yhat = X @ coef
    resid = y - yhat
    dof = max(1, len(x) - 2)
    sigma2 = (resid ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(cov[1, 1]))
    return float(alpha), se_alpha, float(beta), mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--S", type=int, default=2000,
                   help="muestras de nu^*_T para estimador MMD^2")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["pSGLD", "MMD$^2$", "Sinkhorn"])
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print("Seccion 6.1 -- MFL rates: MMD^2 vs M")
    print(f"  M list : {args.M}")
    print(f"  seeds  : {args.seeds}")
    print(f"  epochs : {args.epochs}")
    print(f"  eps    : {args.eps}")
    print("=" * 72)

    # --- Datos make_moons ---------------------------------------------------
    X, y, X_np, y_np = get_moons()
    y_emb = embed_labels(y_np)
    print(f"  dataset: make_moons N={len(y)}  d=2")

    # --- nu^*_T explicito ---------------------------------------------------
    A_grid, _ = build_grid_nd(d1=2, R=2.0, M_per_dim=4)
    print(f"  grid R^5  |A|={A_grid.shape[0]}  Picard con "
          "eps_schedule=[0.5,0.2,0.1,0.05]")
    t0 = time.time()
    results = picard_continuation_nd(
        X_np.astype(float), y_emb.astype(float), A_grid, d1=2,
        eps_schedule=[0.5, 0.2, 0.1, args.eps],
        T=1.0, K=10, omega=0.2, max_iter=200, tol=1e-4, verbose=False,
    )
    nu_T = results[-1][1].nu[-1]
    print(f"  Picard listo en {time.time()-t0:.1f}s  eps_final={args.eps}")

    rng = np.random.default_rng(0)
    samples_explicit = sample_from_nu(A_grid, nu_T, args.S, rng)
    sigma = median_heuristic_sigma(samples_explicit)
    print(f"  sigma (kernel) fijo = {sigma:.3f} "
          f"(mediana heuristica sobre {args.S} muestras de nu^*_T)")

    # --- Prior comun para MMD^2 / Sinkhorn ----------------------------------
    prior = sample_prior_langevin(n_samples=400, epsilon=args.eps, dim=5)

    # --- Barrido en M -------------------------------------------------------
    rows = []  # (seed, M, method, mmd2)
    for seed in args.seeds:
        for M in args.M:
            for method in args.methods:
                t0 = time.time()
                model = train_one(method, M, X, y, args.eps, args.epochs,
                                  prior, seed=seed)
                neur = extract_neurons(model, t=1.0)
                mmd2 = mmd2_unbiased(neur, samples_explicit, sigma=sigma)
                dt = time.time() - t0
                rows.append((seed, M, method, mmd2))
                print(f"  seed={seed} M={M:<4d} {method:<10s} "
                      f"MMD^2={mmd2:+.5f}  ({dt:5.1f}s)")

    # --- Guardar CSV --------------------------------------------------------
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "M", "method", "mmd2"])
        w.writerows(rows)
    print(f"\n  -> {CSV_OUT}")

    # --- Ajustes por metodo -------------------------------------------------
    fits = {}
    print("\n" + "=" * 72)
    print(f"  {'metodo':<12s} {'alpha':>10s} {'SE':>10s} {'intercept':>12s} "
          f"{'n puntos':>10s}")
    print("-" * 72)
    for method in args.methods:
        subset = [(M, v) for (_, M, m, v) in rows if m == method]
        Ms = [s[0] for s in subset]
        vs = [s[1] for s in subset]
        alpha, se, beta, mask = fit_power_law(Ms, vs)
        fits[method] = dict(alpha=alpha, se=se, beta=beta,
                            Ms=np.array(Ms), vs=np.array(vs), mask=mask)
        n_ok = int(mask.sum()) if mask is not None else 0
        print(f"  {method:<12s} {alpha:>10.3f} {se:>10.3f} {beta:>12.3f} "
              f"{n_ok:>10d}")
    print("-" * 72)
    print("  Prediccion Monte Carlo: alpha = 1.0 para pSGLD "
          "(MMD ~ M^{-1/2} => MMD^2 ~ M^{-1})")
    print("=" * 72)

    # --- Figura log-log -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    M_dense = np.logspace(np.log10(min(args.M)), np.log10(max(args.M)), 100)
    for method in args.methods:
        info = fits[method]
        Ms = info["Ms"]
        vs = info["vs"]
        color = METHOD_COLORS.get(method, "gray")
        # Puntos (agregando multiples seeds si los hay)
        ax.scatter(Ms, np.clip(vs, 1e-5, None), color=color, s=55, alpha=0.8,
                   label=method, edgecolor="k", linewidth=0.6)
        # Recta de ajuste si alpha es finito
        if np.isfinite(info["alpha"]):
            y_fit = 10 ** info["beta"] * M_dense ** (-info["alpha"])
            # Exponente visible en la leyenda = -alpha (con signo)
            exp_str = f"{-info['alpha']:+.2f}"
            ax.plot(M_dense, y_fit, color=color, lw=1.5, ls="--",
                    label=rf"  ajuste $\propto M^{{{exp_str}}}$")
    # Linea guia Monte Carlo M^{-1}
    y_guide = 10 ** (np.log10(fits["pSGLD"]["vs"].max()) + 0.5) / M_dense
    ax.plot(M_dense, y_guide, color="grey", lw=1.0, ls=":",
            label=r"guia $M^{-1}$ (Monte Carlo)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$ (nº de neuronas)")
    ax.set_ylabel(r"MMD$^2(\nu_T^{\text{proxy}}, \nu^*_T)$  (unbiased)")
    ax.set_title(r"Propagación del caos: MMD$^2$ vs $M$ "
                 r"en make\_moons, $\varepsilon=0.05$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "06_mfl_rates.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
