"""QMC + importance sampling para el solver explicito (Thm 1.4).

Reemplaza el grid tensorial (M = M_pd^{2 d1 + 1} celdas) por un sample QMC
(Sobol scrambled) desde un proposal Gaussiano N(0, sigma^2 I), con peso de IS:
    log_nu_inf(a) = -ell(a) - log q(a)   (hasta normalizacion)

Comparativa sobre make_moons (d_1=2, dim_a=5):
  - tensorial M_pd in {2, 3, 4}   ->  M in {32, 243, 1024}
  - QMC Sobol N in {32, 256, 1024}
  - Plain MC (Gauss random) N in {32, 256, 1024}
para los mismos eps de continuacion.

Reporta J final y term L final.  Se espera que QMC alcance el mismo coste
con O(10x) menos puntos que el grid tensorial gracias a la baja dimension
efectiva del integrando exp(-ell - phi/eps).

Output:
  paper/figures/04_qmc_convergence.png
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc, norm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "src" / "02_regularizadores_moons"))

from codigo.data import get_moons   # noqa: E402
from explicit_nd import (   # noqa: E402
    build_grid_nd, picard_continuation_nd, ell_nd,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Builders de soporte
# ---------------------------------------------------------------------------
def build_qmc_nd(d1, n_samples, sigma=1.0, c1=0.05, c2=0.5, scramble=True,
                  seed=0):
    """Sobol scrambled en R^{2 d1 + 1} via inverse Gaussian CDF.

    Devuelve A (n, dim_a), log_nu_inf (n,) con peso de IS.
    n se redondea hacia arriba a la potencia de 2 mas cercana (Sobol).
    """
    dim_a = 2 * d1 + 1
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    n = 2 ** m
    sampler = qmc.Sobol(d=dim_a, scramble=scramble, seed=seed)
    u = sampler.random(n)                                 # (n, dim_a) en [0,1]
    u = np.clip(u, 1e-9, 1 - 1e-9)
    A = norm.ppf(u, loc=0.0, scale=sigma)                 # N(0, sigma^2 I)
    log_q = norm.logpdf(A, loc=0.0, scale=sigma).sum(axis=1)
    log_w = -ell_nd(A, c1, c2) - log_q
    log_nu_inf = log_w - logsumexp(log_w)
    return A, log_nu_inf, n


def build_mc_nd(d1, n_samples, sigma=1.0, c1=0.05, c2=0.5, seed=0):
    """MC plano: muestras iid de N(0, sigma^2 I) con peso de IS, sin Sobol."""
    dim_a = 2 * d1 + 1
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(n_samples, dim_a))
    log_q = norm.logpdf(A, loc=0.0, scale=sigma).sum(axis=1)
    log_w = -ell_nd(A, c1, c2) - log_q
    log_nu_inf = log_w - logsumexp(log_w)
    return A, log_nu_inf, n_samples


def build_grid_with_log_nu_inf(d1, M_per_dim, R=2.0, c1=0.05, c2=0.5):
    """Grid tensorial uniforme + log_nu_inf(a) = -ell(a) - log Z (uniforme)."""
    A, dim_a = build_grid_nd(d1, R=R, M_per_dim=M_per_dim)
    log_w = -ell_nd(A, c1, c2)
    log_nu_inf = log_w - logsumexp(log_w)
    return A, log_nu_inf, A.shape[0]


# ---------------------------------------------------------------------------
# Embedding y accuracy (mismos que en run_explicit_moons)
# ---------------------------------------------------------------------------
def embed_labels(y_np):
    emb = np.zeros((len(y_np), 2))
    emb[:, 0] = 2.0 * y_np - 1.0
    return emb


def accuracy(X_T, y_np):
    pred = (X_T[:, 0] > 0).astype(np.float32)
    return float((pred == y_np).mean())


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------
def run_one(name, A, log_nu_inf, X_np, y_emb, y_np, eps_schedule):
    t0 = time.time()
    results = picard_continuation_nd(
        X_np.astype(float), y_emb.astype(float), A, d1=2,
        eps_schedule=eps_schedule, T=1.0, K=10,
        omega=0.2, max_iter=120, tol=1e-4, verbose=False,
        log_nu_inf=log_nu_inf,
    )
    dt = time.time() - t0
    eps_last, res_last = results[-1]
    XT = res_last.X[-1]
    term = 0.5 * ((XT - y_emb) ** 2).sum(axis=1).mean()
    acc = accuracy(XT, y_np)
    n = A.shape[0]
    print(f"  {name:<20s}  M={n:>5d}  J={res_last.cost_history[-1]:.4f}  "
          f"term={term:.4f}  acc={acc:.3f}  iters={res_last.iters:>3d}  "
          f"time={dt:5.1f}s")
    return n, res_last.cost_history[-1], term, acc, dt


def main():
    print("=" * 78)
    print("Tarea 4 — QMC + IS para el solver explicito (Thm 1.4) sobre moons")
    print("=" * 78)
    _, _, X_np, y_np = get_moons()
    y_emb = embed_labels(y_np)
    eps_schedule = [0.5, 0.2, 0.1, 0.05]
    sigma_qmc = 1.0    # proposal: N(0, 1) en cada coord (rango efectivo ~[-3,3])
    print(f"  N={len(y_np)}  d_1=2  dim_a=5  eps_final={eps_schedule[-1]}  "
          f"sigma_proposal={sigma_qmc}")

    # ── Grid tensorial: M_pd in {2, 3, 4}  ⇒ M in {32, 243, 1024} ──────────
    print("\n[A] grid tensorial uniforme")
    grid_rows = []
    for M_pd in [2, 3, 4]:
        A, lni, n = build_grid_with_log_nu_inf(d1=2, M_per_dim=M_pd, R=2.0)
        grid_rows.append(run_one(f"grid M_pd={M_pd}", A, lni,
                                  X_np, y_emb, y_np, eps_schedule))

    # ── QMC Sobol scrambled ────────────────────────────────────────────────
    print("\n[B] QMC Sobol scrambled (proposal N(0, 1))")
    qmc_rows = []
    for n_target in [32, 256, 1024]:
        A, lni, n = build_qmc_nd(d1=2, n_samples=n_target, sigma=sigma_qmc,
                                  scramble=True, seed=0)
        qmc_rows.append(run_one(f"QMC n={n}", A, lni,
                                 X_np, y_emb, y_np, eps_schedule))

    # ── MC plano ───────────────────────────────────────────────────────────
    print("\n[C] MC iid (mismo proposal Gaussiano, sin Sobol)")
    mc_rows = []
    for n_target in [32, 256, 1024]:
        A, lni, n = build_mc_nd(d1=2, n_samples=n_target, sigma=sigma_qmc,
                                 seed=0)
        mc_rows.append(run_one(f"MC n={n}", A, lni,
                                X_np, y_emb, y_np, eps_schedule))

    # ── Resumen ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Resumen (eps final = {})".format(eps_schedule[-1]))
    print("-" * 78)
    print(f"  {'metodo':<14s}  {'M':>6s}  {'J':>8s}  {'term':>8s}  "
          f"{'acc':>6s}  {'time(s)':>8s}")
    for label, rows in [("grid", grid_rows), ("QMC", qmc_rows),
                          ("MC", mc_rows)]:
        for n, J, term, acc, dt in rows:
            print(f"  {label:<14s}  {n:>6d}  {J:8.4f}  {term:8.4f}  "
                  f"{acc:6.3f}  {dt:8.1f}")

    # ── Figura: convergencia J final y term L vs M ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, rows, marker, color in [
        ("grid tensorial", grid_rows, 'o', '#9b59b6'),
        ("QMC Sobol",      qmc_rows,  's', '#e67e22'),
        ("MC iid",         mc_rows,   'v', '#34495e'),
    ]:
        ns = [r[0] for r in rows]
        Js = [r[1] for r in rows]
        terms = [r[2] for r in rows]
        axes[0].plot(ns, Js, marker=marker, color=color, lw=1.5, ms=8,
                      label=label)
        axes[1].plot(ns, terms, marker=marker, color=color, lw=1.5, ms=8,
                      label=label)
    axes[0].set_xscale('log')
    axes[0].set_xlabel("M (puntos de soporte)")
    axes[0].set_ylabel(r"$J$ final (eps={})".format(eps_schedule[-1]))
    axes[0].set_title("Coste $J$ vs tamaño del soporte")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xscale('log')
    axes[1].set_xlabel("M (puntos de soporte)")
    axes[1].set_ylabel(r"term $L$ final (eps={})".format(eps_schedule[-1]))
    axes[1].set_title("Coste terminal $L$ vs tamaño del soporte")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("QMC + importance sampling vs grid tensorial — "
                  "make_moons, $d_1=2$, $\\dim A=5$", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "04_qmc_convergence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
