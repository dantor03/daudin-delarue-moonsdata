"""Tarea 5 - Aplicacion economica: ACSIncome (California 2018, N=500).

Replica el pipeline de make_moons sobre datos demograficos reales:
  (1) Solver explicito (Thm 1.4) con d_1 = 2  (AGEP, SCHL).
       - embedding y -> (-1, 0) / (+1, 0)
       - grid en R^5 con M_per_dim=4 (1024 puntos)
       - continuacion en eps = [0.5, 0.2, 0.1, 0.05]

  (2) Metodos parametricos pSGLD / MMD2 / Sinkhorn con d_1 = 4
       (anyade WKHP horas trabajadas y SEX) y BCE como coste terminal.

  (3) Bridge apples-to-apples (d_1 = 2): mismos proxies parametricos
       reentrenados en d_1=2 para comparar sus M=64 neuronas en t=T contra
       muestras de nu*_T del solver explicito (Thm 1.4) sobre la misma rejilla.

Outputs:
  paper/figures/05_acs_pipeline.png      (X(0) -> X(T) explicito)
  paper/figures/05_acs_param_compare.png (J vs eps + accuracy de proxies d_1=4)
  paper/figures/05_acs_bridge.png        (MMD^2 / mean / cov vs nu*_T en d_1=2)
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "src" / "02_regularizadores_moons"))
sys.path.insert(0, str(ROOT / "paper" / "src" / "04_metodos_alternativos"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo.config import SEED, DEVICE   # noqa: E402
from codigo.model import MeanFieldResNet   # noqa: E402
from codigo.train import train   # noqa: E402
from codigo.metrics import sample_prior_langevin   # noqa: E402

from explicit_nd import build_grid_nd, picard_continuation_nd   # noqa: E402
from bridge_distributions import (   # noqa: E402
    extract_neurons, sample_from_nu, mmd2_gauss,
)

from data_acs import get_acs_income   # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_labels(y_np):
    emb = np.zeros((len(y_np), 2))
    emb[:, 0] = 2.0 * y_np - 1.0
    return emb


def accuracy_explicit(X_T, y_np):
    return float(((X_T[:, 0] > 0).astype(np.float32) == y_np).mean())


def fresh_model(d1, M=64):
    torch.manual_seed(SEED)
    return MeanFieldResNet(d1=d1, M=M, T=1.0, n_steps=10).to(DEVICE)


def train_proxies(X, y, eps, n_epochs, prior, d1):
    """Entrena pSGLD / MMD2 / Sinkhorn (3 metodos) en d_1 dado."""
    out = {}
    print(f"  [d_1={d1}] pSGLD ...")
    m = fresh_model(d1=d1)
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_sgld=True)
    out["pSGLD"] = m
    print(f"  [d_1={d1}] MMD$^2$ ...")
    m = fresh_model(d1=d1)
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_mmd=True, prior_samples=prior)
    out["MMD$^2$"] = m
    print(f"  [d_1={d1}] Sinkhorn ...")
    m = fresh_model(d1=d1)
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_sinkhorn=True, prior_samples=prior,
          sinkhorn_blur=0.05)
    out["Sinkhorn"] = m
    return out


def acc_param(model, X, y):
    model.eval()
    with torch.no_grad():
        logit = model(X)
        return float(((logit > 0).float() == y).float().mean())


# ---------------------------------------------------------------------------
# Bloques del experimento
# ---------------------------------------------------------------------------
def part_explicit_d2(X_np, y_np):
    """Solver explicito con d_1 = 2 (AGEP, SCHL).  Continuacion en eps."""
    print("\n" + "=" * 72)
    print("[A] Solver explicito (Thm 1.4)  -  d_1 = 2  (AGEP, SCHL)")
    print("=" * 72)
    y_emb = embed_labels(y_np)
    A_grid, dim_a = build_grid_nd(d1=2, R=2.0, M_per_dim=4)
    print(f"  grid R^{dim_a}, M={A_grid.shape[0]} puntos")
    eps_schedule = [0.5, 0.2, 0.1, 0.05]
    t0 = time.time()
    results = picard_continuation_nd(
        X_np.astype(float), y_emb.astype(float), A_grid, d1=2,
        eps_schedule=eps_schedule,
        T=1.0, K=10, omega=0.2, max_iter=200, tol=1e-4, verbose=False,
    )
    print(f"  total Picard: {time.time()-t0:.1f}s")
    print(f"  {'eps':>6s} {'iters':>6s} {'J':>10s} {'term':>10s} {'acc':>6s}")
    for eps, res in results:
        XT = res.X[-1]
        term = 0.5 * ((XT - y_emb) ** 2).sum(axis=1).mean()
        print(f"  {eps:6.3f} {res.iters:6d} "
              f"{res.cost_history[-1]:10.5f} {term:10.5f} "
              f"{accuracy_explicit(XT, y_np):6.3f}")
    return results, A_grid, y_emb, eps_schedule


def part_parametric_d4(X_t, y_t, X_np, y_np, eps_list, n_epochs):
    """Proxies parametricos pSGLD/MMD2/Sinkhorn con d_1 = 4."""
    print("\n" + "=" * 72)
    print("[B] Proxies parametricos  -  d_1 = 4  "
          "(AGEP, SCHL, WKHP, SEX)")
    print("=" * 72)
    rows = []
    for eps in eps_list:
        prior = sample_prior_langevin(n_samples=400, epsilon=eps,
                                       dim=2 * 4 + 1)
        models = train_proxies(X_t, y_t, eps, n_epochs, prior, d1=4)
        for name, m in models.items():
            rows.append((name, eps, acc_param(m, X_t, y_t)))
            print(f"  eps={eps:5.3f}  {name:<10s}  acc={rows[-1][2]:.3f}")
    return rows


def part_bridge_d2(X_t, y_t, X_np, y_np, results_explicit, A_grid,
                    eps_bridge, n_epochs):
    """Bridge a d_1 = 2 contra nu*_T del solver explicito.

    Reentrenamos pSGLD/MMD2/Sinkhorn con d_1=2 (apples-to-apples) y comparamos
    sus M=64 neuronas en t=T contra muestras de nu*_T (Thm 1.4) en R^5.
    """
    print("\n" + "=" * 72)
    print(f"[C] Bridge apples-to-apples  -  d_1 = 2,  eps={eps_bridge}")
    print("=" * 72)
    eps_idx = next(i for i, (e, _) in enumerate(results_explicit)
                    if abs(e - eps_bridge) < 1e-9)
    nu_T = results_explicit[eps_idx][1].nu[-1]
    rng = np.random.default_rng(SEED)
    samples_explicit = sample_from_nu(A_grid, nu_T, 2000, rng)

    # Seleccion d_1=2 features: solo AGEP, SCHL  -> primeras dos columnas
    X2_np = X_np[:, :2]
    X2_t = torch.tensor(X2_np, device=DEVICE)
    y_t2 = y_t

    prior = sample_prior_langevin(n_samples=400, epsilon=eps_bridge, dim=5)
    print(f"  reentrenando proxies en d_1=2 con eps={eps_bridge} ...")
    models = train_proxies(X2_t, y_t2, eps_bridge, n_epochs, prior, d1=2)
    neurons = {name: extract_neurons(m, t=1.0)
               for name, m in models.items()}

    mu_explicit = (nu_T[:, None] * A_grid).sum(axis=0)
    cov_explicit = np.einsum('m,mi,mj->ij', nu_T, A_grid - mu_explicit,
                              A_grid - mu_explicit)
    rows = []
    print(f"  {'metodo':<12s} {'MMD^2':>10s} {'mean dist':>11s} "
          f"{'cov frob':>11s}")
    for name, neur in neurons.items():
        mmd2, sigma = mmd2_gauss(neur, samples_explicit)
        mu = neur.mean(axis=0)
        cov = np.cov(neur, rowvar=False)
        d_mu = float(np.linalg.norm(mu - mu_explicit))
        d_cov = float(np.linalg.norm(cov - cov_explicit, ord='fro'))
        rows.append((name, mmd2, d_mu, d_cov))
        print(f"  {name:<12s} {mmd2:10.4f} {d_mu:11.4f} {d_cov:11.4f}")
    print(f"  (sigma kernel = {sigma:.3f})")
    return rows, neurons, nu_T


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_explicit_pipeline(results, X_np, y_np, eps_schedule, feat_names):
    n_eps = len(eps_schedule)
    fig, axes = plt.subplots(2, n_eps, figsize=(4 * n_eps, 8))
    y_emb = embed_labels(y_np)
    for j, (eps, res) in enumerate(results):
        XT = res.X[-1]
        ax = axes[0, j]
        for c, m in [(0, "#e74c3c"), (1, "#3498db")]:
            ax.scatter(X_np[y_np == c, 0], X_np[y_np == c, 1],
                       c=m, s=12, alpha=0.7,
                       label=f"y={c}" if j == 0 else None)
        ax.set_title(f"$\\varepsilon$={eps}: $X(0)$ (datos ACS)")
        ax.set_xlabel(feat_names[0]); ax.set_ylabel(feat_names[1])
        ax.grid(alpha=0.3); ax.set_aspect("equal")
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[1, j]
        for c, m in [(0, "#e74c3c"), (1, "#3498db")]:
            ax.scatter(XT[y_np == c, 0], XT[y_np == c, 1],
                       c=m, s=12, alpha=0.7)
        ax.scatter([-1, 1], [0, 0], c="k", marker="X", s=120,
                   label="$y_{emb}$")
        ax.axvline(0, color="grey", ls="--", alpha=0.4)
        acc = accuracy_explicit(XT, y_np)
        ax.set_title(f"$X(T)$  acc={acc:.3f}")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.3); ax.set_aspect("equal")
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("ACSIncome California 2018 (N=500): "
                 "solver explicito Thm 1.4 con $d_1 = d_2 = 2$",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "05_acs_pipeline.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def fig_param_compare(rows_d4, eps_list):
    cols = {"pSGLD": "#f39c12", "MMD$^2$": "#2ecc71",
             "Sinkhorn": "#3498db"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in cols:
        accs = [r[2] for r in rows_d4 if r[0] == name]
        ax.plot(eps_list, accs, "o-", color=cols[name], lw=1.7, ms=8,
                 label=name)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$ (regularizacion entropica)")
    ax.set_ylabel("accuracy")
    ax.set_title("ACSIncome - accuracy de proxies parametricos "
                  "($d_1 = 4$, $M = 64$, $N = 500$)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "05_acs_param_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def fig_bridge(rows_bridge, eps_bridge):
    cols = {"pSGLD": "#f39c12", "MMD$^2$": "#2ecc71",
             "Sinkhorn": "#3498db"}
    names = [r[0] for r in rows_bridge]
    bar_cols = [cols[n] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(names, [r[1] for r in rows_bridge], color=bar_cols,
                 edgecolor='k')
    axes[0].set_ylabel(r"MMD$^2$"); axes[0].set_title(r"MMD$^2$ vs $\nu^*_T$")
    axes[1].bar(names, [r[2] for r in rows_bridge], color=bar_cols,
                 edgecolor='k')
    axes[1].set_ylabel(r"$\|\mu - \mu^*\|_2$")
    axes[1].set_title("Distancia de medias")
    axes[2].bar(names, [r[3] for r in rows_bridge], color=bar_cols,
                 edgecolor='k')
    axes[2].set_ylabel(r"$\|\Sigma - \Sigma^*\|_F$")
    axes[2].set_title("Distancia de covarianzas")
    for ax in axes:
        ax.grid(axis='y', alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(15)
    fig.suptitle(f"Bridge ACSIncome ($d_1=2$, $\\varepsilon$={eps_bridge}): "
                  r"proxies vs $\nu^*_T$ (Thm 1.4)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "05_acs_bridge.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Tarea 5 - ACSIncome  (California 2018,  N=500,  balanced)")
    print("=" * 72)

    # Datos d_1=2 (para solver explicito y bridge)
    X2_t, y_t, X2_np, y_np, names2 = get_acs_income(
        n_samples=500, feature_set="d2", balance=True)
    print(f"  d_1=2 features: {names2}")

    # Datos d_1=4 (para proxies parametricos)
    X4_t, y4_t, X4_np, _, names4 = get_acs_income(
        n_samples=500, feature_set="d4", balance=True)
    print(f"  d_1=4 features: {names4}")
    # Sanity: y_np y _ son la misma muestra balanceada (mismo seed)
    assert np.array_equal(y_np, _), "y mismatch d2 vs d4"

    # ── A) solver explicito d_1=2 ────────────────────────────────────────
    results, A_grid, y_emb, eps_sched = part_explicit_d2(X2_np, y_np)
    fig_explicit_pipeline(results, X2_np, y_np, eps_sched, names2)

    # ── B) proxies d_1=4 ────────────────────────────────────────────────
    eps_list = [0.05, 0.1, 0.2]
    n_epochs = 500
    rows_d4 = part_parametric_d4(X4_t, y4_t, X4_np, y_np, eps_list, n_epochs)
    fig_param_compare(rows_d4, eps_list)

    # ── C) bridge d_1=2 ──────────────────────────────────────────────────
    eps_bridge = 0.05
    rows_bridge, neurons, nu_T = part_bridge_d2(
        X2_t, y_t, X2_np, y_np, results, A_grid, eps_bridge, n_epochs)
    fig_bridge(rows_bridge, eps_bridge)

    # ── Resumen final ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print("\n[A] Explicito d_1=2:")
    for eps, res in results:
        XT = res.X[-1]
        print(f"  eps={eps:5.3f}  J={res.cost_history[-1]:.4f}  "
              f"acc={accuracy_explicit(XT, y_np):.3f}")
    print("\n[B] Parametricos d_1=4 (mejor eps por metodo):")
    for name in {r[0] for r in rows_d4}:
        best = max((r for r in rows_d4 if r[0] == name), key=lambda r: r[2])
        print(f"  {name:<10s}  eps={best[1]:5.3f}  acc={best[2]:.3f}")
    print("\n[C] Bridge d_1=2 vs nu*_T (eps={}):".format(eps_bridge))
    for name, mmd2, d_mu, d_cov in rows_bridge:
        print(f"  {name:<10s}  MMD^2={mmd2:.4f}  "
              f"|mu-mu*|={d_mu:.3f}  |Sigma-Sigma*|_F={d_cov:.3f}")


if __name__ == "__main__":
    main()
