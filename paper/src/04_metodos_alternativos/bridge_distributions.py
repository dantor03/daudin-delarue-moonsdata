"""Bridge interpretable: nu^*_t (Thm 1.4) vs distribucion empirica de neuronas
de los 4 regularizadores (SGLD vanilla, pSGLD, MMD2, Sinkhorn) en make_moons.

Setup (mismo que Tarea 2):
  - make_moons N=400, d_1 = d_2 = 2 (embedding y -> (-1,0) / (+1,0))
  - red parametrica: M=64 neuronas, K=10 pasos, T=1
  - solver explicito: grid M_per_dim=4 -> 1024 puntos en R^5
  - mismo eps para todos: eps=0.05 (el menor en que el explicito converge bien)

Para cada metodo parametrico se EXTRAE el conjunto de M=64 neuronas en t=T:
  - a_0^m = W0[:, m]               (R^2)
  - a_1^m = W1[m, :2]               (R^2)
  - a_2^m(T) = W1[m, 2] * T + b1[m] (R)
Esto da un sample de M=64 puntos en A = R^5.

Para el solver explicito, nu^*_T es discreta sobre el grid (1024 pesos).
Se muestrea S=2000 puntos categoricamente segun los pesos para tener
muestras comparables.

Comparacion:
  - 5 marginales 1D: histograma parametrico vs marginal de nu^*_T
  - MMD^2 (kernel gaussiano, mediana heuristica) entre los dos samples
  - Distancia entre medias y traza de covarianzas

Output:
  paper/figures/04_bridge_marginals.png  — 4 metodos x 5 marginales
  Tabla impresa con MMD^2 por metodo.
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

from codigo.config import SEED, DEVICE   # noqa: E402
from codigo.data import get_moons   # noqa: E402
from codigo.model import MeanFieldResNet   # noqa: E402
from codigo.train import train   # noqa: E402
from codigo.metrics import sample_prior_langevin   # noqa: E402

from explicit_nd import (   # noqa: E402
    build_grid_nd, picard_continuation_nd,
)
from sgld_vanilla import train_sgld_vanilla   # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)

DIM_LABELS = [r"$a_{0,1}$", r"$a_{0,2}$",
              r"$a_{1,1}$", r"$a_{1,2}$", r"$a_2$"]


# ---------------------------------------------------------------------------
# Extraccion de neuronas en t=T desde un modelo entrenado
# ---------------------------------------------------------------------------
def extract_neurons(model, t):
    """Devuelve (M, 5) con (a_0, a_1, a_2(t)) por neurona.

    En la red, a_2^m(t) = W1[m, 2] * t + b1[m]  (augmentacion temporal lineal).
    Aqui pasamos el t NORMALIZADO en [0, 1] tal y como lo usa la red.
    """
    W1 = model.velocity.W1.weight.detach().cpu().numpy()   # (M, d1+1)=(M, 3)
    b1 = model.velocity.W1.bias.detach().cpu().numpy()     # (M,)
    W0 = model.velocity.W0.weight.detach().cpu().numpy()   # (d1, M)=(2, M)
    a0 = W0.T                                              # (M, 2)
    a1 = W1[:, :2]                                         # (M, 2)
    a2 = W1[:, 2] * t + b1                                 # (M,)
    return np.concatenate([a0, a1, a2[:, None]], axis=1)   # (M, 5)


# ---------------------------------------------------------------------------
# Sample de nu^*_T discreta sobre el grid
# ---------------------------------------------------------------------------
def sample_from_nu(A_grid, nu_T, n_samples, rng):
    """Muestrea n_samples puntos del grid segun pesos nu_T.  Devuelve (n_samples, 5)."""
    idx = rng.choice(A_grid.shape[0], size=n_samples, replace=True, p=nu_T)
    return A_grid[idx]


# ---------------------------------------------------------------------------
# MMD^2 con kernel Gaussiano (mediana heuristica)
# ---------------------------------------------------------------------------
def mmd2_gauss(X, Y, sigma=None):
    """Estimador MMD^2 sesgado, kernel k(x,y) = exp(-|x-y|^2 / (2 sigma^2))."""
    XY = np.vstack([X, Y])
    if sigma is None:
        # Mediana de las distancias en la mezcla (heuristica estandar)
        n = min(len(XY), 500)
        sub = XY[np.random.default_rng(0).choice(len(XY), n, replace=False)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        sigma = float(np.sqrt(0.5 * np.median(d2[d2 > 0])))

    def K(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    Kxx = K(X, X).mean()
    Kyy = K(Y, Y).mean()
    Kxy = K(X, Y).mean()
    return float(Kxx + Kyy - 2 * Kxy), sigma


# ---------------------------------------------------------------------------
# Helpers de entrenamiento
# ---------------------------------------------------------------------------
def fresh_model():
    torch.manual_seed(SEED)
    return MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)


def train_all(X, y, eps, n_epochs, prior):
    out = {}

    print("\n[1/4] SGLD vanilla")
    m = fresh_model()
    train_sgld_vanilla(m, X, y, epsilon=eps, lr=1e-3, n_epochs=n_epochs,
                        verbose=False)
    out["SGLD vanilla"] = m

    print("[2/4] pSGLD")
    m = fresh_model()
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_sgld=True)
    out["pSGLD"] = m

    print("[3/4] MMD$^2$")
    m = fresh_model()
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_mmd=True, prior_samples=prior)
    out["MMD$^2$"] = m

    print("[4/4] Sinkhorn")
    m = fresh_model()
    train(m, X, y, epsilon=eps, lr=0.01, n_epochs=n_epochs,
          verbose=False, use_sinkhorn=True, prior_samples=prior,
          sinkhorn_blur=0.05)
    out["Sinkhorn"] = m
    return out


def embed_labels(y_np):
    emb = np.zeros((len(y_np), 2))
    emb[:, 0] = 2.0 * y_np - 1.0
    return emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Tarea 4 (bridge) — nu^*_T (Thm 1.4) vs distribucion de neuronas")
    print("=" * 72)
    EPS = 0.05
    N_EPOCHS = 800
    rng = np.random.default_rng(SEED)

    X, y, X_np, y_np = get_moons()
    y_emb = embed_labels(y_np)
    print(f"  N={len(y)}  d=2  eps={EPS}  epochs={N_EPOCHS}")

    # -- 1. Solver explicito ------------------------------------------------
    A_grid, dim_a = build_grid_nd(d1=2, R=2.0, M_per_dim=4)
    print(f"\n[explicit] grid R^5 con M={A_grid.shape[0]} puntos, eps={EPS}")
    t0 = time.time()
    eps_schedule = [0.5, 0.2, 0.1, EPS]
    results = picard_continuation_nd(
        X_np.astype(float), y_emb.astype(float), A_grid, d1=2,
        eps_schedule=eps_schedule,
        T=1.0, K=10, omega=0.2, max_iter=200, tol=1e-4, verbose=False,
    )
    nu_T = results[-1][1].nu[-1]   # (M,) en t=T
    print(f"  Picard total: {time.time()-t0:.1f}s   eps_final={results[-1][0]}")

    # Sample S puntos de nu^*_T para comparar via MMD^2
    S = 2000
    samples_explicit = sample_from_nu(A_grid, nu_T, S, rng)
    print(f"  sample {S} puntos de nu^*_T")

    # -- 2. Entrenar parametricos -------------------------------------------
    print(f"\n[parametric] entrenando 4 metodos con eps={EPS}, "
          f"{N_EPOCHS} epochs ...")
    prior = sample_prior_langevin(n_samples=400, epsilon=EPS, dim=5)
    models = train_all(X, y, EPS, N_EPOCHS, prior)

    # Extraer las M=64 neuronas en t=T (t_norm=1.0 = T en escala normalizada)
    neurons = {name: extract_neurons(m, t=1.0) for name, m in models.items()}

    # -- 3. MMD^2 entre cada parametrico y nu^*_T ---------------------------
    print("\n" + "=" * 72)
    print(f"  {'metodo':<18s} {'MMD^2':>10s} {'mean dist':>12s} "
          f"{'cov frob':>12s}")
    print("-" * 72)
    mu_explicit = (nu_T[:, None] * A_grid).sum(axis=0)
    cov_explicit = np.einsum('m,mi,mj->ij', nu_T, A_grid - mu_explicit,
                              A_grid - mu_explicit)
    rows = []
    for name, neur in neurons.items():
        mmd2, sigma = mmd2_gauss(neur, samples_explicit)
        mu = neur.mean(axis=0)
        cov = np.cov(neur, rowvar=False)
        d_mean = float(np.linalg.norm(mu - mu_explicit))
        d_cov = float(np.linalg.norm(cov - cov_explicit, ord='fro'))
        rows.append((name, mmd2, d_mean, d_cov))
        print(f"  {name:<18s} {mmd2:10.4f} {d_mean:12.4f} {d_cov:12.4f}")
    print(f"  (sigma kernel = {sigma:.3f})")

    # -- 4. Figura: 4 metodos x 5 marginales --------------------------------
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex='col')
    bins = 25
    grid_axis = np.linspace(-2, 2, 80)
    method_colors = {"SGLD vanilla": "#e74c3c", "pSGLD": "#f39c12",
                      "MMD$^2$": "#2ecc71", "Sinkhorn": "#3498db"}

    for i, (name, neur) in enumerate(neurons.items()):
        for k in range(5):
            ax = axes[i, k]
            # Histograma del parametrico
            ax.hist(neur[:, k], bins=bins, density=True, alpha=0.55,
                    color=method_colors[name], edgecolor='k', linewidth=0.4,
                    label=name if k == 0 else None)
            # Marginal de nu^*_T sobre la dim k: agrupar pesos por valor unico
            marginal = {}
            for w, val in zip(nu_T, A_grid[:, k]):
                marginal[val] = marginal.get(val, 0.0) + w
            xs = np.array(sorted(marginal.keys()))
            ws = np.array([marginal[x] for x in xs])
            # Aproximacion de densidad: width entre niveles del grid
            dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
            ax.bar(xs, ws / dx, width=dx * 0.85, alpha=0.0,
                   edgecolor='k', linewidth=1.2,
                   label=r"$\nu^*_T$ marginal" if (i == 0 and k == 0) else None)
            ax.plot(xs, ws / dx, '-o', color='k', ms=4, lw=1.2)
            if i == 0:
                ax.set_title(DIM_LABELS[k])
            if k == 0:
                ax.set_ylabel(name, fontsize=11)
            ax.grid(alpha=0.25)
            ax.set_xlim(-2.5, 2.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.55, ec='k',
                      label='neuronas (M=64)'),
        plt.Line2D([0], [0], color='k', marker='o', ms=4, lw=1.2,
                   label=r'$\nu^*_T$ (Thm 1.4)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Bridge Thm 1.4 vs neuronas parametricas — make_moons, "
                 f"$\\varepsilon$={EPS} (5 dim de A=$\\mathbb{{R}}^5$)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = FIG_DIR / "04_bridge_marginals.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out}")

    # -- 5. Figura compacta: barras de MMD^2 --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [r[0] for r in rows]
    cols = [method_colors[n] for n in names]
    axes[0].bar(names, [r[1] for r in rows], color=cols, edgecolor='k')
    axes[0].set_ylabel(r"MMD$^2$")
    axes[0].set_title(r"MMD$^2$ vs $\nu^*_T$")
    axes[1].bar(names, [r[2] for r in rows], color=cols, edgecolor='k')
    axes[1].set_ylabel("$\\|\\mu - \\mu^*\\|_2$")
    axes[1].set_title("Distancia de medias")
    axes[2].bar(names, [r[3] for r in rows], color=cols, edgecolor='k')
    axes[2].set_ylabel(r"$\|\Sigma - \Sigma^*\|_F$")
    axes[2].set_title("Distancia de covarianzas")
    for ax in axes:
        ax.grid(axis='y', alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(15)
    fig.suptitle(r"Distancia a $\nu^*_T$ (Thm 1.4) — menor = mas cercano "
                 "a la distribucion de Gibbs del paper", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "04_bridge_distances.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
