"""
=============================================================================
exp_e.py — Experimento E (robustez): Robustez de la distribución ν*
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train


def experiment_E_robustness(n_seeds: int = 10, n_epochs: int = 500):
    """
    Robustez de ν* a distintas condiciones de entrenamiento (make_moons).

    MOTIVACIÓN:
        El Experimento E muestra la distribución ν* para UN único entrenamiento.
        Pero ¿es esa distribución robusta?  Si el minimizador es único (Meta-
        Teorema 1), distintas inicializaciones y distintos datasets deberían
        producir distribuciones ν* similares — la misma "nube" de puntos en el
        espacio de parámetros.

    DISEÑO:
        E-1 — Robustez a θ₀ (datos fijos, init variable):
            data_seed=42 fijo | init_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con la inicialización de pesos?

        E-2 — Robustez a γ₀ (init fija, datos variables):
            init_seed=4 fijo | data_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con el dataset de entrenamiento?

    OPTIMIZADOR:
        Ambos sub-experimentos usan SGLD (use_sgld=True): SGD + cosine annealing
        + ruido de Langevin.  El ruido permite explorar la distribución de Gibbs
        ν_t* ∝ exp(−J(θ)/ε) en lugar de colapsar a un estimador puntual.

    FIGURA E (E_parameter_robustness.png):
        Layout 1×2 (dos paneles en columnas):
          Panel izquierdo (E-1): importancias ‖a₀ᵐ‖ ordenadas — init variable, datos fijos
          Panel derecho  (E-2): importancias ‖a₀ᵐ‖ ordenadas — datos variables, init fija
        Cada curva de color corresponde a un run (seed distinta).
        La línea blanca es la media y la banda blanca es ±1σ sobre las 10 seeds.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO E (robustez)  —  Robustez de ν* entre entrenamientos")
    print("=" * 62)

    EPS             = 0.01
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 4
    SEEDS           = list(range(n_seeds))

    SEED_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6',
                   '#e67e22', '#1abc9c', '#e84393', '#a29bfe', '#fd79a8']

    def get_params(model):
        with torch.no_grad():
            W1w = model.velocity.W1.weight.cpu().numpy()   # (M, d1+1)
            W0w = model.velocity.W0.weight.cpu().numpy()   # (d1, M)
        a1 = W1w[:, :2]   # (M, 2)
        a0 = W0w.T        # (M, 2)
        return a1, a0

    def importance(a0):
        return np.linalg.norm(a0, axis=1)   # (M,)

    # ── E-1: datos fijos, init variable ───────────────────────────────────────
    print("  E-1: data_seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, _, _ = get_moons(seed=DATA_SEED_FIXED)
    results_E2_1 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        a1, a0 = get_params(model)
        results_E2_1.append({'seed': s, 'a1': a1, 'a0': a0, 'hist': hist})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={hist['accuracy'][-1]:.3f}")

    # ── E-2: init fija, datos variables ───────────────────────────────────────
    print("  E-2: init_seed=4 fijo, 10 datasets distintos...")
    results_E2_2 = []
    for s in SEEDS:
        X_s, y_s, _, _ = get_moons(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_s, y_s, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        a1, a0 = get_params(model)
        results_E2_2.append({'seed': s, 'a1': a1, 'a0': a0, 'hist': hist})
        print(f"    data_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={hist['accuracy'][-1]:.3f}")

    # ── Figura E2 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    def _panel(ax, results, subtitle):
        """Dibuja importancias ‖a₀ᵐ‖ ordenadas por seed."""
        for i, r in enumerate(results):
            imp = np.sort(importance(r['a0']))[::-1]
            ax.plot(np.arange(1, len(imp) + 1), imp,
                    color=SEED_COLORS[i % len(SEED_COLORS)],
                    lw=1.2, alpha=0.70)
        all_imp = np.array([np.sort(importance(r['a0']))[::-1] for r in results])
        ax.plot(np.arange(1, all_imp.shape[1] + 1), all_imp.mean(axis=0),
                color='white', lw=2.2, label='Media')
        ax.fill_between(np.arange(1, all_imp.shape[1] + 1),
                        all_imp.mean(axis=0) - all_imp.std(axis=0),
                        all_imp.mean(axis=0) + all_imp.std(axis=0),
                        color='white', alpha=0.15)
        style_ax(ax,
                 subtitle + '\n' r'Importancia $\|a_0^m\|_2$ ordenada por semilla',
                 'Rank (neurona)', r'$\|a_0^m\|_2$')
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    _panel(axes[0], results_E2_1, 'E-1 — init variable, datos fijos (seed=42)')
    _panel(axes[1], results_E2_2, 'E-2 — datos variables, init fija (seed=4)')

    fig.suptitle(
        r'Experimento E — Robustez de $\nu^*$: importancia neuronal $\|a_0^m\|_2$'
        '\n'
        r'Banda blanca = media ± std sobre 10 semillas  |  $\varepsilon=0.01$',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, 'E_parameter_robustness.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")
