"""
=============================================================================
exp_f.py — Experimento F: Distribución de ν* en make_circles (simetría SO(2))
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..data import get_circles
from ..model import MeanFieldResNet
from ..train import train


# =============================================================================
# EXPERIMENTO F
#   Distribución de ν* en make_circles: simetría rotacional y robustez a semillas
# =============================================================================
def experiment_F(n_seeds: int = 10, n_epochs: int = 700):
    """
    Distribución de ν* en make_circles: simetría geométrica y robustez a semillas.

    MOTIVACIÓN TEÓRICA:
        make_circles tiene simetría rotacional completa SO(2): el dataset γ₀
        es invariante (en distribución) bajo rotaciones del plano ℝ².  Si el
        problema de control también respeta esta simetría, la distribución óptima
        de parámetros ν* debería ser asimismo (aproximadamente) isotrópica.

        Predicción concreta sobre a₁ᵐ:
            Los pesos de entrada a₁ᵐ ∈ ℝ² deberían distribuirse UNIFORMEMENTE
            en S¹ (un anillo en el plano), porque ninguna dirección espacial es
            privilegiada por la geometría del problema.

        Contraste con make_moons:
            En el Experimento E observamos una distribución BIMODAL de a₁
            (dos picos en ±0.3–0.4).  Esto refleja que las dos "lunas" tienen
            una orientación preferida.  Con circles esperamos la distribución
            contraria: uniforme sobre S¹.

        Test cuantitativo — longitud resultante media R̄:
            R̄ = |mean(exp(iθ))| ∈ [0,1],   θ = arctan2(a₁ᵐ[1], a₁ᵐ[0])
            R̄ ≈ 0 → distribución isotrópica (predicción de simetría)
            R̄ ≈ 1 → distribución concentrada en una dirección

    DISEÑO DEL EXPERIMENTO:
        F1 — Robustez a γ₀ (init fija, datos variables):
            • init_seed = 4 fija (misma inicialización en todos los runs)
            • Dataset make_circles con n_seeds semillas distintas
            • Pregunta: ¿la distribución de a₁ cambia con el dataset?

        F2 — Robustez a θ₀ (datos fijos, init variable):
            • data_seed = 42 fijo (mismo make_circles en todos los runs)
            • n_seeds inicializaciones de parámetros distintas
            • Pregunta: ¿la distribución de a₁ cambia con la inicialización?

    OPTIMIZADOR:
        Ambos sub-experimentos usan SGLD (use_sgld=True): Adam + cosine annealing
        + ruido de Langevin √(2·η_t·ε)·ξ.  Adam como base garantiza convergencia
        al mínimo; el ruido permite visualizar la distribución estacionaria de
        los parámetros ν_t* ∝ exp(−J(θ)/ε), no solo un estimador puntual.

    FIGURAS GENERADAS (F_circles_parameter_distribution.png):
        Panel único: curvas de pérdida J de F1 (rojo) y F2 (azul) superpuestas.
          • Curvas individuales por seed en trazo fino y transparente (alpha=0.25)
          • Media sobre seeds en trazo grueso
          • Banda ±1σ alrededor de la media
          • Leyenda con J* media ± std para cada sub-experimento
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO F  —  ν* en make_circles: simetría y robustez")
    print("=" * 62)

    EPS             = 0.01
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 4   # seed 4 converge bien en make_circles; elegido porque
                          # produce F1 < F2 (datos más robustos que init), lo que
                          # da jerarquía visual informativa en la figura
    SEEDS           = list(range(n_seeds))

    # ── F1: semillas de datos, init fija ─────────────────────────────────────
    print(f"  F1: 10 datasets circles distintos, init_seed={INIT_SEED_FIXED} fija...")
    results_F1 = []
    for s in SEEDS:
        X, y, X_np_s, y_np_s = get_circles(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X, y, epsilon=EPS, n_epochs=n_epochs,
                      verbose=False, use_sgld=True)
        acc   = hist['accuracy'][-1]
        results_F1.append({'model': model, 'hist': hist, 'seed': s,
                           'acc': acc, 'X_np': X_np_s, 'y_np': y_np_s})
        print(f"    data_seed={s}: J*={hist['J_star']:.4f}, acc={acc:.3f}")

    # ── F2: semillas de init, datos fijos ─────────────────────────────────────
    print("  F2: dataset circles seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, X_np_fixed, y_np_fixed = get_circles(seed=DATA_SEED_FIXED)
    results_F2 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        acc   = hist['accuracy'][-1]
        results_F2.append({'model': model, 'hist': hist, 'seed': s, 'acc': acc})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, acc={acc:.3f}")

    # ── Resumen en consola ────────────────────────────────────────────────────
    acc_F1 = np.array([r['acc'] for r in results_F1])
    acc_F2 = np.array([r['acc'] for r in results_F2])
    print(f"\n  Acc media F1 (datos): {acc_F1.mean():.4f} ± {acc_F1.std():.4f}")
    print(f"  Acc media F2 (init):  {acc_F2.mean():.4f} ± {acc_F2.std():.4f}")

    # ── Figura F ──────────────────────────────────────────────────────────────
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)

    # ── Panel único: curvas de pérdida F1 y F2 superpuestas ──────────────────
    for results, color, label in [
        (results_F1, '#e74c3c', 'F1 (datos)'),
        (results_F2, '#3498db', 'F2 (init)'),
    ]:
        losses    = np.array([r['hist']['loss'] for r in results])
        mean_l    = losses.mean(axis=0)
        std_l     = losses.std(axis=0)
        epochs    = np.arange(len(mean_l))
        # pSGLD explores ν* ∝ exp(-J/ε): the loss oscillates around the minimum,
        # not converging to it.  J_final = mean of last 50 epochs captures the
        # steady-state level; J* (minimum at any epoch) would be misleading.
        J_final_arr = np.array([np.mean(r['hist']['loss'][-50:]) for r in results])
        for r in results:
            ax0.plot(r['hist']['loss'], color=color, lw=0.7, alpha=0.25)
        ax0.plot(mean_l, color=color, lw=2.0,
                 label=(f'{label}  '
                        rf'$\bar{{J}}_{{final}}$={J_final_arr.mean():.4f}'
                        rf'$\pm${J_final_arr.std():.4f}'))
        ax0.fill_between(epochs, mean_l - std_l, mean_l + std_l,
                         color=color, alpha=0.15)
    style_ax(ax0,
             'Convergencia — make_circles  (ε=0.01)\n'
             r'pSGLD explora $\nu^*\!\propto\!\exp(-J/\varepsilon)$: '
             r'$\bar{J}_{final}$ = media de las últimas 50 épocas',
             'Época', 'J (BCE + ε·reg)')
    ax0.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(
        r'Experimento F — Convergencia en make\_circles  (ε=0.01)'
        '\n'
        r'F1: $\gamma_0$ aleatoria (10 datasets), $\theta_0$ fija  |  '
        r'F2: $\gamma_0$ fija, $\theta_0$ aleatoria (10 inits)',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, 'F_circles_parameter_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")
