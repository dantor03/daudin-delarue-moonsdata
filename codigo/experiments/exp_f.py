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

    FIGURAS GENERADAS (F_circles_parameter_distribution.png):
        Layout 3×3:
        Fila 0 (F1): curvas J ±1σ  |  scatter 2D de a₁  |  hist. ángulo θ(a₁)
        Fila 1 (F2): curvas J ±1σ  |  scatter 2D de a₁  |  hist. ángulo θ(a₁)
        Fila 2 (síntesis): R̄ por run  |  ||a₁|| media ± std por run  |  importancias
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO F  —  ν* en make_circles: simetría y robustez")
    print("=" * 62)

    EPS             = 0.01
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 4
    SEEDS           = list(range(n_seeds))

    # ── Funciones auxiliares ──────────────────────────────────────────────────
    def get_a1(model):
        """Extrae los pesos espaciales a₁ᵐ ∈ ℝ² para cada neurona."""
        W1w = model.velocity.W1.weight.detach().cpu().numpy()  # (M, d1+1)
        return W1w[:, :2]   # (M, 2)

    def get_a0(model):
        """Extrae los pesos de salida a₀ᵐ ∈ ℝ² para cada neurona."""
        W0w = model.velocity.W0.weight.detach().cpu().numpy()  # (d1, M)
        return W0w.T        # (M, 2)

    def importance(a0):
        """||a₀ᵐ||₂ — importancia de cada neurona."""
        return np.linalg.norm(a0, axis=1)

    def mean_resultant_length(a1):
        """
        Longitud resultante media R̄ del ángulo de los vectores a₁ᵐ.

        R̄ = |mean(exp(iθ))| ∈ [0,1], con θ = arctan2(a₁ᵐ[1], a₁ᵐ[0]).
        R̄ ≈ 0: distribución isotrópica (uniforme en S¹).
        R̄ ≈ 1: todos los ángulos apuntan en la misma dirección.
        """
        angles = np.arctan2(a1[:, 1], a1[:, 0])
        return float(np.abs(np.mean(np.exp(1j * angles))))

    # ── F1: semillas de datos, init fija ─────────────────────────────────────
    print("  F1: 10 datasets circles distintos, init_seed=4 fija...")
    results_F1 = []
    for s in SEEDS:
        X, y, X_np_s, y_np_s = get_circles(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X, y, epsilon=EPS, n_epochs=n_epochs, verbose=False)
        acc   = hist['accuracy'][-1]
        a1, a0 = get_a1(model), get_a0(model)
        Rbar   = mean_resultant_length(a1)
        results_F1.append({'model': model, 'hist': hist, 'seed': s,
                           'a1': a1, 'a0': a0, 'acc': acc,
                           'Rbar': Rbar, 'X_np': X_np_s, 'y_np': y_np_s})
        print(f"    data_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={acc:.3f}, R̄={Rbar:.3f}")

    # ── F2: semillas de init, datos fijos ─────────────────────────────────────
    print("  F2: dataset circles seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, X_np_fixed, y_np_fixed = get_circles(seed=DATA_SEED_FIXED)
    results_F2 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False)
        acc   = hist['accuracy'][-1]
        a1, a0 = get_a1(model), get_a0(model)
        Rbar   = mean_resultant_length(a1)
        results_F2.append({'model': model, 'hist': hist, 'seed': s,
                           'a1': a1, 'a0': a0, 'acc': acc, 'Rbar': Rbar})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={acc:.3f}, R̄={Rbar:.3f}")

    # ── Resumen en consola ────────────────────────────────────────────────────
    Rbar_F1 = np.array([r['Rbar'] for r in results_F1])
    Rbar_F2 = np.array([r['Rbar'] for r in results_F2])
    print(f"\n  R̄ medio F1 (datos): {Rbar_F1.mean():.4f} ± {Rbar_F1.std():.4f}")
    print(f"  R̄ medio F2 (init):  {Rbar_F2.mean():.4f} ± {Rbar_F2.std():.4f}")
    print("  R̄≈0 → ν* isotrópico (predicción de simetría circles ✓)")

    # ── Figura F ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor(DARK_BG)

    # ── Panel 0: curvas de pérdida F1 y F2 superpuestas ──────────────────────
    ax0 = axes[0]
    for results, color, label in [
        (results_F1, '#e74c3c', 'F1 (datos)'),
        (results_F2, '#3498db', 'F2 (init)'),
    ]:
        losses    = np.array([r['hist']['loss'] for r in results])
        mean_l    = losses.mean(axis=0)
        std_l     = losses.std(axis=0)
        epochs    = np.arange(len(mean_l))
        Jstar_arr = np.array([r['hist']['J_star'] for r in results])
        for r in results:
            ax0.plot(r['hist']['loss'], color=color, lw=0.7, alpha=0.25)
        ax0.plot(mean_l, color=color, lw=2.0,
                 label=f'{label}  J*={Jstar_arr.mean():.4f}±{Jstar_arr.std():.4f}')
        ax0.fill_between(epochs, mean_l - std_l, mean_l + std_l,
                         color=color, alpha=0.15)
    style_ax(ax0,
             'Convergencia — make_circles  (ε=0.01)\n'
             'Banda = ±1σ sobre 10 semillas',
             'Época', 'J (BCE + ε·reg)')
    ax0.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── Panel 1: histograma de ángulo θ(a₁) — banda media ± std ─────────────
    ax1 = axes[1]
    bins  = np.linspace(-np.pi, np.pi, 25)
    bin_c = 0.5 * (bins[:-1] + bins[1:])
    for results, color, label in [
        (results_F1, '#e74c3c', 'F1 (datos)'),
        (results_F2, '#3498db', 'F2 (init)'),
    ]:
        hists = []
        for r in results:
            angles = np.arctan2(r['a1'][:, 1], r['a1'][:, 0])
            counts, _ = np.histogram(angles, bins=bins)
            hists.append(counts / counts.sum())
        hists   = np.array(hists)
        mean_h  = hists.mean(axis=0)
        std_h   = hists.std(axis=0)
        ax1.plot(bin_c, mean_h, color=color, lw=2.0, label=label)
        ax1.fill_between(bin_c, mean_h - std_h, mean_h + std_h,
                         color=color, alpha=0.20)
    uniform_h = 1.0 / (len(bins) - 1)
    ax1.axhline(uniform_h, color='white', lw=2.0, ls='--', alpha=0.80,
                label='Uniforme')
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'],
                        color=TXT, fontsize=8)
    style_ax(ax1,
             r'Histograma de $\theta(a_1^m)$ — media ± std sobre 10 semillas'
             '\n' r'Curva plana = $\nu^*$ isotrópico (predicción de simetría SO(2))',
             r'$\theta = \arctan2(a_1[1],\,a_1[0])$', 'Densidad')
    ax1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── Panel 2: R̄ por run — test cuantitativo de isotropía ─────────────────
    ax2 = axes[2]
    x_F1 = np.arange(n_seeds)
    x_F2 = np.arange(n_seeds) + n_seeds + 1.5
    ax2.bar(x_F1, Rbar_F1, color='#e74c3c', alpha=0.82, width=0.8,
            label=f'F1 (datos):  R̄={Rbar_F1.mean():.3f}')
    ax2.bar(x_F2, Rbar_F2, color='#3498db', alpha=0.82, width=0.8,
            label=f'F2 (init):   R̄={Rbar_F2.mean():.3f}')
    ax2.axhline(0.0, color='white', lw=1.0, ls='--', alpha=0.4)
    for xi, v in zip(x_F1, Rbar_F1):
        ax2.text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom',
                 color=TXT, fontsize=6.5)
    for xi, v in zip(x_F2, Rbar_F2):
        ax2.text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom',
                 color=TXT, fontsize=6.5)
    ax2.set_xticks(list(x_F1) + list(x_F2))
    ax2.set_xticklabels(
        [f'F1-{s}' for s in SEEDS] + [f'F2-{s}' for s in SEEDS],
        rotation=50, fontsize=6.5, color=TXT)
    ax2.set_ylim(0, max(Rbar_F1.max(), Rbar_F2.max()) * 1.25)
    style_ax(ax2,
             r'Longitud resultante media $\bar{R}$ del ángulo de $a_1^m$'
             '\n' r'$\bar{R} \approx 0$ = isotrópico  |  $\bar{R} \approx 1$ = concentrado',
             'Run', r'$\bar{R}$')
    ax2.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(
        r'Experimento F — Simetría rotacional de $\nu^*$ en make\_circles  (ε=0.01)'
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
