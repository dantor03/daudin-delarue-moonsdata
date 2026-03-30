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
    SEED_CMAP = plt.cm.tab10

    fig = plt.figure(figsize=(21, 18))
    fig.patch.set_facecolor(DARK_BG)
    gs_fig = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    # ── Helpers de figura ─────────────────────────────────────────────────────
    def _loss_band(ax, results, title):
        """Curvas de convergencia con banda ±1σ entre seeds."""
        losses = np.array([r['hist']['loss'] for r in results])
        mean_l = losses.mean(axis=0)
        std_l  = losses.std(axis=0)
        epochs = np.arange(len(mean_l))
        for i, r in enumerate(results):
            ax.plot(r['hist']['loss'], color=SEED_CMAP(i % 10),
                    lw=0.7, alpha=0.35)
        ax.plot(mean_l, color='white', lw=2.0, label='Media')
        ax.fill_between(epochs, mean_l - std_l, mean_l + std_l,
                        color='white', alpha=0.15, label='±1σ')
        Jstar_arr = np.array([r['hist']['J_star'] for r in results])
        ax.text(0.97, 0.97,
                f'σ(J*)={Jstar_arr.std():.4f}\nJ*={Jstar_arr.mean():.4f}±{Jstar_arr.std():.4f}',
                transform=ax.transAxes, ha='right', va='top',
                color=TXT, fontsize=7.5,
                bbox=dict(facecolor=PANEL_BG, alpha=0.75, pad=2))
        ax.set_xlim(0, len(mean_l))
        style_ax(ax, title, 'Época', 'J (BCE + ε·reg)')
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    def _scatter_a1(ax, results, title, X_ref=None, y_ref=None):
        """
        Scatter 2D de los vectores a₁ᵐ para todos los runs superpuestos.
        Si ν* es isotrópico, la nube de puntos debe formar un anillo.
        """
        if X_ref is not None:
            ax.scatter(X_ref[y_ref == 0, 0], X_ref[y_ref == 0, 1],
                       c='#ff6b6b', s=4, alpha=0.10, zorder=1)
            ax.scatter(X_ref[y_ref == 1, 0], X_ref[y_ref == 1, 1],
                       c='#74b9ff', s=4, alpha=0.10, zorder=1)
        for i, r in enumerate(results):
            a1 = r['a1']
            ax.scatter(a1[:, 0], a1[:, 1],
                       color=SEED_CMAP(i % 10), s=22,
                       alpha=0.55, edgecolors='none')
        # Círculo de referencia con radio = norma media de a₁
        r_med = np.median([np.linalg.norm(r['a1'], axis=1).mean()
                           for r in results])
        theta_ref = np.linspace(0, 2 * np.pi, 300)
        ax.plot(r_med * np.cos(theta_ref), r_med * np.sin(theta_ref),
                color='white', lw=1.2, ls='--', alpha=0.5,
                label=f'r̄={r_med:.2f}')
        ax.set_aspect('equal')
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)
        style_ax(ax, title, r'$a_1^m[0]$', r'$a_1^m[1]$')

    def _angle_hist(ax, results, title):
        """
        Histograma del ángulo polar θ = arctan2(a₁[1], a₁[0]).
        Si ν* es isotrópico → distribución plana (uniforme en [-π, π]).
        """
        bins   = np.linspace(-np.pi, np.pi, 25)
        bin_c  = 0.5 * (bins[:-1] + bins[1:])
        for i, r in enumerate(results):
            angles = np.arctan2(r['a1'][:, 1], r['a1'][:, 0])
            counts, _ = np.histogram(angles, bins=bins)
            ax.plot(bin_c, counts / counts.sum(),
                    color=SEED_CMAP(i % 10), lw=1.3, alpha=0.55, marker='.')
        uniform_h = 1.0 / (len(bins) - 1)
        ax.axhline(uniform_h, color='white', lw=2.0, ls='--', alpha=0.80,
                   label='Uniforme')
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'],
                           color=TXT, fontsize=7.5)
        style_ax(ax, title,
                 r'$\theta = \arctan2(a_1^m[1],\,a_1^m[0])$', 'Densidad')
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── Fila 0: F1 ────────────────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs_fig[0, 0])
    _loss_band(ax00, results_F1,
               r'F1 — $\gamma_0$ aleatoria, $\theta_0$ fija  (ε=0.01)'
               '\n10 datasets make_circles distintos')

    # Datos de referencia = primer dataset de F1
    X_ref_F1 = results_F1[0]['X_np']
    y_ref_F1 = results_F1[0]['y_np']
    ax01 = fig.add_subplot(gs_fig[0, 1])
    _scatter_a1(ax01, results_F1,
                r'F1 — Distribución 2D de $a_1^m$  (10 datasets)'
                '\n' r'¿forma anular? → simetría rotacional de circles',
                X_ref=X_ref_F1, y_ref=y_ref_F1)

    ax02 = fig.add_subplot(gs_fig[0, 2])
    _angle_hist(ax02, results_F1,
                r'F1 — Histograma de $\theta(a_1^m)$'
                '\n' r'Curva plana = $\nu^*$ isotrópico (predicción de simetría)')

    # ── Fila 1: F2 ────────────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs_fig[1, 0])
    _loss_band(ax10, results_F2,
               r'F2 — $\gamma_0$ fija, $\theta_0$ aleatoria  (ε=0.01)'
               '\n10 inicializaciones de parámetros distintas')

    ax11 = fig.add_subplot(gs_fig[1, 1])
    _scatter_a1(ax11, results_F2,
                r'F2 — Distribución 2D de $a_1^m$  (10 inits)'
                '\n' r'¿el anillo se preserva con distintos $\theta_0$?',
                X_ref=X_np_fixed, y_ref=y_np_fixed)

    ax12 = fig.add_subplot(gs_fig[1, 2])
    _angle_hist(ax12, results_F2,
                r'F2 — Histograma de $\theta(a_1^m)$'
                '\n' r'Estabilidad angular entre inicializaciones')

    # ── Fila 2: síntesis ──────────────────────────────────────────────────────

    # (2,0): R̄ por run — test cuantitativo de isotropía
    ax20 = fig.add_subplot(gs_fig[2, 0])
    x_F1 = np.arange(n_seeds)
    x_F2 = np.arange(n_seeds) + n_seeds + 1.5
    ax20.bar(x_F1, Rbar_F1, color='#e74c3c', alpha=0.82, width=0.8,
             label=f'F1 (datos):  R̄={Rbar_F1.mean():.3f}')
    ax20.bar(x_F2, Rbar_F2, color='#3498db', alpha=0.82, width=0.8,
             label=f'F2 (init):   R̄={Rbar_F2.mean():.3f}')
    ax20.axhline(0.0, color='white', lw=1.0, ls='--', alpha=0.4)
    # Anotaciones de valor
    for xi, v in zip(x_F1, Rbar_F1):
        ax20.text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom',
                  color=TXT, fontsize=6.5)
    for xi, v in zip(x_F2, Rbar_F2):
        ax20.text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom',
                  color=TXT, fontsize=6.5)
    ax20.set_xticks(list(x_F1) + list(x_F2))
    ax20.set_xticklabels(
        [f'F1-{s}' for s in SEEDS] + [f'F2-{s}' for s in SEEDS],
        rotation=50, fontsize=6.5, color=TXT)
    ax20.set_ylim(0, max(Rbar_F1.max(), Rbar_F2.max()) * 1.25)
    style_ax(ax20,
             r'Longitud resultante media $\bar{R}$ del ángulo de $a_1^m$'
             '\n' r'$\bar{R} \approx 0$ = isotrópico  |  $\bar{R} \approx 1$ = concentrado',
             'Run', r'$\bar{R}$')
    ax20.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # (2,1): norma ||a₁|| media ± std por run — escala de las proyecciones
    ax21 = fig.add_subplot(gs_fig[2, 1])
    nm_F1 = [np.linalg.norm(r['a1'], axis=1).mean() for r in results_F1]
    ns_F1 = [np.linalg.norm(r['a1'], axis=1).std()  for r in results_F1]
    nm_F2 = [np.linalg.norm(r['a1'], axis=1).mean() for r in results_F2]
    ns_F2 = [np.linalg.norm(r['a1'], axis=1).std()  for r in results_F2]
    ax21.errorbar(SEEDS, nm_F1, yerr=ns_F1,
                  fmt='o-', color='#e74c3c', capsize=4, lw=1.8,
                  label='F1 (datos)')
    ax21.errorbar(SEEDS, nm_F2, yerr=ns_F2,
                  fmt='s--', color='#3498db', capsize=4, lw=1.8,
                  label='F2 (init)')
    style_ax(ax21,
             r'Escala de $a_1^m$: $\overline{\|a_1^m\|_2}$ ± std por run'
             '\n' r'Mide cuán "agresiva" es la proyección espacial de cada run',
             'Semilla $s$', r'$\overline{\|a_1^m\|_2}$')
    ax21.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # (2,2): curvas de importancia ||a₀|| para todos los runs (F1 + F2)
    ax22 = fig.add_subplot(gs_fig[2, 2])
    for i, r in enumerate(results_F1):
        imp_s = np.sort(importance(r['a0']))[::-1]
        ax22.plot(np.arange(1, len(imp_s) + 1), imp_s,
                  color='#e74c3c', lw=0.9, alpha=0.35)
    for i, r in enumerate(results_F2):
        imp_s = np.sort(importance(r['a0']))[::-1]
        ax22.plot(np.arange(1, len(imp_s) + 1), imp_s,
                  color='#3498db', lw=0.9, alpha=0.35)
    # Curvas medias
    mean_imp_F1 = np.sort(
        np.mean([importance(r['a0']) for r in results_F1], axis=0))[::-1]
    mean_imp_F2 = np.sort(
        np.mean([importance(r['a0']) for r in results_F2], axis=0))[::-1]
    ax22.plot(np.arange(1, len(mean_imp_F1) + 1), mean_imp_F1,
              color='#e74c3c', lw=2.5, label='F1 media')
    ax22.plot(np.arange(1, len(mean_imp_F2) + 1), mean_imp_F2,
              color='#3498db', lw=2.5, ls='--', label='F2 media')
    style_ax(ax22,
             r'Importancias $\|a_0^m\|_2$ ordenadas — todos los runs'
             '\n' r'Estabilidad del rango efectivo entre semillas',
             'Rank (neurona)', r'$\|a_0^m\|_2$')
    ax22.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(
        r'Experimento F — Distribución de $\nu^*$ en make_circles'
        r': simetría rotacional y robustez a semillas'
        '\n'
        r'F1: $\gamma_0$ aleatoria (10 datasets), $\theta_0$ fija  |  '
        r'F2: $\gamma_0$ fija, $\theta_0$ aleatoria (10 inits)  |  ε=0.01',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, 'F_circles_parameter_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")
