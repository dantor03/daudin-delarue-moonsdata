"""
=============================================================================
exp_d.py — Experimento D: Genericidad del minimizador (Meta-Teorema 1)
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, GRID_C, style_ax
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate


# =============================================================================
# EXPERIMENTO D
#   Genericidad: robustez a semillas de inicialización y de datos
# =============================================================================
def experiment_D(n_seeds: int = 10, n_epochs: int = 500):
    """
    Genericidad del minimizador: robustez a semillas de inicialización y de datos.

    META-TEOREMA 1 (paper, sec. 1.3):
        Para un conjunto abierto y denso 𝒪 de condiciones iniciales γ₀, el
        problema de control tiene un único minimizador ESTABLE.  La palabra
        "genéricamente" significa que casi toda distribución inicial de datos
        produce un paisaje de pérdida con un único mínimo profundo.

    CONEXIÓN CON LA CONDICIÓN PL (Meta-Teorema 2):
        Si la condición PL se cumple con μ > 0, gradient descent NO puede
        quedar atrapado en mínimos locales distintos: la desigualdad garantiza
        que desde cualquier punto del espacio de parámetros hay "cuesta abajo"
        hacia el mínimo global.  Esto implica robustez a la inicialización.

    DISEÑO DEL EXPERIMENTO:
        Dos sub-experimentos para separar dos fuentes de variabilidad:

        D1 — Robustez a la INICIALIZACIÓN (γ₀ fija, semilla de ν₀ variable):
            • Dataset fijo: data_seed = 42  (misma γ₀ en todos los runs)
            • Parámetros inicializados con n_seeds semillas distintas
            • Para ε ∈ {0, 0.01}: ¿convergen al mismo J*?
            • Predicción del paper: con ε > 0 la varianza de J* debe ser baja
              (unicidad del minimizador garantizada por PL)

        D2 — Robustez a γ₀ (inicialización fija, semilla de datos variable):
            • init_seed = 4  (semilla que converge bien en D1)
            • Dataset generado con n_seeds semillas distintas → n_seeds γ₀ distintas
            • Para ε ∈ {0, 0.01}: ¿converge siempre?
            • Predicción del paper: genericidad → casi toda γ₀ admite minimizador
              estable; las fronteras de decisión deben ser cualitativamente similares

        D3 — Variabilidad conjunta (ambas semillas varían simultáneamente):
            • Para la semilla s ∈ {0,...,n_seeds-1}: data_seed=s E init_seed=s
            • Ni el dataset ni los parámetros iniciales se repiten entre runs
            • Es el escenario más realista: en la práctica no se controla ninguna
              de las dos fuentes de aleatoriedad
            • Predicción: mayor dispersión que D1 y D2 por separado; con ε > 0
              la banda debe ser más estrecha que con ε = 0
            • Las fronteras de D3 combinan variabilidad geométrica (γ₀) y
              variabilidad de paisaje (θ₀) — aun así deben ser topológicamente
              similares si el Meta-Teorema 1 se cumple

    FIGURAS GENERADAS (D_genericity.png):
        Layout 3×3:
        Fila 0 (D1):  curvas J ε=0  |  curvas J ε=0.01  |  boxplot J* y μ̂
        Fila 1 (D2):  curvas J ε=0  |  curvas J ε=0.01  |  fronteras superpuestas
        Fila 2 (D3):  curvas J ε=0  |  curvas J ε=0.01  |  fronteras superpuestas

    RESULTADO ESPERADO:
        • D3 debe tener la banda más ancha de los tres sub-experimentos
          (combina ambas fuentes de variabilidad)
        • Con ε=0.01 la banda de D3 debe ser más estrecha que con ε=0
        • Las fronteras de D3 son cualitativamente similares aunque provienen
          de puntos de partida totalmente distintos en (γ₀, θ₀)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO D  —  Genericidad: robustez a semillas")
    print("=" * 62)

    SEEDS = list(range(n_seeds))
    EPS_COMPARE = [0.0, 0.01]
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 2   # seed 2 da el J* más bajo en D1 (J*≈0.00015) → D2 muestra
                          # variabilidad real de γ₀ sin contaminación por
                          # una mala inicialización

    # ── D1: γ₀ fija, inicialización variable ─────────────────────────────────
    print(f"\n  D1 — γ₀ fija (data_seed={DATA_SEED_FIXED}), "
          f"{n_seeds} seeds de inicialización")
    X, y, X_np, y_np = get_moons(seed=DATA_SEED_FIXED)

    d1_results = {}   # d1_results[eps] = lista de dicts {hist, model, mu}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d1_results[eps] = []
        for s in SEEDS:
            # Fijamos la semilla de inicialización para reproducibilidad
            torch.manual_seed(s)
            np.random.seed(s)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, X, y, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      init_seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d1_results[eps].append({'hist': hist, 'model': model, 'mu': mu})

    # ── D2: inicialización fija, γ₀ variable ─────────────────────────────────
    print(f"\n  D2 — init fija (init_seed={INIT_SEED_FIXED}), "
          f"{n_seeds} seeds de datos (γ₀ distintas)")

    d2_results = {}   # d2_results[eps] = lista de dicts {hist, model, mu, X_np, y_np}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d2_results[eps] = []
        for s in SEEDS:
            Xs, ys, Xs_np, ys_np = get_moons(seed=s)
            # Misma inicialización de parámetros en todos los runs de D2
            torch.manual_seed(INIT_SEED_FIXED)
            np.random.seed(INIT_SEED_FIXED)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, Xs, ys, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      data_seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d2_results[eps].append({
                'hist': hist, 'model': model, 'mu': mu,
                'X_np': Xs_np, 'y_np': ys_np
            })

    # ── D3: ambas semillas varían simultáneamente ─────────────────────────────
    print(f"\n  D3 — ambas semillas varían (data_seed=s, init_seed=s), "
          f"{n_seeds} pares distintos")

    d3_results = {}   # d3_results[eps] = lista de dicts {hist, model, mu, X_np, y_np}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d3_results[eps] = []
        for s in SEEDS:
            Xs, ys, Xs_np, ys_np = get_moons(seed=s)
            torch.manual_seed(s)
            np.random.seed(s)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, Xs, ys, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d3_results[eps].append({
                'hist': hist, 'model': model, 'mu': mu,
                'X_np': Xs_np, 'y_np': ys_np
            })

    # ── Resumen estadístico ───────────────────────────────────────────────────
    print("\n  RESUMEN — Robustez a init seeds (D1):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d1_results[eps]]
        mus    = [r['mu']             for r in d1_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    print("\n  RESUMEN — Robustez a data seeds (D2):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d2_results[eps]]
        mus    = [r['mu']             for r in d2_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    print("\n  RESUMEN — Variabilidad conjunta (D3):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d3_results[eps]]
        mus    = [r['mu']             for r in d3_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    # ── Figura D — Layout 3×3 ─────────────────────────────────────────────────
    # Paleta de colores para las seeds individuales
    SEED_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6',
                   '#e67e22', '#1abc9c', '#e84393', '#a29bfe', '#fd79a8']

    fig = plt.figure(figsize=(21, 18))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.34)

    def _plot_loss_curves(ax, results_list, title):
        """
        Curvas de pérdida individuales (transparentes) + media ± 1σ en blanco.

        Cada curva fina de color corresponde a un run con distinta semilla.
        La banda blanca muestra la variabilidad entre seeds: cuanto más estrecha,
        más robusta es la convergencia (lo que predice la condición PL con ε > 0).
        """
        losses = np.array([r['hist']['loss'] for r in results_list])   # (S, E)
        epochs = np.arange(losses.shape[1])
        for i, r in enumerate(results_list):
            ax.plot(r['hist']['loss'],
                    color=SEED_COLORS[i % len(SEED_COLORS)],
                    lw=0.9, alpha=0.45)
        mean_l = losses.mean(axis=0)
        std_l  = losses.std(axis=0)
        ax.plot(epochs, mean_l, color='white', lw=2.0, label='Media')
        ax.fill_between(epochs, mean_l - std_l, mean_l + std_l,
                        color='white', alpha=0.18, label='±1σ')
        style_ax(ax, title, 'Época', '$J$')
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── Fila 0: D1 ────────────────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    _plot_loss_curves(ax00, d1_results[0.0],
                      r'D1: init aleatoria, $\varepsilon=0$'
                      '\n' r'$\gamma_0$ fija')

    ax01 = fig.add_subplot(gs[0, 1])
    _plot_loss_curves(ax01, d1_results[0.01],
                      r'D1: init aleatoria, $\varepsilon=0.01$'
                      '\n' r'$\gamma_0$ fija')

    # D1 col 2: dos subpaneles independientes — μ̂ arriba, J* abajo.
    # Cada métrica tiene su propia escala Y, evitando que J* (escala ~0.1)
    # aplaste μ̂ (escala ~0.003) cuando comparten eje.
    gs02  = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 2], hspace=0.55
    )
    ax02a = fig.add_subplot(gs02[0])   # μ̂_PL
    ax02b = fig.add_subplot(gs02[1])   # J*

    mu_e0   = [r['mu']             for r in d1_results[0.0]]
    mu_e05 = [r['mu']             for r in d1_results[0.01]]
    js_e0   = [r['hist']['J_star'] for r in d1_results[0.0]]
    js_e05 = [r['hist']['J_star'] for r in d1_results[0.01]]

    _box_kw = dict(
        patch_artist=True, widths=0.55,
        medianprops=dict(color='white', lw=2.2),
        whiskerprops=dict(color=TXT, lw=1.2),
        capprops=dict(color=TXT, lw=1.2),
        flierprops=dict(markerfacecolor=TXT, marker='o', markersize=4)
    )

    # — subpanel μ̂ —
    bp_mu = ax02a.boxplot([mu_e0, mu_e05], positions=[1, 2], **_box_kw)
    for patch, c in zip(bp_mu['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax02a.set_xticks([1, 2])
    ax02a.set_xticklabels(['ε=0', 'ε=0.01'], color=TXT, fontsize=8)
    ax02a.axhline(0, color=GRID_C, lw=1.0, ls='--', alpha=0.7)
    style_ax(ax02a, r'$\hat{\mu}_{PL}$ entre init seeds', '', r'$\hat{\mu}$')

    # — subpanel J* —
    bp_js = ax02b.boxplot([js_e0, js_e05], positions=[1, 2], **_box_kw)
    for patch, c in zip(bp_js['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax02b.set_xticks([1, 2])
    ax02b.set_xticklabels(['ε=0', 'ε=0.01'], color=TXT, fontsize=8)
    style_ax(ax02b, r'$J^*$ entre init seeds', '', r'$J^*$')

    # ── Fila 1: D2 ────────────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    _plot_loss_curves(ax10, d2_results[0.0],
                      r'D2: $\gamma_0$ aleatoria, $\varepsilon=0$'
                      '\n' r'init fija')

    ax11 = fig.add_subplot(gs[1, 1])
    _plot_loss_curves(ax11, d2_results[0.01],
                      r'D2: $\gamma_0$ aleatoria, $\varepsilon=0.01$'
                      '\n' r'init fija')

    # D2 col 2: fronteras de decisión superpuestas, ε=0.01, γ₀ variables.
    # Scatter de TODOS los datasets (alpha bajo) como fondo, para que cada
    # frontera tenga el contexto de su propio dataset.
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.set_facecolor(PANEL_BG)

    xmin_g = min(r['X_np'][:, 0].min() for r in d2_results[0.01]) - 0.15
    xmax_g = max(r['X_np'][:, 0].max() for r in d2_results[0.01]) + 0.15
    ymin_g = min(r['X_np'][:, 1].min() for r in d2_results[0.01]) - 0.15
    ymax_g = max(r['X_np'][:, 1].max() for r in d2_results[0.01]) + 0.15
    xx_c, yy_c = np.meshgrid(np.linspace(xmin_g, xmax_g, 150),
                              np.linspace(ymin_g, ymax_g, 150))
    grid_c = torch.tensor(
        np.c_[xx_c.ravel(), yy_c.ravel()].astype(np.float32), device=DEVICE
    )

    for r in d2_results[0.01]:
        ax12.scatter(r['X_np'][r['y_np'] == 0, 0], r['X_np'][r['y_np'] == 0, 1],
                     c='#ff6b6b', s=5, alpha=0.06, zorder=1)
        ax12.scatter(r['X_np'][r['y_np'] == 1, 0], r['X_np'][r['y_np'] == 1, 1],
                     c='#74b9ff', s=5, alpha=0.06, zorder=1)

    for i, r in enumerate(d2_results[0.01]):
        if r['hist']['accuracy'][-1] < 0.95:
            continue
        m = r['model']
        m.eval()
        with torch.no_grad():
            Z = torch.sigmoid(m(grid_c)).cpu().numpy().reshape(xx_c.shape)
        ax12.contourf(xx_c, yy_c, Z, levels=[0.0, 0.5, 1.0],
                      colors=[SEED_COLORS[i], SEED_COLORS[i]], alpha=0.06, zorder=2)
        ax12.contour(xx_c, yy_c, Z, levels=[0.5],
                     colors=[SEED_COLORS[i]], linewidths=1.8,
                     alpha=0.90, zorder=5)

    style_ax(ax12,
             r'D2: fronteras de decisión, $\varepsilon=0.01$'
             '\n' r'$\gamma_0$ variables  (acc $\geq$ 0.95)',
             '$x_1$', '$x_2$')
    ax12.set_aspect('equal')
    ax12.set_xlim(xmin_g, xmax_g)
    ax12.set_ylim(ymin_g, ymax_g)

    # ── Fila 2: D3 ────────────────────────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    _plot_loss_curves(ax20, d3_results[0.0],
                      r'D3: ambas aleatorias, $\varepsilon=0$'
                      '\n' r'data\_seed = init\_seed = $s$')

    ax21 = fig.add_subplot(gs[2, 1])
    _plot_loss_curves(ax21, d3_results[0.01],
                      r'D3: ambas aleatorias, $\varepsilon=0.01$'
                      '\n' r'data\_seed = init\_seed = $s$')

    # D3 col 2: fronteras de decisión, ε=0.01, ambas semillas variables.
    # Scatter de TODOS los datasets (alpha muy bajo) para que cada frontera
    # tenga contexto. Cada modelo se evalúa en la cuadrícula común.
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.set_facecolor(PANEL_BG)

    xmin_d3 = min(r['X_np'][:, 0].min() for r in d3_results[0.01]) - 0.15
    xmax_d3 = max(r['X_np'][:, 0].max() for r in d3_results[0.01]) + 0.15
    ymin_d3 = min(r['X_np'][:, 1].min() for r in d3_results[0.01]) - 0.15
    ymax_d3 = max(r['X_np'][:, 1].max() for r in d3_results[0.01]) + 0.15
    xx_d3, yy_d3 = np.meshgrid(np.linspace(xmin_d3, xmax_d3, 150),
                                np.linspace(ymin_d3, ymax_d3, 150))
    grid_d3 = torch.tensor(
        np.c_[xx_d3.ravel(), yy_d3.ravel()].astype(np.float32), device=DEVICE
    )

    # Scatter de todos los datasets (alpha muy bajo = nube de fondo)
    for r in d3_results[0.01]:
        ax22.scatter(r['X_np'][r['y_np'] == 0, 0], r['X_np'][r['y_np'] == 0, 1],
                     c='#ff6b6b', s=5, alpha=0.06, zorder=1)
        ax22.scatter(r['X_np'][r['y_np'] == 1, 0], r['X_np'][r['y_np'] == 1, 1],
                     c='#74b9ff', s=5, alpha=0.06, zorder=1)

    for i, r in enumerate(d3_results[0.01]):
        if r['hist']['accuracy'][-1] < 0.95:
            continue
        m = r['model']
        m.eval()
        with torch.no_grad():
            Z = torch.sigmoid(m(grid_d3)).cpu().numpy().reshape(xx_d3.shape)
        ax22.contourf(xx_d3, yy_d3, Z, levels=[0.0, 0.5, 1.0],
                      colors=[SEED_COLORS[i], SEED_COLORS[i]], alpha=0.06, zorder=2)
        ax22.contour(xx_d3, yy_d3, Z, levels=[0.5],
                     colors=[SEED_COLORS[i]], linewidths=1.8,
                     alpha=0.90, zorder=5)

    style_ax(ax22,
             r'D3: fronteras de decisión, $\varepsilon=0.01$'
             '\n' r'(data\_seed, init\_seed) distintos  (acc $\geq$ 0.95)',
             '$x_1$', '$x_2$')
    ax22.set_aspect('equal')
    ax22.set_xlim(xmin_d3, xmax_d3)
    ax22.set_ylim(ymin_d3, ymax_d3)

    fig.suptitle(
        r'Experimento D — Genericidad del minimizador (Meta-Teorema 1)'
        '\n'
        r'D1: $\gamma_0$ fija, init aleatoria   |   '
        r'D2: init fija, $\gamma_0$ aleatoria   |   '
        r'D3: ambas aleatorias',
        color=TXT, fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, 'D_genericity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    print("\n  INTERPRETACIÓN:")
    print("  • D3 combina ambas fuentes de variabilidad: esperar banda más ancha.")
    print("  • Con ε=0.01 la banda de D3 debe ser más estrecha que con ε=0.")
    print("  • Las fronteras de D3 deben ser topológicamente similares (Meta-Teorema 1).")

    return d1_results, d2_results, d3_results
