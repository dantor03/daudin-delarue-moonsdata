"""
=============================================================================
exp_e.py — Experimentos E y E2: Análisis y robustez de la distribución ν*
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


# =============================================================================
# EXPERIMENTO E
#   Análisis de la distribución de parámetros ν* aprendida
# =============================================================================
def experiment_E(results_eps: dict):
    """
    Análisis en profundidad de la distribución de parámetros ν* aprendida.

    MOTIVACIÓN:
        El experimento B3 muestra la distribución MARGINAL de todos los
        parámetros combinados comparada con el prior ν^∞.  Pero la estructura
        de los parámetros en el paper es más rica: cada "neurona" m tiene tres
        componentes con roles distintos en el campo prototípico:

            b(x, aᵐ) = σ(a₁ᵐ · x + a₂ᵐ) · a₀ᵐ

            a₁ᵐ ∈ ℝ²   — pesos de ENTRADA: definen la dirección en ℝ² que
                          cada neurona "mira".  Son las "antenas" del campo.
            a₂ᵐ ∈ ℝ    — sesgo: desplaza el umbral de activación σ.  En la
                          implementación se divide en bias fijo (W1.bias[m])
                          y coeficiente temporal (W1.weight[m, 2]) que escala
                          el efecto del tiempo t ∈ [0,1].
            a₀ᵐ ∈ ℝ²   — pesos de SALIDA: escalan la contribución de la
                          neurona al campo vectorial en ℝ².  Su norma ||a₀ᵐ||
                          mide la "importancia" de la neurona m.

        Pese a compartir el mismo prior ν^∞ ∝ exp(-0.05|a|⁴ - 0.5|a|²),
        estos tipos pueden converger a distribuciones distintas porque el
        gradiente de la pérdida les llega de forma diferente.

    DISEÑO DE LA FIGURA (E_parameter_analysis.png) — Layout 3×3:

        FILA 0 — Distribuciones marginales por tipo de parámetro (ε=0.01):
            (0,0): histograma de a₁ ∈ ℝ²  (pesos de entrada, 2×64=128 valores)
            (0,1): histograma de coef. temporal y sesgo (a₂, 2×64 valores)
            (0,2): histograma de a₀ ∈ ℝ²  (pesos de salida, 128 valores)
            Cada panel compara con el prior ν^∞ (curva blanca discontinua).
            Pregunta: ¿los distintos tipos de parámetro convergen de forma
            diferente hacia el prior?

        FILA 1 — Distribución 2D de pesos de entrada a₁ = (a₁[0], a₁[1]):
            Un punto por neurona m, en el plano ℝ² de los pesos de entrada.
            La posición en el plano es a₁ᵐ y el COLOR indica la importancia
            ||a₀ᵐ||₂ de esa neurona.
            Fondo: datos de make_moons a muy baja opacidad como referencia.
            Para ε ∈ {0, 0.01, 0.5} — ¿cómo restringe ε la dispersión y
            afecta a qué neuronas son importantes?

        FILA 2 — Importancia de neuronas y activación temporal:
            (2,0): Scatter de contribución al campo en t=0 vs t=T=1, por neurona.
                   c_m(t) = (1/N) Σᵢ |σ(a₁ᵐ·Xᵢ + t·tcoef_m + bias_m)| · ||a₀ᵐ||
                   La diagonal punteada es identidad. Puntos SOBRE la diagonal:
                   neuronas más activas al final que al principio del flujo.
                   Puntos BAJO la diagonal: neuronas que "apagan" durante la ODE.
            (2,1): Importancias ||a₀ᵐ||₂ ordenadas de mayor a menor para
                   ε ∈ {0, 0.01, 0.5}. Si la curva cae rápidamente (codo
                   pronunciado), pocas neuronas hacen todo el trabajo.
                   Si es plana, el campo es "democrático" entre neuronas.
            (2,2): Correlación ||a₁ᵐ||₂ vs ||a₀ᵐ||₂ por neurona y por ε.
                   Pregunta: ¿las neuronas con proyección de entrada fuerte
                   (a₁ grande) tienden a tener salida fuerte (a₀ grande)?
                   Una correlación positiva indicaría que la red asigna
                   conjuntamente la importancia en entrada y salida.

    NOTA SOBRE EXTRACCIÓN DE PARÁMETROS:
        La implementación usa nn.Linear(d1+1, M) para W1, por lo que:
            W1.weight ∈ ℝ^{M×(d1+1)} = ℝ^{64×3}
            W1.weight[m, :2] = a₁ᵐ  (componente espacial)
            W1.weight[m,  2] = coef. temporal de la neurona m
            W1.bias[m]       = sesgo fijo de la neurona m
            W0.weight ∈ ℝ^{d1×M} = ℝ^{2×64}
            W0.weight[:, m]  = a₀ᵐ  (peso de salida de la neurona m)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO E  —  Distribución de parámetros ν*")
    print("=" * 62)

    X, y, X_np, y_np = get_moons()

    # ── Funciones auxiliares ──────────────────────────────────────────────────
    def get_params(model):
        """Extrae los componentes (a₁, tcoef, bias, a₀) para cada neurona."""
        with torch.no_grad():
            W1w = model.velocity.W1.weight.cpu().numpy()   # (M, d1+1)
            W1b = model.velocity.W1.bias.cpu().numpy()     # (M,)
            W0w = model.velocity.W0.weight.cpu().numpy()   # (d1, M)
        a1    = W1w[:, :2]   # (M, 2) — proyección espacial
        tcoef = W1w[:,  2]   # (M,)   — coeficiente de t
        bias  = W1b          # (M,)   — sesgo en t=0
        a0    = W0w.T        # (M, 2) — peso de salida
        return a1, tcoef, bias, a0

    def importance(a0):
        """||a₀ᵐ||₂ para cada neurona m → (M,)."""
        return np.linalg.norm(a0, axis=1)

    def contribution_at_t(a1, tcoef, bias, a0, X_local, t):
        """
        Contribución media de cada neurona al campo F en el "tiempo" t.

        c_m(t) = (1/N) Σᵢ |σ(a₁ᵐ·Xᵢ + tcoef_m·t + bias_m)| · ||a₀ᵐ||₂

        Nota: usa los datos X₀ (no la trayectoria X_t) para aislar el efecto
        del tiempo codificado en los pesos, independientemente del flujo.
        """
        pre = X_local @ a1.T + tcoef * t + bias   # (N, M)
        act = np.abs(np.tanh(pre)).mean(axis=0)    # (M,)
        return act * importance(a0)                # (M,)

    # Parámetros del modelo de referencia (ε=0.01)
    model_ref = results_eps[0.01]['model']
    a1r, tcoefr, biasr, a0r = get_params(model_ref)

    print(f"  M = {a1r.shape[0]} neuronas | "
          f"||a₁|| media = {np.linalg.norm(a1r,axis=1).mean():.4f} | "
          f"||a₀|| media = {importance(a0r).mean():.4f}")

    # ── Figura E ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 18))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    def _prior_curve(ax, vals, color='#3498db', label_hist='Parámetros aprendidos'):
        """Dibuja histograma + prior ν^∞ con escala adaptativa."""
        p_lo = min(np.percentile(vals, 0.5), -0.5)
        p_hi = max(np.percentile(vals, 99.5),  0.5)
        a_range = np.linspace(p_lo, p_hi, 400)
        log_pr  = -0.05 * a_range**4 - 0.5 * a_range**2
        log_pr -= log_pr.max()
        prior   = np.exp(log_pr) / np.trapz(np.exp(log_pr), a_range)
        vals_in = vals[(vals >= p_lo) & (vals <= p_hi)]
        ax.hist(vals_in, bins=40, density=True, alpha=0.75,
                color=color, edgecolor='none', label=label_hist)
        ax.plot(a_range, prior, 'w--', lw=2, label=r'$\nu^\infty$')
        ax.text(0.97, 0.97, f'std={vals.std():.3f}', transform=ax.transAxes,
                ha='right', va='top', color=TXT, fontsize=8,
                bbox=dict(facecolor=PANEL_BG, alpha=0.7, pad=2))
        ax.set_xlim(p_lo, p_hi)
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── Fila 0: marginales por tipo de parámetro ──────────────────────────────
    COLORS_TYPE = ['#e74c3c', '#f39c12', '#2ecc71']

    # (0,0) — a₁: pesos de entrada espaciales
    ax00 = fig.add_subplot(gs[0, 0])
    _prior_curve(ax00, a1r.ravel(), color=COLORS_TYPE[0])
    style_ax(ax00,
             r'Pesos de entrada $a_1^m \in \mathbb{R}^2$  (ε=0.01)'
             '\n' r'Rol: proyección $a_1^m \cdot x$ → "dirección que mira"',
             r'$a_1^m[k]$', 'Densidad')

    # (0,1) — a₂: coef. temporal + sesgo
    ax01 = fig.add_subplot(gs[0, 1])
    a2_all = np.concatenate([tcoefr, biasr])
    _prior_curve(ax01, a2_all, color=COLORS_TYPE[1])
    style_ax(ax01,
             r'Coef. temporal $W_1[:,2]$ y sesgo $b_1$  (ε=0.01)'
             '\n' r'Rol: umbral $a_1^m \cdot x + \text{tcoef}_m \cdot t + \text{bias}_m$',
             r'valor', 'Densidad')

    # (0,2) — a₀: pesos de salida
    ax02 = fig.add_subplot(gs[0, 2])
    _prior_curve(ax02, a0r.ravel(), color=COLORS_TYPE[2])
    style_ax(ax02,
             r'Pesos de salida $a_0^m \in \mathbb{R}^2$  (ε=0.01)'
             '\n' r'Rol: amplitud $\sigma(\cdot) \cdot a_0^m$ → velocidad en $\mathbb{R}^2$',
             r'$a_0^m[k]$', 'Densidad')

    # ── Fila 1: distribución 2D de a₁ coloreada por ||a₀|| ───────────────────
    EPS_SCATTER = [0.0, 0.01, 0.5]
    for col, eps in enumerate(EPS_SCATTER):
        ax = fig.add_subplot(gs[1, col])
        a1e, _, _, a0e = get_params(results_eps[eps]['model'])
        imp_e = importance(a0e)

        # Fondo: datos make_moons (referencia de escala)
        ax.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1],
                   c='#ff6b6b', s=5, alpha=0.12, zorder=1)
        ax.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1],
                   c='#74b9ff', s=5, alpha=0.12, zorder=1)

        sc = ax.scatter(a1e[:, 0], a1e[:, 1],
                        c=imp_e, cmap='plasma', s=70, alpha=0.88,
                        zorder=3, edgecolors='white', linewidths=0.4,
                        vmin=0, vmax=imp_e.max())
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                     label=r'$\|a_0^m\|_2$')
        style_ax(ax,
                 f'$a_1^m \\in \\mathbb{{R}}^2$ — ε={eps}'
                 '\n' r'color = importancia $\|a_0^m\|_2$',
                 r'$a_1^m[0]$', r'$a_1^m[1]$')
        ax.set_aspect('equal')

    # ── Fila 2: importancia y activación temporal ─────────────────────────────
    EPS_IMP = [0.0, 0.01, 0.5]
    COLORS_IMP = ['#e74c3c', '#2ecc71', '#9b59b6']

    # (2,0): scatter contribución t=0 vs t=T (modelo ε=0.01)
    ax20 = fig.add_subplot(gs[2, 0])
    c0 = contribution_at_t(a1r, tcoefr, biasr, a0r, X_np, t=0.0)
    cT = contribution_at_t(a1r, tcoefr, biasr, a0r, X_np, t=1.0)
    imp_ref = importance(a0r)

    sc20 = ax20.scatter(c0, cT, c=imp_ref, cmap='plasma', s=65,
                        alpha=0.88, edgecolors='white', linewidths=0.4,
                        vmin=0, vmax=imp_ref.max())
    plt.colorbar(sc20, ax=ax20, fraction=0.046, pad=0.04,
                 label=r'$\|a_0^m\|_2$')
    lim = max(c0.max(), cT.max()) * 1.12
    ax20.plot([0, lim], [0, lim], color='#2a2a4a', lw=1.5, ls='--', alpha=0.8)
    # Anotar neuronas extremas
    for m in range(len(c0)):
        if cT[m] > 1.5 * c0[m] + 0.01 or c0[m] > 1.5 * cT[m] + 0.01:
            ax20.annotate(str(m), (c0[m], cT[m]),
                          fontsize=6, color=TXT, alpha=0.7,
                          xytext=(3, 3), textcoords='offset points')
    ax20.set_xlim(0, lim)
    ax20.set_ylim(0, lim)
    style_ax(ax20,
             r'Contribución neuronal $c_m(t)$: $t=0$ vs $t=T$'
             '\n' r'ε=0.01  |  ▲ diagonal: más activas al final',
             r'$c_m(t=0)$', r'$c_m(t=T)$')

    # (2,1): importancias ordenadas para los tres ε
    ax21 = fig.add_subplot(gs[2, 1])
    for eps, col in zip(EPS_IMP, COLORS_IMP):
        _, _, _, a0e = get_params(results_eps[eps]['model'])
        imp_sorted = np.sort(importance(a0e))[::-1]
        ax21.plot(np.arange(1, len(imp_sorted)+1), imp_sorted,
                  color=col, lw=1.8, marker='o', ms=3.5,
                  label=f'ε={eps}')
    style_ax(ax21,
             r'Importancia neuronal $\|a_0^m\|_2$ ordenada'
             '\n' r'Codo pronunciado → pocas neuronas dominan',
             'Rank (neurona)', r'$\|a_0^m\|_2$')
    ax21.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # (2,2): correlación ||a₁ᵐ|| vs ||a₀ᵐ|| por neurona
    ax22 = fig.add_subplot(gs[2, 2])
    for eps, col in zip(EPS_IMP, COLORS_IMP):
        a1e, _, _, a0e = get_params(results_eps[eps]['model'])
        norm_a1 = np.linalg.norm(a1e, axis=1)
        norm_a0 = importance(a0e)
        corr = np.corrcoef(norm_a1, norm_a0)[0, 1]
        ax22.scatter(norm_a1, norm_a0, color=col, s=35, alpha=0.70,
                     edgecolors='none', label=f'ε={eps}  r={corr:.2f}')
    style_ax(ax22,
             r'Correlación $\|a_1^m\|$ vs $\|a_0^m\|$'
             '\n' r'¿entrada fuerte implica salida fuerte?',
             r'$\|a_1^m\|_2$', r'$\|a_0^m\|_2$')
    ax22.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    fig.suptitle(
        r'Experimento E — Análisis de la distribución de parámetros $\nu^*$'
        '\n'
        r'Fila 0: marginales por tipo  |  '
        r'Fila 1: distribución 2D de $a_1$ coloreada por importancia  |  '
        r'Fila 2: importancia y activación temporal',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, 'E_parameter_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")


# =============================================================================
# EXPERIMENTO E2
#   Robustez de ν* entre entrenamientos: ¿la distribución de parámetros
#   es la misma con distintas semillas de inicialización y/o de datos?
# =============================================================================
def experiment_E2(n_seeds: int = 10, n_epochs: int = 500):
    """
    Robustez de ν* a distintas condiciones de entrenamiento (make_moons).

    MOTIVACIÓN:
        El Experimento E muestra la distribución ν* para UN único entrenamiento.
        Pero ¿es esa distribución robusta?  Si el minimizador es único (Meta-
        Teorema 1), distintas inicializaciones y distintos datasets deberían
        producir distribuciones ν* similares — la misma "nube" de puntos en el
        espacio de parámetros.

    DISEÑO:
        E2-1 — Robustez a θ₀ (datos fijos, init variable):
            data_seed=42 fijo | init_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con la inicialización de pesos?

        E2-2 — Robustez a γ₀ (init fija, datos variables):
            init_seed=4 fijo | data_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con el dataset de entrenamiento?

    FIGURA E2 (2×3):
        Fila 0 (E2-1):  scatter 2D a₁ por seed  |  hist. a₁[k] por seed + prior ν∞  |  importancias ordenadas
        Fila 1 (E2-2):  scatter 2D a₁ por seed  |  hist. a₁[k] por seed + prior ν∞  |  importancias ordenadas
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO E2  —  Robustez de ν* entre entrenamientos")
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

    # ── E2-1: datos fijos, init variable ──────────────────────────────────────
    print("  E2-1: data_seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, _, _ = get_moons(seed=DATA_SEED_FIXED)
    results_E2_1 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False)
        a1, a0 = get_params(model)
        results_E2_1.append({'seed': s, 'a1': a1, 'a0': a0, 'hist': hist})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={hist['accuracy'][-1]:.3f}")

    # ── E2-2: init fija, datos variables ──────────────────────────────────────
    print("  E2-2: init_seed=4 fijo, 10 datasets distintos...")
    results_E2_2 = []
    for s in SEEDS:
        X_s, y_s, _, _ = get_moons(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_s, y_s, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False)
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

    _panel(axes[0], results_E2_1, 'E2-1 — init variable, datos fijos (seed=42)')
    _panel(axes[1], results_E2_2, 'E2-2 — datos variables, init fija (seed=4)')

    fig.suptitle(
        r'Experimento E2 — Robustez de $\nu^*$: importancia neuronal $\|a_0^m\|_2$'
        '\n'
        r'Banda blanca = media ± std sobre 10 semillas  |  $\varepsilon=0.01$',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, 'E2_parameter_robustness.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")
