"""
=============================================================================
exp_h.py — Experimento H: Diagnóstico de Convergencia pSGLD → ν*
=============================================================================

MOTIVACIÓN:
    pSGLD tiene distribución estacionaria ν* ∝ exp(−J(θ)/ε).  Pero no podemos
    verificarlo directamente con KL(ν_N || ν^∞), ya que ν_N es discreta y la KL
    sería infinita.  En su lugar usamos dos métricas bien definidas para
    distribuciones de soporte mixto:
        • MMD²  (Maximum Mean Discrepancy con kernel Gaussiano)
        • W₁    (Wasserstein-1 exacta, por dimensión)

PREGUNTA:
    ¿Muestrea realmente pSGLD de ν*?
    ¿Cuánto difiere ν* del prior ν^∞ (efecto de los datos)?

PROTOCOLO:
    1. Entrenar un modelo con pSGLD (ε=0.01, 800 epochs).
    2. Post-entrenamiento: continuar pSGLD 2000 pasos con lr fijo → snapshots
       de partículas cada `thin` pasos.  El pool de todos los snapshots aproxima
       la distribución tiempo-promedio ν̂* ≈ ν* (teorema ergódico).
    3. Generar muestras de ν^∞ ∝ exp(−ℓ(a)/ε) por ULA (Langevin MCMC sin datos).
    4. Calcular:
         MMD²(ν_N_final, ν̂*)    — partículas finales ≈ estacionario?
         MMD²(ν̂*,        ν^∞)   — ν* ≠ prior (datos hacen algo)?
         MMD²(ν_N_final, ν^∞)   — partículas vs prior (directamente)
         W₁ por dimensión        — resolución fina por componente de a

FIGURAS:
    H1 — Histogramas marginales 1D (5 dimensiones: a₀₀, a₀₁, a₁₀, a₁₁, a₂)
         Azul = ν̂* (tiempo-promedio) | Rojo tick = ν_N final | Gris = ν^∞ prior
    H2 — MMD² global entre las tres distribuciones (barras)
    H3 — W₁ por dimensión (barras agrupadas, tres comparaciones)
    H4 — Scatter 2D en la proyección (a₀₀, a₀₁) con contornos KDE
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import (
    extract_particles,
    mmd_rbf,
    wasserstein_1d,
    sample_prior_langevin,
    collect_psgld_snapshots,
)

# ── Etiquetas para las 5 dimensiones del espacio A ──────────────────────────
DIM_LABELS = [
    r'$a_0^{(0)}$',   # W0.weight[0, m]  — componente 0 de a₀^m
    r'$a_0^{(1)}$',   # W0.weight[1, m]  — componente 1 de a₀^m
    r'$a_1^{(0)}$',   # W1.weight[m, 0]  — componente 0 de a₁^m
    r'$a_1^{(1)}$',   # W1.weight[m, 1]  — componente 1 de a₁^m
    r'$a_2$',         # W1.bias[m]        — umbral (bias) de neurona m
]

# ── Colores ──────────────────────────────────────────────────────────────────
C_TIMEAVG = '#3498db'   # azul  — ν̂* tiempo-promedio (referencia empírica de ν*)
C_FINAL   = '#e74c3c'   # rojo  — ν_N final (64 partículas al acabar training)
C_PRIOR   = '#95a5a6'   # gris  — ν^∞ prior analítico


def _prior_density_1d(theta, epsilon, c1=0.05, c2=0.5):
    """
    Densidad marginal no normalizada del prior ν^∞ ∝ exp(−ℓ(a)/ε),
    evaluada en una dimensión con las demás integradas.

    Como ℓ(a) = Σⱼ (c₁aⱼ⁴ + c₂aⱼ²), las dimensiones son independientes bajo
    el prior, y la marginal 1D es también una medida de Gibbs:
        p(θ) ∝ exp(−(c₁θ⁴ + c₂θ²)/ε)
    """
    return np.exp(-(c1 * theta ** 4 + c2 * theta ** 2) / epsilon)


def experiment_H(n_epochs=800, n_snapshot_steps=2_000, thin=10,
                 n_prior_samples=8_000, epsilon=0.01):
    """
    Diagnóstico de calidad del muestreo pSGLD mediante MMD y W₁.

    Args:
        n_epochs          : épocas de entrenamiento pSGLD
        n_snapshot_steps  : pasos adicionales para construir ν̂*
        thin              : sub-muestreo de snapshots (1 de cada `thin`)
        n_prior_samples   : muestras de ν^∞ a generar por Langevin
        epsilon           : temperatura (ε)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO H  —  Diagnóstico: pSGLD → ν*")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs train  "
          f"|  {n_snapshot_steps} pasos snapshots (thin={thin})")

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    X_np, y_np = make_moons(n_samples=500, noise=0.12, random_state=42)
    scaler = StandardScaler().fit(X_np)
    X_np   = scaler.transform(X_np).astype(np.float32)
    y_np   = y_np.astype(np.float32)
    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    # ── 2. Entrenamiento pSGLD ──────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    model = MeanFieldResNet().to(DEVICE)
    print("\n  [1/4] Entrenando con pSGLD...")
    train(model, X, y, epsilon=epsilon, n_epochs=n_epochs,
          verbose=True, use_sgld=True)

    # ── 3. Partículas finales ν_N ──────────────────────────────────────────
    particles_final = extract_particles(model)   # (64, 5)
    print(f"\n  [2/4] Recogiendo snapshots pSGLD "
          f"({n_snapshot_steps} pasos, thin={thin})...")
    snapshots = collect_psgld_snapshots(
        model, X, y, epsilon,
        n_steps=n_snapshot_steps, thin=thin
    )   # (n_snapshot_steps//thin, 64, 5)
    particles_timeavg = snapshots.reshape(-1, 5)   # (n_snaps*64, 5)
    print(f"     → {particles_timeavg.shape[0]} muestras de ν̂*")

    # ── 4. Prior ν^∞ por Langevin MCMC ────────────────────────────────────
    print(f"\n  [3/4] Generando {n_prior_samples} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(
        n_prior_samples, epsilon=epsilon, dim=5
    )   # (n_prior_samples, 5)

    # ── 5. Métricas ────────────────────────────────────────────────────────
    print("\n  [4/4] Calculando MMD² y W₁...")
    # Sub-muestra para que MMD sea O(n²) manejable
    N_mmd = min(600, particles_timeavg.shape[0])
    idx_t = torch.randperm(particles_timeavg.shape[0])[:N_mmd]
    idx_p = torch.randperm(prior_samples.shape[0])[:N_mmd]
    pt    = particles_timeavg[idx_t]
    ps    = prior_samples[idx_p]
    pf    = particles_final       # solo 64, no es necesario sub-muestrear

    mmd_fn_ta = mmd_rbf(pf, pt)
    mmd_ta_pr = mmd_rbf(pt, ps)
    mmd_fn_pr = mmd_rbf(pf, ps)

    print(f"\n  MMD²(ν_N_final, ν̂*)   = {mmd_fn_ta:.4e}"
          f"  ← pequeño si ν_N ≈ ν*")
    print(f"  MMD²(ν̂*,        ν^∞)  = {mmd_ta_pr:.4e}"
          f"  ← grande si datos desplazan ν*")
    print(f"  MMD²(ν_N_final, ν^∞)  = {mmd_fn_pr:.4e}")

    # W₁ por dimensión
    w1_fn_ta = [wasserstein_1d(pf[:, d], particles_timeavg[:, d]) for d in range(5)]
    w1_ta_pr = [wasserstein_1d(particles_timeavg[:, d], prior_samples[:, d])
                for d in range(5)]
    w1_fn_pr = [wasserstein_1d(pf[:, d], prior_samples[:, d]) for d in range(5)]

    print("\n  W₁ por dimensión:")
    print(f"  {'dim':<6} {'ν_N vs ν̂*':>12} {'ν̂* vs ν^∞':>12} {'ν_N vs ν^∞':>12}")
    for d in range(5):
        print(f"  {DIM_LABELS[d]:<6} "
              f"{w1_fn_ta[d]:>12.4f} {w1_ta_pr[d]:>12.4f} {w1_fn_pr[d]:>12.4f}")

    # ── 6. Figuras ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(DARK_BG)
    gs  = fig.add_gridspec(3, 5, hspace=0.50, wspace=0.38)

    theta_grid = np.linspace(-3.5, 3.5, 500)

    # ── H1: Histogramas marginales ─────────────────────────────────────────
    for d in range(5):
        ax = fig.add_subplot(gs[0, d])
        ax.set_facecolor(PANEL_BG)

        # Prior analítico normalizado
        pv   = _prior_density_1d(theta_grid, epsilon)
        pv  /= np.trapz(pv, theta_grid)
        ax.plot(theta_grid, pv, color=C_PRIOR, lw=1.6, ls='--',
                alpha=0.85, label=r'$\nu^\infty$ prior')

        # ν̂* tiempo-promedio (KDE de Gaussian con ancho Scott)
        ta_dim = particles_timeavg[:, d].numpy()
        try:
            kde_ta = gaussian_kde(ta_dim, bw_method='scott')
            ax.fill_between(theta_grid, kde_ta(theta_grid),
                            alpha=0.30, color=C_TIMEAVG)
            ax.plot(theta_grid, kde_ta(theta_grid), color=C_TIMEAVG,
                    lw=1.8, label=r'$\hat\nu^*$ (t-avg)')
        except Exception:
            pass

        # ν_N final — rug plot (64 puntos)
        fn_dim = pf[:, d].numpy()
        yrug   = -0.012 * pv.max()
        ax.plot(fn_dim, np.full(len(fn_dim), yrug), '|',
                color=C_FINAL, alpha=0.9, markersize=9,
                label=r'$\nu_N$ final')

        style_ax(ax, DIM_LABELS[d], '', 'densidad' if d == 0 else '')
        ax.tick_params(labelsize=6.5)
        if d == 2:
            ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=6.5,
                      loc='upper right')

    # ── H2: MMD² global ────────────────────────────────────────────────────
    ax_mmd = fig.add_subplot(gs[1, :2])
    ax_mmd.set_facecolor(PANEL_BG)
    labels_mmd = [r'$\nu_N$ vs $\hat\nu^*$',
                  r'$\hat\nu^*$ vs $\nu^\infty$',
                  r'$\nu_N$ vs $\nu^\infty$']
    vals_mmd   = [mmd_fn_ta, mmd_ta_pr, mmd_fn_pr]
    cols_mmd   = [C_FINAL, C_TIMEAVG, C_PRIOR]
    bars = ax_mmd.bar(labels_mmd, [max(v, 0) for v in vals_mmd],
                      color=cols_mmd, alpha=0.80, width=0.5, zorder=3)
    y_scale = max(abs(v) for v in vals_mmd)
    for bar, v in zip(bars, vals_mmd):
        ax_mmd.text(bar.get_x() + bar.get_width() / 2,
                    max(v, 0) + y_scale * 0.03,
                    f'{v:.2e}', ha='center', va='bottom',
                    color=TXT, fontsize=8)
    style_ax(ax_mmd,
             r'H2 — MMD² global'
             '\n'
             r'Rojo $\ll$ azul/gris → $\nu_N \approx \hat\nu^* \neq \nu^\infty$',
             '', 'MMD²')
    ax_mmd.tick_params(axis='x', labelsize=8.5)
    ax_mmd.set_ylim(bottom=0)

    # ── H3: W₁ por dimensión ───────────────────────────────────────────────
    ax_w1 = fig.add_subplot(gs[1, 2:])
    ax_w1.set_facecolor(PANEL_BG)
    x_pos = np.arange(5)
    w = 0.26
    ax_w1.bar(x_pos - w, w1_fn_ta, w, color=C_FINAL,   alpha=0.80,
              label=r'$\nu_N$ vs $\hat\nu^*$',   zorder=3)
    ax_w1.bar(x_pos,     w1_ta_pr, w, color=C_TIMEAVG, alpha=0.80,
              label=r'$\hat\nu^*$ vs $\nu^\infty$', zorder=3)
    ax_w1.bar(x_pos + w, w1_fn_pr, w, color=C_PRIOR,   alpha=0.80,
              label=r'$\nu_N$ vs $\nu^\infty$',   zorder=3)
    ax_w1.set_xticks(x_pos)
    ax_w1.set_xticklabels(DIM_LABELS, fontsize=8.5)
    style_ax(ax_w1,
             r'H3 — $W_1$ por dimensión de parámetro'
             '\n'
             r'Rojo $\ll$ azul/gris → partículas finales ≈ estacionario ≠ prior',
             '', r'$W_1$')
    ax_w1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5, loc='upper right')

    # ── H4: Scatter 2D (a₀₀, a₀₁) ────────────────────────────────────────
    ax_2d = fig.add_subplot(gs[2, :])
    ax_2d.set_facecolor(PANEL_BG)

    pr_2d = prior_samples[:, :2].numpy()
    ta_2d = particles_timeavg[:, :2].numpy()
    fn_2d = pf[:, :2].numpy()

    # Rango común para el grid de KDE
    all_2d = np.vstack([pr_2d, ta_2d, fn_2d])
    pad    = 0.4
    x_lo, x_hi = all_2d[:, 0].min() - pad, all_2d[:, 0].max() + pad
    y_lo, y_hi = all_2d[:, 1].min() - pad, all_2d[:, 1].max() + pad
    xx, yy = np.mgrid[x_lo:x_hi:100j, y_lo:y_hi:100j]
    pos    = np.vstack([xx.ravel(), yy.ravel()])

    # KDE del prior 2D
    try:
        kde_pr = gaussian_kde(pr_2d.T)
        z_pr   = kde_pr(pos).reshape(xx.shape)
        ax_2d.contourf(xx, yy, z_pr, levels=7, cmap='Greys',  alpha=0.35)
        ax_2d.contour( xx, yy, z_pr, levels=7, colors=[C_PRIOR],
                       linewidths=0.9, alpha=0.75)
    except Exception:
        pass

    # KDE de ν̂* 2D (sub-muestreado para velocidad)
    try:
        idx_ta2d = np.random.choice(len(ta_2d), size=min(3000, len(ta_2d)),
                                    replace=False)
        kde_ta = gaussian_kde(ta_2d[idx_ta2d].T)
        z_ta   = kde_ta(pos).reshape(xx.shape)
        ax_2d.contourf(xx, yy, z_ta, levels=7, cmap='Blues', alpha=0.30)
        ax_2d.contour( xx, yy, z_ta, levels=7, colors=[C_TIMEAVG],
                       linewidths=1.3)
    except Exception:
        pass

    # Partículas finales como puntos
    ax_2d.scatter(fn_2d[:, 0], fn_2d[:, 1],
                  c=C_FINAL, s=45, zorder=6, alpha=0.90,
                  edgecolors='white', linewidths=0.4)

    legend_handles = [
        mlines.Line2D([], [], color=C_PRIOR,   lw=2, ls='-',
                      label=r'$\nu^\infty$ prior (ULA sin datos)'),
        mlines.Line2D([], [], color=C_TIMEAVG, lw=2, ls='-',
                      label=r'$\hat\nu^*$ t-avg (pSGLD continuado)'),
        mlines.Line2D([], [], color=C_FINAL, lw=0, marker='o', markersize=7,
                      markerfacecolor=C_FINAL,
                      label=r'$\nu_N$ final (64 partículas)'),
    ]
    style_ax(ax_2d,
             r'H4 — Proyección 2D: $(a_0^{(0)},\ a_0^{(1)})$'
             '\n'
             r'Si $\nu^*$ ≠ $\nu^\infty$: la concentración azul no coincide con la gris',
             r'$a_0^{(0)}$', r'$a_0^{(1)}$')
    ax_2d.legend(handles=legend_handles, facecolor=PANEL_BG,
                 labelcolor=TXT, fontsize=8.5)

    fig.suptitle(
        rf'Experimento H — Diagnóstico pSGLD: ¿muestrea de $\nu^*$?  ($\varepsilon={epsilon}$)'
        '\n'
        rf'$M=64$ partículas  |  {n_snapshot_steps} pasos post-convergencia (thin={thin})'
        rf'  |  {n_prior_samples} muestras $\nu^\infty$ por ULA',
        color=TXT, fontsize=11, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUTPUT_DIR, 'H_psgld_diagnostic.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    return {
        'particles_final':    particles_final,
        'particles_timeavg':  particles_timeavg,
        'prior_samples':      prior_samples,
        'mmd_final_timeavg':  mmd_fn_ta,
        'mmd_timeavg_prior':  mmd_ta_pr,
        'mmd_final_prior':    mmd_fn_pr,
        'w1_final_timeavg':   w1_fn_ta,
        'w1_timeavg_prior':   w1_ta_pr,
        'w1_final_prior':     w1_fn_pr,
    }
