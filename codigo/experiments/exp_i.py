"""
=============================================================================
exp_i.py — Experimento I: MMD y Sinkhorn como regularizadores de entrenamiento
=============================================================================

MOTIVACIÓN:
    El experimento H mostró que ν* ≠ ν^∞ (MMD ≈ 0.22): los datos desplazan
    significativamente la distribución óptima respecto al prior.
    pSGLD implementa esto implícitamente via ruido Langevin.

    Este experimento pregunta: ¿se puede reemplazar el ruido Langevin por una
    penalización explícita de distancia (MMD o Sinkhorn) entre las partículas
    actuales y el prior?

    Loss original (pSGLD implícito):
        J = BCE + ε·E_ν[ℓ(a)]   +   ruido Langevin

    Loss nuevo (explícito, sin ruido):
        J_MMD      = BCE + ε·MMD²(ν_N, ν^∞_samples)
        J_Sinkhorn = BCE + ε·S_blur(ν_N, ν^∞_samples)

PROTOCOLO:
    1. Precomputar N_prior=2000 muestras de ν^∞ por ULA (una sola vez).
    2. Entrenar 3 modelos (misma semilla, mismo dataset):
         (a) pSGLD   — baseline, sin prior_samples
         (b) MMD-reg — Adam puro, regularizador MMD²
         (c) Sinkhorn-reg — Adam puro, regularizador Sinkhorn
    3. Para cada modelo, recoger ν̂* (pSGLD post-convergencia 1000 pasos).
    4. Comparar con la ν̂* del baseline pSGLD como referencia.

FIGURAS (I_regularizer_comparison.png):
    I1 — BCE convergence (3 curvas suavizadas)
    I2 — Valor del regularizador durante training (MMD² / W para nuevos;
         L4+L2 para pSGLD en eje derecho)
    I3 — W₁ por dimensión: ν̂* de cada método vs ν̂*_psgld (referencia)
    I4 — Histogramas 1D de ν̂* para la dimensión más informativa (a₀₀)
    I5 — Scatter 2D (a₀₀, a₀₁) de las partículas finales de los 3 métodos
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, GRID_C, style_ax
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import (
    extract_particles,
    mmd_rbf,
    wasserstein_1d,
    sample_prior_langevin,
    collect_psgld_snapshots,
)

# ── Configuración visual ─────────────────────────────────────────────────────
C_PSGLD    = '#3498db'   # azul  — pSGLD (baseline)
C_MMD      = '#e67e22'   # naranja — MMD-reg
C_SINKHORN = '#2ecc71'   # verde  — Sinkhorn-reg
C_PRIOR    = '#95a5a6'   # gris   — prior ν^∞

DIM_LABELS = [r'$a_0^{(0)}$', r'$a_0^{(1)}$',
              r'$a_1^{(0)}$', r'$a_1^{(1)}$', r'$a_2$']

N_SNAPSHOT_STEPS = 1_000   # pasos post-training para estimar ν̂*
THIN             = 10


def _smooth(x, w=20):
    return uniform_filter1d(x, size=w, mode='nearest')


def experiment_I(n_epochs=700, epsilon=0.01, n_prior=2_000,
                 sinkhorn_blur=0.05):
    """
    Compara pSGLD, MMD-reg y Sinkhorn-reg como métodos de regularización.

    Args:
        n_epochs      : épocas de entrenamiento para los 3 métodos
        epsilon       : temperatura / coeficiente de regularización
        n_prior       : muestras de ν^∞ precomputadas para MMD/Sinkhorn
        sinkhorn_blur : regularización del OT en Sinkhorn
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO I  —  MMD y Sinkhorn como regularizadores")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs  |  "
          f"{n_prior} prior samples  |  blur={sinkhorn_blur}")

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    X_np, y_np = make_moons(n_samples=500, noise=0.12, random_state=42)
    scaler = StandardScaler().fit(X_np)
    X_np   = scaler.transform(X_np).astype(np.float32)
    y_np   = y_np.astype(np.float32)
    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    # ── 2. Prior samples (una sola vez) ────────────────────────────────────
    print(f"\n  Generando {n_prior} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(n_prior, epsilon=epsilon, dim=5)
    print(f"  → {prior_samples.shape}")

    # ── 3. Entrenamiento de los 3 métodos ───────────────────────────────────
    histories = {}
    models    = {}

    configs = [
        ('pSGLD',    dict(use_sgld=True)),
        ('MMD',      dict(use_mmd=True,      prior_samples=prior_samples)),
        ('Sinkhorn', dict(use_sinkhorn=True,  prior_samples=prior_samples,
                          sinkhorn_blur=sinkhorn_blur)),
    ]

    for name, kwargs in configs:
        print(f"\n  [{name}] Entrenando {n_epochs} epochs...")
        torch.manual_seed(42)
        np.random.seed(42)
        m = MeanFieldResNet().to(DEVICE)
        h = train(m, X, y, epsilon=epsilon, n_epochs=n_epochs,
                  verbose=True, **kwargs)
        histories[name] = h
        models[name]    = m

    # ── 4. ν̂* por método (snapshots pSGLD post-convergencia) ────────────────
    timeavg = {}
    for name, m in models.items():
        print(f"\n  [{name}] Recogiendo ν̂* ({N_SNAPSHOT_STEPS} pasos)...")
        snaps = collect_psgld_snapshots(
            m, X, y, epsilon,
            n_steps=N_SNAPSHOT_STEPS, thin=THIN
        )
        timeavg[name] = snaps.reshape(-1, 5)
        print(f"    → {timeavg[name].shape[0]} muestras")

    # ── 5. Métricas: W₁ de cada ν̂* vs ν̂*_pSGLD ──────────────────────────────
    ref = timeavg['pSGLD']
    print("\n  W₁ de ν̂*_método vs ν̂*_pSGLD (referencia):")
    print(f"  {'dim':<8} {'MMD':>8} {'Sinkhorn':>10}")
    w1_mmd  = [wasserstein_1d(timeavg['MMD'][:, d],      ref[:, d]) for d in range(5)]
    w1_sink = [wasserstein_1d(timeavg['Sinkhorn'][:, d], ref[:, d]) for d in range(5)]
    for d in range(5):
        print(f"  {DIM_LABELS[d]:<8} {w1_mmd[d]:>8.4f} {w1_sink[d]:>10.4f}")

    # MMD global entre métodos
    N_sub = 500
    mmd_mmd_vs_psgld  = mmd_rbf(
        timeavg['MMD'][:N_sub],      ref[:N_sub])
    mmd_sink_vs_psgld = mmd_rbf(
        timeavg['Sinkhorn'][:N_sub], ref[:N_sub])
    print(f"\n  MMD²(ν̂*_MMD, ν̂*_pSGLD)      = {mmd_mmd_vs_psgld:.4e}")
    print(f"  MMD²(ν̂*_Sinkhorn, ν̂*_pSGLD) = {mmd_sink_vs_psgld:.4e}")

    # ── 6. Figuras ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(DARK_BG)
    gs  = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.38)

    style_map = {
        'pSGLD':    (C_PSGLD,    '-',  'pSGLD (baseline)'),
        'MMD':      (C_MMD,      '-',  'MMD-reg'),
        'Sinkhorn': (C_SINKHORN, '-',  'Sinkhorn-reg'),
    }

    # ── I1: BCE convergencia ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    for name, (col, ls, lbl) in style_map.items():
        bce = np.array(histories[name]['loss_term'])
        ax1.plot(_smooth(bce), color=col, lw=1.8, ls=ls, alpha=0.9, label=lbl)
        ax1.plot(bce, color=col, lw=0.5, alpha=0.25)
    style_ax(ax1, 'I1 — BCE convergencia\n(línea fina = raw, gruesa = suavizado)',
             'época', 'BCE')
    ax1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    # ── I2: Regularizador ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    # pSGLD: L4+L2 en eje izquierdo (escala distinta)
    reg_psgld = np.array(histories['pSGLD']['loss_reg'])
    ax2.plot(_smooth(reg_psgld), color=C_PSGLD, lw=1.8, ls='--',
             alpha=0.9, label='pSGLD  L4+L2 (eje izq)')
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor(PANEL_BG)
    for name, col in [('MMD', C_MMD), ('Sinkhorn', C_SINKHORN)]:
        reg = np.array(histories[name]['loss_reg'])
        ax2_r.plot(_smooth(reg), color=col, lw=1.8, alpha=0.9, label=name)
        ax2_r.plot(reg, color=col, lw=0.5, alpha=0.2)
    ax2_r.tick_params(colors=TXT, labelsize=7)
    ax2_r.set_ylabel('MMD² / Sinkhorn', color=TXT, fontsize=8)
    style_ax(ax2, 'I2 — Regularizador durante training\n'
             r'Discontinuo = L4+L2 (pSGLD)  |  Sólido = MMD²/W',
             'época', 'L4+L2 (pSGLD)')
    lines1, lbls1 = ax2.get_legend_handles_labels()
    lines2, lbls2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbls1 + lbls2,
               facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── I3: Accuracy convergencia ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL_BG)
    for name, (col, ls, lbl) in style_map.items():
        acc = np.array(histories[name]['accuracy'])
        ax3.plot(_smooth(acc), color=col, lw=1.8, ls=ls, alpha=0.9, label=lbl)
    ax3.set_ylim(0.5, 1.02)
    style_ax(ax3, 'I3 — Accuracy convergencia', 'época', 'acc')
    ax3.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    # ── I4: W₁ por dimensión vs ν̂*_pSGLD ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(PANEL_BG)
    x_pos = np.arange(5)
    w = 0.35
    ax4.bar(x_pos - w/2, w1_mmd,  w, color=C_MMD,      alpha=0.85,
            label='MMD-reg vs pSGLD')
    ax4.bar(x_pos + w/2, w1_sink, w, color=C_SINKHORN,  alpha=0.85,
            label='Sinkhorn-reg vs pSGLD')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(DIM_LABELS, fontsize=9)
    style_ax(ax4,
             r'I4 — $W_1$ por dimensión: $\hat\nu^*_\mathrm{método}$ vs $\hat\nu^*_\mathrm{pSGLD}$'
             '\n'
             r'Pequeño → método converge a la misma $\nu^*$ que pSGLD',
             '', r'$W_1$')
    ax4.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # MMD global como texto
    ax4.text(0.98, 0.95,
             f'MMD²(MMD vs pSGLD) = {mmd_mmd_vs_psgld:.3e}\n'
             f'MMD²(W vs pSGLD) = {mmd_sink_vs_psgld:.3e}',
             transform=ax4.transAxes, ha='right', va='top',
             color=TXT, fontsize=8,
             bbox=dict(facecolor=DARK_BG, alpha=0.7, edgecolor=GRID_C))

    # ── I5: Histograma 1D de ν̂* en dim a₀₀ (más informativa del exp H) ─────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL_BG)
    from scipy.stats import gaussian_kde
    theta_g = np.linspace(-3.5, 3.5, 400)
    for name, col in [('pSGLD', C_PSGLD), ('MMD', C_MMD), ('Sinkhorn', C_SINKHORN)]:
        data = timeavg[name][:, 0].numpy()
        try:
            kde = gaussian_kde(data, bw_method='scott')
            ax5.fill_between(theta_g, kde(theta_g), alpha=0.20, color=col)
            ax5.plot(theta_g, kde(theta_g), color=col, lw=1.8, label=name)
        except Exception:
            pass
    style_ax(ax5, r'I5 — $\hat\nu^*$ en $a_0^{(0)}$ (dim más desplazada)'
             '\nKDE de muestras tiempo-promedio',
             r'$a_0^{(0)}$', 'densidad')
    ax5.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    # ── I6: Scatter 2D partículas finales ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(PANEL_BG)
    markers = {'pSGLD': 'o', 'MMD': 's', 'Sinkhorn': '^'}
    for name, (col, _, lbl) in style_map.items():
        pts = extract_particles(models[name])[:, :2].numpy()
        ax6.scatter(pts[:, 0], pts[:, 1],
                    c=col, marker=markers[name], s=50,
                    alpha=0.85, edgecolors='white', linewidths=0.4,
                    label=lbl, zorder=5)
    # Prior samples como fondo
    pr_2d = prior_samples[:300, :2].numpy()
    ax6.scatter(pr_2d[:, 0], pr_2d[:, 1],
                c=C_PRIOR, marker='.', s=8, alpha=0.3,
                label=r'$\nu^\infty$ (prior, sub-muestreado)', zorder=1)
    style_ax(ax6,
             r'I6 — Partículas finales proyectadas en $(a_0^{(0)}, a_0^{(1)})$'
             '\nGris = muestras del prior  |  Coloreados = partículas entrenadas',
             r'$a_0^{(0)}$', r'$a_0^{(1)}$')
    ax6.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8.5,
               ncol=4, loc='upper right')

    fig.suptitle(
        rf'Experimento I — MMD² y Sinkhorn como regularizadores  ($\varepsilon={epsilon}$)'
        '\n'
        rf'MMD²(MMD vs pSGLD) = {mmd_mmd_vs_psgld:.3e}  '
        rf'|  MMD²(Sinkhorn vs pSGLD) = {mmd_sink_vs_psgld:.3e}  '
        rf'|  {n_epochs} epochs, {n_prior} prior samples',
        color=TXT, fontsize=11, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUTPUT_DIR, 'I_regularizer_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    return {
        'histories':        histories,
        'models':           models,
        'timeavg':          timeavg,
        'w1_mmd':           w1_mmd,
        'w1_sinkhorn':      w1_sink,
        'mmd_mmd_psgld':    mmd_mmd_vs_psgld,
        'mmd_sink_psgld':   mmd_sink_vs_psgld,
    }
