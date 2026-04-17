"""
=============================================================================
exp_j.py — Experimento J: Regresión en California Housing (datos reales)
=============================================================================

PREGUNTA:
    ¿Funcionan pSGLD, MMD-reg y Sinkhorn-reg igual de bien en una tarea de
    regresión con datos reales que en clasificación sintética?

    El setup es idéntico al exp_i salvo:
        • Dataset: California Housing proyectado a ℝ² via PCA (8 features → 2)
        • Loss: MSE en lugar de BCE
        • Métrica de calidad: R² (train y test) en lugar de accuracy
        • Se separa train/test para medir generalización

PROTOCOLO:
    1. Cargar California Housing, proyectar a ℝ², estandarizar target.
    2. Precomputar muestras del prior ν^∞ (mismas que exp_i — el prior no
       depende de los datos, solo de la dimensión d=5 del espacio A).
    3. Entrenar 3 modelos (pSGLD, MMD-reg, Sinkhorn-reg) con MSE loss.
    4. Evaluar MSE y R² en train y test para cada método.
    5. Recoger ν̂* por método y comparar vs referencia pSGLD (W₁, MMD²).

FIGURAS (J_california_regression.png):
    J1 — MSE train + test convergencia (3 curvas × 2 splits)
    J2 — R² train + test convergencia
    J3 — Regularizador durante training
    J4 — Predicted vs Actual (scatter, 3 métodos)
    J5 — W₁ por dimensión vs ν̂*_pSGLD (como en exp_i pero con regresión)
    J6 — Scatter 2D partículas finales en (a₀₀, a₀₁)
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.ndimage import uniform_filter1d

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, GRID_C, style_ax
from ..data import get_california_regression
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import (
    extract_particles,
    mmd_rbf,
    wasserstein_1d,
    sample_prior_langevin,
    collect_psgld_snapshots,
)

C_PSGLD    = '#3498db'
C_MMD      = '#e67e22'
C_SINKHORN = '#2ecc71'
C_PRIOR    = '#95a5a6'
DIM_LABELS = [r'$a_0^{(0)}$', r'$a_0^{(1)}$',
              r'$a_1^{(0)}$', r'$a_1^{(1)}$', r'$a_2$']

N_SNAPSHOT_STEPS = 1_000
THIN             = 10


def _smooth(x, w=25):
    return uniform_filter1d(np.array(x, dtype=float), size=w, mode='nearest')


def _r2_numpy(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def experiment_J(n_epochs=700, epsilon=0.01, n_prior=2_000,
                 sinkhorn_blur=0.05, n_train=800, n_test=400):
    """
    Compara pSGLD, MMD-reg y Sinkhorn-reg en regresión con California Housing.

    Args:
        n_epochs      : épocas de entrenamiento
        epsilon       : coeficiente de regularización
        n_prior       : muestras del prior ν^∞ para MMD/Sinkhorn
        sinkhorn_blur : regularización del OT
        n_train       : muestras de entrenamiento
        n_test        : muestras de test
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO J  —  Regresión: California Housing")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs  |  "
          f"train={n_train}  test={n_test}")

    # ── 1. Datos ────────────────────────────────────────────────────────────
    ds = get_california_regression(n_train=n_train, n_test=n_test)
    X_tr, y_tr = ds['X_train'], ds['y_train']
    X_te, y_te = ds['X_test'],  ds['y_test']
    print(f"  Varianza explicada PCA(2): {ds['explained_var']:.1%}")

    # ── 2. Prior samples ────────────────────────────────────────────────────
    print(f"\n  Generando {n_prior} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(n_prior, epsilon=epsilon, dim=5)

    # ── 3. Entrenamiento ────────────────────────────────────────────────────
    histories = {}
    models    = {}
    test_metrics = {}

    configs = [
        ('pSGLD',    dict(use_sgld=True)),
        ('MMD',      dict(use_mmd=True,     prior_samples=prior_samples)),
        ('Sinkhorn', dict(use_sinkhorn=True, prior_samples=prior_samples,
                          sinkhorn_blur=sinkhorn_blur)),
    ]

    for name, kwargs in configs:
        print(f"\n  [{name}] Entrenando {n_epochs} epochs (MSE + regresión)...")
        torch.manual_seed(42)
        np.random.seed(42)
        m = MeanFieldResNet(task='regression').to(DEVICE)
        h = train(m, X_tr, y_tr, epsilon=epsilon, n_epochs=n_epochs,
                  verbose=True, **kwargs)
        histories[name] = h
        models[name]    = m

        # Métricas test al final del entrenamiento
        m.eval()
        with torch.no_grad():
            pred_tr = m(X_tr).cpu().numpy()
            pred_te = m(X_te).cpu().numpy()
        y_tr_np = y_tr.cpu().numpy()
        y_te_np = y_te.cpu().numpy()
        mse_tr  = float(np.mean((pred_tr - y_tr_np) ** 2))
        mse_te  = float(np.mean((pred_te - y_te_np) ** 2))
        r2_tr   = _r2_numpy(y_tr_np, pred_tr)
        r2_te   = _r2_numpy(y_te_np, pred_te)
        test_metrics[name] = dict(mse_tr=mse_tr, mse_te=mse_te,
                                   r2_tr=r2_tr,   r2_te=r2_te,
                                   pred_te=pred_te, y_te=y_te_np)
        print(f"    MSE train={mse_tr:.4f}  test={mse_te:.4f}  "
              f"R² train={r2_tr:.3f}  test={r2_te:.3f}")

    # ── 4. ν̂* por método ────────────────────────────────────────────────────
    timeavg = {}
    for name, m in models.items():
        print(f"\n  [{name}] Recogiendo ν̂* ({N_SNAPSHOT_STEPS} pasos)...")
        snaps = collect_psgld_snapshots(
            m, X_tr, y_tr, epsilon,
            n_steps=N_SNAPSHOT_STEPS, thin=THIN
        )
        timeavg[name] = snaps.reshape(-1, 5)

    # ── 5. Métricas de distribución ─────────────────────────────────────────
    ref = timeavg['pSGLD']
    w1_mmd  = [wasserstein_1d(timeavg['MMD'][:, d],      ref[:, d]) for d in range(5)]
    w1_sink = [wasserstein_1d(timeavg['Sinkhorn'][:, d], ref[:, d]) for d in range(5)]
    N_sub   = 500
    mmd_mmd_psgld  = mmd_rbf(timeavg['MMD'][:N_sub],      ref[:N_sub])
    mmd_sink_psgld = mmd_rbf(timeavg['Sinkhorn'][:N_sub], ref[:N_sub])

    print("\n  Comparación ν̂* vs referencia pSGLD:")
    print(f"  MMD²(MMD, pSGLD)      = {mmd_mmd_psgld:.4e}")
    print(f"  MMD²(Sinkhorn, pSGLD) = {mmd_sink_psgld:.4e}")
    print(f"\n  {'método':<10} {'MSE_tr':>8} {'MSE_te':>8} {'R²_tr':>7} {'R²_te':>7}")
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        m = test_metrics[name]
        print(f"  {name:<10} {m['mse_tr']:>8.4f} {m['mse_te']:>8.4f} "
              f"{m['r2_tr']:>7.3f} {m['r2_te']:>7.3f}")

    # ── 6. Figuras ──────────────────────────────────────────────────────────
    style_map = {
        'pSGLD':    (C_PSGLD,    '-',  'pSGLD'),
        'MMD':      (C_MMD,      '-',  'MMD-reg'),
        'Sinkhorn': (C_SINKHORN, '-',  'Sinkhorn-reg'),
    }

    fig = plt.figure(figsize=(22, 17))
    fig.patch.set_facecolor(DARK_BG)
    gs  = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38)

    # ── J1: MSE train + test ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    for name, (col, ls, lbl) in style_map.items():
        mse = np.array(histories[name]['loss_term'])
        ax1.plot(_smooth(mse), color=col, lw=2.0, label=f'{lbl} train')
        ax1.plot(mse, color=col, lw=0.4, alpha=0.2)
    # Añadir MSE test como puntos al final de cada curva
    for name, (col, _, lbl) in style_map.items():
        ep = len(histories[name]['loss_term'])
        ax1.scatter([ep - 1], [test_metrics[name]['mse_te']],
                    c=col, marker='*', s=120, zorder=6,
                    label=f'{lbl} test*')
    style_ax(ax1, 'J1 — MSE train (curva) + test (★ final)',
             'época', 'MSE')
    ax1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=6.5, ncol=2)

    # ── J2: R² train ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    for name, (col, ls, lbl) in style_map.items():
        r2 = np.array(histories[name]['accuracy'])   # R² en modo regresión
        ax2.plot(_smooth(r2), color=col, lw=2.0, label=lbl)
        ax2.plot(r2, color=col, lw=0.4, alpha=0.2)
    # Añadir R² test final
    for name, (col, _, lbl) in style_map.items():
        ep = len(histories[name]['accuracy'])
        ax2.scatter([ep - 1], [test_metrics[name]['r2_te']],
                    c=col, marker='*', s=120, zorder=6)
    style_ax(ax2, 'J2 — R² train (curva) + test (★ final)',
             'época', r'$R^2$')
    ax2.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    # ── J3: Regularizador ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL_BG)
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(PANEL_BG)
    reg_psgld = np.array(histories['pSGLD']['loss_reg'])
    ax3.plot(_smooth(reg_psgld), color=C_PSGLD, lw=1.8, ls='--',
             alpha=0.9, label='pSGLD L4+L2 (izq)')
    for name, col in [('MMD', C_MMD), ('Sinkhorn', C_SINKHORN)]:
        reg = np.array(histories[name]['loss_reg'])
        ax3_r.plot(_smooth(reg), color=col, lw=1.8, label=name)
        ax3_r.plot(reg, color=col, lw=0.4, alpha=0.2)
    ax3_r.tick_params(colors=TXT, labelsize=7)
    ax3_r.set_ylabel('MMD²/Sinkhorn', color=TXT, fontsize=8)
    style_ax(ax3, 'J3 — Regularizador durante training',
             'época', 'L4+L2')
    lines1, l1 = ax3.get_legend_handles_labels()
    lines2, l2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1+lines2, l1+l2, facecolor=PANEL_BG,
               labelcolor=TXT, fontsize=7)

    # ── J4: Predicted vs Actual (test set) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(PANEL_BG)
    offsets = {'pSGLD': -0.06, 'MMD': 0.0, 'Sinkhorn': 0.06}
    y_te_np = test_metrics['pSGLD']['y_te']
    lims = [y_te_np.min() - 0.3, y_te_np.max() + 0.3]
    ax4.plot(lims, lims, color='white', lw=1.0, ls='--', alpha=0.4,
             label='predicción perfecta')
    for name, (col, _, lbl) in style_map.items():
        pred = test_metrics[name]['pred_te']
        ax4.scatter(y_te_np + offsets[name], pred,
                    c=col, s=12, alpha=0.45, label=lbl)
        # R² en la leyenda
    handles = [
        mlines.Line2D([], [], color='white', lw=1, ls='--',
                      label='perfecta'),
    ] + [
        mlines.Line2D([], [], color=c, lw=0, marker='o', markersize=6,
                      markerfacecolor=c,
                      label=f"{lbl}  R²={test_metrics[n]['r2_te']:.3f}")
        for n, (c, _, lbl) in style_map.items()
    ]
    style_ax(ax4, 'J4 — Predicción vs Real (test set, target estandarizado)',
             'Real (y)', 'Predicho (ŷ)')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.legend(handles=handles, facecolor=PANEL_BG, labelcolor=TXT,
               fontsize=8.5, ncol=4)

    # ── J5: W₁ por dimensión ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(PANEL_BG)
    x_pos = np.arange(5)
    w = 0.35
    ax5.bar(x_pos - w/2, w1_mmd,  w, color=C_MMD,      alpha=0.85,
            label='MMD vs pSGLD')
    ax5.bar(x_pos + w/2, w1_sink, w, color=C_SINKHORN,  alpha=0.85,
            label='Sinkhorn vs pSGLD')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(DIM_LABELS, fontsize=9)
    ax5.text(0.98, 0.95,
             f'MMD²(MMD, pSGLD) = {mmd_mmd_psgld:.3e}\n'
             f'MMD²(W, pSGLD) = {mmd_sink_psgld:.3e}',
             transform=ax5.transAxes, ha='right', va='top',
             color=TXT, fontsize=8,
             bbox=dict(facecolor=DARK_BG, alpha=0.7, edgecolor=GRID_C))
    style_ax(ax5,
             r'J5 — $W_1$ por dimensión: $\hat\nu^*_\mathrm{método}$ vs $\hat\nu^*_\mathrm{pSGLD}$',
             '', r'$W_1$')
    ax5.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── J6: Scatter 2D partículas finales ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL_BG)
    markers = {'pSGLD': 'o', 'MMD': 's', 'Sinkhorn': '^'}
    for name, (col, _, lbl) in style_map.items():
        pts = extract_particles(models[name])[:, :2].numpy()
        ax6.scatter(pts[:, 0], pts[:, 1], c=col,
                    marker=markers[name], s=45, alpha=0.85,
                    edgecolors='white', linewidths=0.4, label=lbl, zorder=5)
    pr_2d = prior_samples[:200, :2].numpy()
    ax6.scatter(pr_2d[:, 0], pr_2d[:, 1], c=C_PRIOR,
                marker='.', s=8, alpha=0.3,
                label=r'$\nu^\infty$', zorder=1)
    style_ax(ax6,
             r'J6 — Partículas $(a_0^{(0)}, a_0^{(1)})$',
             r'$a_0^{(0)}$', r'$a_0^{(1)}$')
    ax6.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    fig.suptitle(
        r'Experimento J — Regresión: California Housing PCA(2)'
        '\n'
        rf'pSGLD vs MMD-reg vs Sinkhorn-reg  |  $\varepsilon={epsilon}$  '
        rf'|  varianza explicada={ds["explained_var"]:.1%}  '
        rf'|  train={n_train}  test={n_test}',
        color=TXT, fontsize=11, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUTPUT_DIR, 'J_california_regression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    return {
        'histories':       histories,
        'models':          models,
        'test_metrics':    test_metrics,
        'timeavg':         timeavg,
        'w1_mmd':          w1_mmd,
        'w1_sinkhorn':     w1_sink,
        'mmd_mmd_psgld':   mmd_mmd_psgld,
        'mmd_sink_psgld':  mmd_sink_psgld,
        'dataset':         ds,
    }
