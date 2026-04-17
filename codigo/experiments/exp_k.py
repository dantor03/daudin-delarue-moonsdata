"""
=============================================================================
exp_k.py — Experimento K: Barrido de ε para regresión (California Housing)
=============================================================================

MOTIVACIÓN:
    El experimento J usó ε=0.01 (igual que clasificación), pero la escala de
    la loss cambia drásticamente:

        Clasificación (exp_i, época 600):
            BCE = 0.0004    ε·reg = 0.01 × 0.133 = 0.00133
            → regularizador = 333% del BCE

        Regresión (exp_j, época 600):
            MSE = 0.7178    ε·reg = 0.01 × 0.176 = 0.00176
            → regularizador = 0.25% del MSE

    Con ε=0.01, el regularizador es prácticamente invisible en regresión.
    La inferioridad de pSGLD puede ser un artefacto del valor de ε, no
    una propiedad fundamental del método.

PROTOCOLO:
    Para cada ε en una rejilla logarítmica {0.001, 0.01, 0.1, 0.5, 1, 5}:
        1. Generar prior_samples de ν^∞ ∝ exp(-ℓ(a)/ε)  ← cambia con ε
        2. Entrenar pSGLD, MMD-reg y Sinkhorn-reg en California Housing
        3. Registrar MSE_train, MSE_test, R²_train, R²_test
        4. Calcular fracción de regularización: ε·reg / MSE_train

FIGURAS (K_epsilon_sweep.png):
    K1 — R²_test vs ε (log)  — curva principal por método
    K2 — R²_train vs ε       — para detectar overfitting
    K3 — Fracción de regularización (ε·reg/MSE) vs ε
         muestra en qué ε el regularizador pasa a ser relevante
    K4 — Comparación final al ε óptimo por método y al ε óptimo compartido
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..data import get_california_regression
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import sample_prior_langevin

C_PSGLD    = '#3498db'
C_MMD      = '#e67e22'
C_SINKHORN = '#2ecc71'

STYLE = {
    'pSGLD':    (C_PSGLD,    'o', '-'),
    'MMD':      (C_MMD,      's', '-'),
    'Sinkhorn': (C_SINKHORN, '^', '-'),
}


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(float(ss_tot), 1e-8))


def experiment_K(epsilon_grid=None, n_epochs=500, n_prior=1_000,
                 sinkhorn_blur=0.05, n_train=800, n_test=400):
    """
    Barrido de ε para encontrar el régimen óptimo en regresión.

    Args:
        epsilon_grid  : lista de valores de ε a probar
        n_epochs      : épocas de entrenamiento por run (reducidas para velocidad)
        n_prior       : muestras del prior por ε (1000 es suficiente para el barrido)
        sinkhorn_blur : regularización del OT Sinkhorn
        n_train       : muestras de entrenamiento
        n_test        : muestras de test
    """
    if epsilon_grid is None:
        epsilon_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]

    print("\n" + "=" * 62)
    print("EXPERIMENTO K  —  Barrido de ε (California Housing)")
    print("=" * 62)
    print(f"  ε ∈ {epsilon_grid}")
    print(f"  {n_epochs} epochs  |  train={n_train}  test={n_test}")

    # ── 1. Datos ────────────────────────────────────────────────────────────
    ds   = get_california_regression(n_train=n_train, n_test=n_test)
    X_tr = ds['X_train']
    y_tr = ds['y_train']
    X_te = ds['X_test']
    y_te = ds['y_test']
    y_tr_np = y_tr.cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    print(f"  Varianza explicada PCA(2): {ds['explained_var']:.1%}")

    # ── 2. Precomputar prior samples para cada ε ─────────────────────────────
    # ν^∞ ∝ exp(-ℓ(a)/ε) depende de ε: hay que regenerar para cada valor
    print("\n  Precomputando prior samples para cada ε...")
    prior_by_eps = {}
    for eps in epsilon_grid:
        prior_by_eps[eps] = sample_prior_langevin(n_prior, epsilon=eps, dim=5)
        print(f"    ε={eps:<5}  → {prior_by_eps[eps].shape}")

    # ── 3. Barrido ──────────────────────────────────────────────────────────
    # results[método][ε] = dict con métricas
    results = {name: {} for name in ['pSGLD', 'MMD', 'Sinkhorn']}

    for eps in epsilon_grid:
        print(f"\n  ── ε = {eps} " + "─" * 40)
        prior_s = prior_by_eps[eps]

        configs = [
            ('pSGLD',    dict(use_sgld=True)),
            ('MMD',      dict(use_mmd=True,     prior_samples=prior_s)),
            ('Sinkhorn', dict(use_sinkhorn=True, prior_samples=prior_s,
                              sinkhorn_blur=sinkhorn_blur)),
        ]

        for name, kwargs in configs:
            torch.manual_seed(42)
            np.random.seed(42)
            m = MeanFieldResNet(task='regression').to(DEVICE)
            h = train(m, X_tr, y_tr, epsilon=eps, n_epochs=n_epochs,
                      verbose=False, **kwargs)

            m.eval()
            with torch.no_grad():
                p_tr = m(X_tr).cpu().numpy()
                p_te = m(X_te).cpu().numpy()

            mse_tr = float(np.mean((p_tr - y_tr_np) ** 2))
            mse_te = float(np.mean((p_te - y_te_np) ** 2))
            r2_tr  = _r2(y_tr_np, p_tr)
            r2_te  = _r2(y_te_np, p_te)
            # Fracción de regularización: ε·reg / MSE_train al final
            reg_final = np.array(h['loss_reg'])[-1]
            reg_frac  = float(eps * reg_final / max(mse_tr, 1e-8))

            results[name][eps] = dict(
                mse_tr=mse_tr, mse_te=mse_te,
                r2_tr=r2_tr,   r2_te=r2_te,
                reg_frac=reg_frac,
                reg_final=reg_final,
            )
            print(f"    [{name:<10}]  MSE={mse_te:.4f}  R²_te={r2_te:.3f}"
                  f"  reg_frac={reg_frac:.3f}")

    # ── 4. Resumen ──────────────────────────────────────────────────────────
    print("\n  R²_test por método y ε:")
    header = f"  {'ε':<7}" + "".join(f"{'pSGLD':>10}{'MMD':>10}{'Sinkhorn':>10}")
    print(header)
    for eps in epsilon_grid:
        row = f"  {eps:<7}"
        for name in ['pSGLD', 'MMD', 'Sinkhorn']:
            row += f"{results[name][eps]['r2_te']:>10.3f}"
        print(row)

    # ε óptimo por método (máximo R²_test)
    best_eps = {}
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        best_eps[name] = max(epsilon_grid,
                             key=lambda e: results[name][e]['r2_te'])
        print(f"  ε óptimo {name}: {best_eps[name]}"
              f"  →  R²_test = {results[name][best_eps[name]]['r2_te']:.3f}")

    # ε óptimo compartido (maximiza el mínimo entre métodos)
    best_shared = max(epsilon_grid,
                      key=lambda e: min(results[n][e]['r2_te']
                                        for n in ['pSGLD', 'MMD', 'Sinkhorn']))
    print(f"\n  ε compartido óptimo: {best_shared}")
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        r = results[name][best_shared]
        print(f"    {name:<10}  R²_te={r['r2_te']:.3f}  MSE_te={r['mse_te']:.4f}"
              f"  reg_frac={r['reg_frac']:.3f}")

    # ── 5. Figuras ──────────────────────────────────────────────────────────
    eps_arr = np.array(epsilon_grid)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    (ax1, ax2), (ax3, ax4) = axes

    for ax in axes.ravel():
        ax.set_facecolor(PANEL_BG)

    # ── K1: R²_test vs ε ────────────────────────────────────────────────────
    for name, (col, mk, ls) in STYLE.items():
        y_vals = [results[name][e]['r2_te'] for e in epsilon_grid]
        ax1.plot(eps_arr, y_vals, color=col, lw=2.2, ls=ls,
                 marker=mk, markersize=7, label=name)
        # Marcar ε óptimo
        be = best_eps[name]
        ax1.axvline(be, color=col, lw=0.8, ls=':', alpha=0.5)
    ax1.axvline(best_shared, color='white', lw=1.2, ls='--', alpha=0.6,
                label=f'ε compartido óptimo ({best_shared})')
    ax1.set_xscale('log')
    style_ax(ax1, r'K1 — $R^2_{test}$ vs $\varepsilon$'
             '\nPunto = ε óptimo por método  |  Blanco = ε compartido',
             r'$\varepsilon$', r'$R^2_{test}$')
    ax1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)

    # ── K2: R²_train vs ε ───────────────────────────────────────────────────
    for name, (col, mk, ls) in STYLE.items():
        y_tr_vals = [results[name][e]['r2_tr'] for e in epsilon_grid]
        y_te_vals = [results[name][e]['r2_te'] for e in epsilon_grid]
        ax2.plot(eps_arr, y_tr_vals, color=col, lw=2.0, ls='--',
                 marker=mk, markersize=6, alpha=0.8, label=f'{name} train')
        ax2.plot(eps_arr, y_te_vals, color=col, lw=2.0, ls='-',
                 marker=mk, markersize=6, label=f'{name} test')
    ax2.set_xscale('log')
    style_ax(ax2, r'K2 — $R^2$ train (--) vs test (—)'
             '\nBrecha grande = overfitting',
             r'$\varepsilon$', r'$R^2$')
    ax2.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7, ncol=2)

    # ── K3: Fracción de regularización vs ε ─────────────────────────────────
    for name, (col, mk, ls) in STYLE.items():
        fracs = [results[name][e]['reg_frac'] for e in epsilon_grid]
        ax3.plot(eps_arr, fracs, color=col, lw=2.0, ls=ls,
                 marker=mk, markersize=7, label=name)
    ax3.axhline(1.0, color='white', lw=1.0, ls=':', alpha=0.5,
                label='reg = 100% MSE')
    ax3.axhline(0.1, color='white', lw=0.8, ls=':', alpha=0.3,
                label='reg = 10% MSE')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    style_ax(ax3,
             r'K3 — Fracción de regularización: $\varepsilon \cdot \mathrm{reg} / \mathrm{MSE_{train}}$'
             '\nEn clasificación era ~3  |  Con ε=0.01 en regresión era ~0.003',
             r'$\varepsilon$', r'$\varepsilon \cdot \mathrm{reg} / \mathrm{MSE}$')
    ax3.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7.5)

    # ── K4: Comparación al ε óptimo compartido ───────────────────────────────
    names   = ['pSGLD', 'MMD', 'Sinkhorn']
    cols    = [C_PSGLD, C_MMD, C_SINKHORN]
    x_pos   = np.arange(len(names))

    r2_best = [results[n][best_eps[n]]['r2_te']     for n in names]
    r2_shar = [results[n][best_shared]['r2_te']     for n in names]
    r2_orig = [results[n][0.01]['r2_te']            for n in names]

    w = 0.26
    ax4.bar(x_pos - w,   r2_orig, w, color=cols, alpha=0.40, hatch='//',
            edgecolor='white', linewidth=0.5)
    ax4.bar(x_pos,       r2_shar, w, color=cols, alpha=0.70)
    ax4.bar(x_pos + w,   r2_best, w, color=cols, alpha=1.00)

    for i, (n, c) in enumerate(zip(names, cols)):
        for j, (val, offset) in enumerate([
            (r2_orig[i], -w), (r2_shar[i], 0), (r2_best[i], w)
        ]):
            ax4.text(i + offset, val + 0.005, f'{val:.3f}',
                     ha='center', va='bottom', color=TXT, fontsize=7.5)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, fontsize=10)
    handles = [
        mlines.Line2D([], [], color='gray',  lw=0, marker='s', markersize=9,
                      markerfacecolor='gray', alpha=0.40,
                      label='ε=0.01 (exp J)'),
        mlines.Line2D([], [], color='gray',  lw=0, marker='s', markersize=9,
                      markerfacecolor='gray', alpha=0.70,
                      label=f'ε={best_shared} (compartido óptimo)'),
        mlines.Line2D([], [], color='gray',  lw=0, marker='s', markersize=9,
                      markerfacecolor='gray', alpha=1.00,
                      label='ε óptimo por método'),
    ]
    style_ax(ax4,
             r'K4 — $R^2_{test}$ comparado: ε original vs óptimo'
             '\n¿Cambia el ranking con el ε correcto?',
             '', r'$R^2_{test}$')
    ax4.legend(handles=handles, facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)
    ax4.set_ylim(bottom=max(0, min(r2_orig) - 0.05))

    fig.suptitle(
        r'Experimento K — Barrido de $\varepsilon$: California Housing PCA(2)'
        '\n'
        rf'¿Es la inferioridad de pSGLD en exp_J un artefacto de $\varepsilon=0.01$?'
        '\n'
        rf'{n_epochs} epochs  |  {n_prior} prior samples por $\varepsilon$  '
        rf'|  varianza explicada={ds["explained_var"]:.1%}',
        color=TXT, fontsize=11, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(OUTPUT_DIR, 'K_epsilon_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    return {
        'results':      results,
        'best_eps':     best_eps,
        'best_shared':  best_shared,
        'epsilon_grid': epsilon_grid,
        'dataset':      ds,
    }
