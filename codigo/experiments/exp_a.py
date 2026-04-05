"""
=============================================================================
exp_a.py — Experimento A: Neural ODE + evolución de features γ_t
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from ..config import SEED, DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plots import plot_decision_boundary


# =============================================================================
# EXPERIMENTO A
#   Neural ODE con regularización entrópica + evolución de features γ_t
# =============================================================================
def experiment_A(n_epochs: int = 800):
    """
    Neural ODE con regularización entrópica + evolución de features γ_t.

    OPTIMIZADOR:
        Usa Adam + cosine annealing (modo por defecto de train()).  Al ser un
        experimento de demostración (no de muestreo de ν*), no se necesita SGLD.

    OBJETIVO MATEMÁTICO:
        Mostrar empíricamente que la ODE de campo medio puede transformar una
        distribución γ_0 (make_moons, no separable linealmente) en γ_T que sí
        lo es.  Esto es la "separabilidad emergente" del paper: la red aprende
        un flujo F(x,t) que reorganiza las features sin etiquetar explícitamente
        qué puntos mueve a dónde.

    CONEXIÓN CON EL PAPER:
        • γ_t = distribución push-forward de γ_0 bajo la ODE (ec. 1.5):
              γ_t = (ϕ_t)_# γ_0   donde ϕ_t es el flujo del campo F
        • La ecuación de continuidad (ec. 1.3) describe cómo evoluciona γ_t:
              ∂_t γ_t + div_x(F(x,t) γ_t) = 0
          Esta PDE garantiza que la "masa" (densidad de datos) se conserva
          y se transporta según el campo F — no se crean ni destruyen puntos.
        • X_T es linealmente separable por el clasificador W·X_T + b porque
          el campo F ha "alineado" las dos clases durante el transporte.

    FIGURAS GENERADAS (A_feature_evolution.png):
        Fila 1: γ_0, γ_{T/4}, γ_{T/2}, curvas de pérdida (J, BCE, entrópica)
        Fila 2: γ_{3T/4}, γ_T, trayectorias individuales X_t, frontera decisión
        El rectángulo discontinuo en cada panel indica la extensión inicial γ_0,
        permitiendo ver cuánto se han desplazado los puntos.
        Las trayectorias (panel inferior izquierdo) muestran 40 partículas
        seleccionadas con su camino completo de t=0 (punto) a t=T (estrella).

    RESULTADO ESPERADO:
        • γ_0 tiene forma de lunas entrelazadas (no separable)
        • γ_T muestra las dos clases separadas (linealmente separables)
        • La separación es gradual y suave: no hay "saltos" bruscos
        • acc ≈ 1.0 con ε=0.01, M=64, T=1.0
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO A  —  Feature evolution γ_t")
    print("=" * 62)

    X, y, X_np, y_np = get_moons()

    print("  Entrenando modelo base  ε=0.01, M=64, T=1.0, n_steps=10 …")
    torch.manual_seed(SEED)
    model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
    hist  = train(model, X, y, epsilon=0.01, n_epochs=n_epochs, verbose=True)
    acc   = hist['accuracy'][-1]
    mu    = mu_pl_estimate(hist)
    print(f"  → J* = {hist['J_star']:.5f} | acc = {acc:.4f} | μ_PL = {mu:.4f}")

    # ── Obtener trayectoria γ_t ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, traj = model.integrate(X, return_trajectory=True)

    # ── Figura ──────────────────────────────────────────────────────────────
    # Layout: 2×4 grid
    #   Fila 0: t=0  |  t=T/4  |  t=T/2  |  curvas de pérdida
    #   Fila 1: t=3T/4  |  t=T  |  trayectorias completas  |  frontera decisión
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.32)

    # 5 snapshots: t=0, T/4, T/2, 3T/4, T
    n_t   = len(traj) - 1           # = n_steps = 10
    snaps = [0,
             n_t // 4,
             n_t // 2,
             3 * n_t // 4,
             n_t]
    snap_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]   # (row, col)

    for (row, col), si in zip(snap_positions, snaps):
        ax = fig.add_subplot(gs[row, col])
        t_val, xt = traj[si]
        xt_np = xt.cpu().numpy()

        ax.scatter(xt_np[y_np == 0, 0], xt_np[y_np == 0, 1],
                   c='#ff6b6b', s=20, alpha=0.88, label='Clase 0', zorder=4)
        ax.scatter(xt_np[y_np == 1, 0], xt_np[y_np == 1, 1],
                   c='#74b9ff', s=20, alpha=0.88, label='Clase 1', zorder=4)

        # ── Ejes adaptativos (percentil 2-98 para ignorar outliers extremos) ──
        xq  = np.percentile(xt_np[:, 0], [2, 98])
        yq  = np.percentile(xt_np[:, 1], [2, 98])
        cx, cy = (xq[0] + xq[1]) / 2, (yq[0] + yq[1]) / 2
        half   = max(xq[1] - xq[0], yq[1] - yq[0]) / 2 + 0.45
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect('equal')

        # Rectángulo de referencia: extensión original de los datos (t=0)
        _, x0 = traj[0]
        x0_np = x0.cpu().numpy()
        x0_xq = np.percentile(x0_np[:, 0], [2, 98])
        x0_yq = np.percentile(x0_np[:, 1], [2, 98])
        rect = Rectangle((x0_xq[0], x0_yq[0]),
                          x0_xq[1] - x0_xq[0], x0_yq[1] - x0_yq[0],
                          linewidth=1, edgecolor='white', facecolor='none',
                          alpha=0.25, linestyle='--', zorder=2)
        ax.add_patch(rect)

        style_ax(ax,
                 f'$\\gamma_{{t={t_val:.2f}}}$  (paso {si}/{n_t})',
                 '$x_1$', '$x_2$')
        if (row, col) == (0, 0):
            ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7,
                      framealpha=0.8)

    # ── Panel (1,2): trayectorias completas X_t ─────────────────────────────
    ax_tr = fig.add_subplot(gs[1, 2])

    # Seleccionar 20 puntos por clase para mostrar sus trayectorias
    np.random.seed(SEED)
    idx0  = np.random.choice(np.where(y_np == 0)[0], size=20, replace=False)
    idx1  = np.random.choice(np.where(y_np == 1)[0], size=20, replace=False)
    all_coords = []
    for j in np.concatenate([idx0, idx1]):
        coords = np.array([traj[k][1].cpu().numpy()[j] for k in range(len(traj))])
        all_coords.append(coords)
        c = '#ff6b6b' if y_np[j] == 0 else '#74b9ff'
        ax_tr.plot(coords[:, 0], coords[:, 1], color=c, alpha=0.45, lw=1.0)
        ax_tr.scatter(coords[0,  0], coords[0,  1], c=c, s=18, zorder=5)
        ax_tr.scatter(coords[-1, 0], coords[-1, 1], c=c, s=40, zorder=6,
                      marker='*', edgecolors='white', linewidths=0.3)
    # Ejes adaptativos para las trayectorias
    all_c = np.vstack(all_coords)
    xq = np.percentile(all_c[:, 0], [1, 99])
    yq = np.percentile(all_c[:, 1], [1, 99])
    cx, cy = (xq[0]+xq[1])/2, (yq[0]+yq[1])/2
    half   = max(xq[1]-xq[0], yq[1]-yq[0])/2 + 0.5
    ax_tr.set_xlim(cx-half, cx+half)
    ax_tr.set_ylim(cy-half, cy+half)
    ax_tr.set_aspect('equal')
    style_ax(ax_tr,
             'Trayectorias $X_t$  (t=0 $\\bullet$ → T $\\star$)\n'
             '40 partículas seleccionadas',
             '$x_1$', '$x_2$')

    # ── Panel (0,3): curvas de pérdida ───────────────────────────────────────
    ax_l = fig.add_subplot(gs[0, 3])
    ep   = np.arange(n_epochs)
    ax_l.plot(ep, hist['loss'],      color='#2ecc71', lw=1.5, label='$J$ total')
    ax_l.plot(ep, hist['loss_term'], color='#3498db', lw=1.3, ls='--', label='BCE')
    ax_l.plot(ep, hist['loss_reg'],  color='#f39c12', lw=1.3, ls=':',  label='Prior energético')
    style_ax(ax_l, 'Curvas de pérdida', 'Época', '$J$')
    ax_l.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    # ── Panel (1,3): frontera de decisión en espacio original ────────────────
    ax_db = fig.add_subplot(gs[1, 3])
    plot_decision_boundary(ax_db, model, X_np, y_np,
                           f'Frontera de decisión en $\\mathbb{{R}}^2$\nacc={acc:.3f}')

    fig.suptitle(
        r'Evolución de features $\gamma_t$ — ODE de campo medio en espacio original $\mathbb{R}^2$'
        '\n'
        r'$\partial_t \gamma_t + \mathrm{div}_x(F(x,t)\,\gamma_t)=0$'
        f'      [ε=0.01  M=64  T=1.0  acc={acc:.3f}]',
        color=TXT, fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'A_feature_evolution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")
    return model, hist
