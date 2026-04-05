"""
=============================================================================
exp_b.py — Experimento B: Efecto del parámetro de regularización entrópica ε
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..config import SEED, DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, COLORS_EPS, style_ax
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plots import plot_decision_boundary


# =============================================================================
# EXPERIMENTO B
#   Efecto del parámetro de regularización entrópica ε
# =============================================================================
def experiment_B(epsilons=None, n_epochs: int = 700):
    """
    Efecto del parámetro de regularización entrópica ε.

    Todos los modelos se inicializan con la MISMA semilla aleatoria (torch.manual_seed
    antes de cada creación), de forma que la única diferencia entre ellos es ε.
    Esto aísla el efecto de la regularización de posibles diferencias por inicialización.

    EFECTO TEÓRICO DE ε (papel sec. 1.1 y 1.3):
        ε = 0 : Sin regularización.  La optimización es libre pero sin garantías
                teóricas.  En datos simples puede funcionar bien, pero en general
                puede quedar atrapada en mínimos locales.

        ε > 0 : La regularización entrópica fuerza ν_t hacia el prior ν^∞.
                • Da lugar a la FORMA DE GIBBS del control óptimo (ec. 1.9):
                      ν_t*(a) ∝ exp(-ℓ(a) - (1/ε) ∫_A b(x,a)·∇u_t dγ_t)
                  donde u_t es la función de valor (solución de la ecuación HJB).
                  Interpretación: el control óptimo concentra ν_t* alrededor de
                  los parámetros que minimizan L(x,y) pero "penalizados" por ℓ(a).
                • Garantiza desigualdad log-Sobolev → condición PL → convergencia exp.
                • No necesita ser grande: ε arbitrariamente pequeño da garantías.

        ε grande: Mayor sesgo hacia el prior → parámetros más concentrados cerca
                  de 0 → fronteras más suaves.  Mayor J* (el óptimo global es
                  "peor" en clasificación pura porque la regularización penaliza).

    TRADE-OFF EMPÍRICO ε vs μ_PL:
        Observamos empíricamente que μ̂_PL es MENOR para ε grande.  Esto no
        contradice el paper: la teoría garantiza μ > 0 para todo ε > 0, pero
        no dice que μ crezca con ε.  El trade-off es:
          • ε grande → mejor regularización, J* más alto, μ̂ posiblemente menor
          • ε pequeño → menor regularización, J* más bajo, μ̂ posiblemente mayor
          • ε = 0 → sin garantía, pero en datos fáciles funciona bien

    OPTIMIZADOR:
        Usa SGLD (use_sgld=True): SGD + cosine annealing + ruido de Langevin
        √(2·η·ε)·ξ.  Implementa la dinámica de Gibbs ν_t* ∝ exp(−J(θ)/ε),
        por lo que el ruido es coherente con el rol teórico de ε como temperatura.
        Con ε=0 el ruido es exactamente cero (se recupera SGD puro).

    FIGURAS GENERADAS:
        B1 — Curvas de convergencia: J total, accuracy, reg. supercoerciva de energía
             Esperado: todas las ε convergen a ~100% acc; J* crece con ε
        B2 — Figura 2×n_eps: fila superior = fronteras de decisión para cada ε,
             fila inferior = campo de velocidad F(x, t=0.5) para cada ε.
             Esperado: fronteras más suaves para ε mayor; campo más uniforme.
    """
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.01, 0.1, 0.5]

    print("\n" + "=" * 62)
    print("EXPERIMENTO B  —  Efecto del parámetro ε")
    print("=" * 62)

    X, y, X_np, y_np = get_moons()
    results = {}

    for eps, col in zip(epsilons, COLORS_EPS):
        print(f"\n  ε = {eps} ─────────────────────────────────────────")
        # Semilla fija → misma inicialización para todos los ε.
        # Así la única diferencia entre modelos es ε (temperatura Langevin
        # y coeficiente del prior energético).
        torch.manual_seed(SEED)
        model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
        hist  = train(model, X, y, epsilon=eps, n_epochs=n_epochs,
                      verbose=False, use_sgld=True)
        mu    = mu_pl_estimate(hist)
        results[eps] = {'model': model, 'hist': hist, 'color': col}
        print(f"    J*={hist['J_star']:.5f} | μ_PL={mu:.5f} "
              f"| acc={hist['accuracy'][-1]:.4f}")

    n_eps = len(epsilons)

    # ── B1: Curvas de convergencia ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)
    for eps, res in results.items():
        h = res['hist']
        axes[0].plot(h['loss'],      color=res['color'], lw=1.5, label=f'ε={eps}')
        axes[1].plot(h['accuracy'],  color=res['color'], lw=1.5, label=f'ε={eps}')
        axes[2].plot(h['loss_reg'],  color=res['color'], lw=1.5, label=f'ε={eps}')
    style_ax(axes[0], 'Pérdida total $J$ vs época', 'Época', '$J$')
    style_ax(axes[1], 'Accuracy vs época', 'Época', 'Acc')
    style_ax(axes[2],
             r'Reg. supercoerciva de energía $\mathcal{E}/N_{params}$', 'Época',
             r'$\mathcal{E}$')
    axes[1].set_ylim(0.45, 1.05)
    for ax in axes:
        ax.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=8)
    fig.suptitle('Efecto de ε en la convergencia del Mean-Field ResNet',
                 color=TXT, fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'B1_convergence_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"\n  → {out}")

    # ── B2: Fronteras de decisión + campo de velocidad (2 filas) ────────────
    xv = np.linspace(-2.5, 2.5, 16)
    Xg, Yg = np.meshgrid(xv, xv)
    grid_v  = torch.tensor(
        np.c_[Xg.ravel(), Yg.ravel()].astype(np.float32), device=DEVICE
    )

    fig, axes = plt.subplots(2, n_eps, figsize=(5 * n_eps, 10))
    fig.patch.set_facecolor(DARK_BG)

    for col, (eps, res) in enumerate(results.items()):
        # Fila 0: fronteras de decisión
        acc = res['hist']['accuracy'][-1]
        plot_decision_boundary(axes[0, col], res['model'], X_np, y_np,
                               f'ε={eps}   acc={acc:.3f}')

        # Fila 1: campo de velocidad F(x, t=0.5)
        ax = axes[1, col]
        m  = res['model']
        m.eval()
        with torch.no_grad():
            vel = m.velocity(0.5, grid_v).cpu().numpy()
        U     = vel[:, 0].reshape(Xg.shape)
        V     = vel[:, 1].reshape(Xg.shape)
        speed = np.hypot(U, V)
        q     = ax.quiver(Xg, Yg, U / (speed + 1e-8), V / (speed + 1e-8),
                          speed, cmap='plasma', alpha=0.85,
                          scale=18, width=0.004)
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                   c='#ff6b6b', s=10, alpha=0.45, zorder=4)
        ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                   c='#74b9ff', s=10, alpha=0.45, zorder=4)
        style_ax(ax, f'$F(x,t=0.5)$  ε={eps}', '$x_1$', '$x_2$')
        ax.set_aspect('equal')

    fig.suptitle(
        r'Fronteras de decisión y campo de velocidad  —  Mean-Field ODE  (make\_moons)'
        '\n'
        r'Fila superior: frontera en $\mathbb{R}^2$  |  '
        r'Fila inferior: $F(x,t=0.5)$, dirección normalizada, color = magnitud',
        color=TXT, fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'B2_decision_boundaries.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")

    return results
