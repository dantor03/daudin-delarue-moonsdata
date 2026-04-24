"""Comparacion limpia de 4 metodos sobre make_moons:
    1. SGLD vanilla       (SGD + ruido isotropico, sin precondicionar)
    2. pSGLD              (Adam + ruido escalado por M_t = 1/sqrt(v_hat))
    3. MMD2 regularizer   (Adam + eps * MMD^2(particulas, prior))
    4. Sinkhorn debiased  (Adam + eps * S_blur(particulas, prior))

Misma inicializacion (torch.manual_seed(SEED)) ⇒ unica diferencia = el metodo.
Mismos hiperparametros que A-K (N=400, noise=0.12, M=64, K=10, T=1.0).

Genera:
    paper/figures/02_metodos_comparacion.png  — 2x4 panel:
        fila 0: J(t)       [4 metodos en colores distintos]
        fila 1: accuracy   [smoothed]
    paper/figures/02_metodos_fronteras.png    — 1x4 fronteras de decision
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from codigo.config import SEED, DEVICE
from codigo.data import get_moons
from codigo.model import MeanFieldResNet
from codigo.train import train
from codigo.metrics import particles_live, sample_prior_langevin
from codigo.plots import plot_decision_boundary

from sgld_vanilla import train_sgld_vanilla   # local

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fresh_model():
    torch.manual_seed(SEED)
    return MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)


def smooth(arr, w=15):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    sm = np.convolve(arr, kernel, mode='valid')
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, sm])


def main():
    print("=" * 70)
    print("Tarea 2 — Comparacion pSGLD/MMD/Sinkhorn (make_moons)")
    print("=" * 70)

    X, y, X_np, y_np = get_moons()
    EPS = 0.01
    N_EPOCHS = 800
    print(f"  N={len(y)}  d=2  eps={EPS}  epochs={N_EPOCHS}")

    # Prior samples (compartidas por MMD y Sinkhorn)
    print("\n[prior]  ULA: 400 muestras de nu^infty ...")
    prior = sample_prior_langevin(n_samples=400, epsilon=EPS, dim=5)

    results = {}

    # 1. SGLD vanilla — SGD + ruido isotropico
    print("\n[1/4] SGLD vanilla (SGD + ruido isotropico, sin clipping)")
    m = fresh_model()
    hist = train_sgld_vanilla(m, X, y, epsilon=EPS, lr=1e-3,
                              n_epochs=N_EPOCHS, verbose=True)
    results["SGLD vanilla"] = (m, hist, "#e74c3c")

    # 2. pSGLD — Adam + ruido escalado por precondicionador
    print("\n[2/4] pSGLD (Adam + ruido escalado por M_t)")
    m = fresh_model()
    hist = train(m, X, y, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_sgld=True)
    print(f"    J*={hist['J_star']:.4f}  acc_final={hist['accuracy'][-1]:.3f}")
    results["pSGLD"] = (m, hist, "#f39c12")

    # 3. MMD2 regularizer
    print("\n[3/4] MMD^2 (Adam + eps * MMD^2(particulas, prior))")
    m = fresh_model()
    hist = train(m, X, y, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_mmd=True, prior_samples=prior)
    print(f"    J*={hist['J_star']:.4f}  acc_final={hist['accuracy'][-1]:.3f}")
    results["MMD$^2$"] = (m, hist, "#2ecc71")

    # 4. Sinkhorn debiased
    print("\n[4/4] Sinkhorn debiased (Adam + eps * S_blur(particulas, prior))")
    m = fresh_model()
    hist = train(m, X, y, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_sinkhorn=True, prior_samples=prior,
                 sinkhorn_blur=0.05)
    print(f"    J*={hist['J_star']:.4f}  acc_final={hist['accuracy'][-1]:.3f}")
    results["Sinkhorn"] = (m, hist, "#3498db")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'metodo':<18s} {'J*':>10s} {'acc final':>12s} {'finite?':>10s}")
    print("-" * 70)
    for name, (m, h, _) in results.items():
        J_star = h.get('J_star', float('nan'))
        acc = h['accuracy'][-1]
        finite = "yes" if np.isfinite(h['loss'][-1]) else "NO"
        print(f"  {name:<18s} {J_star:10.4f} {acc:12.3f} {finite:>10s}")

    # ── Figura 1: curvas ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, (m, h, col) in results.items():
        loss = np.asarray(h['loss'], dtype=float)
        axes[0].plot(loss, color=col, lw=1.4, label=name, alpha=0.9)
        acc_sm = smooth(h['accuracy'], w=15)
        axes[1].plot(acc_sm, color=col, lw=1.6, label=name)
        axes[1].plot(h['accuracy'], color=col, lw=0.5, alpha=0.25)
    axes[0].set_yscale('log')
    axes[0].set_xlabel("epoca"); axes[0].set_ylabel("J (log)")
    axes[0].set_title(f"Coste J vs epoca — eps={EPS}")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoca"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy vs epoca (suavizada, ventana 15)")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle("Comparacion de regularizadores en make_moons "
                 "(MeanField ResNet, N=400, M=64, K=10)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "02_metodos_comparacion.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out}")

    # ── Figura 2: fronteras de decision ──────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, (m, h, col)) in zip(axes, results.items()):
        acc = h['accuracy'][-1]
        try:
            plot_decision_boundary(ax, m, X_np, y_np,
                                    f"{name}\nacc={acc:.3f}")
        except Exception as e:
            ax.set_title(f"{name}: failed ({e})")
            ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, s=10)
    fig.suptitle("Fronteras de decision aprendidas — eps=0.01, 800 epochs",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "02_metodos_fronteras.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
