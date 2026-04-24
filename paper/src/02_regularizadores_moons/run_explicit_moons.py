"""Solucion explicita (Thm 1.4) sobre make_moons con embedding de etiquetas.

Embedding:
    y = 0  ->  (-1, 0)
    y = 1  ->  (+1, 0)

Asi d1 = d2 = 2 (encajado exactamente en el setup del paper, Ej. 1.3) y el
coste terminal es L(x, y_emb) = 0.5 |x - y_emb|^2.

Estrategia:
  1. Cargar make_moons (mismo seed que A-K)
  2. Construir grid en R^5 (d1 = 2 -> dim_a = 2 d1 + 1 = 5)
  3. Continuacion en eps con warm-start
  4. Reportar accuracy: predict(x) = sign(X(T)[0])
     (la coordenada 0 separa las clases por construccion)
"""
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo.data import get_moons
from explicit_nd import build_grid_nd, picard_continuation_nd

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def embed_labels(y_np):
    """y in {0,1} -> (-1,0) / (+1,0)."""
    emb = np.zeros((len(y_np), 2))
    emb[:, 0] = 2.0 * y_np - 1.0   # 0 -> -1, 1 -> +1
    return emb


def accuracy(X_T, y_np):
    pred = (X_T[:, 0] > 0).astype(np.float32)
    return float((pred == y_np).mean())


def main():
    print("=" * 70)
    print("Tarea 2 (parte explicita) — Solucion explicita en make_moons")
    print("=" * 70)

    _, _, X_np, y_np = get_moons()
    y_emb = embed_labels(y_np)
    print(f"  N={len(y_np)}  d1=d2=2  embedding: y=0 -> (-1,0), y=1 -> (+1,0)")

    # Grid: 4^5 = 1024 puntos en R^5
    A, dim_a = build_grid_nd(d1=2, R=2.0, M_per_dim=4)
    print(f"  grid: M_per_dim=4 -> M={A.shape[0]} puntos, dim_a={dim_a}")

    eps_schedule = [0.5, 0.2, 0.1, 0.05, 0.02]

    # Baseline: predecir media (= 0.5) -> term = 0.5 * E[(X-y)^2] = 0.5 * 1 = 0.5
    # Trivial: si X = 0 (sin red), term = 0.5 * (1+0)/2 + 0.5*(1+0)/2 = 0.5
    print("\n  baseline trivial X=0:  term = 0.5")

    t0 = time.time()
    results = picard_continuation_nd(
        X_np.astype(float), y_emb.astype(float), A, d1=2,
        eps_schedule=eps_schedule,
        T=1.0, K=10, omega=0.2, max_iter=150, tol=1e-4, verbose=False,
    )
    print(f"\n  total Picard time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"  {'eps':>6s} {'iters':>6s} {'conv':>5s} {'J':>10s} {'term':>10s} {'acc':>6s}")
    print("-" * 70)
    summary = []
    for eps, res in results:
        term = 0.5 * ((res.X[-1] - y_emb) ** 2).sum(axis=1).mean()
        acc = accuracy(res.X[-1], y_np)
        summary.append((eps, res.iters, res.converged, res.cost_history[-1], term, acc))
        print(f"  {eps:6.3f} {res.iters:6d} {str(res.converged):>5s} "
              f"{res.cost_history[-1]:10.5f} {term:10.5f} {acc:6.3f}")

    # ── Figura: para cada eps, scatter de (X(0), X(T), y_emb) y datos ──────
    n_eps = len(eps_schedule)
    fig, axes = plt.subplots(2, n_eps, figsize=(4 * n_eps, 8))

    for j, (eps, res) in enumerate(results):
        XT = res.X[-1]
        # fila 0: X(0) coloreado por clase (datos originales)
        ax = axes[0, j]
        ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                   c="#e74c3c", s=12, label="y=0", alpha=0.7)
        ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                   c="#3498db", s=12, label="y=1", alpha=0.7)
        ax.set_title(f"eps={eps}: X(0) (datos)")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7); ax.set_aspect("equal")

        # fila 1: X(T) coloreado por clase + anchors y_emb
        ax = axes[1, j]
        ax.scatter(XT[y_np == 0, 0], XT[y_np == 0, 1],
                   c="#e74c3c", s=12, alpha=0.7)
        ax.scatter(XT[y_np == 1, 0], XT[y_np == 1, 1],
                   c="#3498db", s=12, alpha=0.7)
        ax.scatter([-1, 1], [0, 0], c="k", marker="X", s=120,
                   label="$y_{emb}$", zorder=5)
        ax.axvline(0, color="grey", ls="--", alpha=0.4)
        acc = accuracy(XT, y_np)
        ax.set_title(f"X(T)  acc={acc:.3f}")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7); ax.set_aspect("equal")

    fig.suptitle("Solucion explicita Thm 1.4 sobre make_moons "
                 "(d1=d2=2, embedding etiquetas)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "02_explicit_moons.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out}")

    # ── Curvas J(eps) y term(eps) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    eps_arr = [s[0] for s in summary]
    J_arr = [s[3] for s in summary]
    term_arr = [s[4] for s in summary]
    acc_arr = [s[5] for s in summary]
    ax.plot(eps_arr, J_arr, "o-", label="J explicito (term + ent)")
    ax.plot(eps_arr, term_arr, "s--", label="term L solo")
    ax2 = ax.twinx()
    ax2.plot(eps_arr, acc_arr, "v:", color="#2ecc71", label="acc")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("eps (continuacion)")
    ax.set_ylabel("coste"); ax2.set_ylabel("accuracy", color="#2ecc71")
    ax.set_title("make_moons explicito: continuacion en eps")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "02_explicit_moons_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
