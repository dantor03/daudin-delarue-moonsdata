"""Solver explicito (Thm 1.4) sobre California Housing usando solo las top-2 features.

En d_1 = 8, dim_a = 17 -> grid M_pd^17 inviable (M_pd=3 ya da 1.3e8).
Por tanto se restringe a 2 features (las mas correlacionadas con el target),
recuperando la situacion de Tarea 2 (d_1 = 2, dim_a = 5).
Esto NO sustituye a la solucion completa: documenta que el solver funciona
en regresion multivariada simplificada, y limita el alcance del enfoque
"explicito" a dimensiones bajas.
"""
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Importa explicit_nd antes de tocar el path local
sys.path.insert(0, str(ROOT / "paper" / "src" / "02_regularizadores_moons"))
from explicit_nd import build_grid_nd, picard_continuation_nd   # noqa: E402
# Ahora pone PRIMERO el dir local (tiene prioridad sobre 02_*)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_california_full, FEATURE_NAMES, top2_features_by_correlation   # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def main():
    print("=" * 70)
    print("Tarea 3 (parte explicita) — California Housing top-2 features")
    print("=" * 70)

    d = get_california_full(n_train=400, n_test=200)   # subset, mas rapido
    cols, corrs = top2_features_by_correlation(d, k=2)
    print(f"  top-2 features: {[FEATURE_NAMES[i] for i in cols]}  "
          f"corrs={corrs}")

    Xtr = d["X_train_np"][:, cols].astype(float)        # (N_tr, 2)
    Xte = d["X_test_np"][:, cols].astype(float)
    ytr = d["y_train_np"].astype(float)
    yte = d["y_test_np"].astype(float)

    # Embedding de y como (y, 0) para que d_1 = d_2 = 2
    y_emb_tr = np.stack([ytr, np.zeros_like(ytr)], axis=1)

    A, dim_a = build_grid_nd(d1=2, R=2.0, M_per_dim=4)   # 4^5 = 1024
    print(f"  grid: M_per_dim=4 -> M={A.shape[0]}, dim_a={dim_a}")

    eps_schedule = [0.5, 0.2, 0.1, 0.05, 0.02]

    t0 = time.time()
    results = picard_continuation_nd(
        Xtr, y_emb_tr, A, d1=2, eps_schedule=eps_schedule,
        T=1.0, K=10, omega=0.2, max_iter=120, tol=1e-4, verbose=False,
    )
    print(f"  Picard total: {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"  {'eps':>6s} {'iters':>6s} {'conv':>5s} {'J':>10s} "
          f"{'term':>10s} {'R²_tr':>8s}")
    print("-" * 70)
    rows = []
    for eps, res in results:
        XT = res.X[-1]
        pred_tr = XT[:, 0]
        r2 = r2_score(ytr, pred_tr)
        term = 0.5 * ((XT - y_emb_tr) ** 2).sum(axis=1).mean()
        rows.append((eps, res.iters, res.converged, res.cost_history[-1], term, r2))
        print(f"  {eps:6.3f} {res.iters:6d} {str(res.converged):>5s} "
              f"{res.cost_history[-1]:10.4f} {term:10.4f} {r2:8.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    eps_arr = [r[0] for r in rows]
    J_arr = [r[3] for r in rows]
    term_arr = [r[4] for r in rows]
    r2_arr = [r[5] for r in rows]
    ax = axes[0]
    ax.plot(eps_arr, J_arr, "o-", label="J explicito")
    ax.plot(eps_arr, term_arr, "s--", label="term L solo")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("eps"); ax.set_ylabel("coste")
    ax.set_title("Continuacion en eps")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(eps_arr, r2_arr, "v-", color="#2ecc71")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("eps"); ax.set_ylabel("R² train")
    ax.set_title("R² train vs eps")
    ax.grid(alpha=0.3)
    fig.suptitle("Solucion explicita en California Housing (top-2 features) "
                 f"[{FEATURE_NAMES[cols[0]]}, {FEATURE_NAMES[cols[1]]}]",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "03_explicit_top2.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
