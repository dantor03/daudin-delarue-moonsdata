"""Regresion multivariada en California Housing SIN PCA (d_1 = 8).

Compara baselines clasicos (OLS, Ridge, RandomForest) y los 4 metodos
paramétricos de Tarea 2 (SGLD vanilla, pSGLD, MMD2, Sinkhorn) en el espacio
original de 8 features. Mismo seed, mismo split.

Genera:
  paper/figures/03_regression_curves.png       — R^2 vs epoca para los 4 paramétricos
  paper/figures/03_regression_predicted.png    — predicho vs real (test) por metodo
  paper/figures/03_regression_residuals.png    — histograma de residuos test
  paper/figures/03_regression_importance.png   — permutation importance de la mejor red
  paper/figures/03_explicit_top2.png           — opcional, explicito en top-2 features
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from codigo.config import SEED, DEVICE
from codigo.model import MeanFieldResNet
from codigo.train import train
from codigo.metrics import sample_prior_langevin

from data import get_california_full, FEATURE_NAMES, top2_features_by_correlation
from sgld_vanilla_reg import train_sgld_vanilla_reg

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fresh_model(d1, M):
    torch.manual_seed(SEED)
    return MeanFieldResNet(d1=d1, M=M, T=1.0, n_steps=10,
                           task='regression').to(DEVICE)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()


def smooth(arr, w=15):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr
    sm = np.convolve(arr, np.ones(w) / w, mode='valid')
    return np.concatenate([np.full(w - 1, np.nan), sm])


def main():
    print("=" * 70)
    print("Tarea 3 — Regresion multivariada SIN PCA (California Housing, d=8)")
    print("=" * 70)

    d = get_california_full(n_train=800, n_test=400)
    X_tr, y_tr = d["X_train"], d["y_train"]
    X_te, y_te = d["X_test"],  d["y_test"]
    Xtr_np, ytr_np = d["X_train_np"], d["y_train_np"]
    Xte_np, yte_np = d["X_test_np"],  d["y_test_np"]
    print(f"  N_train={len(ytr_np)}  N_test={len(yte_np)}  d_1=8")

    # ── Baselines clasicos ────────────────────────────────────────────────────
    print("\n[baselines clasicos]")
    baselines = {}
    for name, model in [
        ("OLS",          LinearRegression()),
        ("Ridge(α=1.0)", Ridge(alpha=1.0, random_state=SEED)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ]:
        model.fit(Xtr_np, ytr_np)
        pr_tr = model.predict(Xtr_np)
        pr_te = model.predict(Xte_np)
        baselines[name] = {
            "model": model,
            "pred_train": pr_tr, "pred_test": pr_te,
            "r2_train": r2_score(ytr_np, pr_tr),
            "r2_test":  r2_score(yte_np, pr_te),
            "rmse_test": np.sqrt(mean_squared_error(yte_np, pr_te)),
        }
        print(f"  {name:14s}  R²_train={baselines[name]['r2_train']:.3f}  "
              f"R²_test={baselines[name]['r2_test']:.3f}  "
              f"RMSE_test={baselines[name]['rmse_test']:.3f}")

    # ── Metodos paramétricos: 4 regularizadores ───────────────────────────────
    M = 128
    EPS = 0.01
    N_EPOCHS = 800
    print(f"\n[paramétricos]  M={M}  eps={EPS}  epochs={N_EPOCHS}")

    # Prior samples para MMD/Sinkhorn (dim_a = 2 d_1 + 1 = 17)
    print("[prior]  ULA: 600 muestras de nu^infty en R^17 ...")
    prior = sample_prior_langevin(n_samples=600, epsilon=EPS, dim=2 * 8 + 1,
                                   n_burnin=8000, thin=20)

    methods = {}

    print("\n[1/4] SGLD vanilla")
    m = fresh_model(d1=8, M=M)
    hist = train_sgld_vanilla_reg(m, X_tr, y_tr, epsilon=EPS, lr=1e-3,
                                   n_epochs=N_EPOCHS, verbose=True)
    methods["SGLD vanilla"] = (m, hist, "#e74c3c")

    print("\n[2/4] pSGLD")
    m = fresh_model(d1=8, M=M)
    hist = train(m, X_tr, y_tr, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_sgld=True)
    print(f"    R²_train_final={hist['accuracy'][-1]:.3f}")
    methods["pSGLD"] = (m, hist, "#f39c12")

    print("\n[3/4] MMD^2")
    m = fresh_model(d1=8, M=M)
    hist = train(m, X_tr, y_tr, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_mmd=True, prior_samples=prior)
    print(f"    R²_train_final={hist['accuracy'][-1]:.3f}")
    methods["MMD$^2$"] = (m, hist, "#2ecc71")

    print("\n[4/4] Sinkhorn debiased")
    m = fresh_model(d1=8, M=M)
    hist = train(m, X_tr, y_tr, epsilon=EPS, lr=0.01, n_epochs=N_EPOCHS,
                 verbose=False, use_sinkhorn=True, prior_samples=prior,
                 sinkhorn_blur=0.05)
    print(f"    R²_train_final={hist['accuracy'][-1]:.3f}")
    methods["Sinkhorn"] = (m, hist, "#3498db")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'modelo':<22s} {'R²_train':>10s} {'R²_test':>10s} {'RMSE_test':>11s}")
    print("-" * 70)
    rows = []
    for name, info in baselines.items():
        rows.append((name, info["r2_train"], info["r2_test"], info["rmse_test"]))
        print(f"  {name:<22s} {info['r2_train']:10.3f} {info['r2_test']:10.3f} "
              f"{info['rmse_test']:11.3f}")
    for name, (m, h, _) in methods.items():
        pr_tr = predict(m, X_tr)
        pr_te = predict(m, X_te)
        r2_tr = r2_score(ytr_np, pr_tr)
        r2_te = r2_score(yte_np, pr_te)
        rmse_te = np.sqrt(mean_squared_error(yte_np, pr_te))
        rows.append((name, r2_tr, r2_te, rmse_te))
        print(f"  {name:<22s} {r2_tr:10.3f} {r2_te:10.3f} {rmse_te:11.3f}")

    # ── Figura 1: curvas R^2 vs epoca ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (m, h, col) in methods.items():
        r2 = np.asarray(h['accuracy'], dtype=float)   # 'accuracy' = R² en regresion
        ax.plot(smooth(r2, 15), color=col, lw=1.6, label=name)
        ax.plot(r2, color=col, lw=0.4, alpha=0.25)
    # Lineas horizontales de los baselines en R²_test
    for nm, info, ls in [
        ("OLS", baselines["OLS"], "--"),
        ("Ridge", baselines["Ridge(α=1.0)"], "-."),
        ("RF", baselines["RandomForest"], ":"),
    ]:
        ax.axhline(info["r2_train"], color="k", ls=ls, lw=0.8, alpha=0.5,
                   label=f"{nm} train R²={info['r2_train']:.2f}")
    ax.set_xlabel("epoca"); ax.set_ylabel("$R^2$ train (suavizado)")
    ax.set_title(f"$R^2$ train vs epoca — California Housing d=8 — eps={EPS}")
    ax.set_ylim(-0.1, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_regression_curves.png", dpi=140, bbox_inches="tight")
    print(f"\n  -> {FIG_DIR / '03_regression_curves.png'}")

    # ── Figura 2: predicho vs real (test) por metodo ─────────────────────────
    n_panels = len(baselines) + len(methods)
    fig, axes = plt.subplots(2, (n_panels + 1) // 2, figsize=(4 * ((n_panels + 1) // 2), 9))
    axes = axes.ravel()
    yall = np.concatenate([yte_np])
    lim = (yall.min() - 0.3, yall.max() + 0.3)
    items = list(baselines.items()) + [(n, (m, h, c)) for n, (m, h, c) in methods.items()]
    for ax, item in zip(axes, items):
        name, info = item
        if isinstance(info, dict):
            pr_te = info["pred_test"]; col = "#888"
        else:
            m, h, col = info
            pr_te = predict(m, X_te)
        r2 = r2_score(yte_np, pr_te)
        ax.scatter(yte_np, pr_te, s=10, alpha=0.5, c=col)
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("y test"); ax.set_ylabel("$\\hat y$ test")
        ax.set_title(f"{name}  R²={r2:.3f}")
        ax.grid(alpha=0.3)
    for ax in axes[len(items):]:
        ax.set_visible(False)
    fig.suptitle("Predicho vs real (test) — California Housing d=8", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_regression_predicted.png", dpi=140, bbox_inches="tight")
    print(f"  -> {FIG_DIR / '03_regression_predicted.png'}")

    # ── Figura 3: histograma de residuos (test) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-3, 3, 41)
    for name, info in baselines.items():
        ax.hist(yte_np - info["pred_test"], bins=bins, alpha=0.35, label=name)
    for name, (m, h, col) in methods.items():
        pr_te = predict(m, X_te)
        ax.hist(yte_np - pr_te, bins=bins, alpha=0.45, label=name, color=col,
                histtype="step", lw=2)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("residuo test = y - $\\hat y$"); ax.set_ylabel("frecuencia")
    ax.set_title("Distribucion de residuos en test (d=8)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_regression_residuals.png", dpi=140, bbox_inches="tight")
    print(f"  -> {FIG_DIR / '03_regression_residuals.png'}")

    # ── Figura 4: permutation importance del mejor metodo paramétrico ────────
    best_name = max(methods, key=lambda n: r2_score(yte_np, predict(methods[n][0], X_te)))
    best_model = methods[best_name][0]
    print(f"\n[permutation importance]  modelo={best_name}")

    rng = np.random.RandomState(SEED)
    base_pred = predict(best_model, X_te)
    base_mse = mean_squared_error(yte_np, base_pred)
    importances = []
    for j in range(8):
        deltas = []
        for _ in range(10):
            X_perm = Xte_np.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pr = predict(best_model, torch.tensor(X_perm, device=DEVICE))
            deltas.append(mean_squared_error(yte_np, pr) - base_mse)
        importances.append((np.mean(deltas), np.std(deltas)))
    importances = np.array(importances)
    order = np.argsort(-importances[:, 0])

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(8)
    ax.barh(y_pos, importances[order, 0], xerr=importances[order, 1],
            color="#3498db", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("ΔMSE al permutar la feature (mayor = mas importante)")
    ax.set_title(f"Importancia por permutacion — {best_name} — California Housing d=8")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_regression_importance.png", dpi=140, bbox_inches="tight")
    print(f"  -> {FIG_DIR / '03_regression_importance.png'}")

    return rows


if __name__ == "__main__":
    main()
