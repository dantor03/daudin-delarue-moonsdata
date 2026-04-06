"""
=============================================================================
exp_g.py — Experimento G: Convergence Problem (N → ∞)
=============================================================================

Daudin & Delarue (2025) dejan explícitamente fuera del análisis el caso de
medidas iniciales empíricas, declarando que "la convergencia cuando N → ∞
se abordará en un trabajo futuro".  Este experimento da la primera evidencia
empírica de esa convergencia.

PREGUNTA CENTRAL:
    A medida que el tamaño del dataset de entrenamiento N crece,
    ¿converge el minimizador J*_N y la pérdida generalizada BCE_test(N)
    hacia un valor asintótico J*_∞?  ¿Acelera ε > 0 esa convergencia?

PROTOCOLO:
    - Oráculo de test: N_test = 10.000 puntos (seed=999, fijo).
      Aproxima la medida continua γ₀.  El StandardScaler se ajusta sobre
      este oráculo y se aplica también a los datos de entrenamiento, para
      que todos los runs vivan en el mismo espacio de features.

    - Entrenamiento: N ∈ {50, 100, 200, 400, 800, 1600, 3200}.
      Para cada N y cada semilla, se muestrean N puntos frescos de
      make_moons (misma distribución, seed = semilla del run).

    - ε ∈ {0.0, 0.01}.  Con ε=0 y use_sgld=True el ruido es exactamente
      cero (= Adam estándar).  Con ε=0.01 se usa pSGLD precondicionado.

    - n_seeds runs independientes por combinación (N, ε).

MÉTRICAS:
    - BCE_train : entropía cruzada sobre los N puntos de entrenamiento
                 (evaluada sin ruido al final del entrenamiento).
    - J*_train  : mínimo de la pérdida total J = BCE + ε·E durante el
                 entrenamiento (lo que el optimizador realmente minimizó).
    - BCE_test  : entropía cruzada sobre el oráculo de 10.000 puntos.
                 NO incluye el término de regularización: mide la calidad
                 predictiva pura del modelo aprendido.
    - acc_train / acc_test : accuracy sobre entrenamiento y oráculo.
    - Gap       : BCE_test − BCE_train (gap de generalización).

FIGURAS (G_convergence_problem.png):
    G1 — BCE_train y BCE_test vs N (escala log en X).
         Banda ±1σ sobre semillas.  Rojo = ε=0, azul = ε=0.01.
         Línea continua = BCE_test, discontinua = BCE_train.
         Se espera que ambas converjan y que el gap se cierre con N.

    G2 — Gap de generalización (BCE_test − BCE_train) vs N.
         Si ε > 0 acelera la convergencia, la curva azul debe caer
         más rápido que la roja (o llegar a un gap menor para N pequeño).
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from ..config import DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, style_ax
from ..model import MeanFieldResNet
from ..train import train


def _save_results_G(results, epsilons, n_values, path):
    """Serializa el dict results a .npz para poder reanudar o regenerar la figura."""
    save_dict = {'EPSILONS': np.array(epsilons), 'N_VALUES': np.array(n_values)}
    for ei, eps in enumerate(epsilons):
        for ni, N in enumerate(n_values):
            r = results[eps][N]
            for metric in ('BCE_train', 'J_star_train', 'BCE_test', 'acc_train', 'acc_test'):
                save_dict[f'{metric}_{ei}_{ni}'] = np.array(r[metric])
    np.savez(path, **save_dict)


def _load_results_G(path, epsilons, n_values):
    """Carga results desde .npz.  Devuelve None si el archivo no existe."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    results = {
        eps: {N: {'BCE_train': [], 'J_star_train': [], 'BCE_test': [],
                  'acc_train': [], 'acc_test': []}
              for N in n_values}
        for eps in epsilons
    }
    for ei, eps in enumerate(epsilons):
        for ni, N in enumerate(n_values):
            for metric in ('BCE_train', 'J_star_train', 'BCE_test', 'acc_train', 'acc_test'):
                key = f'{metric}_{ei}_{ni}'
                if key in data:
                    results[eps][N][metric] = data[key].tolist()
    print(f"  ✓ Resultados parciales cargados desde {path}")
    return results


def experiment_G(n_seeds: int = 5, n_epochs: int = 700):
    """
    Convergence Problem: estudio empírico de J*_N → J*_∞ cuando N → ∞.

    Args:
        n_seeds  : número de semillas por combinación (N, ε)
        n_epochs : épocas de entrenamiento por run
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO G  —  Convergence Problem: N → ∞")
    print("=" * 62)

    N_VALUES = [50, 100, 200, 400, 800, 1600, 3200]
    EPSILONS = [0.0, 0.01]
    N_TEST   = 10_000
    SEEDS    = list(range(n_seeds))
    RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                                '..', '..', 'figuras', 'G_results.npz')

    # ── Oráculo de test ───────────────────────────────────────────────────────
    # Se usa seed=999 para evitar solapamiento con las semillas de entrenamiento.
    # El scaler se ajusta sobre el oráculo y se aplica a TODOS los runs, así
    # todos viven en el mismo espacio de features.
    X_oracle_raw, y_oracle_raw = make_moons(
        n_samples=N_TEST, noise=0.12, random_state=999
    )
    oracle_scaler = StandardScaler().fit(X_oracle_raw)
    X_oracle = oracle_scaler.transform(X_oracle_raw).astype(np.float32)
    y_oracle = y_oracle_raw.astype(np.float32)
    X_test   = torch.tensor(X_oracle, device=DEVICE)
    y_test   = torch.tensor(y_oracle, device=DEVICE)
    print(f"  Oráculo test: {N_TEST} puntos (seed=999, scaler fijo)")

    # ── Carga de resultados previos (reanudación) ─────────────────────────────
    # Si existe G_results.npz de una ejecución anterior (o parcial en Colab),
    # se cargan los resultados ya calculados y se saltan esas combinaciones.
    results = _load_results_G(RESULTS_FILE, EPSILONS, N_VALUES)
    if results is None:
        results = {
            eps: {
                N: {'BCE_train': [], 'J_star_train': [], 'BCE_test': [],
                    'acc_train': [], 'acc_test': []}
                for N in N_VALUES
            }
            for eps in EPSILONS
        }

    # ── Bucle principal ───────────────────────────────────────────────────────
    for eps in EPSILONS:
        print(f"\n  ε = {eps} ──────────────────────────────────────────────")
        for N in N_VALUES:
            # Saltar si ya tenemos suficientes semillas para esta combinación
            already = len(results[eps][N]['BCE_test'])
            if already >= len(SEEDS):
                r  = results[eps][N]
                bt = np.array(r['BCE_train'])
                te = np.array(r['BCE_test'])
                print(f"    N={N:4d}: ya calculado ({already} seeds) — "
                      f"BCE_test={te.mean():.4f}±{te.std():.4f}  "
                      f"gap={(te - bt).mean():.4f}")
                continue
            for s in SEEDS[already:]:
                # Dataset de entrenamiento: N puntos frescos, escalados con el
                # scaler del oráculo para consistencia entre runs.
                X_raw, y_raw = make_moons(
                    n_samples=N, noise=0.12, random_state=s
                )
                X_np = oracle_scaler.transform(X_raw).astype(np.float32)
                y_np = y_raw.astype(np.float32)
                X    = torch.tensor(X_np, device=DEVICE)
                y    = torch.tensor(y_np, device=DEVICE)

                torch.manual_seed(s)
                np.random.seed(s)
                model = MeanFieldResNet().to(DEVICE)
                hist  = train(model, X, y, epsilon=eps,
                              n_epochs=n_epochs, verbose=False,
                              use_sgld=True)

                # ── Métricas de entrenamiento ─────────────────────────────
                J_star_train = hist['J_star']
                BCE_train    = hist['loss_term'][-1]   # BCE final (sin reg.)
                acc_train    = hist['accuracy'][-1]

                # ── Métricas sobre el oráculo ─────────────────────────────
                model.eval()
                with torch.no_grad():
                    # compute_loss con eps=0 → solo BCE (sin regularización)
                    _, BCE_test_t, _ = model.compute_loss(X_test, y_test, 0.0)
                    BCE_test = BCE_test_t  # ya es float (loss_term es scalar)
                    acc_test = (
                        (model(X_test) > 0).float() == y_test
                    ).float().mean().item()

                results[eps][N]['BCE_train'].append(BCE_train)
                results[eps][N]['J_star_train'].append(J_star_train)
                results[eps][N]['BCE_test'].append(BCE_test)
                results[eps][N]['acc_train'].append(acc_train)
                results[eps][N]['acc_test'].append(acc_test)

            # Resumen por N + guardado incremental
            r  = results[eps][N]
            bt = np.array(r['BCE_train'])
            te = np.array(r['BCE_test'])
            print(f"    N={N:4d}: BCE_train={bt.mean():.4f}±{bt.std():.4f}  "
                  f"BCE_test={te.mean():.4f}±{te.std():.4f}  "
                  f"gap={(te - bt).mean():.4f}  "
                  f"acc_test={np.mean(r['acc_test']):.3f}")
            # Guardado incremental: si Colab se corta, los resultados hasta aquí
            # están en G_results.npz y se pueden reanudar en la siguiente sesión.
            _save_results_G(results, EPSILONS, N_VALUES, RESULTS_FILE)
            print("      → checkpoint guardado en G_results.npz")

    # ── Figura G ──────────────────────────────────────────────────────────────
    COLOR = {0.0: '#e74c3c', 0.01: '#3498db'}
    Ns    = np.array(N_VALUES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax1, ax2, ax3 = axes

    rate_info = {}   # para imprimir en consola

    for eps in EPSILONS:
        col   = COLOR[eps]
        label = f'ε={eps}'

        bt_mean = np.array([np.mean(results[eps][N]['BCE_train']) for N in N_VALUES])
        bt_std  = np.array([np.std( results[eps][N]['BCE_train']) for N in N_VALUES])
        te_mean = np.array([np.mean(results[eps][N]['BCE_test'])  for N in N_VALUES])
        te_std  = np.array([np.std( results[eps][N]['BCE_test'])  for N in N_VALUES])
        gap     = te_mean - bt_mean
        gap_std = np.sqrt(bt_std**2 + te_std**2)

        # ── G1: BCE_train y BCE_test vs N ─────────────────────────────────
        ax1.plot(Ns, te_mean, color=col, lw=2.2, ls='-',
                 label=f'{label}  BCE$_{{test}}$')
        ax1.fill_between(Ns, np.maximum(te_mean - te_std, 0),
                         te_mean + te_std, color=col, alpha=0.18)
        ax1.plot(Ns, bt_mean, color=col, lw=1.5, ls='--',
                 label=f'{label}  BCE$_{{train}}$')
        ax1.fill_between(Ns, np.maximum(bt_mean - bt_std, 0),
                         bt_mean + bt_std, color=col, alpha=0.10)

        # ── G2: gap de generalización vs N (escala lineal, sin N=50) ──────
        # Excluye N=50 cuya varianza domina el eje y oscurece la tendencia
        mask  = Ns >= 100
        ax2.plot(Ns[mask], gap[mask], color=col, lw=2.2, label=label)
        ax2.fill_between(Ns[mask],
                         np.maximum(gap[mask] - gap_std[mask], 0),
                         gap[mask] + gap_std[mask],
                         color=col, alpha=0.20)

        # ── G3: log-log del gap → tasa de convergencia N^{-α} ────────────
        # Ajuste lineal en log-log: log(gap) = -α·log(N) + c
        # Se usa N ≥ 100 para evitar el régimen de N muy pequeño (ruidoso).
        log_N   = np.log(Ns[mask])
        log_gap = np.log(np.maximum(gap[mask], 1e-6))
        alpha, intercept = np.polyfit(log_N, log_gap, 1)
        N_fit   = np.logspace(np.log10(100), np.log10(3200), 50)
        gap_fit = np.exp(intercept) * N_fit ** alpha
        rate_info[eps] = alpha

        ax3.scatter(Ns[mask], gap[mask], color=col, s=40, zorder=5)
        ax3.plot(N_fit, gap_fit, color=col, lw=2.0,
                 label=rf'{label}  $\alpha={-alpha:.2f}$  (gap $\propto N^{{{alpha:.2f}}}$)')
        ax3.fill_between(Ns[mask],
                         np.maximum(gap[mask] - gap_std[mask], 1e-6),
                         gap[mask] + gap_std[mask],
                         color=col, alpha=0.18)

    # Referencia teórica 1/√N en G3
    N_ref  = np.logspace(np.log10(100), np.log10(3200), 50)
    # Escalar la referencia para que pase por el gap medio de ε=0 en N=100
    ref_scale = np.mean(results[0.0][100]['BCE_test']) - \
                np.mean(results[0.0][100]['BCE_train'])
    ax3.plot(N_ref, ref_scale * (100 / N_ref) ** 0.5,
             color='white', lw=1.2, ls=':', alpha=0.5,
             label=r'$\propto N^{-0.5}$ (referencia clásica)')

    # Formato ejes
    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([100, 200, 400, 800, 1600, 3200])
    ax3.set_xticklabels(['100', '200', '400', '800', '1600', '3200'])

    style_ax(ax1,
             'G1 — Convergencia de la pérdida con $N$\n'
             r'Continua = BCE$_{test}$  |  Discontinua = BCE$_{train}$',
             '$N$', 'BCE')
    ax1.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    style_ax(ax2,
             r'G2 — Gap de generalización  (BCE$_{test}$ $-$ BCE$_{train}$)'
             '\n'
             r'$N \geq 100$  (sin $N=50$: demasiada varianza)',
             '$N$', 'Gap')
    ax2.axhline(0, color='white', lw=0.8, ls=':', alpha=0.4)
    ax2.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    style_ax(ax3,
             r'G3 — Tasa de convergencia del gap  (log-log)'
             '\n'
             r'Ajuste gap $\propto N^{-\alpha}$  |  $\alpha$ estimado por regresión lineal',
             '$N$', 'Gap (log)')
    ax3.legend(facecolor=PANEL_BG, labelcolor=TXT, fontsize=7)

    fig.suptitle(
        r'Experimento G — Convergence Problem: BCE$_{test}(N) \to$ BCE$_\infty$  '
        r'($N \to \infty$)'
        '\n'
        rf'Banda = $\pm 1\sigma$ sobre {n_seeds} semillas  |  '
        r'$N_\mathrm{test} = 10\,000$ (oráculo make\_moons, seed=999)',
        color=TXT, fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, 'G_convergence_problem.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()

    print("\n  Tasas de convergencia estimadas (gap ∝ N^α, ajuste N≥100):")
    for eps, alpha in rate_info.items():
        print(f"    ε={eps}: α = {alpha:.3f}  (gap ∝ N^{{{alpha:.2f}}})")
    print(f"\n  → {out}")

    return results
