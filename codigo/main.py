"""
=============================================================================
main.py — Punto de entrada con CLI para ejecutar experimentos individuales
=============================================================================

Uso:
    python -m codigo                        # todos los experimentos
    python -m codigo --experiment A         # solo experimento A
    python -m codigo --experiment B --epochs 500
    python codigo/main.py --experiment all  # equivalente al primero
"""

import argparse

from .config import DEVICE, OUTPUT_DIR
from .experiments import (
    experiment_A, experiment_B, experiment_C,
    experiment_D, experiment_E_robustness, experiment_F, experiment_G,
)


def main(experiment: str = 'all', epochs: int | None = None) -> None:
    print("=" * 65)
    print("  MEAN-FIELD NEURAL ODE — MAKE MOONS")
    print("  Aplicación de arXiv:2507.08486  (Daudin & Delarue, 2025)")
    print("=" * 65)
    print(f"\n  Dispositivo : {DEVICE}")
    print(f"  Figuras     : {OUTPUT_DIR}")
    print()
    print("  Campo vectorial: b(x,a) = σ(a₁·x + a₂)·a₀   (σ = tanh)")
    print("  Prior:           ν^∞ ∝ exp(-0.05|a|⁴ - 0.5|a|²)")
    print("  ODE:             dX_t/dt = F(X_t,t)  en  ℝ²  (sin embedding)")
    print("  Integrador:      RK4  (n_steps=10)")
    print()

    run_all = experiment == 'all'

    # A — Evolución de γ_t
    if run_all or experiment == 'A':
        experiment_A(n_epochs=epochs or 800)

    # B — Efecto de ε
    if run_all or experiment == 'B':
        experiment_B(
            epsilons=[0.0, 0.001, 0.01, 0.1, 0.5],
            n_epochs=epochs or 700,
        )

    # C — Verificación de la condición PL (autocontenido, entrena con SGD)
    if run_all or experiment == 'C':
        experiment_C(
            epsilons=[0.0, 0.001, 0.01, 0.1, 0.5],
            n_epochs=epochs or 700,
        )

    # D — Genericidad
    if run_all or experiment == 'D':
        experiment_D(n_seeds=10, n_epochs=epochs or 500)

    # E (robustez) — Robustez de ν*
    if run_all or experiment == 'E':
        experiment_E_robustness(n_seeds=10, n_epochs=epochs or 500)

    # F — ν* en make_circles
    if run_all or experiment == 'F':
        experiment_F(n_seeds=10, n_epochs=epochs or 700)

    # G — Convergence problem: J*_N → J*_∞
    if run_all or experiment == 'G':
        experiment_G(n_seeds=20, n_epochs=epochs or 700)

    print("\n" + "=" * 65)
    print("  TODOS LOS EXPERIMENTOS COMPLETADOS")
    print()
    print("  Archivos generados:")
    for fname in [
        'A_feature_evolution.png',
        'B1_convergence_curves.png',
        'B2_decision_boundaries.png',
        'C_pl_verification.png',
        'D_genericity.png',
        'E_parameter_robustness.png',
        'F_circles_parameter_distribution.png',
        'G_convergence_problem.png',
    ]:
        print(f"    {OUTPUT_DIR}/{fname}")
    print("=" * 65)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Mean-Field Neural ODE — Make Moons (arXiv:2507.08486)'
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['all', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
        default='all',
        help='Experimento a ejecutar (por defecto: all)',
    )
    parser.add_argument(
        '--epochs', '-n',
        type=int,
        default=None,
        help='Número de épocas (sobrescribe el valor por defecto de cada experimento)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(experiment=args.experiment, epochs=args.epochs)
