"""
=============================================================================
CAMPO MEDIO + NEURAL ODE EN ESPACIO DE FEATURES ℝ² — Make Moons
Aplicación de arXiv:2507.08486 (Daudin & Delarue, 2025)
=============================================================================

Este fichero es un SHIM DE COMPATIBILIDAD que mantiene la interfaz original
(python daudin_delarue_moons.py) mientras el código ha sido refactorizado en
el paquete `codigo/`.

Para ejecutar experimentos individuales usa el nuevo CLI:
    python -m codigo --experiment A
    python -m codigo --experiment B --epochs 500
    python -m codigo --help
=============================================================================
"""

# Re-exportar todos los símbolos públicos del paquete para compatibilidad
from .config import (  # noqa: F401
    SEED, DEVICE, OUTPUT_DIR,
    DARK_BG, PANEL_BG, TXT, GRID_C, COLORS_EPS, style_ax,
)
from .data import get_moons, get_circles              # noqa: F401
from .model import MeanFieldVelocity, MeanFieldResNet  # noqa: F401
from .train import train, mu_pl_estimate              # noqa: F401
from .plots import plot_decision_boundary             # noqa: F401
from .experiments import (                            # noqa: F401
    experiment_A, experiment_B, experiment_C,
    experiment_D, experiment_E, experiment_E2, experiment_F,
)
from .main import main

if __name__ == '__main__':
    main()
