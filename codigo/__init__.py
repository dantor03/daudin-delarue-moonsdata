"""
codigo — Paquete principal de la implementación Mean-Field Neural ODE.

Aplicación de arXiv:2507.08486 (Daudin & Delarue, 2025) al dataset make_moons.
"""

from .config import SEED, DEVICE, OUTPUT_DIR, DARK_BG, PANEL_BG, TXT, GRID_C, COLORS_EPS, style_ax
from .data import get_moons, get_circles
from .model import MeanFieldVelocity, MeanFieldResNet
from .train import train, mu_pl_estimate
from .plots import plot_decision_boundary

__all__ = [
    'SEED', 'DEVICE', 'OUTPUT_DIR',
    'DARK_BG', 'PANEL_BG', 'TXT', 'GRID_C', 'COLORS_EPS', 'style_ax',
    'get_moons', 'get_circles',
    'MeanFieldVelocity', 'MeanFieldResNet',
    'train', 'mu_pl_estimate',
    'plot_decision_boundary',
]
