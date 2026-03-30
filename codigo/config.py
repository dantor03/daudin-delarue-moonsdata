import os
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # noqa: E402

warnings.filterwarnings('ignore')

# ── Reproducibilidad y dispositivo ──────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'figuras'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Dispositivo: {DEVICE}")
print(f"✓ Figuras → {OUTPUT_DIR}")

# ── Tema visual oscuro ───────────────────────────────────────────────────────
DARK_BG   = '#0f0f1a'
PANEL_BG  = '#1a1a2e'
TXT       = '#e0e0e0'
GRID_C    = '#2a2a4a'
COLORS_EPS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=TXT, fontsize=9.5, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TXT, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TXT, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)
