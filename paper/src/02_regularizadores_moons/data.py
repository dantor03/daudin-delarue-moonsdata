"""Dataset make_moons identico al de los experimentos A-K (mismo seed/n/noise)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from codigo.data import get_moons   # noqa: E402

__all__ = ["get_moons"]
