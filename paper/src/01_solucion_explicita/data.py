"""Datasets reales 1D->1D para la solucion explicita.

Cacheados en paper/data/. Originalmente de vincentarelbundock/Rdatasets.

- Old Faithful (faithful.csv): eruption_duration -> waiting_time, N=272,
  relacion casi lineal (corr ~0.9).  Sirve como sanity check (el solver
  deberia recuperar el optimo OLS).

- Auto MPG (auto.csv): horsepower -> mpg, N~390, relacion claramente
  no lineal (saturante decreciente).  Aqui la dinamica de la mean-field
  ResNet sí necesita trabajar para mejorar al baseline lineal.
"""
from pathlib import Path
import numpy as np
import csv

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _standardize(x, y):
    return (x - x.mean()) / x.std(), (y - y.mean()) / y.std()


def load_faithful(standardize: bool = True):
    """eruption_duration -> waiting.  Returns (x, y) arrays of shape (N,)."""
    raw = np.genfromtxt(DATA_DIR / "faithful.csv", delimiter=",", skip_header=1)
    x = raw[:, 1].astype(float)
    y = raw[:, 2].astype(float)
    return _standardize(x, y) if standardize else (x, y)


def load_trees(standardize: bool = True):
    """Girth -> Volume (R built-in trees dataset).  N=31, claramente no lineal
    (V ~ Girth^2) y monotonicamente CRECIENTE — apta para X(T) = network output."""
    raw = np.genfromtxt(DATA_DIR / "trees.csv", delimiter=",", skip_header=1)
    x = raw[:, 1].astype(float)   # Girth
    y = raw[:, 3].astype(float)   # Volume
    return _standardize(x, y) if standardize else (x, y)


def load_auto(standardize: bool = True):
    """horsepower -> mpg.  Returns (x, y) arrays.  Drops rows con NA en horsepower."""
    rows = []
    with open(DATA_DIR / "auto.csv", newline="") as f:
        for r in csv.DictReader(f):
            try:
                hp = float(r["horsepower"])
                mpg = float(r["mpg"])
                rows.append((hp, mpg))
            except ValueError:
                continue   # skip '?' o NA
    arr = np.array(rows)
    x, y = arr[:, 0], arr[:, 1]
    return _standardize(x, y) if standardize else (x, y)
