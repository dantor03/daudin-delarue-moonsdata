"""California Housing SIN PCA: 8 features originales estandarizadas.

A diferencia de codigo/data.get_california_regression() que proyecta a R^2 con
PCA antes de pasar al modelo, aqui mantenemos d_1 = 8 (todas las features
originales).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from codigo.config import SEED, DEVICE   # noqa: E402

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def get_california_full(n_train: int = 800, n_test: int = 400, seed: int = SEED):
    """California Housing con 8 features estandarizadas.

    Pipeline:
        1. Submuestrear n_train+n_test filas
        2. StandardScaler en X (8 features)
        3. StandardScaler en y (target)
        4. train/test split

    Returns:
        dict con X_train, y_train, X_test, y_test (tensores y arrays NumPy),
        scaler_X, scaler_y, feature_names.
    """
    data = fetch_california_housing()
    X_full = data.data.astype(np.float32)
    y_full = data.target.astype(np.float32)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_full), size=n_train + n_test, replace=False)
    X_s, y_s = X_full[idx], y_full[idx]

    scaler_X = StandardScaler().fit(X_s)
    X_std = scaler_X.transform(X_s).astype(np.float32)

    scaler_y = StandardScaler().fit(y_s.reshape(-1, 1))
    y_std = scaler_y.transform(y_s.reshape(-1, 1)).ravel().astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_std, y_std,
        train_size=n_train, test_size=n_test, random_state=seed,
    )
    return {
        "X_train": torch.tensor(X_tr, device=DEVICE),
        "y_train": torch.tensor(y_tr, device=DEVICE),
        "X_test":  torch.tensor(X_te, device=DEVICE),
        "y_test":  torch.tensor(y_te, device=DEVICE),
        "X_train_np": X_tr, "y_train_np": y_tr,
        "X_test_np":  X_te, "y_test_np":  y_te,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "feature_names": FEATURE_NAMES,
    }


def top2_features_by_correlation(d, k: int = 2):
    """Devuelve los indices de las k features mas correlacionadas con y_train."""
    X = d["X_train_np"]
    y = d["y_train_np"]
    corrs = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    order = np.argsort(-np.abs(corrs))[:k]
    return order, corrs[order]
