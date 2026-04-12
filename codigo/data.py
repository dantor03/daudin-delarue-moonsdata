import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SEED, DEVICE


# =============================================================================
# § 1  DATASET
# =============================================================================
def get_moons(n=400, noise=0.12, seed=SEED):
    """
    Genera el dataset make_moons y lo devuelve como tensores PyTorch.

    En el contexto del paper, este dataset representa γ_0: la distribución
    empírica CONJUNTA (features, etiquetas) en el "tiempo de red" t=0:

        γ_0 = (1/N) Σᵢ δ_{(X₀ⁱ, Y₀ⁱ)}     (medida empírica sobre ℝ² × ℝ)

    donde (X₀ⁱ, Y₀ⁱ) ∈ ℝ² × ℝ es el i-ésimo par (feature, etiqueta).
    La ODE desplaza solo la componente de features X₀ⁱ → X_tⁱ; la etiqueta
    Y₀ⁱ permanece fija en todos los tiempos de red ("pasajero del flujo"):

        γ_t = (1/N) Σᵢ δ_{(X_tⁱ, Y₀ⁱ)}     (medida empírica sobre ℝ² × ℝ)

    Lo que se visualiza en las figuras es la MARGINAL EN x de γ_t:
        (γ_t)_x = (1/N) Σᵢ δ_{X_tⁱ}  ∈ P(ℝ²)
    que muestra cómo evolucionan los features, con colores que indican Y₀ⁱ.

    Se aplica StandardScaler para que γ_0 tenga media ≈ 0 y std ≈ 1,
    condición de regularidad implícita en el paper (datos acotados).

    Returns:
        X    : (N, 2) tensor en DEVICE — features estandarizadas
        y    : (N,) tensor en DEVICE  — etiquetas {0, 1}
        X_np : (N, 2) array NumPy     — para visualización
        y_np : (N,) array NumPy       — para visualización
    """
    X_np, y_np = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_np = StandardScaler().fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.float32)
    return (torch.tensor(X_np, device=DEVICE),
            torch.tensor(y_np, device=DEVICE),
            X_np, y_np)


def get_circles(n: int = 400, noise: float = 0.08, factor: float = 0.5,
                seed: int = SEED):
    """
    Genera el dataset make_circles y lo devuelve como tensores PyTorch.

    Análogo a get_moons() pero con simetría rotacional COMPLETA: el dato γ₀
    es invariante bajo rotaciones del plano ℝ². Esto implica una predicción
    teórica sobre la distribución óptima de parámetros ν*:

        Si γ₀ es isotrópico (simétrico bajo SO(2)), entonces ν* también debería
        ser (aproximadamente) isotrópico. En particular, los pesos de entrada
        a₁ᵐ deberían distribuirse uniformemente en S¹ (un anillo en ℝ²),
        en lugar de los dos picos simétricos observados en make_moons.

    Parámetros:
        n      : número de muestras totales (mitad por clase)
        noise  : desviación estándar del ruido gaussiano añadido a cada punto
        factor : ratio de radio interior / radio exterior (∈ (0,1))
                 Con factor=0.5: clase 0 en círculo externo, clase 1 en interno
        seed   : semilla de aleatoriedad para reproducibilidad

    Returns:
        X    : (N, 2) tensor en DEVICE — features estandarizadas
        y    : (N,) tensor en DEVICE  — etiquetas {0, 1}
        X_np : (N, 2) array NumPy     — para visualización
        y_np : (N,) array NumPy       — para visualización
    """
    X_np, y_np = make_circles(n_samples=n, noise=noise, factor=factor,
                              random_state=seed)
    X_np = StandardScaler().fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.float32)
    return (torch.tensor(X_np, device=DEVICE),
            torch.tensor(y_np, device=DEVICE),
            X_np, y_np)


def get_california_regression(n_train: int = 800, n_test: int = 400,
                               n_components: int = 2, seed: int = SEED):
    """
    California Housing proyectado a ℝ² via PCA para regresión.

    El dataset original tiene 8 features (MedInc, HouseAge, AveRooms,
    AveBedrms, Population, AveOccup, Latitude, Longitude) y target continuo
    (valor mediano de la vivienda en $100k, rango ~0.15–5.0).

    Pipeline:
        1. Submuestrear n_train+n_test filas (para velocidad de entrenamiento)
        2. StandardScaler en features
        3. PCA(n_components) → ℝ²  (el espacio donde corre la ODE)
        4. StandardScaler en target  (media=0, std=1 → MSE estable)
        5. train/test split

    La varianza explicada por PCA(2) se reporta para informar sobre la
    calidad de la proyección.  En California Housing, las dos primeras
    componentes capturan principalmente la señal geográfica (Lat/Lon) y
    de riqueza (MedInc), que son las más predictivas del precio.

    Args:
        n_train      : muestras de entrenamiento
        n_test       : muestras de test
        n_components : dimensión de la proyección PCA (= d1 del modelo)
        seed         : semilla de aleatoriedad

    Returns:
        dict con:
            X_train, y_train : tensores en DEVICE
            X_test,  y_test  : tensores en DEVICE
            X_train_np, y_train_np, X_test_np, y_test_np : arrays NumPy
            pca              : objeto PCA ajustado (para visualización)
            scaler_y         : StandardScaler del target (para invertir escala)
            explained_var    : fracción de varianza explicada por PCA(2)
    """
    from sklearn.datasets import fetch_california_housing

    data    = fetch_california_housing()
    X_full  = data.data.astype(np.float32)
    y_full  = data.target.astype(np.float32)

    rng  = np.random.RandomState(seed)
    idx  = rng.choice(len(X_full), size=n_train + n_test, replace=False)
    X_s, y_s = X_full[idx], y_full[idx]

    scaler_X  = StandardScaler().fit(X_s)
    X_scaled  = scaler_X.transform(X_s)
    pca       = PCA(n_components=n_components, random_state=seed).fit(X_scaled)
    X_2d      = pca.transform(X_scaled).astype(np.float32)

    scaler_y  = StandardScaler().fit(y_s.reshape(-1, 1))
    y_norm    = scaler_y.transform(y_s.reshape(-1, 1)).ravel().astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_2d, y_norm,
        train_size=n_train, test_size=n_test,
        random_state=seed
    )

    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  California Housing PCA({n_components}): "
          f"varianza explicada = {explained:.1%}  "
          f"| train={n_train}  test={n_test}")

    return {
        'X_train':    torch.tensor(X_tr, device=DEVICE),
        'y_train':    torch.tensor(y_tr, device=DEVICE),
        'X_test':     torch.tensor(X_te, device=DEVICE),
        'y_test':     torch.tensor(y_te, device=DEVICE),
        'X_train_np': X_tr,
        'y_train_np': y_tr,
        'X_test_np':  X_te,
        'y_test_np':  y_te,
        'pca':        pca,
        'scaler_y':   scaler_y,
        'explained_var': explained,
    }
