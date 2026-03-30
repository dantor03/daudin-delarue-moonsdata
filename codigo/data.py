import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles
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
