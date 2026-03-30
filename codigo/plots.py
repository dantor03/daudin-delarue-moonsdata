import numpy as np
import torch

from .config import DEVICE, style_ax


# =============================================================================
# § 5  HELPERS DE VISUALIZACIÓN
# =============================================================================
def plot_decision_boundary(ax, model, X_np, y_np, title=''):
    """
    Visualiza la frontera de decisión P(y=1|x) = 0.5 en el espacio original ℝ².

    Evalúa el modelo completo (ODE + clasificador) en una malla densa y colorea
    cada punto según la probabilidad predicha.  La línea blanca es la isocurva
    σ(logit)=0.5, que corresponde a la frontera de clasificación.

    En el contexto del paper, esta frontera refleja la acción CONJUNTA de:
      1. La ODE que transforma X_0 → X_T (separa las clases en ℝ²)
      2. El clasificador lineal W·X_T + b sobre el espacio transformado
    """
    r = 0.5
    xmin, xmax = X_np[:, 0].min() - r, X_np[:, 0].max() + r
    ymin, ymax = X_np[:, 1].min() - r, X_np[:, 1].max() + r
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32),
                        device=DEVICE)
    model.eval()
    with torch.no_grad():
        Z = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.72, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=1.5)
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
               c='#ff6b6b', s=14, alpha=0.85, zorder=3)
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
               c='#74b9ff', s=14, alpha=0.85, zorder=3)
    style_ax(ax, title, '$x_1$', '$x_2$')
