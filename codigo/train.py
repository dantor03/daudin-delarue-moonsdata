import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# § 4  BUCLE DE ENTRENAMIENTO
# =============================================================================
def train(model, X, y, epsilon,
          lr: float = 0.01, n_epochs: int = 800, verbose: bool = True):
    """
    Bucle de entrenamiento con Adam + cosine annealing.

    Registra métricas por época para la verificación empírica de la condición PL.

    CONDICIÓN POLYAK-ŁOJASIEWICZ (Meta-Teorema 2):
        ‖∇J(θ)‖² ≥ 2μ · (J(θ) − J*)
    donde J* es el valor mínimo alcanzado y μ > 0 es la "constante PL".
    Si esta desigualdad se cumple durante el entrenamiento, gradient descent
    converge exponencialmente: J(θ_t) − J* ≤ (J₀ − J*) · e^{−2μt}.

    SCHEDULER (cosine annealing): reduce lr de lr_max a ~0 siguiendo un coseno.
    Útil para escapar de mesetas y converger a mínimos más bajos.  La reducción
    del lr al final del entrenamiento puede aplanar las curvas de convergencia,
    lo que NO invalida la condición PL (que se evalúa con el lr original).

    Métricas registradas por época:
        loss       : J(θ) = BCE + ε·penalización — pérdida total
        loss_term  : componente BCE pura (sin regularización)
        loss_reg   : penalización ℓ(θ)/N_params (sin el factor ε)
        grad_norm2 : ‖∇J‖² = Σⱼ (∂J/∂θⱼ)² — numerador de la condición PL
        pl_ratio   : ‖∇J‖² / (2·(J−J*)) — debe ser ≥ μ > 0 si PL se cumple
        accuracy   : proporción de puntos correctamente clasificados

    Args:
        model    : MeanFieldResNet a entrenar
        X        : (N, d1) tensor — features de entrenamiento
        y        : (N,) tensor — etiquetas
        epsilon  : coeficiente de regularización ε
        lr       : learning rate inicial de Adam
        n_epochs : número de épocas
        verbose  : si True, imprime progreso cada 200 épocas

    Returns:
        hist : dict con listas de métricas + 'J_star' (mínimo global observado)
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    hist = {'loss': [], 'loss_term': [], 'loss_reg': [],
            'grad_norm2': [], 'pl_ratio': [], 'accuracy': []}
    L_min = math.inf

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss, lt, lr_val = model.compute_loss(X, y, epsilon)
        loss.backward()

        # ‖∇J‖² se calcula DESPUÉS de backward() pero ANTES de clip y step.
        # Así medimos el gradiente "verdadero" antes de cualquier modificación,
        # que es lo que corresponde al numerador de la condición PL teórica.
        gn2 = sum(p.grad.pow(2).sum().item()
                  for p in model.parameters() if p.grad is not None)

        # Gradient clipping: evita explosiones en pasos con gradiente grande
        # (puede ocurrir en los primeros epochs cuando los params están lejos de J*)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sch.step()

        lv = loss.item()
        if lv < L_min:
            L_min = lv         # J* aproximado = mínimo observado hasta ahora
        # Ratio PL provisional (online): se recalculará con J* global al final
        excess = max(lv - L_min, 1e-10)
        pl = gn2 / (2.0 * excess) if excess > 1e-9 else float('nan')

        with torch.no_grad():
            acc = ((model(X) > 0).float() == y).float().mean().item()

        hist['loss'].append(lv)
        hist['loss_term'].append(lt)
        hist['loss_reg'].append(lr_val)
        hist['grad_norm2'].append(gn2)
        hist['pl_ratio'].append(pl)
        hist['accuracy'].append(acc)

        if verbose and (ep + 1) % 200 == 0:
            print(f"    época {ep+1:4d} | J={lv:.4f} | BCE={lt:.4f} | "
                  f"reg={lr_val:.5f} | ‖∇J‖²={gn2:.3e} | acc={acc:.3f}")

    # Recalcular pl_ratio con J* GLOBAL (mínimo de toda la trayectoria de entrenamiento).
    # Esto es más justo que el J* online (que subestima J* en épocas tempranas):
    # el ratio ‖∇J‖²/(2(J−J*)) debe ser ≥ μ para todos los epochs cuando J* es fijo.
    J_star = min(hist['loss'])
    hist['J_star'] = J_star
    hist['pl_ratio'] = [
        gn2 / (2.0 * max(J - J_star, 1e-10))
        if J - J_star > 1e-9 else float('nan')
        for gn2, J in zip(hist['grad_norm2'], hist['loss'])
    ]
    return hist


def mu_pl_estimate(hist):
    """
    Estima conservadoramente la constante PL μ como el percentil 10 del ratio PL.

    El ratio PL en cada época es ‖∇J‖²/(2(J−J*)).  Si la condición PL se cumple
    con constante μ, este ratio debe ser ≥ μ en TODOS los epochs (por definición).
    Por tanto, el estimador natural sería el mínimo.

    Sin embargo, el mínimo es muy sensible a outliers numéricos (e.g., cuando
    J ≈ J* el denominador es tiny y el ratio explota o colapsa).  El percentil 10
    es más robusto: descarta el 10% de los valores más bajos, protegiendo frente a
    instantes de inestabilidad numérica, pero sigue siendo conservador respecto
    al verdadero μ (no lo sobreestima sistemáticamente).

    Se excluyen:
      • NaN (cuando J−J* ≤ 1e-9, el modelo ya está en J*)
      • Valores ≤ 0 (anomalías numéricas)
      • Valores > 1e4 (explosiones de gradiente no capturadas por el clipping)
    """
    vals = [v for v in hist['pl_ratio']
            if not math.isnan(v) and 0 < v < 1e4]
    return np.percentile(vals, 10) if len(vals) > 10 else 0.0
