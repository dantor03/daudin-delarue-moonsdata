import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# § 4  BUCLE DE ENTRENAMIENTO
# =============================================================================
def train(model, X, y, epsilon,
          lr: float = 0.01, n_epochs: int = 800, verbose: bool = True,
          use_sgd: bool = False, use_sgld: bool = False):
    """
    Bucle de entrenamiento.  Tres modos de optimizador:

    - Por defecto (use_sgd=False, use_sgld=False):
        Adam + cosine annealing.  Convergencia rápida; el lr→0 al final
        puede aplanar artificialmente las curvas (artefacto del scheduler).

    - use_sgd=True:
        SGD puro + lr constante, SIN ruido.  Gradient flow determinístico
        limpio.  Recomendado para exp_c (verificación PL) donde J* debe ser
        el mínimo geométrico real sin artefactos.

    - use_sgld=True:
        SGD + cosine annealing + ruido de Langevin.  Implementa la dinámica:
            θ ← θ − η·∇J(θ) + √(2ηε)·ξ,   ξ ~ N(0,I)
        El ruido inyectado representa la entropía verdadera del problema de
        campo medio: hace que los parámetros exploren la distribución de Gibbs
            ν* ∝ exp(−J(θ)/ε)
        en lugar de colapsar a un estimador puntual.  La escala del ruido
        √(2ηε) está directamente derivada del paper (sección 1.3): ε es la
        temperatura entrópica y η el paso de discretización de Euler-Maruyama.
        El cosine annealing aplica simulated annealing natural (η→0 reduce el
        ruido progresivamente, satisfaciendo las condiciones de Robbins-Monro).
        La penalización L4+L2 en compute_loss actúa como prior energético
        ν^∞ ∝ exp(−ℓ(a)); el ruido proporciona el término entrópico −H(ν_t).
        Con ε=0 el ruido es cero y se recupera SGD estándar.
        Recomendado para exp_b, exp_e, exp_f (verificación de ν*).

    CONDICIÓN POLYAK-ŁOJASIEWICZ (Meta-Teorema 2):
        ‖∇J(θ)‖² ≥ 2μ · (J(θ) − J*)

    Métricas registradas por época:
        loss       : J(θ) = BCE + ε·prior energético (evaluado SIN ruido)
        loss_term  : BCE pura
        loss_reg   : prior energético ℓ(θ)/N_params
        grad_norm2 : ‖∇J‖² antes de clipping — numerador de PL
        pl_ratio   : ‖∇J‖² / (2·(J−J*))
        accuracy   : proporción de puntos correctamente clasificados

    Args:
        model    : MeanFieldResNet a entrenar
        X        : (N, d1) tensor — features (batch completo, N/M = 1)
        y        : (N,) tensor — etiquetas
        epsilon  : coeficiente ε (regularización + temperatura Langevin)
        lr       : learning rate inicial
        n_epochs : número de épocas
        verbose  : imprime progreso cada 200 épocas
        use_sgd  : SGD + lr constante, sin ruido (exp_c)
        use_sgld : SGD + cosine annealing + ruido Langevin (exp_b, E, F)

    Returns:
        hist : dict con listas de métricas + 'J_star'
    """
    if use_sgld:
        opt = optim.SGD(model.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    elif use_sgd:
        opt = optim.SGD(model.parameters(), lr=lr)
        sch = None
    else:
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

        # ── Ruido de Langevin (solo con use_sgld=True) ───────────────────────
        # Se inyecta DESPUÉS del paso de gradiente y ANTES del scheduler,
        # usando el lr ACTUAL (antes de que cosine lo actualice).
        # Magnitud: √(2·η·ε) por componente, discretización de Euler-Maruyama
        # de la SDE dθ = −∇J dt + √(2ε) dW.
        # Con ε=0 el ruido es exactamente cero (sin ramas condicionales).
        if use_sgld and epsilon > 0:
            current_lr = opt.param_groups[0]['lr']
            noise_std  = math.sqrt(2.0 * current_lr * epsilon)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * noise_std)

        if sch is not None:
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
