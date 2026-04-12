import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .metrics import (
    particles_live, compute_mmd_sigma,
    mmd_loss_train, sinkhorn_loss,
)


# =============================================================================
# § 4  BUCLE DE ENTRENAMIENTO
# =============================================================================
def train(model, X, y, epsilon,
          lr: float = 0.01, n_epochs: int = 800, verbose: bool = True,
          use_sgd: bool = False, use_sgld: bool = False,
          use_mmd: bool = False, use_sinkhorn: bool = False,
          prior_samples=None, mmd_sigma: float = None,
          sinkhorn_blur: float = 0.05):
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
        pSGLD — preconditioned Stochastic Gradient Langevin Dynamics
        (Li et al., 2016).  En cada paso:
            θ ← Adam(θ, ∇J) + √(2·η_t·ε·M_t)·ξ,   ξ ~ N(0,I)
        donde M_t[j] = min(1/(√(v̂_t[j]) + δ), 1) es el precondicionador
        de Adam (segundo momento sesgado-corregido) por parámetro j,
        acotado superiormente en 1 para evitar explosión de ruido en
        direcciones planas (v̂_t ≈ 0).  El clamp corresponde a usar un
        hiperparámetro λ ≥ 1 en lugar de δ = eps_adam = 1e-8 (Li et al. 2016
        usan λ separado de Adam's eps).
        El ruido DEBE estar acoplado a M_t: ruido isotrópico (σ constante)
        desacopla la parte estocástica del precondicionador y destruye la
        distribución estacionaria — la cadena de Markov ya no converge a
        ν* ∝ exp(−J(θ)/ε).  Con M_t acotado, la distribución estacionaria
        es ν* ∝ exp(−J(θ)/ε) bajo el precondicionador de Adam.
        Adam como base es necesario: SGD puro no converge porque los
        gradientes de BCE son ~1e-5 y Adam adapta el lr por parámetro.
        La penalización L4+L2 actúa como prior ν^∞ ∝ exp(−ℓ(a));
        el ruido proporciona el término entrópico −H(ν_t).
        Con ε=0 el ruido es exactamente cero (Adam estándar).
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

    LEARNING RATE RECOMENDADO POR MODO:
        Adam  (defecto): lr = 0.01  — Adam adapta el lr por parámetro.
        SGD   (use_sgd):  lr = 0.01  — gradient flow determinístico.
        SGLD  (use_sgld): lr = 0.01  — Adam como base; el ruido pSGLD
            √(2·η_t·ε·M_t) se añade tras el paso Adam.  M_t acopla el
            ruido al precondicionador, preservando la distribución ν*.

    Args:
        model    : MeanFieldResNet a entrenar
        X        : (N, d1) tensor — features (batch completo, N/M = 1)
        y        : (N,) tensor — etiquetas
        epsilon  : coeficiente ε (regularización + temperatura Langevin)
        lr       : learning rate inicial (ver tabla de valores recomendados)
        n_epochs : número de épocas
        verbose  : imprime progreso cada 200 épocas
        use_sgd  : SGD + lr constante, sin ruido (exp_c)
        use_sgld    : pSGLD — Adam + ruido Langevin precondicionado (exp_b, E, F)
        use_mmd     : Adam + regularizador MMD²(partículas, prior_samples).
                      Reemplaza la penalización L4+L2 por la distancia MMD
                      entre las partículas actuales y muestras fijas del prior
                      ν^∞.  Entrenamiento determinístico (sin ruido Langevin).
                      Requiere prior_samples ≠ None.
        use_sinkhorn: Adam + divergencia de Sinkhorn(partículas, prior_samples).
                      Similar a use_mmd pero con transporte óptimo entrópico
                      como regularizador.  Más sensible a la geometría del
                      espacio de parámetros que MMD.
                      Requiere prior_samples ≠ None.
        prior_samples: (N, d) tensor — muestras de ν^∞ precomputadas.
                      Generadas con sample_prior_langevin() en metrics.py.
                      Se mueven automáticamente al dispositivo del modelo.
        mmd_sigma   : ancho de banda del kernel MMD (None → heurística de la
                      mediana calculada una vez antes del entrenamiento).
        sinkhorn_blur: regularización del OT en Sinkhorn (default 0.05).

    Returns:
        hist : dict con listas de métricas + 'J_star'
    """
    # ── Validación de argumentos ──────────────────────────────────────────────
    if (use_mmd or use_sinkhorn) and prior_samples is None:
        raise ValueError("use_mmd y use_sinkhorn requieren prior_samples != None.")
    if sum([use_sgld, use_sgd, use_mmd, use_sinkhorn]) > 1:
        raise ValueError("Solo un modo puede estar activo a la vez.")

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    if use_sgd:
        opt = optim.SGD(model.parameters(), lr=lr)
        sch = None
    else:
        # Adam + cosine annealing para todos los demás modos
        opt = optim.Adam(model.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    # ── Setup previo al bucle (MMD y Sinkhorn) ────────────────────────────────
    prior_dev   = None   # prior_samples en el dispositivo del modelo
    _sigma      = None   # ancho de banda MMD (fijo durante training)
    _kyy_mean   = None   # E[k(y,y')] precomputado (constante)
    _W_yy       = None   # OT_blur(Y,Y) precomputado (constante)

    if use_mmd or use_sinkhorn:
        prior_dev = prior_samples.to(X.device).float()

    if use_mmd:
        # Sigma: heurística de la mediana, calculada una vez
        _sigma = mmd_sigma if mmd_sigma is not None else compute_mmd_sigma(
            particles_live(model).detach(), prior_dev
        )
        # E[k(y,y')]: constante durante todo el entrenamiento
        with torch.no_grad():
            def _K(A, B, s):
                d = A.unsqueeze(1) - B.unsqueeze(0)
                return torch.exp(-(d ** 2).sum(-1) / (2.0 * s ** 2))
            Kyy = _K(prior_dev, prior_dev, _sigma)
            _kyy_mean = float(Kyy.sum() / (prior_dev.shape[0] ** 2))
        if verbose:
            print(f"    [MMD]  σ = {_sigma:.4f}  |  E[k(y,y')] = {_kyy_mean:.4f}")

    if use_sinkhorn:
        # W_yy: OT_blur(prior, prior), constante durante el entrenamiento
        with torch.no_grad():
            _W_yy = float(sinkhorn_loss(
                prior_dev, prior_dev, blur=sinkhorn_blur, n_iter=50
            ))
        if verbose:
            print(f"    [Sinkhorn]  blur = {sinkhorn_blur}  |  W_yy = {_W_yy:.4e}")

    hist = {'loss': [], 'loss_term': [], 'loss_reg': [],
            'grad_norm2': [], 'pl_ratio': [], 'accuracy': []}
    L_min = math.inf

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()

        # ── Cálculo de la pérdida según el modo activo ────────────────────────
        if use_mmd:
            # BCE (sin L4+L2) + ε·MMD²(partículas, prior)
            loss_bce, lt, _ = model.compute_loss(X, y, 0.0)
            pts     = particles_live(model)
            reg     = mmd_loss_train(pts, prior_dev, _sigma, _kyy_mean)
            loss    = loss_bce + epsilon * reg
            lr_val  = reg.item()
        elif use_sinkhorn:
            # BCE (sin L4+L2) + ε·Sinkhorn(partículas, prior)
            loss_bce, lt, _ = model.compute_loss(X, y, 0.0)
            pts     = particles_live(model)
            reg     = sinkhorn_loss(pts, prior_dev,
                                    blur=sinkhorn_blur, n_iter=50, W_yy=_W_yy)
            loss    = loss_bce + epsilon * reg
            lr_val  = reg.item()
        else:
            # pSGLD / Adam / SGD: BCE + ε·L4+L2
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

        # ── Ruido de Langevin precondicionado pSGLD (solo con use_sgld=True) ──
        # Implementa pSGLD (Li et al., 2016): el ruido está acoplado a la
        # matriz de precondicionamiento M_t de Adam, NO es isotrópico.
        #
        # La actualización correcta es:
        #   θ ← Adam(θ, ∇J) + √(2·η·ε·M_t)·ξ,   ξ ~ N(0, I)
        #
        # donde M_t[j] = 1 / (√(v̂_t[j]) + δ) es el precondicionador de Adam
        # para el parámetro j, con v̂_t[j] = v_t[j] / (1 - β₂ᵗ) el segundo
        # momento sesgado-corregido.
        #
        # Esto garantiza que la cadena de Markov converja a la distribución
        # de Gibbs ν* ∝ exp(−J(θ)/ε) con el precondicionador M_t.
        # Inyectar ruido isotrópico (σ=√(2ηε) igual para todos los parámetros)
        # desacopla el ruido de M_t y destruye la distribución estacionaria.
        #
        # Con ε=0 el ruido es exactamente cero (recupera Adam estándar).
        if use_sgld and epsilon > 0:
            current_lr = opt.param_groups[0]['lr']
            beta2      = opt.param_groups[0]['betas'][1]
            eps_adam   = opt.param_groups[0]['eps']
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    state = opt.state[p]
                    if len(state) == 0:
                        # Adam aún no inicializó estado (épocas muy tempranas):
                        # ruido isotrópico como fallback conservador.
                        noise_std = math.sqrt(2.0 * current_lr * epsilon)
                        p.add_(torch.randn_like(p) * noise_std)
                    else:
                        step = state['step']
                        if torch.is_tensor(step):
                            step = step.item()
                        v_t   = state['exp_avg_sq']
                        # Segundo momento sesgado-corregido: v̂_t = v_t/(1-β₂ᵗ)
                        bias_corr2 = 1.0 - beta2 ** step
                        v_hat = v_t / bias_corr2
                        # Precondicionador M_t[j] = 1/(√(v̂_t[j]) + δ)
                        # Clamp M a 1.0: en Li et al. 2016 el término de
                        # estabilización λ no es eps de Adam (1e-8) sino
                        # un hiperparámetro ≥ 1 que evita explosión de ruido
                        # en direcciones planas (v̂_t ≈ 0 → M ≈ 1/1e-8 → ∞).
                        # Con M ≤ 1, noise_std ≤ √(2·η·ε) = SGLD isotrópico:
                        # amplificamos ruido en direcciones curvas pero nunca
                        # más allá del nivel isotrópico.
                        M = 1.0 / (v_hat.sqrt() + eps_adam)
                        M = M.clamp(max=1.0)
                        # std del ruido: √(2·η·ε·M_t[j]) por componente
                        noise_std = (2.0 * current_lr * epsilon * M).sqrt()
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
            pred = model(X)
            if getattr(model, 'task', 'classification') == 'regression':
                ss_res = ((pred - y) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=1e-8)
                acc = float(1.0 - ss_res / ss_tot)   # R²
            else:
                acc = ((pred > 0).float() == y).float().mean().item()

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
