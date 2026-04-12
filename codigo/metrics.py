"""
metrics.py — Métricas para diagnosticar y regularizar con ν_N vs ν^∞.

Dado que ν_N = (1/M)Σ δ_{a^m} es discreta y el prior ν^∞ es continuo,
KL(ν_N||ν^∞) = +∞.  Este módulo implementa alternativas bien definidas:

  DIAGNÓSTICO (sin grafo de cómputo):
    extract_particles        : extrae los M vectores de parámetros (M, 5)
    mmd_rbf                  : MMD² con kernel Gaussiano (ancho = mediana)
    wasserstein_1d           : W₁ exacta en 1D via scipy
    sample_prior_langevin    : muestras de ν^∞ ∝ exp(-ℓ(a)/ε) por ULA
    collect_psgld_snapshots  : continúa pSGLD y recoge trayectoria de partículas

  ENTRENAMIENTO (con grafo de cómputo, diferenciables respecto a partículas):
    particles_live           : extrae partículas CON grad (para usar como loss)
    compute_mmd_sigma        : heurística de la mediana para ancho de banda
    mmd_loss_train           : MMD²(partículas, prior) diferenciable
    sinkhorn_loss            : divergencia de Sinkhorn (OT entrópico) diferenciable
"""

import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Extracción de partículas
# =============================================================================

def extract_particles(model):
    """
    Extrae los M vectores de parámetros por neurona: a^m = (a₀^m, a₁^m, a₂^m).

    Mapeado desde la estructura de MeanFieldVelocity:
        W0.weight : (d1, M)    — columna m = a₀^m ∈ ℝ^{d1}
        W1.weight : (M, d1+1)  — fila m, primeras d1 cols = a₁^m ∈ ℝ^{d1}
        W1.bias   : (M,)       — a₂^m ∈ ℝ  (bias / umbral por neurona)

    Nota: W1.weight[:, d1] es el peso de la codificación temporal (artefacto
    de la augmentación t → no forma parte del espacio A del paper).

    Returns:
        (M, 2*d1+1) tensor en CPU — coordenadas de las M partículas
    """
    d1 = model.velocity.d1
    a0 = model.velocity.W0.weight.T.detach().cpu()           # (M, d1)
    a1 = model.velocity.W1.weight[:, :d1].detach().cpu()     # (M, d1)
    a2 = model.velocity.W1.bias.unsqueeze(1).detach().cpu()  # (M, 1)
    return torch.cat([a0, a1, a2], dim=1)                     # (M, 2*d1+1)


# =============================================================================
# MMD² con kernel Gaussiano
# =============================================================================

def mmd_rbf(X, Y, sigma=None):
    """
    Estimador no sesgado de MMD² entre muestras X (N, d) e Y (M, d).

    Kernel:  k(x, y) = exp(−‖x−y‖² / (2σ²))
    sigma=None → heurística de la mediana sobre el pool X∪Y.

    El estimador no sesgado descarta los términos diagonales de Kxx y Kyy:
        MMD² = E[k(x,x')] − 2E[k(x,y)] + E[k(y,y')]
    con la corrección 1/(n(n-1)) en lugar de 1/n².

    Returns:
        float — MMD² estimado (puede ser ligeramente negativo con n, m pequeños)
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
    X, Y = X.float().cpu(), Y.float().cpu()

    if sigma is None:
        Z     = torch.cat([X, Y], dim=0)
        diffs = Z.unsqueeze(1) - Z.unsqueeze(0)
        dists = (diffs ** 2).sum(-1).sqrt()
        pos   = dists[dists > 0]
        sigma = float(pos.median()) if len(pos) > 0 else 1.0
        sigma = max(sigma, 1e-8)

    def K(A, B):
        d = A.unsqueeze(1) - B.unsqueeze(0)           # (n, m, d)
        return torch.exp(-(d ** 2).sum(-1) / (2.0 * sigma ** 2))

    Kxx = K(X, X); Kxx.fill_diagonal_(0.0)
    Kyy = K(Y, Y); Kyy.fill_diagonal_(0.0)
    Kxy = K(X, Y)
    n, m = X.shape[0], Y.shape[0]

    return float(
        Kxx.sum() / (n * (n - 1))
        - 2.0 * Kxy.sum() / (n * m)
        + Kyy.sum() / (m * (m - 1))
    )


# =============================================================================
# Wasserstein-1 en 1D (exacto)
# =============================================================================

def wasserstein_1d(x, y):
    """
    Distancia de Wasserstein-1 exacta para distribuciones univariadas.

    W₁(P, Q) = ∫|F_P(t) − F_Q(t)| dt = media de |x_(i) − y_(j)| (CDF empíricas)

    Usa scipy.stats.wasserstein_distance, que implementa esto exactamente
    a coste O(n log n) por ordenamiento.

    Args:
        x, y: tensores 1D o arrays
    Returns:
        float — W₁
    """
    from scipy.stats import wasserstein_distance
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return float(wasserstein_distance(x.ravel(), y.ravel()))


# =============================================================================
# Muestreo del prior ν^∞ por Langevin MCMC no ajustado (ULA)
# =============================================================================

def sample_prior_langevin(n_samples, epsilon, dim=5,
                           n_burnin=5_000, thin=20, step=None):
    """
    Genera muestras de ν^∞ ∝ exp(−ℓ(a)/ε) por ULA (Langevin MCMC).

    Potencial:  ℓ(a) = c₁ Σⱼ aⱼ⁴ + c₂ Σⱼ aⱼ²    (c₁=0.05, c₂=0.5)
    Fuerza:    −∇ℓ(a)/ε = −(4c₁ a³ + 2c₂ a) / ε   (elemento a elemento)

    Actualización ULA:
        a ← a − η · ∇ℓ(a)/ε + √(2η) · ξ,   ξ ~ N(0, I)

    La distribución estacionaria del SDE continuo es exp(−ℓ(a)/ε).  ULA
    introduce un sesgo O(η) que se controla con `step` pequeño y `thin`
    grande (sub-muestreo).

    Args:
        n_samples : número de muestras a devolver
        epsilon   : temperatura (misma ε del entrenamiento)
        dim       : dimensión del espacio A (2*d1+1 = 5 para d1=2)
        n_burnin  : pasos de burn-in descartados
        thin      : devuelve 1 de cada `thin` pasos tras burn-in
        step      : tamaño de paso η (ULA)

    Returns:
        (n_samples, dim) tensor CPU
    """
    # Constante de Lipschitz de ∇U(a) = ∇ℓ(a)/ε escala como 1/ε.
    # Condición de estabilidad ULA: step ≤ 2/L → step ∝ ε.
    if step is None:
        step = min(5e-3, 0.5 * epsilon)

    c1, c2    = 0.05, 0.5
    a         = torch.randn(dim) * 0.3
    noise_std = math.sqrt(2.0 * step)
    total     = n_burnin + n_samples * thin
    samples   = []

    for i in range(total):
        grad = (4.0 * c1 * a ** 3 + 2.0 * c2 * a) / epsilon
        a    = a - step * grad + noise_std * torch.randn(dim)
        if i >= n_burnin and (i - n_burnin) % thin == 0:
            samples.append(a.clone())

    return torch.stack(samples)   # (n_samples, dim)


# =============================================================================
# Snapshots de trayectoria pSGLD post-entrenamiento
# =============================================================================

def collect_psgld_snapshots(model, X, y, epsilon,
                             n_steps=2_000, thin=10, lr=1e-3):
    """
    Continúa pSGLD desde el estado actual del modelo y recoge snapshots
    de las posiciones de las M partículas.

    Opera sobre una copia profunda del modelo: el modelo original no se modifica.

    Tras la convergencia del entrenamiento, la cadena pSGLD está (aproximadamente)
    en régimen estacionario.  El conjunto de snapshots acumulados aproxima la
    distribución tiempo-promedio ν̂*, que por el teorema ergódico converge a la
    distribución estacionaria ν* ∝ exp(−J(θ)/ε).

    Se usa un lr fijo (no cosine-annealing) para mantener temperatura constante
    durante el muestreo.

    Args:
        model    : MeanFieldResNet ya entrenado (no se modifica)
        X, y     : datos de entrenamiento (necesarios para evaluar ∇J)
        epsilon  : temperatura pSGLD (misma ε del entrenamiento)
        n_steps  : número de pasos pSGLD adicionales
        thin     : guarda 1 de cada `thin` pasos
        lr       : learning rate fijo para la fase de muestreo

    Returns:
        (n_steps // thin, M, dim) tensor CPU — trayectoria de partículas
    """
    m_copy = copy.deepcopy(model).to(X.device)
    m_copy.train()

    opt    = optim.Adam(m_copy.parameters(), lr=lr)
    beta2  = opt.param_groups[0]['betas'][1]
    eps_a  = opt.param_groups[0]['eps']
    snapshots = []

    for step_i in range(n_steps):
        opt.zero_grad()
        loss, _, _ = m_copy.compute_loss(X, y, epsilon)
        loss.backward()
        nn.utils.clip_grad_norm_(m_copy.parameters(), max_norm=5.0)
        opt.step()

        if epsilon > 0:
            with torch.no_grad():
                for p in m_copy.parameters():
                    if p.grad is None:
                        continue
                    st = opt.state[p]
                    if len(st) == 0:
                        p.add_(torch.randn_like(p) * math.sqrt(2.0 * lr * epsilon))
                    else:
                        s_val = st['step']
                        if torch.is_tensor(s_val):
                            s_val = s_val.item()
                        v_hat = st['exp_avg_sq'] / (1.0 - beta2 ** s_val)
                        M_t   = (1.0 / (v_hat.sqrt() + eps_a)).clamp(max=1.0)
                        p.add_(torch.randn_like(p) * (2.0 * lr * epsilon * M_t).sqrt())

        if (step_i + 1) % thin == 0:
            snapshots.append(extract_particles(m_copy))

    return torch.stack(snapshots)   # (n_steps // thin, M, dim)


# =============================================================================
# Funciones de entrenamiento diferenciables
# =============================================================================

def particles_live(model):
    """
    Extrae los M vectores de parámetros CON grafo de cómputo.

    Igual que extract_particles() pero SIN .detach(), de modo que los
    gradientes fluyen hacia W0, W1 y bias de MeanFieldVelocity.
    Usar únicamente dentro del bucle de entrenamiento como parte del loss.

    Returns:
        (M, 2*d1+1) tensor en el dispositivo del modelo — CON grad
    """
    d1 = model.velocity.d1
    a0 = model.velocity.W0.weight.T           # (M, d1) — CON grad
    a1 = model.velocity.W1.weight[:, :d1]     # (M, d1) — CON grad
    a2 = model.velocity.W1.bias.unsqueeze(1)  # (M, 1)  — CON grad
    return torch.cat([a0, a1, a2], dim=1)      # (M, 2*d1+1)


def compute_mmd_sigma(particles, prior_samples):
    """
    Heurística de la mediana para el ancho de banda σ del kernel Gaussiano.

    Calcula la mediana de todas las distancias inter-muestras en el pool
    (partículas ∪ prior).  Se usa una vez antes del entrenamiento y se
    mantiene fijo durante todo el entrenamiento para que el paisaje de loss
    sea estacionario.

    Args:
        particles     : (M, d) tensor (puede estar en cualquier dispositivo)
        prior_samples : (N, d) tensor CPU
    Returns:
        float — σ
    """
    X = particles.detach().cpu().float()
    Y = prior_samples.cpu().float()
    # Submuestrea Y para que el cálculo sea O((M+N_sub)²) manejable
    N_sub = min(Y.shape[0], max(X.shape[0] * 4, 200))
    Y_sub = Y[torch.randperm(Y.shape[0])[:N_sub]]
    Z     = torch.cat([X, Y_sub], dim=0)
    diffs = Z.unsqueeze(1) - Z.unsqueeze(0)
    dists = (diffs ** 2).sum(-1).sqrt()
    pos   = dists[dists > 0]
    sigma = float(pos.median()) if len(pos) > 0 else 1.0
    return max(sigma, 1e-8)


def mmd_loss_train(X, Y, sigma, kyy_mean=None):
    """
    MMD² diferenciable entre partículas X y muestras del prior Y.

    Estimador sesgado (1/n² en lugar de 1/(n(n-1))): el sesgo es O(1/n) y
    no afecta la dirección del gradiente.  Más estable que el no sesgado
    cuando n=M=64 es pequeño.

    El término E[k(y,y')] = kyy_mean es constante respecto a X: se puede
    precomputar una vez antes del entrenamiento y pasarlo para evitar
    recalcularlo en cada paso.

    Args:
        X        : (M, d) partículas — CON grad (salida de particles_live)
        Y        : (N, d) muestras del prior — fijas, en el mismo dispositivo
        sigma    : ancho de banda del kernel (precomputado con compute_mmd_sigma)
        kyy_mean : float — E[k(y,y')] precomputado (None → se calcula aquí)

    Returns:
        Tensor escalar — MMD²  (diferenciable respecto a X)
    """
    def K(A, B):
        d = A.unsqueeze(1) - B.unsqueeze(0)      # (n, m, dim)
        return torch.exp(-(d ** 2).sum(-1) / (2.0 * sigma ** 2))

    n, m = X.shape[0], Y.shape[0]
    Kxx  = K(X, X)
    Kxy  = K(X, Y)

    kxx_mean = Kxx.sum() / (n * n)
    kxy_mean = Kxy.sum() / (n * m)

    if kyy_mean is None:
        with torch.no_grad():
            Kyy      = K(Y, Y)
            kyy_mean = float(Kyy.sum() / (m * m))

    return kxx_mean - 2.0 * kxy_mean + kyy_mean


def sinkhorn_loss(X, Y, blur=0.05, n_iter=50, W_yy=None):
    """
    Divergencia de Sinkhorn (OT entrópico debiasado) entre partículas y prior.

    S_blur(X, Y) = OT_blur(X, Y) − (1/2)·OT_blur(X, X) − (1/2)·OT_blur(Y, Y)

    donde OT_blur es el coste de transporte óptimo con regularización entrópica
    blur.  La versión debiasada es siempre ≥ 0, vale 0 iff X e Y tienen la
    misma distribución, y elimina el sesgo de contracción del OT entrópico simple.

    La implementación usa log-Sinkhorn (numericamente estable) con la matriz de
    coste normalizada a [0,1] para evitar desbordamiento aritmético.

    Gradientes:
        ∂S/∂X = ∂OT_blur(X,Y)/∂X − (1/2)·∂OT_blur(X,X)/∂X
    Ambos términos dependen de X → fluyen hacia las partículas.
    OT_blur(Y,Y) no depende de X: se puede precomputar (argumento W_yy).

    Args:
        X     : (n, d) partículas — CON grad
        Y     : (m, d) prior samples — fijos, mismo dispositivo
        blur  : regularización del OT (>0; más pequeño ≈ Wasserstein exacto
                pero más inestable; 0.05–0.1 es un rango razonable)
        n_iter: iteraciones de Sinkhorn (50 suele ser suficiente para blur≥0.05)
        W_yy  : float — OT_blur(Y,Y) precomputado (None → se calcula aquí)

    Returns:
        Tensor escalar — S_blur(X, Y)  (diferenciable respecto a X)
    """
    def _cost(A, B):
        diff = A.unsqueeze(1) - B.unsqueeze(0)   # (n, m, d)
        return (diff ** 2).sum(-1)                # (n, m)

    def _ot(C, blur, n_iter):
        """OT_blur con log-Sinkhorn.  C: (n, m) coste normalizado."""
        n, m  = C.shape
        log_K = -C / blur
        log_u = torch.zeros(n, device=C.device, dtype=C.dtype)
        log_v = torch.zeros(m, device=C.device, dtype=C.dtype)
        ln_n, ln_m = math.log(n), math.log(m)
        for _ in range(n_iter):
            log_u = -ln_n - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = -ln_m - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        return (log_P.exp() * C).sum()

    C_xy    = _cost(X, Y)
    scale   = C_xy.detach().max().clamp(min=1e-8)   # normalizar en escala original
    W_xy    = _ot(C_xy / scale, blur, n_iter) * scale

    C_xx    = _cost(X, X)
    W_xx    = _ot(C_xx / scale, blur, n_iter) * scale

    if W_yy is None:
        with torch.no_grad():
            C_yy = _cost(Y.detach(), Y.detach())
            W_yy = float(_ot(C_yy / scale.detach(), blur, n_iter) * scale.detach())

    return W_xy - 0.5 * W_xx - 0.5 * W_yy
