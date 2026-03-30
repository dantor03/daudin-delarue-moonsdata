import torch
import torch.nn as nn


# =============================================================================
# § 2  CAMPO VECTORIAL PROTOTÍPICO  b(x, a) = σ(a₁·x + a₂)·a₀
#
#  Paper, Ejemplo 1.1 (ec. 1.8):
#      A = ℝ^{d₁} × ℝ^{d₁} × ℝ,    a = (a₀, a₁, a₂)
#      b(x, a) = σ(a₁ · x + a₂) · a₀,     σ = tanh
#
#  Aquí d₁=2 (features ∈ ℝ²), a₀ ∈ ℝ², a₁ ∈ ℝ², a₂ ∈ ℝ.
#  Es la "neurona prototípica": a₁·x+a₂ es una proyección lineal (pre-activación),
#  σ(·) es la activación no lineal, y a₀ reescala la salida en ℝ^{d₁}.
#
#  Con M "partículas" (aproximación de ν_t por M muestras discretas aᵐ):
#      F(x, t) = ∫_A b(x,a) dν_t(a) ≈ (1/M) Σₘ b(x, aᵐ(t))
#              = (1/M) Σₘ σ(a₁ᵐ(t)·x + a₂ᵐ(t)) · a₀ᵐ(t)
#
#  Esto es una red neuronal de 1 capa oculta con M neuronas con parámetros
#  que deberían variar libremente en t.  El paper optimiza sobre trayectorias
#  arbitrarias (ν_t)_{t∈[0,T]}, lo que en M partículas equivale a optimizar
#  M caminos aᵐ : [0,T] → A.
#
#  APROXIMACIÓN POR AUGMENTACIÓN TEMPORAL:
#      En la implementación los pesos (W₁, W₀) son ESTÁTICOS y t se concatena
#      al input, de modo que el campo efectivo es:
#          F(x,t) = W₀ tanh(W₁[:,​:d₁] x + W₁[:,d₁] t + b₁)
#      Esto equivale a restringir la familia de controles a aquellos donde
#      a₀ᵐ y a₁ᵐ son CONSTANTES en t, y a₂ᵐ(t) = W₁[m,d₁]·t + b₁[m] varía
#      linealmente.  El campo F(x,t) SÍ depende de t, pero la trayectoria
#      de ν_t está restringida a esta familia paramétrica concreta.
#
#  Implementación matricial:
#      W₁ ∈ ℝ^{M×(d₁+1)} agrupa (a₁ᵐ, col_temporal) como filas
#      W₀ ∈ ℝ^{M×d₁}     agrupa a₀ᵐ como filas
#      h = σ( [x, t] @ W₁ᵀ + b₁ )   → (N, M)
#      F = h @ W₀ᵀ / M               → (N, d₁)   ← /M absorbido en init
#
#  Activación tanh: satisface la "propiedad discriminante" (sec. 1.2) que
#  garantiza que el campo b(x,·) puede aproximar cualquier función continua.
#
#  PENALIZACIÓN ENTRÓPICA (Assumption Regularity (i) y (ii)):
#      ε · E(ν_t | ν^∞)   con   ν^∞ ∝ exp(-ℓ(a)),   ℓ(a) = c₁|a|⁴ + c₂|a|²
#
#  Por qué potencial CUÁRTICO (c₁|a|⁴):
#      La condición "supercoercividad" exige que ∇²ℓ(a) ≥ c(1+|a|²)I,
#      que c₂|a|² (cuadrático) NO cumple en solitario pero sí con c₁|a|⁴.
#      Este crecimiento más rápido que cuadrático es lo que garantiza la
#      desigualdad log-Sobolev para ν^∞, que a su vez implica la condición PL.
# =============================================================================
class MeanFieldVelocity(nn.Module):
    """
    Campo vectorial prototípico del paper (ec. 1.8) aproximado con M partículas.

    Implementa F(x,t) = (1/M) Σₘ σ(a₁ᵐ(t)·x + a₂ᵐ(t)) · a₀ᵐ(t), que es el
    campo vectorial efectivo que mueve las features según la ecuación de control:
        dX_t/dt = F(X_t, t)

    La dependencia temporal a(t) se codifica augmentando el input con el escalar t.
    Esto corresponde a parámetros que varían suavemente con el "tiempo de red",
    análogamente a una Neural ODE con parámetros compartidos entre capas.

    Atributos:
        d1  : dimensión del espacio de features (d₁=2 para make_moons)
        M   : número de "partículas" que aproximan la medida de control ν_t
        W1  : matriz de pesos [d₁+1 → M] — codifica (a₁ᵐ, a₂ᵐ) para m=1..M
        W0  : matriz de pesos [M → d₁]  — codifica a₀ᵐ para m=1..M
    """

    def __init__(self, d1: int = 2, M: int = 64):
        super().__init__()
        self.d1, self.M = d1, M
        # W₁: cada fila m es el parámetro (a₁ᵐ ∈ ℝ^{d₁}, a₂ᵐ ∈ ℝ) de la neurona m
        # La columna extra de input es para el tiempo t (augmentación temporal)
        self.W1 = nn.Linear(d1 + 1, M, bias=True)
        # W₀: cada fila m es a₀ᵐ ∈ ℝ^{d₁} — escala la salida de cada neurona
        # Sin bias: la simetría b(x,a)=0 cuando a₀=0 debe preservarse
        self.W0 = nn.Linear(M, d1, bias=False)
        # Inicialización pequeña (std=0.1):
        #   • Evita que la ODE "explote" en los primeros pasos de entrenamiento
        #   • Coherente con la teoría: parámetros cerca del prior ν^∞ (que también
        #     concentra masa cerca del origen) al inicio del entrenamiento
        nn.init.normal_(self.W1.weight, std=0.1)
        nn.init.zeros_(self.W1.bias)
        nn.init.normal_(self.W0.weight, std=0.1)

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Evalúa el campo vectorial efectivo F(x, t).

        Calcula F(x,t) = (1/M) Σₘ σ(a₁ᵐ·x + a₂ᵐ) · a₀ᵐ para N puntos
        simultáneamente en forma matricial.  Este valor es la VELOCIDAD
        con la que la ODE mueve cada feature x en el tiempo t.

        Args:
            t : tiempo normalizado ∈ [0, 1].  Se broadcast a todos los N puntos.
            x : (N, d1) — componente de features de los N puntos de γ_t;
                           cada "partícula" de γ_t es el par (x, y), pero F
                           solo actúa sobre x (la etiqueta y viaja fija)

        Returns:
            (N, d1) — velocidad dX/dt en cada uno de los N puntos
        """
        t_val = t.item() if torch.is_tensor(t) else float(t)
        # Construir input aumentado [x, t] ∈ ℝ^{d₁+1} para cada punto
        t_col = x.new_full((x.size(0), 1), t_val)           # (N, 1) — constante t
        # h = σ(W₁·[x,t]ᵀ + b₁) — activaciones de las M neuronas para cada punto
        h = torch.tanh(self.W1(torch.cat([x, t_col], dim=1)))   # (N, M)
        # F = W₀ᵀ·h — combinación lineal de las M salidas → velocidad en ℝ^{d₁}
        # El factor 1/M está absorbido en la escala de inicialización de W₀
        return self.W0(h)                                        # (N, d1)

    def entropic_penalty(self, c1: float = 0.05, c2: float = 0.5) -> torch.Tensor:
        """
        Aproximación de la penalización entrópica ε · E(ν_t | ν^∞).

        TEORÍA (paper):
            E(ν_t | ν^∞) = KL(ν_t || ν^∞) = ∫ log(dν_t/dν^∞) dν_t
                          = E_{ν_t}[ℓ(a)] − H(ν_t)
            donde ℓ(a) = c₁|a|⁴ + c₂|a|² y H(ν_t) = −∫log(dν_t)dν_t es la
            entropía diferencial de ν_t.

        LIMITACIÓN DE LA IMPLEMENTACIÓN:
            Los parámetros son estimadores PUNTUALES (deterministicos), por lo
            que ν_t = (1/M) Σₘ δ_{θₘ} es una suma de deltas de Dirac.  La
            entropía diferencial de una medida discreta es −∞ respecto a un
            prior continuo → la KL completa es técnicamente +∞.

            Solo es accesible el TÉRMINO DE ENERGÍA:
                E_{ν_t}[ℓ(a)] ≈ (1/N_params) Σⱼ [c₁ θⱼ⁴ + c₂ θⱼ²]

            Esta aproximación es equivalente a regularización L4+L2 (weight
            decay polinomial) y es el sustituto práctico estándar para la KL
            cuando se usan estimadores puntuales en lugar de distribuciones.
            Para la verdadera regularización entrópica sería necesario usar
            dinámicas de Langevin (ruido en el gradiente) o inferencia
            variacional (distribuir probabilísticamente cada peso).

        Elección de hiperparámetros (Assumption Regularity (i)):
            c₁ = 0.05 — término cuártico (supercoercividad: garantiza log-Sobolev)
            c₂ = 0.5  — término cuadrático (convexidad básica)
        El término c₁|a|⁴ es ESENCIAL: c₂|a|² solo garantiza convexidad
        cuadrática (L2), pero no la desigualdad log-Sobolev que implica PL.

        Returns:
            Escalar — término de energía medio por parámetro (aprox. de KL)
        """
        pen, n = torch.tensor(0.0, device=next(self.parameters()).device), 0
        for p in self.parameters():
            pen = pen + c1 * (p ** 4).sum() + c2 * (p ** 2).sum()
            n += p.numel()
        return pen / n


# =============================================================================
# § 3  MEAN-FIELD RESNET — Neural ODE en espacio de features ℝ^{d₁}
#
#  Implementa el sistema de control óptimo del paper:
#
#      ┌─ ODE de transporte (ec. 1.3/1.5):
#      │   dX_t/dt = F(X_t, t) = ∫_A b(X_t, a) dν_t(a),   t ∈ [0, T]
#      │   X_0 = dato de entrada ∈ ℝ^{d₁}
#      │   γ_t = distribución de X_t dado (X_0, Y_0)  (ec. 1.5)
#      │
#      └─ Clasificador lineal (coste terminal):
#          logit = W · X_T + b  →  P(y=1|X_T) = σ(logit)
#          L(x, y) = BCE(W·x + b, y)   (coste terminal del problema de control)
#
#  POR QUÉ ESPACIO ORIGINAL ℝ² (sin embedding):
#      En el paper, γ_t ∈ P(ℝ^{d₁} x ℝ^{d}) — la distribución vive en el mismo espacio
#      que los datos. Si embeddiéramos x ∈ ℝ² → h ∈ ℝ^H, la ODE correría en ℝ^H
#      y γ_t sería no visualizable, alejándose del setup teórico.
#
#  INTEGRACIÓN RK4 vs EULER:
#      El paper trabaja con el flujo continuo (tiempo continuo), cuya
#      discretización más fiel es RK4.
#        • Euler: error local O(dt²), error GLOBAL acumulado O(dt)
#        • RK4:   error local O(dt⁵), error GLOBAL acumulado O(dt⁴)
#      Con dt = T/n_steps = 0.1:
#        • Error global RK4  ~ dt⁴ = 10⁻⁴  (cuatro órdenes de magnitud)
#        • Error global Euler ~ dt  = 10⁻¹
#      Las 4 evaluaciones del campo por paso (k1..k4) capturan mejor
#      la curvatura del flujo sin aumentar el número de pasos.
# =============================================================================
class MeanFieldResNet(nn.Module):
    """
    Neural ODE de campo medio que transforma γ_0 (make_moons) en γ_T (separable).

    La "red" no tiene capas fijas sino un flujo continuo parametrizado:
        X_0 → [ODE: dX/dt = F(X,t)] → X_T → [lineal] → logit → P(y=1)

    Atributos:
        velocity   : MeanFieldVelocity — el campo vectorial F(x,t)
        classifier : nn.Linear(d1,1)  — clasificador lineal sobre X_T
        T          : horizonte temporal de la ODE (por defecto T=1.0)
        n_steps    : número de pasos RK4 para discretizar [0,T]
    """

    def __init__(self, d1: int = 2, M: int = 64, T: float = 1.0,
                 n_steps: int = 10):
        super().__init__()
        self.velocity   = MeanFieldVelocity(d1=d1, M=M)
        self.classifier = nn.Linear(d1, 1)
        self.T, self.n_steps, self.d1 = T, n_steps, d1

        # Inicialización pequeña del clasificador: para que al inicio
        # el logit sea ≈ 0 y la pérdida BCE empiece cerca de log(2) ≈ 0.693
        nn.init.normal_(self.classifier.weight, std=0.1)
        nn.init.zeros_(self.classifier.bias)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  MeanFieldResNet: d1={d1}, M={M}, T={T}, "
              f"n_steps={n_steps}, params={n_p:,}")

    # ── Integrador RK4 ───────────────────────────────────────────────────────
    def _rk4(self, t_norm: float, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Avanza un paso de la ODE dX/dt = F(X,t) usando el método de Runge-Kutta 4.

        El método RK4 evalúa el campo en 4 puntos intermedios (k1..k4) y combina
        con pesos [1,2,2,1]/6 para obtener error local O(dt⁵) [global O(dt⁴)]
        frente a error local O(dt²) [global O(dt)] de Euler:
            k1 = F(x,         t)
            k2 = F(x + dt/2·k1, t+dt/2)
            k3 = F(x + dt/2·k2, t+dt/2)
            k4 = F(x + dt·k3,   t+dt)
            x_new = x + (dt/6)(k1 + 2k2 + 2k3 + k4)

        Args:
            t_norm : tiempo normalizado al inicio del paso, ∈ [0, 1)
            x      : (N, d1) — posiciones actuales de los N puntos
            dt     : tamaño del paso normalizado (= 1/n_steps)

        Returns:
            (N, d1) — posiciones después del paso
        """
        k1 = self.velocity(t_norm,        x)
        k2 = self.velocity(t_norm + dt/2, x + dt/2 * k1)
        k3 = self.velocity(t_norm + dt/2, x + dt/2 * k2)
        k4 = self.velocity(t_norm + dt,   x + dt    * k3)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, x0: torch.Tensor,
                  return_trajectory: bool = False):
        """
        Integra la ODE dX/dt = F(X,t) desde t=0 hasta t=T.

        Cada paso avanza el "tiempo de red" en dt=T/n_steps.  Al final,
        X_T es la representación aprendida de los features que el clasificador
        lineal puede separar.

        Internamente el tiempo se normaliza a [0,1] para que la velocidad
        F(x, t_norm) sea independiente de T (facilita la generalización a
        distintos horizontes T).

        Args:
            x0               : (N, d1) — features iniciales (= puntos de γ_0)
            return_trajectory: si True, guarda snapshots intermedios de γ_t.
                               Útil para visualizar la evolución de las features.

        Returns:
            Si return_trajectory=False: x_T (N, d1)
            Si return_trajectory=True : (x_T, traj) donde
                traj = [(t_real, x_t_detached)] con t_real ∈ {0, T/n, 2T/n, …, T}
                Los tensores en traj están detachados del grafo para ahorrar memoria.
        """
        x  = x0.clone()
        dt = 1.0 / self.n_steps       # paso normalizado ∈ [0,1]
        traj = [(0.0, x.detach().clone())] if return_trajectory else None

        for i in range(self.n_steps):
            t_i = i * dt                              # tiempo normalizado al inicio del paso
            x   = self._rk4(t_i, x, dt)
            if return_trajectory:
                t_real = (i + 1) * self.T / self.n_steps   # tiempo real en [0, T]
                traj.append((t_real, x.detach().clone()))

        return (x, traj) if return_trajectory else x

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Integra la ODE y aplica el clasificador lineal sobre X_T."""
        x_T = self.integrate(x0)
        return self.classifier(x_T).squeeze(-1)   # (N,) — logits sin activar

    def compute_loss(self, x0, y, epsilon: float = 0.01):
        """
        Calcula el coste total J del problema de control (ec. 1.6 del paper):

            J(γ_0, ν) = ∫ L(x,y) dγ_T(x,y)   ←— coste terminal (BCE)
                      + ε · ∫₀ᵀ E(ν_t | ν^∞) dt  ←— penalización entrópica

        El primer término mide qué tan bien clasifica el modelo.
        El segundo término penaliza cuánto se aleja ν_t del prior ν^∞,
        lo que REGULARIZA el espacio de parámetros y garantiza (con ε>0)
        la condición log-Sobolev → PL → convergencia exponencial.

        Nota: en la aproximación de M partículas, ∫ E(ν_t|ν^∞) dt se
        aproxima como la penalización del potencial ℓ evaluado en los
        parámetros aprendidos θ (los "M puntos de soporte" de ν_t).

        Args:
            x0      : (N, d1) — features de entrada
            y       : (N,) — etiquetas {0,1}
            epsilon : coeficiente de regularización entrópica (ε ≥ 0)

        Returns:
            loss_total : J (tensor con grafo de cómputo para autograd)
            loss_term  : BCE puro (float) — componente de clasificación
            loss_reg   : penalización entrópica (float) — componente de regularización
        """
        logit     = self.forward(x0)
        loss_term = nn.BCEWithLogitsLoss()(logit, y)
        loss_reg  = self.velocity.entropic_penalty()
        return loss_term + epsilon * loss_reg, loss_term.item(), loss_reg.item()
