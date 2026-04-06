# Neural ODEs de Campo Medio sobre Make Moons

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08486-b31b1b.svg)](https://arxiv.org/abs/2507.08486)

Verificación empírica de **Daudin & Delarue (2025)**: Neural ODEs de campo medio con regularización entrópica, la condición de Polyak-Łojasiewicz y garantías de convergencia exponencial — aplicadas a los datasets `make_moons` y `make_circles`.

> Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* [arXiv:2507.08486](https://arxiv.org/abs/2507.08486)

---

## En qué consiste

Una Neural ODE aprende un campo vectorial $F(x,t)$ que transforma **continuamente** la distribución de datos $\gamma_0$ (dos medias lunas entrelazadas) en una distribución $\gamma_T$ linealmente separable — todo en el espacio original $\mathbb{R}^2$, sin ningún embedding a un espacio latente.

$$\frac{dX_t}{dt} = F(X_t, t) = \frac{1}{M}\sum_{m=1}^{M} \tanh(a_1^m \cdot X_t + a_2^m)\cdot a_0^m, \qquad t \in [0, 1]$$

La función objetivo incluye un término de **regularización entrópica** (divergencia KL respecto a un prior supercoercivo $\nu^\infty \propto e^{-\ell(a)}$):

$$J(\gamma_0, \nu) = \underbrace{\int L(x,y)\, d\gamma_T(x,y)}_{\text{coste terminal (BCE)}} + \varepsilon \underbrace{\int_0^T \mathcal{E}(\nu_t \,\|\, \nu^\infty)\, dt}_{\text{regularización entrópica}}$$

con $\ell(a) = 0.05|a|^4 + 0.5|a|^2$ (supercoercivo). Para cualquier $\varepsilon > 0$, esto garantiza:

- **Meta-Teorema 1** — Un minimizador estable único (genericidad)
- **Meta-Teorema 2** — La condición de Polyak-Łojasiewicz $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ → **convergencia exponencial** sin necesidad de convexidad

---

## Experimentos

| | Experimento | Qué verifica | Figuras |
|---|---|---|---|
| **A** | Evolución de $\gamma_t$ | La ODE transporta las lunas a una distribución separable | `A_feature_evolution.png` |
| **B** | Efecto de $\varepsilon$ | Convergencia, fronteras de decisión, campo de velocidad para $\varepsilon \in \{0, 0.001, 0.01, 0.1, 0.5\}$ | `B1_convergence_curves.png`, `B2_decision_boundaries.png` |
| **C** | Verificación de la condición PL | $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ durante todo el entrenamiento | `C_pl_verification.png` |
| **D** | Genericidad del minimizador | Robustez a semillas de init y de datos (Meta-Teorema 1) | `D_genericity.png` |
| **E** | Robustez de $\nu^*$ (make\_moons) | La distribución $\nu^*$ es estable entre entrenamientos | `E_parameter_robustness.png` |
| **F** | Convergencia en make\_circles | Simetría SO(2): robustez de la convergencia a semillas de datos e init | `F_circles_parameter_distribution.png` |
| **G** | Problema de convergencia ($N \to \infty$) | Primera evidencia empírica de $J^*_N \to J^*_\infty$; tasa estimada gap $\propto N^{-\alpha}$ con $\alpha \approx 1$ | `G_convergence_problem.png` |

---

## Resultados

| Resultado del paper | Evidencia empírica |
|---|---|
| La ODE transforma $\gamma_0$ en $\gamma_T$ separable | Exp. A: lunas → clases separadas, acc ≈ 100% |
| $\varepsilon > 0$ penaliza parámetros lejos del prior | Exp. B: $J^*$ crece con $\varepsilon$; campo de velocidad más uniforme |
| La condición PL se cumple con $\mu > 0$ | Exp. C: ratio PL > 0 en todas las épocas para todo $\varepsilon > 0$ |
| Convergencia exponencial bajo PL | Exp. C2: decaimiento lineal en escala log (SGD + lr constante) |
| Genericidad del minimizador único | Exp. D: fronteras de decisión similares entre data seeds |
| $\nu^*$ es robusta a las condiciones de entrenamiento | Exp. E: curvas de importancia $\|a_0^m\|_2$ estables entre seeds |
| La geometría del dataset determina la estructura de $\nu^*$ | Exp. F: make\_circles (simétrico SO(2)) → convergencia isotrópica |
| *Problema abierto*: $J^*_N \to J^*_\infty$ cuando $N \to \infty$ | Exp. G: BCE$_\text{test}$ converge con tasa emp. $\alpha \approx 1.0$–$1.1$ (más rápida que $N^{-1/2}$ clásico) |

---

## Instalación

```bash
git clone https://github.com/dantor03/daudin-delarue-moonsdata.git
cd daudin-delarue-moonsdata
pip install -r requirements.txt
```

Requiere Python 3.10+.

---

## Uso

### Ejecutar todos los experimentos

```bash
python -m codigo
```

### Ejecutar un experimento concreto

```bash
python -m codigo --experiment A          # evolución de γ_t
python -m codigo --experiment B          # efecto de ε
python -m codigo --experiment C          # verificación PL
python -m codigo --experiment D          # genericidad
python -m codigo --experiment E          # robustez de ν*
python -m codigo --experiment F          # simetría en make_circles
python -m codigo --experiment G          # convergencia N → ∞
python -m codigo --experiment B --epochs 1000  # con número de épocas distinto
```

Las figuras se guardan en `figuras/`.

---

## Estructura del proyecto

```
daudin-delarue-moonsdata/
├── codigo/
│   ├── config.py          # Constantes, dispositivo, tema visual
│   ├── data.py            # get_moons(), get_circles()
│   ├── model.py           # MeanFieldVelocity, MeanFieldResNet
│   ├── train.py           # train() con Adam/SGD/pSGLD, mu_pl_estimate()
│   ├── plots.py           # plot_decision_boundary()
│   ├── main.py            # CLI con argparse (--experiment {A,B,C,D,E,F})
│   └── experiments/
│       ├── exp_a.py       # Experimento A
│       ├── exp_b.py       # Experimento B
│       ├── exp_c.py       # Experimento C
│       ├── exp_d.py       # Experimento D
│       ├── exp_e.py       # Experimento E (robustez de ν*)
│       ├── exp_f.py       # Experimento F (make_circles)
│       └── exp_g.py       # Experimento G (convergencia N → ∞)
├── docs/
│   ├── DOCUMENTACION.md          # Documentación técnica completa
│   └── arquitectura_matematica.tex  # Derivaciones matemáticas en LaTeX
├── figuras/               # Figuras generadas (no versionadas)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Arquitectura del modelo

```
X_0 ∈ ℝ²  →  [ODE: dX/dt = F(X,t), t ∈ [0,1]]  →  X_T ∈ ℝ²  →  [lineal W,b]  →  logit
```

| Componente | Detalle |
|---|---|
| Campo vectorial | $F(x,t) = W_0 \tanh(W_1 [x,t]^\top + b_1)$, $M = 64$ neuronas, 384 parámetros |
| Integrador | Runge-Kutta 4 (RK4), 10 pasos — error global $O(dt^4)$ |
| Clasificador | lineal sobre $X_T$, 3 parámetros |
| Penalización energética | $\varepsilon \cdot \frac{1}{N_p}\sum_j [0.05\,\theta_j^4 + 0.5\,\theta_j^2]$ |
| Clipping de gradiente | max\_norm = 5 (todos los modos) |

### Modos de optimización

| Modo | Experimentos | Descripción |
|---|---|---|
| Adam + cosine annealing | A, D | Convergencia rápida |
| SGD + lr constante | C | Verificación PL sin artefactos del scheduler |
| **pSGLD** + cosine annealing | B, E, F | Adam base + ruido acoplado al precondicionador $M_t$: $\theta \leftarrow \text{Adam}(\theta,\nabla J) + \sqrt{2\eta\varepsilon M_t}\,\xi$ |

**pSGLD** (Li et al. 2016): el ruido de Langevin está acoplado al precondicionador de Adam componente a componente, $M_t[j] = \min(1/(\sqrt{\hat{v}_t[j]} + \delta),\, 1)$. La cota superior en 1 evita explosión de ruido en direcciones planas. El acoplamiento es necesario para que la distribución estacionaria sea la de Gibbs $\nu^* \propto \exp(-J/\varepsilon)$; ruido isotrópico rompe esta propiedad.

---

## Cita

```bibtex
@article{daudin2025genericity,
  title   = {Genericity of the {Polyak-{\L}ojasiewicz} inequality for
             mean-field {Neural ODEs} with entropic regularization},
  author  = {Daudin, Samuel and Delarue, Fran{\c{c}}ois},
  journal = {arXiv preprint arXiv:2507.08486},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.08486}
}
```

---

## Referencias

- Chen et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Welling, M. & Teh, Y. W. (2011). *Bayesian Learning via Stochastic Gradient Langevin Dynamics.* ICML.
- Li, C., Chen, C., Carlson, D., & Carin, L. (2016). *Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks.* AAAI. [arXiv:1512.07666](https://arxiv.org/abs/1512.07666)
- Polyak, B.T. (1963). *Gradient methods for minimizing functionals.* USSR Comput. Math. Math. Phys.
- Villani, C. (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics.
