# Neural ODEs de Campo Medio sobre Make Moons

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08486-b31b1b.svg)](https://arxiv.org/abs/2507.08486)
[![CI](https://github.com/dantor03/daudin-delarue-moons/actions/workflows/ci.yml/badge.svg)](https://github.com/dantor03/daudin-delarue-moons/actions)

Verificación empírica de **Daudin & Delarue (2025)**: Neural ODEs de campo medio con regularización entrópica, la condición de Polyak-Łojasiewicz y garantías de convergencia exponencial — aplicadas al dataset `make_moons`.

> Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* [arXiv:2507.08486](https://arxiv.org/abs/2507.08486)

---

## En qué consiste

Una Neural ODE aprende un campo vectorial $F(x,t)$ que transforma **continuamente** la distribución de datos $\gamma_0$ (dos medias lunas entrelazadas) en una distribución $\gamma_T$ linealmente separable — todo en el espacio original $\mathbb{R}^2$, sin ningún embedding a un espacio latente.

$$\frac{dX_t}{dt} = F(X_t, t) = \frac{1}{M}\sum_{m=1}^{M} \underbrace{\tanh(a_1^m \cdot X_t + a_2^m)}_{\text{activación}} \cdot a_0^m, \qquad t \in [0, 1]$$

La función objetivo incluye un término de **regularización entrópica** (divergencia KL respecto a un prior supercoercivo $\nu^\infty \propto e^{-\ell(a)}$):

$$J(\gamma_0, \nu) = \underbrace{\int L(x,y)\, d\gamma_T(x,y)}_{\text{coste terminal (BCE)}} + \varepsilon \underbrace{\int_0^T \mathcal{E}(\nu_t \,\|\, \nu^\infty)\, dt}_{\text{regularización entrópica}}$$

con $\ell(a) = c_1|a|^4 + c_2|a|^2$ (supercoercivo). Para cualquier $\varepsilon > 0$, esto garantiza:

- **Meta-Teorema 1** — Un minimizador estable único (genericidad)
- **Meta-Teorema 2** — La condición de Polyak-Łojasiewicz $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ → **convergencia exponencial** sin necesidad de convexidad

---

## Experimentos

| | Experimento | Qué verifica | Figuras |
|---|---|---|---|
| **A** | Evolución de $\gamma_t$ | La ODE transporta las lunas a una distribución separable | `A_feature_evolution.png` |
| **B** | Efecto de $\varepsilon$ | Convergencia, fronteras de decisión, forma de Gibbs de $\nu^*$, campo de velocidad | `B1`–`B4` |
| **C** | Verificación de la condición PL | $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ durante todo el entrenamiento | `C_pl_verification.png` |
| **D** | Genericidad del minimizador | Robustez a semillas de init y de datos (Meta-Teorema 1) | `D_genericity.png` |
| **E** | Análisis profundo de $\nu^*$ | Marginales por tipo de parámetro, distribución 2D de $a_1$, importancias | `E_parameter_analysis.png` |
| **E2** | Robustez de $\nu^*$ | La distribución $\nu^*$ es estable entre entrenamientos | `E2_parameter_robustness.png` |
| **F** | $\nu^*$ en `make_circles` | Simetría SO(2) → $a_1^m$ uniformes en $S^1$ (R̄ ≈ 0) | `F_circles_parameter_distribution.png` |

---

## Resultados

| Resultado del paper | Evidencia empírica |
|---|---|
| La ODE transforma $\gamma_0$ en $\gamma_T$ separable | Exp. A: lunas → clases separadas, acc ≈ 100% |
| $\varepsilon > 0$ fuerza la forma de Gibbs en $\nu^*$ | Exp. B3: std($\theta$) decrece con $\varepsilon$ |
| La condición PL se cumple con $\mu > 0$ | Exp. C: ratio PL > 0 en todas las épocas |
| Convergencia exponencial bajo PL | Exp. C2: decaimiento lineal en escala log |
| Genericidad del minimizador único | Exp. D: Std$(J^*) \approx 0$ para datos limpios |
| Simetría de $\nu^*$ refleja la simetría de $\gamma_0$ | Exp. F: R̄ ≈ 0 en `make_circles` |

---

## Instalación

```bash
git clone https://github.com/dantor03/daudin-delarue-moons.git
cd daudin-delarue-moons
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
python -m codigo --experiment A          # solo evolución de γ_t
python -m codigo --experiment B          # efecto de ε
python -m codigo --experiment C          # verificación PL
python -m codigo --experiment D --epochs 200  # genericidad, 200 épocas
python -m codigo --experiment F          # simetría en make_circles
```

### Compatibilidad con versión original

```bash
python codigo/daudin_delarue_moons.py    # ejecuta todos los experimentos
```

Las figuras se guardan en `figuras/`.

---

## Estructura del proyecto

```
daudin-delarue-moons/
├── codigo/
│   ├── config.py          # Constantes, dispositivo, tema visual
│   ├── data.py            # get_moons(), get_circles()
│   ├── model.py           # MeanFieldVelocity, MeanFieldResNet
│   ├── train.py           # train(), mu_pl_estimate()
│   ├── plots.py           # plot_decision_boundary()
│   ├── main.py            # CLI con argparse
│   ├── experiments/
│   │   ├── exp_a.py       # Experimento A
│   │   ├── exp_b.py       # Experimento B
│   │   ├── exp_c.py       # Experimento C
│   │   ├── exp_d.py       # Experimento D
│   │   ├── exp_e.py       # Experimentos E y E2
│   │   └── exp_f.py       # Experimento F
│   └── daudin_delarue_moons.py  # Shim de compatibilidad
├── docs/
│   ├── EXPERIMENTO.md     # Documentación matemática detallada
│   ├── slides_moons.tex   # Presentación LaTeX (experimentos A–C)
│   └── slides_DEF.tex     # Presentación LaTeX (experimentos D–F)
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
| Campo vectorial | $F(x,t) = W_0 \tanh(W_1 [x,t]^\top + b_1)$, $M = 64$ neuronas |
| Integrador | Runge-Kutta 4 (RK4), 10 pasos — error global $O(dt^4)$ |
| Penalización | $\varepsilon \cdot \frac{1}{N_p}\sum_j [0.05\,\theta_j^4 + 0.5\,\theta_j^2]$ |
| Optimizador | Adam + cosine annealing, clipping de gradiente (max\_norm=5) |

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

## Referencias adicionales

- Chen et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Polyak, B.T. (1963). *Gradient methods for minimizing functionals.* USSR Comput. Math. Math. Phys.
- Villani, C. (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics.
