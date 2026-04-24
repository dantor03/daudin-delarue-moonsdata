# Paper — Aplicación computacional de Daudin & Delarue (2025)

Trabajo posterior a los experimentos A–K (en `codigo/experiments/`).
Cada subcarpeta corresponde a un **work-stream de tareas**, no a una sección del paper.
La estructura de secciones del paper se decide en `draft/` cuando haya resultados.

## Estructura

```
paper/
  src/
    01_solucion_explicita/      # implementación fiel del Teorema 1.4
    02_regularizadores_moons/   # re-implementación pSGLD/MMD²/Sinkhorn en make_moons
    03_regresion_multivariada/  # regresión sin PCA (California / Breast Cancer)
    04_metodos_alternativos/    # otras alternativas a los 3 actuales
    05_economia/                # aplicación económica
  figures/                      # figuras del paper
  notebooks/                    # exploratorio antes de consolidar
  draft/                        # LaTeX del paper
```

Los módulos núcleo (`model.py`, `metrics.py`, `data.py`, ...) viven en `codigo/`
y se importan desde `paper/src/...`. No duplicar lógica.

## Estado de tareas (post reunión 17-Abr-2026)

| # | Tarea | Carpeta | Estado |
|---|---|---|---|
| 1 | Solución explícita del paper, dataset pequeño | `01_solucion_explicita/` | hecho |
| 2 | Re-implementar pSGLD/MMD²/Sinkhorn en make_moons | `02_regularizadores_moons/` | hecho |
| 3 | Regresión multivariada sin PCA | `03_regresion_multivariada/` | hecho |
| 4 | Métodos alternativos (bridge ν* exacta vs neuronas) | `04_metodos_alternativos/` | hecho |
| 5 | Aplicación económica | `05_economia/` | pendiente |
