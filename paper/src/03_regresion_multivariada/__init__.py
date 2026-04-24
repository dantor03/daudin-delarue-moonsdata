"""Tarea 3: regresion multivariada de verdad (sin PCA) sobre California Housing.

Aplica los mismos 4 metodos paramétricos de Tarea 2 (SGLD vanilla, pSGLD,
MMD2, Sinkhorn) a un MeanFieldResNet con d_1 = 8 (las 8 features originales).
Anade baselines clasicos (OLS, Ridge, Random Forest) y, si es factible, el
solver explicito sobre un subconjunto de 2 features.
"""
