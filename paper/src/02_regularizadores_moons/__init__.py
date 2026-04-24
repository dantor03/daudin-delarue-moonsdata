"""Re-implementacion limpia de pSGLD / MMD2 / Sinkhorn sobre make_moons.

Reutiliza las primitivas de `codigo/` (MeanFieldResNet, train, metricas)
para evitar duplicacion. Anade SGLD vanilla (SGD + ruido isotropico) como
referencia explicita de por que pSGLD es necesario.
"""
