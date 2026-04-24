"""SGLD vanilla para regresion (MSE).  Igual que el de Tarea 2 pero registra R^2."""
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def train_sgld_vanilla_reg(model, X, y, epsilon, lr=1e-3, n_epochs=800,
                            verbose=False):
    """SGD + ruido isotropico de Langevin sobre MeanFieldResNet en regresion.

    Returns:
        hist : dict con loss, accuracy (R²), J_star.
    """
    opt = optim.SGD(model.parameters(), lr=lr)
    noise_std = math.sqrt(2.0 * lr * epsilon) if epsilon > 0 else 0.0

    hist = {'loss': [], 'loss_term': [], 'loss_reg': [],
            'grad_norm2': [], 'accuracy': []}

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss, lt, lr_val = model.compute_loss(X, y, epsilon)
        loss.backward()
        gn2 = sum(p.grad.pow(2).sum().item()
                  for p in model.parameters() if p.grad is not None)
        opt.step()

        if epsilon > 0:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * noise_std)

        with torch.no_grad():
            pred = model(X)
            ss_res = ((pred - y) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=1e-8)
            r2 = float(1.0 - ss_res / ss_tot)

        hist['loss'].append(loss.item())
        hist['loss_term'].append(lt)
        hist['loss_reg'].append(lr_val)
        hist['grad_norm2'].append(gn2)
        hist['accuracy'].append(r2)

        if verbose and (ep + 1) % 200 == 0:
            print(f"    [SGLD vanilla] ep {ep+1:4d} | J={loss.item():.4f} | "
                  f"R²={r2:.3f} | gn2={gn2:.2e}")

        if not np.isfinite(loss.item()):
            if verbose:
                print(f"    [SGLD vanilla] diverged at epoch {ep+1}")
            for k in hist:
                hist[k].extend([float('nan')] * (n_epochs - ep - 1))
            break

    valid = [v for v in hist['loss'] if np.isfinite(v)]
    hist['J_star'] = min(valid) if valid else float('nan')
    return hist
