"""Regenera paper/figures/06_mfl_rates.png a partir del CSV sin reentrenar."""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from run_mfl_rates import fit_power_law, METHOD_COLORS

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "rates_results.csv"
FIG_OUT = HERE.parents[1] / "figures" / "06_mfl_rates.png"


def main():
    rows = []
    with open(CSV_IN) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((int(row["seed"]), int(row["M"]), row["method"],
                         float(row["mmd2"])))
    methods = sorted({m for _, _, m, _ in rows},
                     key=["pSGLD", "MMD$^2$", "Sinkhorn"].index)

    by_method = defaultdict(list)
    for _, M, m, v in rows:
        by_method[m].append((M, v))

    fits = {}
    print(f"  {'metodo':<12s} {'alpha':>10s} {'SE':>10s} {'intercept':>12s}")
    print("-" * 52)
    for m in methods:
        Ms = [a for a, _ in by_method[m]]
        vs = [b for _, b in by_method[m]]
        alpha, se, beta, mask = fit_power_law(Ms, vs)
        fits[m] = dict(alpha=alpha, se=se, beta=beta, Ms=np.array(Ms),
                       vs=np.array(vs))
        print(f"  {m:<12s} {alpha:>10.3f} {se:>10.3f} {beta:>12.3f}")
    print("-" * 52)

    M_min = min(M for _, M, _, _ in rows)
    M_max = max(M for _, M, _, _ in rows)
    M_dense = np.logspace(np.log10(M_min), np.log10(M_max), 100)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m in methods:
        info = fits[m]
        color = METHOD_COLORS.get(m, "gray")
        ax.scatter(info["Ms"], np.clip(info["vs"], 1e-5, None), color=color,
                   s=55, alpha=0.85, label=m, edgecolor="k", linewidth=0.6)
        if np.isfinite(info["alpha"]):
            y_fit = 10 ** info["beta"] * M_dense ** (-info["alpha"])
            exp_str = f"{-info['alpha']:+.2f}"
            ax.plot(M_dense, y_fit, color=color, lw=1.5, ls="--",
                    label=rf"  ajuste $\propto M^{{{exp_str}}}$")
    # Guia M^{-1} (Monte Carlo) anclada en el minimo de pSGLD
    anchor = fits["pSGLD"]["vs"].min()
    y_guide = anchor * (M_dense / M_min) ** (-1.0)
    ax.plot(M_dense, y_guide, color="grey", lw=1.0, ls=":",
            label=r"guía $M^{-1}$ (Monte Carlo)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$ (nº de neuronas)")
    ax.set_ylabel(r"MMD$^2(\nu_T^{\text{proxy}}, \nu^*_T)$  (unbiased)")
    ax.set_title(r"Propagación del caos: MMD$^2$ vs $M$ "
                 r"en make\_moons, $\varepsilon=0.05$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=140, bbox_inches="tight")
    print(f"  -> {FIG_OUT}")


if __name__ == "__main__":
    main()
