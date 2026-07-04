"""Figures for EX-A / EX-B: measured points vs closed-form curves."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_margin(path="../outputs/exa_margin.json", out="../figures/fig_margin.pdf"):
    d = json.load(open(path))
    rows = d["results"]
    Cs = sorted({r["C"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    n_max = max(r["n"] for r in rows)
    fig, axes = plt.subplots(1, len(betas), figsize=(3.2 * len(betas), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, beta in zip(axes, betas):
        for C in Cs:
            sel = sorted([r for r in rows
                          if r["beta"] == beta and r["C"] == C and r["n"] == n_max],
                         key=lambda r: r["eta"])
            eta = [r["eta"] for r in sel]
            ax.plot(eta, [r["margin_theory"] for r in sel], "-", lw=1.2,
                    label=f"theory C={C}")
            ax.errorbar(eta, [r["margin_meas"] for r in sel],
                        yerr=[3 * r["mc_se"] for r in sel],
                        fmt="o", ms=2.5, capsize=1.5, label=f"measured C={C}")
        ax.axhline(beta / (1 + 2 * beta), color="gray", ls=":", lw=0.8)
        ax.set_title(rf"$\beta={beta}$ (n={n_max})", fontsize=9)
        ax.set_xlabel(r"noise rate $\eta$")
    axes[0].set_ylabel("domination margin")
    axes[0].legend(fontsize=6)
    fig.suptitle("EX-A: min-combiner vs confidence filter — measured vs Thm 16",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def fig_lponly(path="../outputs/exb_lponly.json", out="../figures/fig_lponly.pdf"):
    d = json.load(open(path))
    rows = d["results"]
    n_max = max(r["n"] for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    for ax, kind in zip(axes, ("ri", "max")):
        for eps in sorted({r["eps"] for r in rows}):
            sel = sorted([r for r in rows if r["sigma"] == kind and r["eps"] == eps
                          and r["n"] == n_max and abs(r["m"] / r["n"] - 1.0) < 1e-9],
                         key=lambda r: r["eta"])
            if not sel:
                continue
            eta = [r["eta"] for r in sel]
            ax.plot(eta, [r["margin_theory_pg"] for r in sel], "-", lw=1.2,
                    label=rf"theory $\varepsilon$={eps}")
            ax.plot(eta, [r["margin_meas_pg"] for r in sel], "o", ms=2.5,
                    label=rf"measured $\varepsilon$={eps}")
            ax.axhline(sel[0]["margin_floor"], color="gray", ls=":", lw=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"canonicalization: {kind}", fontsize=9)
        ax.set_xlabel(r"noise rate $\eta$")
    axes[0].set_ylabel("per-gadget margin (LP-commit $-$ filter)")
    axes[0].legend(fontsize=6)
    fig.suptitle("EX-B: prediction breaks LP degeneracy — measured vs Thm 17",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    fig_margin()
    fig_lponly()
