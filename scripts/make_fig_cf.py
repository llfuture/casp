#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E12 figure (v2, legend/annotation placement fixed). -> figures/fig_cf.pdf
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/projects/casp_max/outputs/run")
FIG = os.path.expanduser("~/projects/casp_max/figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})
C_MC = "#d1495b"; C_CF = "#1b6ca8"; C_FB = "#7a7a7a"; C_ACC = "#2a9d8f"

d = json.load(open(f"{R}/r3_harden.json"))
scf = d["sc_curves"]["flip"]; vc = d["vc_curves"]
eta = [r["eta"] for r in scf]

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))

# ---- (a) Set Cover ----
mc = [r["mincomb"] for r in scf]; cf = [r["CF"] for r in scf]; fb = [r["fallback"] for r in scf]
ax[0].fill_between(eta, cf, mc, color=C_CF, alpha=0.12, label="domination margin")
ax[0].plot(eta, fb, ":", color=C_FB, lw=1.6, label="fallback")
ax[0].plot(eta, mc, "o-", color=C_MC, lw=2.2, ms=6, label="min-combiner")
ax[0].plot(eta, cf, "s-", color=C_CF, lw=2.2, ms=6, label="confidence-filter (ours)")
ax[0].set_xlabel(r"prediction noise $\eta$"); ax[0].set_ylabel("mean cost ratio")
ax[0].set_title("(a) Set Cover"); ax[0].legend(fontsize=8.2, loc="lower right")
ax[0].set_ylim(0.99, 1.13)

# ---- (b) Vertex Cover ----
mc = [r["mincomb"] for r in vc]; cf = [r["CF"] for r in vc]; fbv = vc[0]["fallback"]
ax[1].fill_between(eta, cf, mc, color=C_CF, alpha=0.12, label="domination margin")
ax[1].axhline(fbv, ls=":", color=C_FB, lw=1.6, label="fallback (%.2f)" % fbv)
ax[1].plot(eta, mc, "o-", color=C_MC, lw=2.2, ms=6, label="min-combiner")
ax[1].plot(eta, cf, "s-", color=C_CF, lw=2.2, ms=6, label="confidence-filter (ours)")
ax[1].annotate("margin grows\nwith noise", xy=(0.47, (mc[-1]+cf[-1])/2), xytext=(0.30, 1.075),
               fontsize=8.5, color=C_CF, ha="center",
               arrowprops=dict(arrowstyle="->", color=C_CF, lw=1.2))
ax[1].set_xlabel(r"prediction noise $\eta$"); ax[1].set_ylabel("mean cost ratio")
ax[1].set_title("(b) Vertex Cover"); ax[1].legend(fontsize=8.2, loc="upper left")
ax[1].set_ylim(1.0, 1.38)

# ---- (c) Domination margin ----
styles = {"flip": ("SC (flip)", "s-", C_CF), "fp": ("SC (false-pos)", "^-", C_ACC),
          "drop": ("SC (drop)", "v-", "#8e7cc3")}
for k, (lab, mk, col) in styles.items():
    m = [r["dom_margin"] for r in d["sc_curves"][k]]
    ax[2].plot(eta, m, mk, color=col, lw=1.8, ms=5, label=lab)
ax[2].plot(eta, [r["dom_margin"] for r in vc], "o-", color=C_MC, lw=2.4, ms=6, label="VC")
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xlabel(r"prediction noise $\eta$"); ax[2].set_ylabel(r"domination margin (min-comb $-$ CF)")
ax[2].set_title(r"(c) Margin $\geq 0$ everywhere"); ax[2].legend(fontsize=8.2, loc="upper left")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_cf.pdf")
plt.close(fig)
print("wrote", f"{FIG}/fig_cf.pdf")
