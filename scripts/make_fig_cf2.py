#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fig_cf2.pdf : 4-panel E12 = (a)(b)(c) as in fig_cf + (d) sample curve. NEW FILE (keeps fig_cf.pdf).
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/projects/casp_max/outputs/run")
FIG = os.path.expanduser("~/projects/casp_max/figures")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})
C_MC = "#d1495b"; C_CF = "#1b6ca8"; C_FB = "#7a7a7a"; C_ACC = "#2a9d8f"

d = json.load(open(f"{R}/r3_harden.json"))
sm = json.load(open(f"{R}/r3_sample.json"))
scf = d["sc_curves"]["flip"]; vc = d["vc_curves"]; eta = [r["eta"] for r in scf]

fig, ax = plt.subplots(1, 4, figsize=(17.4, 3.7))

# (a) Set Cover
mc = [r["mincomb"] for r in scf]; cf = [r["CF"] for r in scf]; fb = [r["fallback"] for r in scf]
ax[0].fill_between(eta, cf, mc, color=C_CF, alpha=0.12, label="domination margin")
ax[0].plot(eta, fb, ":", color=C_FB, lw=1.6, label="fallback")
ax[0].plot(eta, mc, "o-", color=C_MC, lw=2.2, ms=6, label="min-combiner")
ax[0].plot(eta, cf, "s-", color=C_CF, lw=2.2, ms=6, label="confidence-filter (ours)")
ax[0].set_xlabel(r"prediction noise $\eta$"); ax[0].set_ylabel("mean cost ratio")
ax[0].set_title("(a) Set Cover"); ax[0].legend(fontsize=8.0, loc="lower right"); ax[0].set_ylim(0.99, 1.13)

# (b) Vertex Cover
mcv = [r["mincomb"] for r in vc]; cfv = [r["CF"] for r in vc]; fbv = vc[0]["fallback"]
ax[1].fill_between(eta, cfv, mcv, color=C_CF, alpha=0.12, label="domination margin")
ax[1].axhline(fbv, ls=":", color=C_FB, lw=1.6, label="fallback (%.2f)" % fbv)
ax[1].plot(eta, mcv, "o-", color=C_MC, lw=2.2, ms=6, label="min-combiner")
ax[1].plot(eta, cfv, "s-", color=C_CF, lw=2.2, ms=6, label="confidence-filter (ours)")
ax[1].annotate("margin grows\nwith noise", xy=(0.47, (mcv[-1]+cfv[-1])/2), xytext=(0.30, 1.075),
               fontsize=8.5, color=C_CF, ha="center", arrowprops=dict(arrowstyle="->", color=C_CF, lw=1.2))
ax[1].set_xlabel(r"prediction noise $\eta$"); ax[1].set_ylabel("mean cost ratio")
ax[1].set_title("(b) Vertex Cover"); ax[1].legend(fontsize=8.0, loc="upper left"); ax[1].set_ylim(1.0, 1.38)

# (c) Domination margin
styles = {"flip": ("SC (flip)", "s-", C_CF), "fp": ("SC (false-pos)", "^-", C_ACC),
          "drop": ("SC (drop)", "v-", "#8e7cc3")}
for k, (lab, mk, col) in styles.items():
    ax[2].plot(eta, [r["dom_margin"] for r in d["sc_curves"][k]], mk, color=col, lw=1.8, ms=5, label=lab)
ax[2].plot(eta, [r["dom_margin"] for r in vc], "o-", color=C_MC, lw=2.4, ms=6, label="VC")
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xlabel(r"prediction noise $\eta$"); ax[2].set_ylabel(r"domination margin (min-comb $-$ CF)")
ax[2].set_title(r"(c) Margin $\geq 0$ everywhere"); ax[2].legend(fontsize=8.0, loc="upper left")

# (d) Sample curve
scN = [p["N"] for p in sm["sc"]["curve"]]; scE = [p["excess"] for p in sm["sc"]["curve"]]; scS = [p["sd"] for p in sm["sc"]["curve"]]
vcN = [p["N"] for p in sm["vc"]["curve"]]; vcE = [p["excess"] for p in sm["vc"]["curve"]]; vcS = [p["sd"] for p in sm["vc"]["curve"]]
ax[3].errorbar(scN, scE, yerr=scS, fmt="s-", color=C_CF, lw=2.0, ms=6, capsize=2, label="Set Cover")
ax[3].errorbar(vcN, vcE, yerr=vcS, fmt="o-", color=C_MC, lw=2.0, ms=6, capsize=2, label="Vertex Cover")
ax[3].axhline(0, color="k", lw=0.8)
ax[3].axvline(5, ls="--", color="gray", lw=1.2, alpha=0.8)
ax[3].set_ylim(bottom=-0.004)
ymax = ax[3].get_ylim()[1]
ax[3].text(5.4, ymax*0.88, r"$N\!\approx\!5$", fontsize=9, color="gray", va="top")
ax[3].set_xlabel(r"training instances $N$"); ax[3].set_ylabel(r"excess test loss $\ell(\hat\theta)-\ell(\theta^\star)$")
ax[3].set_title(r"(d) $\theta^\star$ learnable from $N\!\approx\!5$"); ax[3].legend(fontsize=8.5, loc="upper right")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_cf2.pdf"); plt.close(fig)
print("wrote", f"{FIG}/fig_cf2.pdf")
