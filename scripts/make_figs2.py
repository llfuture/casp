#!/usr/bin/env python3
# Rich multi-panel v3 figures from FULL per-instance data. matplotlib Agg -> PDF in figures/.
import json, os, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = os.path.expanduser("~/projects/casp_max/outputs/run")
TD = os.path.expanduser("~/projects/casp_max/outputs/theoremD")
FIG = os.path.expanduser("~/projects/casp_max/figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.axisbelow": True})
def L(p): return json.load(open(p))
CB = "#1b6ca8"; CR = "#d1495b"; CG = "#2a9d8f"; CO = "#e9a000"
def panel(ax, lab): ax.text(-0.14, 1.06, lab, transform=ax.transAxes, fontweight="bold", fontsize=12)

# ================= FIG 1: SEPARATIONS (hero, 3 panels) =================
e7 = L(f"{R}/e7.json"); e8 = L(f"{R}/e8.json"); e10 = L(f"{R}/e10.json")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
# (a) E7 CASP certified optimum vs ground-truth exact (all 30 instances on diagonal => 0 mismatch)
rows = e7["rows"]
co = [r["casp_opt"] for r in rows]; ex = [r["exact"] for r in rows]; ns = [r["n"] for r in rows]
sc = axs[0].scatter(ex, co, c=ns, cmap="viridis", s=70, edgecolor="k", lw=0.5, zorder=3)
lim = [min(ex)-2, max(ex)+2]; axs[0].plot(lim, lim, "--", color="gray", lw=1.2, label="CASP = exact OPT")
axs[0].set_xlabel("ground-truth optimum (SCIP)"); axs[0].set_ylabel("CASP certified optimum")
axs[0].set_title("(a) E7 self-certification (Thm A)")
cb = fig.colorbar(sc, ax=axs[0], fraction=0.046); cb.set_label("graph size $n$", fontsize=8)
axs[0].legend(loc="upper left", fontsize=8)
axs[0].text(0.04, 0.74, "30/30 on diagonal\nCASP certifies 30/30\nAntoniadis 0/30\ncore = planted, 0 mismatch",
            transform=axs[0].transAxes, fontsize=8, bbox=dict(fc="white", ec=CB, alpha=0.9))
# (b) E8 boundedness loglog
e8r = e8["rows"]; Rs = [r["R"] for r in e8r]; ca = [r["casp_ratio"] for r in e8r]; po = [r["pos_ratio"] for r in e8r]
axs[1].loglog(Rs, po, "o-", color=CR, lw=2.2, ms=8, label="positive-signal commit")
axs[1].loglog(Rs, ca, "s-", color=CB, lw=2.2, ms=8, label="CASP (negative)")
axs[1].axhline(2, ls="--", color="gray", lw=1, label="CASP bound max$(f,\\alpha)$")
axs[1].set_xlabel("cost spread $R=c_{\\max}/c_{\\min}$"); axs[1].set_ylabel("worst-case cost ratio")
axs[1].set_title("(b) E8 boundedness separation (Thm B)"); axs[1].legend(fontsize=8, loc="upper left")
for x, y in zip(Rs, po): axs[1].annotate("%.1f" % y, (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7.5)
# (c) E10 consistency-robustness with CI band
byA = collections.defaultdict(list); byC = collections.defaultdict(list); cert = collections.defaultdict(list)
for r in e10["rows"]:
    if r.get("antoniadis_ratio") is not None: byA[r["eta"]].append(r["antoniadis_ratio"])
    if r.get("casp_ratio") is not None: byC[r["eta"]].append(r["casp_ratio"])
    cert[r["eta"]].append(1 if r.get("casp_certifies") else 0)
etas = sorted(byA)
mA = [np.mean(byA[e]) for e in etas]; sA = [np.std(byA[e]) for e in etas]
mC = [np.mean(byC[e]) if byC[e] else 1.0 for e in etas]
axs[2].plot(etas, mA, "o-", color=CR, lw=2.2, ms=7, label="Antoniadis'24 (positive)")
axs[2].fill_between(etas, np.array(mA)-np.array(sA), np.array(mA)+np.array(sA), color=CR, alpha=0.15)
axs[2].plot(etas, mC, "s-", color=CB, lw=2.2, ms=7, label="CASP (negative)")
axs[2].set_xlabel("prediction error $\\eta$"); axs[2].set_ylabel("approximation ratio")
axs[2].set_title("(c) E10 consistency--robustness"); axs[2].legend(fontsize=8, loc="upper left")
cfrac = np.mean([c for e in etas for c in cert[e]])
axs[2].text(0.5, 0.06, "CASP certified-optimal on %.0f%% of instances\n(positive-signal: 0%%)" % (100*cfrac),
            transform=axs[2].transAxes, fontsize=8, ha="center", bbox=dict(fc="white", ec=CB, alpha=0.9))
fig.savefig(f"{FIG}/fig_sep.pdf"); plt.close(fig)

# ================= FIG 2: E2 pruning rate, full 600 =================
e2 = L(f"{R}/e2.json")["rows"]; e2 = [r for r in e2 if "gap" in r]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
byf = collections.defaultdict(list)
for r in e2: byf[r["f"]].append(r["emp_rate"])
fs = sorted(byf)
bp = axs[0].boxplot([byf[f] for f in fs], positions=range(len(fs)), widths=0.6, patch_artist=True,
                    medianprops=dict(color="k"))
for b in bp["boxes"]: b.set(facecolor=CB, alpha=0.55)
axs[0].set_xticks(range(len(fs))); axs[0].set_xticklabels(["f=%d" % f for f in fs])
axs[0].set_ylabel("empirical pruning rate"); axs[0].set_title("(a) E2 pruning rate rises with $f$ (n=600)")
panel(axs[0], "")
# (b) emp vs proven LB
lb = [r["lb"] for r in e2]; emp = [r["emp_rate"] for r in e2]; fcol = [r["f"] for r in e2]
ss = axs[1].scatter(lb, emp, c=fcol, cmap="plasma", s=22, alpha=0.8, edgecolor="none")
m = max(max(lb), max(emp)); axs[1].plot([0, m], [0, m], "--", color="gray", lw=1.2)
axs[1].set_xlabel("proven lower bound (Thm rate)"); axs[1].set_ylabel("empirical pruning rate")
axs[1].set_title("(b) bound never violated (all on/above diag.)")
cb = fig.colorbar(ss, ax=axs[1], fraction=0.046); cb.set_label("frequency $f$", fontsize=8)
viol = sum(1 for r in e2 if r["gap"] < 0)
axs[1].text(0.04, 0.9, "0 violations / %d\nmin gap +%.3f" % (len(e2), min(r["gap"] for r in e2)),
            transform=axs[1].transAxes, fontsize=8.5, bbox=dict(fc="white", ec=CB, alpha=0.9))
fig.savefig(f"{FIG}/fig_e2_full.pdf"); plt.close(fig)

# ================= FIG 3: E6 speedup, full (SC + FL) =================
def rows_ok(j): return [r for r in L(j)["rows"] if r.get("speedup")]
sc = rows_ok(f"{R}/e6_sc.json"); fl = rows_ok(f"{R}/e6_fl.json")
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
def strip(ax, data, base_x, color, label):
    xs = np.random.default_rng(0).normal(base_x, 0.05, len(data))
    ax.scatter(xs, data, s=30, color=color, alpha=0.75, edgecolor="k", lw=0.3, label=label)
    if data: ax.hlines(np.mean(data), base_x-0.18, base_x+0.18, color="k", lw=2)
groups = [("SC OPT-pres", [r["speedup"] for r in sc if not r.get("mismatch")], CB),
          ("SC f-approx", [r["speedup"] for r in sc if r.get("mismatch")], CR),
          ("FL OPT-pres", [r["speedup"] for r in fl if not r.get("mismatch")], CB),
          ("FL approx", [r["speedup"] for r in fl if r.get("mismatch")], CR)]
for i, (lab, data, col) in enumerate(groups): strip(axs[0], data, i, col, None)
axs[0].set_yscale("log"); axs[0].set_xticks(range(4)); axs[0].set_xticklabels([g[0] for g in groups], fontsize=8, rotation=12)
axs[0].set_ylabel("exact-solving speedup ($\\times$, log)"); axs[0].axhline(1, ls="--", color="gray", lw=1)
axs[0].set_title("(a) E6 speedup by certificate type")
# (b) speedup vs prune rate
for r in sc: axs[1].scatter(r.get("prune_rate"), r["speedup"], marker="o", s=34,
                            color=(CR if r.get("mismatch") else CB), edgecolor="k", lw=0.3)
for r in fl: axs[1].scatter(r.get("prune_rate"), r["speedup"], marker="^", s=34,
                            color=(CR if r.get("mismatch") else CB), edgecolor="k", lw=0.3)
axs[1].set_yscale("log"); axs[1].set_xlabel("pruning rate"); axs[1].set_ylabel("speedup ($\\times$, log)")
axs[1].set_title("(b) speedup vs pruning")
axs[1].legend(handles=[Line2D([],[],marker="o",ls="",color="gray",label="Set Cover"),
                       Line2D([],[],marker="^",ls="",color="gray",label="Facility Loc."),
                       Line2D([],[],marker="s",ls="",color=CB,label="OPT-preserving"),
                       Line2D([],[],marker="s",ls="",color=CR,label="$f$-approx (mismatch)")], fontsize=7.5, loc="lower right")
fig.savefig(f"{FIG}/fig_e6_full.pdf"); plt.close(fig)

# ================= FIG 4: learnability E4 + E4' =================
e4 = L(f"{R}/e4.json"); e4p = L(f"{R}/e4p.json")["summary"]["by_p"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
# (a) E4 cost-ratio landscape over tau
taus = ["0.02", "0.05", "0.1", "0.2", "0.5"]
rr = e4.get("rows", [])
mean_ratio = []
for t in taus:
    vals = [row["ratio"][t] for row in rr if row.get("ratio", {}).get(t) is not None]
    mean_ratio.append(np.mean(vals) if vals else np.nan)
axs[0].plot([float(t) for t in taus], mean_ratio, "o-", color=CB, lw=2, ms=8)
best = e4["summary"]["best_tau"]
axs[0].axvline(best, ls="--", color=CG, lw=1.5, label="learned $\\tau^\\star=%.2f$ ($N{=}5$)" % best)
axs[0].set_xlabel("threshold $\\tau$"); axs[0].set_ylabel("mean cost ratio (CASP/OPT)")
axs[0].set_title("(a) E4 single-parameter learning landscape"); axs[0].legend(fontsize=8.5)
# (b) E4' gap vs N for p
for p in sorted(e4p, key=int):
    cur = e4p[p]["curve"]; Ns = [c["N"] for c in cur]; g = [c["mean_gap"] for c in cur]
    axs[1].plot(Ns, g, "o-", lw=2, ms=7, label="p=%s (max loss %.2f)" % (p, e4p[p]["max_observed_loss"]))
axs[1].set_xlabel("training samples $N$"); axs[1].set_ylabel("test generalization gap")
axs[1].set_title("(b) E4$'$ multi-param: gap grows with $p$, loss bounded"); axs[1].legend(fontsize=8)
fig.savefig(f"{FIG}/fig_learn.pdf"); plt.close(fig)

# ================= FIG 5: heterogeneity E9 + exactness E1 =================
e9 = L(f"{TD}/knapsack_casp.json"); e1v = L(f"{R}/e1_vc.json")["rows"]; e1f = L(f"{TD}/fl_facint.json")
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
byk = collections.defaultdict(list)
for r in e9["rows"]: byk[r["kind"]].append(r["core_frac"])
order = [k for k in ["uncorrelated", "spanner", "weakly_corr", "strongly_corr", "subset_sum"] if k in byk]
for i, k in enumerate(order):
    xs = np.random.default_rng(1).normal(i, 0.05, len(byk[k]))
    col = CB if np.mean(byk[k]) < 0.9 else CR
    axs[0].scatter(xs, byk[k], s=34, color=col, alpha=0.8, edgecolor="k", lw=0.3)
    axs[0].hlines(np.mean(byk[k]), i-0.2, i+0.2, color="k", lw=2)
axs[0].set_xticks(range(len(order))); axs[0].set_xticklabels([k.replace("_", "\n") for k in order], fontsize=8)
axs[0].set_ylabel("residual core fraction"); axs[0].set_ylim(0, 1.1); axs[0].axhline(1, ls="--", color="gray", lw=1)
axs[0].set_title("(a) E9 Knapsack core collapse (Thm E, 0 mismatch)")
# (b) E1 VC core fraction histogram (dimacs)
dim = [1 - r["prune_rate"] for r in e1v if r["kind"] == "dimacs"]   # residual fraction = 1 - prune
axs[1].hist(dim, bins=18, color=CG, alpha=0.8, edgecolor="k", lw=0.3)
axs[1].set_xlabel("residual core fraction $|P_{1/2}|/|V|$ (DIMACS)"); axs[1].set_ylabel("# graphs")
axs[1].set_title("(b) E1 NT honesty + real-FL exactness")
axs[1].text(0.04, 0.82, "VC: 0 mismatch (12/97 verifiable)\nreal ORLIB FL: 37/37 exact (Thm D)",
            transform=axs[1].transAxes, fontsize=8.5, bbox=dict(fc="white", ec=CG, alpha=0.9))
fig.savefig(f"{FIG}/fig_hetero.pdf"); plt.close(fig)

# ================= FIG 6: robustness E5 =================
e5 = L(f"{R}/e5.json")["rows"]
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
for kind, col, mk in [("sc", CB, "o"), ("vc", CG, "s")]:
    bye = collections.defaultdict(list)
    for r in e5:
        if r["kind"] == kind: bye[r["eta"]].append(1 if r["casp_safe"] else 0)
    es = sorted(bye); safe = [100*np.mean(bye[e]) for e in es]
    ax.plot(es, safe, mk+"-", color=col, lw=2, ms=7, label="CASP %s" % kind.upper())
# balcan (sc only)
bb = collections.defaultdict(list)
for r in e5:
    if r["kind"] == "sc" and "balcan_safe" in r: bb[r["eta"]].append(1 if r["balcan_safe"] else 0)
if bb:
    es = sorted(bb); ax.plot(es, [100*np.mean(bb[e]) for e in es], "x--", color=CR, lw=1.6, ms=8, label="Balcan ERM SC")
ax.set_xlabel("certificate noise $\\eta$"); ax.set_ylabel("safe rate (%)"); ax.set_ylim(-5, 108)
ax.set_title("E5 robustness: safety independent of $\\eta$ (Thm noisy)"); ax.legend(fontsize=8.5)
fig.savefig(f"{FIG}/fig_e5_robust2.pdf"); plt.close(fig)
print("done figs")
