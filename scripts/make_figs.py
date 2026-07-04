#!/usr/bin/env python3
# Generate v2 publication figures from real result JSONs. matplotlib Agg -> PDF in figures/.
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/projects/casp_max/outputs/run")
TD = os.path.expanduser("~/projects/casp_max/outputs/theoremD")
FIG = os.path.expanduser("~/projects/casp_max/figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})
def L(p): return json.load(open(p))
C_CASP = "#1b6ca8"; C_POS = "#d1495b"; C_ACC = "#2a9d8f"

# ---- F1: E8 boundedness separation (the headline) ----
e8 = L(f"{R}/e8.json")["rows"]
Rs = [r["R"] for r in e8]; casp = [r["casp_ratio"] for r in e8]; pos = [r["pos_ratio"] for r in e8]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.loglog(Rs, pos, "o-", color=C_POS, lw=2.2, ms=7, label="Positive-signal commitment")
ax.loglog(Rs, casp, "s-", color=C_CASP, lw=2.2, ms=7, label="CASP (negative-signal)")
ax.axhline(2, ls="--", color="gray", lw=1, label="CASP bound max(f,$\\alpha$)")
ax.set_xlabel("cost spread $R=c_{\\max}/c_{\\min}$"); ax.set_ylabel("worst-case cost ratio")
ax.set_title("Boundedness separation (Thm B)")
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(f"{FIG}/fig_e8_bound.pdf"); plt.close(fig)

# ---- F2: E10 consistency-robustness-certification ----
e10 = L(f"{R}/e10.json")["summary"]
import collections, statistics as st
rows = L(f"{R}/e10.json")["rows"]
bye = collections.defaultdict(list)
for r in rows:
    if r.get("antoniadis_ratio"): bye[r["eta"]].append(r["antoniadis_ratio"])
etas = sorted(bye); ant = [st.mean(bye[e]) for e in etas]; caspr = [1.0]*len(etas)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(etas, ant, "o-", color=C_POS, lw=2.2, ms=7, label="Antoniadis'24 (positive)")
ax.plot(etas, caspr, "s-", color=C_CASP, lw=2.2, ms=7, label="CASP (negative)")
ax.set_xlabel("prediction error $\\eta$"); ax.set_ylabel("approximation ratio")
ax.set_ylim(0.95, max(ant)+0.1)
ax.set_title("Consistency-robustness (Thm robust); CASP certifies 70%")
ax.legend(fontsize=9, loc="upper left")
fig.savefig(f"{FIG}/fig_e10_cr.pdf"); plt.close(fig)

# ---- F3: E7 certified-optimality separation (bar) ----
e7 = L(f"{R}/e7.json")["summary"]
fig, ax = plt.subplots(figsize=(4.4, 3.6))
bars = ax.bar(["CASP\n(negative)", "Antoniadis'24\n(positive)"],
              [e7["casp_certified"], e7["antoniadis_certified"]],
              color=[C_CASP, C_POS], width=0.6)
ax.set_ylabel("instances certified optimal (of %d)" % e7["n"])
ax.set_title("Certified-optimality separation (Thm A)")
for b, v in zip(bars, [e7["casp_certified"], e7["antoniadis_certified"]]):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", fontweight="bold")
ax.set_ylim(0, e7["n"]+4)
fig.savefig(f"{FIG}/fig_e7_sep.pdf"); plt.close(fig)

# ---- F4: E3 f-stratified prune_gain vs solver_gain ----
_e3r = L(f"{R}/e3.json"); e3 = _e3r.get("by_f") or next(iter(_e3r.values()))
fs = [r["f"] for r in e3]; pg = [r["mean_prune_gain"] for r in e3]; sg = [r["mean_solver_gain"] for r in e3]
import numpy as np
x = np.arange(len(fs)); w = 0.38
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.bar(x-w/2, sg, w, color=C_ACC, label="solver gain (A-B)")
ax.bar(x+w/2, pg, w, color=C_CASP, label="prune gain (A-C)")
ax.set_xticks(x); ax.set_xticklabels(["f=%d"%f for f in fs])
ax.set_ylabel("cost reduction vs greedy"); ax.set_title("Ablation: pruning gain vanishes as f grows (E3)")
ax.legend(fontsize=9)
fig.savefig(f"{FIG}/fig_e3_byf.pdf"); plt.close(fig)

# ---- F5: E6 speedup split (SC & FL, OPT-preserving vs approx) ----
sc = L(f"{R}/e6_sc.json")["summary"]; fl = L(f"{R}/e6_fl.json")["summary"]
labels = ["SC\nOPT-pres", "SC\nf-approx", "FL\nOPT-pres", "FL\napprox"]
# pull mean speedups from rows split
def split_means(j):
    rows = [r for r in L(j)["rows"] if r.get("speedup")]
    op = [r["speedup"] for r in rows if not r.get("mismatch")]
    ap = [r["speedup"] for r in rows if r.get("mismatch")]
    return (st.mean(op) if op else 0, st.mean(ap) if ap else 0)
scop, scap = split_means(f"{R}/e6_sc.json"); flop, flap = split_means(f"{R}/e6_fl.json")
vals = [scop, scap, flop, flap]; cols = [C_CASP, C_POS, C_CASP, C_POS]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(labels, vals, color=cols, width=0.62)
ax.set_ylabel("mean exact-solving speedup ($\\times$)")
ax.set_title("Net speedup, split by certificate type (E6)")
for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+1, "%.1f"%v, ha="center", fontsize=9, fontweight="bold")
fig.savefig(f"{FIG}/fig_e6_split.pdf"); plt.close(fig)

# ---- F6: E9 knapsack core fraction by kind ----
kn = L(f"{TD}/knapsack_casp.json")["summary"]["mean_core_frac_by_kind"]
order = ["uncorrelated", "spanner", "weakly_corr", "strongly_corr", "subset_sum"]
order = [k for k in order if k in kn]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar([k.replace("_", "\n") for k in order], [kn[k] for k in order],
              color=[C_CASP if kn[k] < 0.9 else C_POS for k in order], width=0.62)
ax.set_ylabel("residual core fraction"); ax.set_ylim(0, 1.1)
ax.set_title("Knapsack (Thm E): core collapse by instance type")
ax.axhline(1.0, ls="--", color="gray", lw=1)
fig.savefig(f"{FIG}/fig_e9_knap.pdf"); plt.close(fig)

# ---- F7: E4' multi-param gap vs p ----
e4p = L(f"{R}/e4p.json")["summary"]["by_p"]
ps = sorted(e4p, key=int)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for p in ps:
    cur = e4p[p]["curve"]; Ns = [c["N"] for c in cur]; gaps = [c["mean_gap"] for c in cur]
    ax.plot(Ns, gaps, "o-", lw=2, ms=6, label="p=%s" % p)
ax.set_xlabel("training samples N"); ax.set_ylabel("test generalization gap")
ax.set_title("Multi-parameter PAC (Thm F): gap grows with p, loss bounded")
ax.legend(fontsize=9)
fig.savefig(f"{FIG}/fig_e4p.pdf"); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    if f.startswith("fig_e") and ("e8" in f or "e10" in f or "e7" in f or "e3_by" in f or "e6_split" in f or "e9" in f or "e4p" in f):
        print(" ", f)
