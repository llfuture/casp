#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 MVE evaluator (G1 decision gate). Three arms on heterogeneous Set Cover:
  (1) LP-default : prune x*_i < 1/f                 (no learning)
  (2) single-tau*: ERM over a single global threshold (1 param)
  (3) per-bucket*: ERM over per-bucket thresholds theta in R^p (p params)
Reward = TRUE OPT-preserving prune rate: re-solve exact on survivors, credit
prune_rate iff residual OPT == full OPT (loss==1), else 0. Fair to alternate
optima; memoized on the pruned-set signature. Bucketing feature = mean element
frequency of a set (observable; high in redundant region, low in tight region).
"""
import sys, os, glob, json, random, itertools, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/hetero_sc")
OUT  = os.path.expanduser("~/projects/casp_max/outputs/run")
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
FEAT_RANGE = (1.5, 6.5)
EPS = 1e-9
_CACHE = {}

def bucket_edges(p):
    lo, hi = FEAT_RANGE
    return [lo + (hi - lo) * k / p for k in range(1, p)]

def bucket_of(feat, edges):
    b = 0
    for e in edges:
        if feat >= e: b += 1
    return b

def prep(path):
    sc = L.load_sc_synth(path); U, S, C = sc["universe"], sc["sets"], sc["costs"]
    if len(S) > 900 or len(U) > 200: return None
    opt, sol, sto, tful = L.sc_exact(U, S, C, tlim=15)
    if opt is None or sto != "optimal" or opt <= 0: return None
    lpval, xs = L.sc_lp(U, S, C)
    f = L.sc_freq(U, S)
    freq = {e: 0 for e in U}
    for s in S:
        for e in s: freq[e] += 1
    meanfreq = [(sum(freq[e] for e in s) / len(s)) if s else 0.0 for s in S]
    return {"file": os.path.basename(path), "m": len(S), "x": xs, "f": f,
            "opt_sets": sorted(sol), "meanfreq": meanfreq,
            "t_full": tful, "U": list(U), "S": S, "C": C, "opt": opt}

def survivors_of(inst, thr_fn):
    xs = inst["x"]
    return [i for i in range(inst["m"]) if xs[i] >= thr_fn(i) - EPS]

def true_reward_fn(inst, thr_fn):
    surv = survivors_of(inst, thr_fn)
    m = inst["m"]; prune_rate = 1.0 - len(surv) / m
    if len(surv) == m:
        return 0.0
    key = (inst["file"], hash(frozenset(surv)))
    if key in _CACHE:
        preserved = _CACHE[key]
    else:
        opt_r, _, sto, _ = L.sc_exact(inst["U"], inst["S"], inst["C"], restrict=surv, tlim=10)
        preserved = (opt_r is not None and sto == "optimal" and abs(opt_r - inst["opt"]) < 1e-6)
        _CACHE[key] = preserved
    return prune_rate if preserved else 0.0

def reward_single(inst, tau):        return true_reward_fn(inst, lambda i: tau)
def reward_bucket(inst, theta, edges):
    mf = inst["meanfreq"]
    return true_reward_fn(inst, lambda i: theta[bucket_of(mf[i], edges)])
def reward_lpdefault(inst):          return true_reward_fn(inst, lambda i: 1.0 / inst["f"])

def bucket_candidates(p):
    if p == 1: return [[g] for g in GRID]
    return [list(t) for t in itertools.product(GRID, repeat=p)]

def erm_single(train):
    return max(GRID, key=lambda tau: st.mean(reward_single(i, tau) for i in train))
def erm_bucket(train, p, edges):
    return max(bucket_candidates(p), key=lambda th: st.mean(reward_bucket(i, th, edges) for i in train))
def mean_reward_single(g, tau): return st.mean(reward_single(i, tau) for i in g)
def mean_reward_bucket(g, th, edges): return st.mean(reward_bucket(i, th, edges) for i in g)

def measure_speedup(insts, theta, edges, k=12):
    ratios = []; preserved = 0; n = 0
    for inst in insts[:k]:
        mf = inst["meanfreq"]; xs = inst["x"]
        surv = [i for i in range(inst["m"]) if xs[i] >= theta[bucket_of(mf[i], edges)] - EPS]
        if len(surv) == inst["m"]: continue
        opt_r, _, sto_r, t_r = L.sc_exact(inst["U"], inst["S"], inst["C"], restrict=surv, tlim=15)
        n += 1
        if opt_r is not None and abs(opt_r - inst["opt"]) < 1e-6: preserved += 1
        if opt_r is not None and t_r > 0:
            ratios.append(inst["t_full"] / max(t_r, 1e-4))
    return {"n": n, "opt_preserved": preserved,
            "mean_speedup": round(st.mean(ratios), 2) if ratios else None,
            "max_speedup": round(max(ratios), 2) if ratios else None}

def load(dist, cap=40):
    files = sorted(glob.glob(os.path.join(DATA, dist, "*.json.gz")))[:cap]
    with Pool(16) as pool:
        return [r for r in pool.map(prep, files) if r]

def diagnose(insts, p=2):
    import collections
    edges = bucket_edges(p)
    print("  [diagnose] p=%d edges=%s n=%d" % (p, [round(e, 2) for e in edges], len(insts)))
    buck_x = collections.defaultdict(list)
    lowrate = collections.defaultdict(lambda: [0, 0])
    for inst in insts:
        opt = set(inst["opt_sets"]); mf = inst["meanfreq"]; xs = inst["x"]
        for i in range(inst["m"]):
            b = bucket_of(mf[i], edges); buck_x[b].append(xs[i])
            if xs[i] < 0.25:
                lowrate[b][1] += 1
                if i in opt: lowrate[b][0] += 1
    for b in sorted(buck_x):
        xv = sorted(buck_x[b]); q = lambda t: xv[min(len(xv) - 1, int(t * len(xv)))]
        io, tot = lowrate[b]
        print("    bucket %d: n=%4d x* q10/50/90=%.3f/%.3f/%.3f low-x* in-OPT=%s (%d/%d)"
              % (b, len(xv), q(.1), q(.5), q(.9), (round(io / tot, 3) if tot else "NA"), io, tot))

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    print("loading D1, D2 ...")
    D1 = load("D1"); D2 = load("D2")
    print("D1=%d D2=%d instances" % (len(D1), len(D2)))
    if mode == "diagnose":
        print("== D1 =="); diagnose(D1, 2)
        print("== D2 =="); diagnose(D2, 2)
        return

    res = {}; edges2 = bucket_edges(2)
    for dname, D in [("D2", D2), ("D1", D1)]:
        idx = list(range(len(D))); random.Random(0).shuffle(idx)
        h = len(idx) // 2; tr = [D[i] for i in idx[:h]]; te = [D[i] for i in idx[h:]]
        tau = erm_single(tr); th2 = erm_bucket(tr, 2, edges2)
        lp = st.mean(reward_lpdefault(i) for i in te)
        s1 = mean_reward_single(te, tau); b2 = mean_reward_bucket(te, th2, edges2)
        sp = measure_speedup(te, th2, edges2)
        res[dname] = {"lp_default": round(lp, 4), "single_tau": round(s1, 4),
                      "perbucket_p2": round(b2, 4), "tau_star": tau, "theta2": th2,
                      "gain_p2_vs_single": round(b2 - s1, 4),
                      "gain_p2_vs_lpdefault": round(b2 - lp, 4), "speedup": sp}
        print("[%s] LP-def=%.3f single-tau*=%.3f per-bucket-p2*=%.3f gain(p2-single)=%.3f speedup=%s"
              % (dname, lp, s1, b2, b2 - s1, sp))

    D = D2; idx = list(range(len(D))); random.Random(0).shuffle(idx)
    h = len(idx) // 2; pool = [D[i] for i in idx[:h]]; te = [D[i] for i in idx[h:]]
    ncurve = {}
    for p in [1, 2]:
        edges = bucket_edges(p) if p > 1 else []
        row = []
        for N in [5, 10, 20]:
            if N > len(pool): break
            vals = []
            for tr in range(10):
                sub = random.Random(tr).sample(pool, N)
                if p == 1: vals.append(mean_reward_single(te, erm_single(sub)))
                else: vals.append(mean_reward_bucket(te, erm_bucket(sub, p, edges), edges))
            row.append({"N": N, "eff_prune": round(st.mean(vals), 4)})
        ncurve[str(p)] = row
        print("N-curve p=%d: %s" % (p, row))

    th_cross = erm_bucket(D1, 2, edges2)
    cross = {"train_D1_test_D2_perbucket_p2": round(mean_reward_bucket(D2, th_cross, edges2), 4),
             "test_D2_single_from_D1": round(mean_reward_single(D2, erm_single(D1)), 4)}
    print("cross:", cross)

    out = {"within": res, "N_curve_D2": ncurve, "cross": cross,
           "n": {"D1": len(D1), "D2": len(D2)}, "grid": GRID}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "mve_p1.json"), "w"), indent=1)
    print("SAVED", os.path.join(OUT, "mve_p1.json"))
    g = res["D2"]["gain_p2_vs_single"]
    print("\n===== G1 DECISION SIGNAL =====")
    print("primary gain (D2, per-bucket-p2 - single-tau, TRUE OPT-preserving) = %.3f prune-rate" % g)
    print("GO if gain>=0.15 & N-curve rises with p & speedup>1; weak if 0<gain<0.15; no-go if ~0")

if __name__ == "__main__":
    main()
