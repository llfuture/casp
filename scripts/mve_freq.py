#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 MVE iteration 2 (principled): FREQUENCY-heterogeneous Set Cover.
The f-safe theorem (Thm lpsafe) says pruning x*<tau is safe for tau<=1/f.
f (element frequency) is OBSERVABLE and here VARIES by region:
  region A: low f (~2)  -> safe threshold up to ~0.5
  region B: high f (~8) -> safe threshold up to ~0.125
A single global tau is bottlenecked by the high-f region (~0.125); a per-bucket
threshold keyed on a set's mean element-frequency can prune region A far more
aggressively.  This is a provable, feature-dependent safe-threshold gap -> the
cleanest chance for learned per-bucket to dominate single-tau.
Reuses the TRUE-OPT-preserving reward of mve_p1.
"""
import sys, os, glob, gzip, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import mve_p1 as M
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/hetero_sc")

def region(elems, red, mn, mx, rng, cost_lo, cost_hi):
    """Cover elems with `red` layers of random sets -> each element freq ~= red."""
    sets, costs = [], []
    for _ in range(red):
        pool = list(elems); rng.shuffle(pool); i = 0
        while i < len(pool):
            sz = rng.randint(mn, mx); S = pool[i:i+sz]; i += sz
            if S: sets.append(sorted(S)); costs.append(round(rng.uniform(cost_lo, cost_hi), 3))
    return sets, costs

def gen_instance(rng, nA, nB, fA, fB):
    A = list(range(nA)); B = list(range(nA, nA + nB))
    sA, cA = region(A, fA, 2, 4, rng, 1.0, 1.6)   # low-f region
    sB, cB = region(B, fB, 2, 4, rng, 1.0, 1.6)   # high-f region
    sets = sA + sB; costs = cA + cB
    return sets, costs, nA + nB

def gen(dist, n, nA, nB, fA, fB, seed):
    d = os.path.join(DATA, dist); os.makedirs(d, exist_ok=True)
    rng = random.Random(seed)
    for k in range(n):
        sets, costs, N = gen_instance(rng, nA, nB, fA, fB)
        with gzip.open(os.path.join(d, "h_%s_%04d.json.gz" % (dist, k)), "wt") as f:
            json.dump({"num_elements": N, "sets": sets, "costs": costs}, f)
    print("wrote %d %s (nA=%d fA=%d, nB=%d fB=%d)" % (n, dist, nA, fA, nB, fB))

def load_dir(dist, cap=40):
    files = sorted(glob.glob(os.path.join(DATA, dist, "*.json.gz")))[:cap]
    with Pool(16) as pool:
        return [r for r in pool.map(M.prep, files) if r]

def evaluate(D, name):
    idx = list(range(len(D))); random.Random(0).shuffle(idx)
    h = len(idx)//2; tr = [D[i] for i in idx[:h]]; te = [D[i] for i in idx[h:]]
    edges2 = M.bucket_edges(2)
    lp = st.mean(M.reward_lpdefault(i) for i in te)
    tau = M.erm_single(tr); s1 = M.mean_reward_single(te, tau)
    th2 = M.erm_bucket(tr, 2, edges2); b2 = M.mean_reward_bucket(te, th2, edges2)
    sp = M.measure_speedup(te, th2, edges2)
    print("[%s] LP-def=%.3f single-tau*=%.3f(tau=%.2f) per-bucket-p2*=%.3f(th=%s) gain=%.3f speedup=%s"
          % (name, lp, s1, tau, b2, th2, b2 - s1, sp))
    return {"lp": round(lp,4), "single": round(s1,4), "tau": tau,
            "perbucket_p2": round(b2,4), "theta2": th2,
            "gain": round(b2 - s1, 4), "speedup": sp}

def main():
    # F2: strong frequency contrast (fA=2 vs fB=8); F1: mild (fA=3 vs fB=4) control
    gen("F2", 40, nA=45, nB=45, fA=2, fB=8, seed=11)
    gen("F1", 40, nA=45, nB=45, fA=3, fB=4, seed=12)
    F2 = load_dir("F2"); F1 = load_dir("F1")
    print("F2=%d F1=%d" % (len(F2), len(F1)))
    print("== diagnose F2 =="); M.diagnose(F2, 2)
    res = {"F2": evaluate(F2, "F2"), "F1": evaluate(F1, "F1")}
    json.dump(res, open(os.path.expanduser("~/projects/casp_max/outputs/run/mve_freq.json"), "w"), indent=1)
    g = res["F2"]["gain"]
    print("\n===== FREQ-HETERO DECISION =====")
    print("F2 gain(per-bucket-p2 - single, TRUE OPT-preserving) = %.3f" % g)
    print("GO if >=0.15 & speedup>1 ; this is the principled (1/f) per-bucket lever")

if __name__ == "__main__":
    main()
