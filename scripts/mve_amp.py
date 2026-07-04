#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 MVE amplification round: stronger, 3-region heterogeneous Set Cover (D3),
to test whether the per-bucket gain crosses the GO bar (0.15) and grows with p.
Three regions with distinct redundancy -> distinct observable mean-frequency ->
p=4 buckets can separate them where p=1 (single tau) cannot.
Reuses true-OPT-preserving reward machinery from mve_p1.
"""
import sys, os, glob, gzip, json, random, itertools, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import mve_p1 as M
import casp_lib as L
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/hetero_sc")

def layers(elems, red, mn, mx, rng):
    out = []
    for _ in range(red):
        pool = list(elems); rng.shuffle(pool); i = 0
        while i < len(pool):
            sz = rng.randint(mn, mx); out.append(pool[i:i+sz]); i += sz
    return out

def gen_instance3(rng, nH, nM, nC):
    """3 regions: H(red6,cheap,big) M(red4,mid) C(red2,pricey,small)."""
    sets, costs = [], []
    H = list(range(nH)); Mr = list(range(nH, nH+nM)); Cr = list(range(nH+nM, nH+nM+nC))
    for S in layers(H, 6, 3, 7, rng):
        if S: sets.append(sorted(S)); costs.append(round(rng.uniform(1.0, 1.3), 3))
    for S in layers(Mr, 4, 2, 5, rng):
        if S: sets.append(sorted(S)); costs.append(round(rng.uniform(1.3, 1.7), 3))
    for S in layers(Cr, 2, 2, 3, rng):
        if S: sets.append(sorted(S)); costs.append(round(rng.uniform(1.7, 2.3), 3))
    n = nH+nM+nC
    for _ in range(max(1, n // 25)):
        a = rng.choice(H); b = rng.choice(Cr)
        sets.append(sorted({a, b})); costs.append(round(rng.uniform(1.4, 1.9), 3))
    return sets, costs, n

def gen_D3(n=40, seed=7):
    d = os.path.join(DATA, "D3"); os.makedirs(d, exist_ok=True)
    rng = random.Random(seed)
    for k in range(n):
        sets, costs, N = gen_instance3(rng, 60, 45, 60)   # strong 3-region contrast
        with gzip.open(os.path.join(d, "h_D3_%04d.json.gz" % k), "wt") as f:
            json.dump({"num_elements": N, "sets": sets, "costs": costs}, f)
    print("wrote %d D3 instances" % n)

def erm_bucket_capped(train, p, edges, cap=300, seed=1):
    if p <= 2:
        cand = [list(t) for t in itertools.product(M.GRID, repeat=p)]
    else:
        rng = random.Random(seed)
        cand = [[rng.choice(M.GRID) for _ in range(p)] for _ in range(cap)]
    return max(cand, key=lambda th: st.mean(M.reward_bucket(i, th, edges) for i in train))

def load_dir(dist, cap=40):
    files = sorted(glob.glob(os.path.join(DATA, dist, "*.json.gz")))[:cap]
    with Pool(16) as pool:
        return [r for r in pool.map(M.prep, files) if r]

def main():
    gen_D3(40)
    D3 = load_dir("D3")
    print("D3=%d instances" % len(D3))
    # diagnose 4 buckets
    M.diagnose(D3, 4)
    idx = list(range(len(D3))); random.Random(0).shuffle(idx)
    h = len(idx)//2; tr = [D3[i] for i in idx[:h]]; te = [D3[i] for i in idx[h:]]
    lp = st.mean(M.reward_lpdefault(i) for i in te)
    tau = M.erm_single(tr); s1 = M.mean_reward_single(te, tau)
    row = {"lp_default": round(lp, 4), "single_tau": round(s1, 4), "tau_star": tau}
    for p in [2, 4]:
        edges = M.bucket_edges(p)
        th = erm_bucket_capped(tr, p, edges)
        bp = M.mean_reward_bucket(te, th, edges)
        sp = M.measure_speedup(te, th, edges)
        row["perbucket_p%d" % p] = round(bp, 4)
        row["gain_p%d_vs_single" % p] = round(bp - s1, 4)
        row["speedup_p%d" % p] = sp
        print("[D3] p=%d per-bucket=%.3f gain=%.3f speedup=%s" % (p, bp, bp - s1, sp))
    print("[D3] LP-def=%.3f single-tau*=%.3f" % (lp, s1))
    json.dump(row, open(os.path.expanduser("~/projects/casp_max/outputs/run/mve_amp.json"), "w"), indent=1)
    g4 = row["gain_p4_vs_single"]
    print("\n===== AMPLIFIED DECISION =====")
    print("D3 gain(p4 - single, TRUE OPT-preserving) = %.3f" % g4)
    print("GO if >=0.15 ; the p2->p4 trend shows whether more buckets help")

if __name__ == "__main__":
    main()
