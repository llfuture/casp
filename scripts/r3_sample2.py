#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample-complexity curve for theta* (FAST: precompute per-(instance,theta) loss table,
so ERM/eval are lookups). Excess test loss vs N on Set Cover and Vertex Cover (eta=0.2).
-> outputs/run/r3_sample.json"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import r3_harden as RH
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
NS = [1, 2, 3, 5, 8, 12, 20]
ETA = 0.2

def curve_from_table(tab, thetas, pool_idx, test_idx, ntrials=40, seed0=1):
    """tab[i] = dict theta->loss for instance i. Return excess-vs-N curve."""
    def mean_loss(idxs, th): return st.mean(tab[i][th] for i in idxs)
    best = min(thetas, key=lambda th: mean_loss(test_idx, th))
    bv = mean_loss(test_idx, best)
    out = []
    for N in NS:
        if N > len(pool_idx): break
        ex = []
        for tr in range(ntrials):
            sub = random.Random(tr*7+seed0).sample(pool_idx, N)
            th = min(thetas, key=lambda t: mean_loss(sub, t))
            ex.append(mean_loss(test_idx, th) - bv)
        out.append({"N": N, "excess": round(st.mean(ex), 4), "sd": round(st.pstdev(ex), 4)})
    return {"curve": out, "best_loss": round(bv, 4), "n_test": len(test_idx)}

def sc_sample():
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        insts = [r for r in pool.map(R3.prep, files) if r]
    thetas = RH.THETA_SC
    tab = []
    for x in insts:
        sd = hash((x["file"], ETA, "flip")) & 0xffff
        pred = RH.predict_sc(x, ETA, "flip", sd)
        tab.append({th: RH.sc_arms(x, pred, th)[2] for th in thetas})
    idx = list(range(len(insts))); random.Random(3).shuffle(idx)
    ntr = min(20, len(idx)//2)
    res = curve_from_table(tab, thetas, idx[:ntr], idx[ntr:], seed0=1)
    print("SC:", res["curve"], "best", res["best_loss"])
    return res

def vc_sample():
    with Pool(16) as pool:
        recs = [r for r in pool.map(RH.prep_vc, range(160)) if r]
    thetas = [0.0, 0.5, 1.0, 1.01]
    tab = []
    for r in recs:
        sd = hash((r["seed"], ETA)) & 0xffff
        pred = RH.vc_predict(r, ETA, sd)
        tab.append({th: RH.vc_arms(r, pred, th)[2] for th in thetas})
    idx = list(range(len(recs))); random.Random(3).shuffle(idx)
    ntr = min(40, len(idx)//2)
    res = curve_from_table(tab, thetas, idx[:ntr], idx[ntr:], seed0=2)
    print("VC:", res["curve"], "best", res["best_loss"])
    return res

def main():
    out = {"eta": ETA, "sc": sc_sample(), "vc": vc_sample()}
    json.dump(out, open(os.path.join(OUT, "r3_sample.json"), "w"), indent=1)
    print("SAVED r3_sample.json")

if __name__ == "__main__":
    main()
