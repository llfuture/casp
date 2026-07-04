#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Richer sample-complexity curve for the confidence-filter threshold theta*:
excess test loss vs N training instances, on Set Cover and Vertex Cover (eta=0.2).
Shows theta* is learnable from N ~ 5. -> outputs/run/r3_sample.json"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import r3_harden as RH
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
NS = [1, 2, 3, 5, 8, 12, 20]
ETA = 0.2

def sc_sample():
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        insts = [r for r in pool.map(R3.prep, files) if r]
    idx = list(range(len(insts))); random.Random(3).shuffle(idx)
    ntr = min(20, len(idx)//2); pool_tr = [insts[i] for i in idx[:ntr]]; te = [insts[i] for i in idx[ntr:]]
    sd = lambda x: hash((x["file"], ETA, "flip")) & 0xffff
    best = min(RH.THETA_SC, key=lambda t: st.mean(RH.sc_arms(x, RH.predict_sc(x, ETA, "flip", sd(x)), t)[2] for x in te))
    bv = st.mean(RH.sc_arms(x, RH.predict_sc(x, ETA, "flip", sd(x)), best)[2] for x in te)
    curve = []
    for N in NS:
        if N > len(pool_tr): break
        ex = []
        for tr in range(40):
            sub = random.Random(tr*7+1).sample(pool_tr, N)
            th = min(RH.THETA_SC, key=lambda t: st.mean(RH.sc_arms(x, RH.predict_sc(x, ETA, "flip", sd(x)), t)[2] for x in sub))
            v = st.mean(RH.sc_arms(x, RH.predict_sc(x, ETA, "flip", sd(x)), th)[2] for x in te)
            ex.append(v - bv)
        curve.append({"N": N, "excess": round(st.mean(ex), 4), "sd": round(st.pstdev(ex), 4)})
    print("SC sample:", curve, "best_loss", round(bv,4))
    return {"curve": curve, "best_loss": round(bv, 4), "n_test": len(te)}

def vc_sample():
    with Pool(16) as pool:
        recs = [r for r in pool.map(RH.prep_vc, range(160)) if r]
    idx = list(range(len(recs))); random.Random(3).shuffle(idx)
    ntr = min(40, len(idx)//2); pool_tr = [recs[i] for i in idx[:ntr]]; te = [recs[i] for i in idx[ntr:]]
    TVC = [0.0, 0.5, 1.0, 1.01]
    sd = lambda r: hash((r["seed"], ETA)) & 0xffff
    best = min(TVC, key=lambda t: st.mean(RH.vc_arms(r, RH.vc_predict(r, ETA, sd(r)), t)[2] for r in te))
    bv = st.mean(RH.vc_arms(r, RH.vc_predict(r, ETA, sd(r)), best)[2] for r in te)
    curve = []
    for N in NS:
        if N > len(pool_tr): break
        ex = []
        for tr in range(40):
            sub = random.Random(tr*7+2).sample(pool_tr, N)
            th = min(TVC, key=lambda t: st.mean(RH.vc_arms(r, RH.vc_predict(r, ETA, sd(r)), t)[2] for r in sub))
            v = st.mean(RH.vc_arms(r, RH.vc_predict(r, ETA, sd(r)), th)[2] for r in te)
            ex.append(v - bv)
        curve.append({"N": N, "excess": round(st.mean(ex), 4), "sd": round(st.pstdev(ex), 4)})
    print("VC sample:", curve, "best_loss", round(bv,4))
    return {"curve": curve, "best_loss": round(bv, 4), "n_test": len(te)}

def main():
    out = {"eta": ETA, "sc": sc_sample(), "vc": vc_sample()}
    json.dump(out, open(os.path.join(OUT, "r3_sample.json"), "w"), indent=1)
    print("SAVED r3_sample.json")

if __name__ == "__main__":
    main()
