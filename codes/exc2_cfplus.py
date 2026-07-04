#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-C2: two-threshold confidence filter CF+ (repairs the EX-C interface gap).

CF+ commits (pred & {x >= th1}) | {x >= th2}, then completes, then min-fb.
Its policy family CONTAINS: CF (th2=1.01), LP-threshold-only (th1=1.01),
min-combiner (th1=0, th2=1.01) and fallback (both 1.01) -- so with joint ERM
it should weakly dominate every single-threshold arm at every noise level.
PAC cost: two parameters => Pdim = O(2 log K) (Thm multipac, p=2).
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import r3_harden as H
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
TH1 = H.THETA_SC                    # prediction gate
TH2 = [0.4, 0.8, 1.01]              # LP-only gate (1.01 = off)
ETAS = H.ETAS


def cfplus_loss(inst, pred, th1, th2):
    xs = inst["x"]
    com = {i for i in pred if xs[i] >= th1 - 1e-9} | \
          {i for i in range(inst["m"]) if xs[i] >= th2 - 1e-9}
    return min(R3.complete_cost(inst, com), inst["gfull"]) / inst["opt"]


def main():
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        insts = [r for r in pool.map(R3.prep, files) if r]
    print("SC prepared %d" % len(insts), flush=True)
    idx = list(range(len(insts))); random.Random(0).shuffle(idx); h = len(idx) // 2
    tr, te = [insts[i] for i in idx[:h]], [insts[i] for i in idx[h:]]

    res = {}
    for model in ["flip", "fp", "drop"]:
        rows = []
        for eta in ETAS:
            sd = lambda x: hash((x["file"], eta, model)) & 0xffff
            pr = lambda x: H.predict_sc(x, eta, model, sd(x))
            # single-threshold arms (as in EX-C)
            th_cf = min(TH1, key=lambda t: st.mean(
                H.sc_arms(x, pr(x), t)[2] for x in tr))
            cf = st.mean(H.sc_arms(x, pr(x), th_cf)[2] for x in te)
            th_lp = min(TH1, key=lambda t: st.mean(
                cfplus_loss(x, set(), 1.01, t) for x in tr))
            lpt = st.mean(cfplus_loss(x, set(), 1.01, th_lp) for x in te)
            mc = st.mean(H.sc_arms(x, pr(x), 0.0)[1] for x in te)
            # joint ERM for CF+
            best = min(((t1, t2) for t1 in TH1 for t2 in TH2),
                       key=lambda p: st.mean(cfplus_loss(x, pr(x), *p) for x in tr))
            cfp = st.mean(cfplus_loss(x, pr(x), *best) for x in te)
            rows.append({"eta": eta, "mincomb": round(mc, 4), "CF": round(cf, 4),
                         "LPt": round(lpt, 4), "CFplus": round(cfp, 4),
                         "th": list(best),
                         "gain_vs_best_single": round(min(cf, lpt) - cfp, 4)})
            print("SC[%s] e%.2f: mc=%.4f CF=%.4f LPt=%.4f CF+=%.4f (th1=%.2f th2=%.2f)"
                  % (model, eta, mc, cf, lpt, cfp, best[0], best[1]), flush=True)
        res[model] = rows
    json.dump({"curves": res, "n": len(insts)},
              open(os.path.join(OUT, "exc2_cfplus.json"), "w"), indent=1)
    print("SAVED exc2_cfplus.json")
    worst = min(r["gain_vs_best_single"] for m in res for r in res[m])
    print("\n===== EX-C2 VERDICT =====")
    print("min gain of CF+ over best single-threshold arm: %.4f" % worst)
    print("(>= -0.005 acceptable: joint ERM generalization slack; "
          ">0 means strict improvement)")


if __name__ == "__main__":
    main()
