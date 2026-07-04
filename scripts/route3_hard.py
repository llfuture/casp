#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route-3 HARDENING (honest gate): is the win genuine ADAPTIVE learning, or just
one-time constant tuning?  Deploy under UNKNOWN/MIXED noise (each instance drawn
eta ~ {0.1,0.2,0.4}). Compare:
  MINCOMB      : min(commit-all, fallback)                       (collapse baseline)
  BEST-FIXED   : single best theta* over the mixture             (strongest non-adaptive tuning)
  ADAPTIVE     : theta chosen per-instance from an OBSERVABLE noise proxy (learned per-bucket)
  ORACLE       : per-instance best theta (upper bound)
Proxy = fraction of committed predicted sets with low LP confidence (x*<0.1) -- observable,
no OPT access. If ADAPTIVE < BEST-FIXED by a real margin -> genuine learning under noise drift.
If ADAPTIVE ~= BEST-FIXED -> a constant suffices; 'learning' is thin.
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import casp_lib as L
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
THETA = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.01]
PROXY_EDGES = [0.15, 0.35]   # 3 proxy buckets
ETAS = [0.1, 0.2, 0.4]

def prep_all():
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        return [r for r in pool.map(R3.prep, files) if r]

def make(inst):
    """Assign one random eta; build prediction; return per-instance record."""
    eta = random.Random(hash(inst["file"]) & 0xffff).choice(ETAS)
    seed = hash((inst["file"], "h")) & 0xffff
    pred = R3.predict(inst, eta, seed)
    xs = inst["x"]
    proxy = (sum(1 for i in pred if xs[i] < 0.1) / len(pred)) if pred else 0.0
    return {"inst": inst, "pred": pred, "eta": eta, "proxy": proxy}

def loss_at(rec, theta):
    inst = rec["inst"]; xs = inst["x"]
    committed = {i for i in rec["pred"] if xs[i] >= theta - 1e-9}
    cost = R3.complete_cost(inst, committed)
    return min(cost, inst["gfull"]) / inst["opt"]

def pbucket(proxy):
    b = 0
    for e in PROXY_EDGES:
        if proxy >= e: b += 1
    return b

def main():
    insts = prep_all()
    recs = [make(x) for x in insts]
    print("prepared %d recs; eta mix=%s" % (len(recs), {e: sum(r['eta']==e for r in recs) for e in ETAS}))
    idx = list(range(len(recs))); random.Random(0).shuffle(idx)
    h = len(idx)//2; tr = [recs[i] for i in idx[:h]]; te = [recs[i] for i in idx[h:]]

    # MINCOMB (theta=1.01 => commit nothing => fallback; but min-combiner uses commit-all)
    def mincomb(rec):
        inst = rec["inst"]
        cost_all = R3.complete_cost(inst, rec["pred"])
        return min(cost_all, inst["gfull"]) / inst["opt"]
    mc = st.mean(mincomb(r) for r in te)

    # BEST-FIXED theta over mixture
    tbest = min(THETA, key=lambda th: st.mean(loss_at(r, th) for r in tr))
    bf = st.mean(loss_at(r, tbest) for r in te)

    # ADAPTIVE: learn theta per proxy-bucket
    theta_by_b = {}
    for b in range(len(PROXY_EDGES)+1):
        grp = [r for r in tr if pbucket(r["proxy"]) == b]
        theta_by_b[b] = min(THETA, key=lambda th: st.mean(loss_at(r, th) for r in grp)) if grp else tbest
    ad = st.mean(loss_at(r, theta_by_b[pbucket(r["proxy"])]) for r in te)

    # ORACLE per-instance best theta
    orc = st.mean(min(loss_at(r, th) for th in THETA) for r in te)

    print("MINCOMB=%.4f  BEST-FIXED=%.4f (theta=%.2f)  ADAPTIVE=%.4f  ORACLE=%.4f"
          % (mc, bf, tbest, ad, orc))
    print("theta_by_proxybucket:", theta_by_b)
    print("gap ADAPTIVE beats BEST-FIXED = %.4f" % (bf - ad))
    print("gap BEST-FIXED beats MINCOMB  = %.4f" % (mc - bf))
    out = {"mincomb": round(mc,4), "best_fixed": round(bf,4), "best_theta": tbest,
           "adaptive": round(ad,4), "oracle": round(orc,4),
           "theta_by_bucket": theta_by_b,
           "adaptive_beats_fixed": round(bf-ad,4), "fixed_beats_mincomb": round(mc-bf,4),
           "eta_mix": {str(e): sum(r['eta']==e for r in recs) for e in ETAS}}
    json.dump(out, open(os.path.join(OUT, "route3_hard.json"), "w"), indent=1)
    print("\n===== ROUTE-3 HARDENING VERDICT =====")
    if (bf - ad) >= 0.01 and (mc - bf) >= 0.03:
        print("STRONG: adaptive learning beats best-fixed AND best-fixed beats min-combiner -> real learning under drift")
    elif (mc - bf) >= 0.03:
        print("MODERATE: beats min-combiner via a tuned constant; adaptivity adds little (learning=tuning)")
    else:
        print("WEAK: even best-fixed ~ min-combiner -> route-3 also thin")

if __name__ == "__main__":
    main()
