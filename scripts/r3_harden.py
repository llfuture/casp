#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route-3 HARDENING for the JMLR domination result:
  (1) SC consistency/robustness curves under 3 noise models (flip / false-positive / drop),
      arms: fallback, MIN-COMBINER, CONFIDENCE-FILTER (learned theta on verifiable LP value).
  (2) SC sample-complexity curve for theta (scalar learnability).
  (3) VC second problem: same domination on Vertex Cover (confidence = LP half-integral value).
Confidence-filter family CONTAINS the min-combiner (theta=0) and fallback (theta>max),
so best-theta CF <= min-combiner by construction; we quantify the strict margin.
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import casp_lib as L
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
THETA_SC = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.01]
ETAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# =============================== SET COVER ===============================
def predict_sc(inst, eta, model, seed):
    rng = random.Random(seed); star = inst["star"]; m = inst["m"]
    non = [i for i in range(m) if i not in star]; rng.shuffle(non)
    if model == "flip":
        pred = set(i for i in star if rng.random() > eta); nfp = int(eta * len(star))
    elif model == "fp":          # false-positive heavy: keep all OPT, add lots of junk
        pred = set(star); nfp = int(2 * eta * len(star))
    else:                        # drop heavy: miss OPT sets, little junk
        pred = set(i for i in star if rng.random() > 2 * eta); nfp = int(0.3 * eta * len(star))
    pred |= set(non[:max(0, nfp)])
    return pred

def sc_arms(inst, pred, theta):
    xs = inst["x"]; opt = inst["opt"]; fb = inst["gfull"]
    ca = R3.complete_cost(inst, pred)
    den = {i for i in pred if xs[i] >= theta - 1e-9}
    cl = R3.complete_cost(inst, den)
    return fb / opt, min(ca, fb) / opt, min(cl, fb) / opt   # fallback, mincomb, CF

def sc_curves(insts):
    idx = list(range(len(insts))); random.Random(0).shuffle(idx); h = len(idx)//2
    tr = [insts[i] for i in idx[:h]]; te = [insts[i] for i in idx[h:]]
    res = {}
    for model in ["flip", "fp", "drop"]:
        rows = []
        for eta in ETAS:
            sd = lambda x: hash((x["file"], eta, model)) & 0xffff
            th = min(THETA_SC, key=lambda t:
                     st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), t)[2] for x in tr))
            fb = st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), 0.0)[0] for x in te)
            mc = st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), 0.0)[1] for x in te)
            cf = st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), th)[2] for x in te)
            rows.append({"eta": eta, "fallback": round(fb,4), "mincomb": round(mc,4),
                         "CF": round(cf,4), "dom_margin": round(mc-cf,4), "theta": th})
        res[model] = rows
        print("SC[%s]: " % model + " ".join("η%.2f dom=%.3f" % (r["eta"], r["dom_margin"]) for r in rows))
    return res

def sc_sample_curve(insts, eta=0.2, model="flip"):
    idx = list(range(len(insts))); random.Random(1).shuffle(idx); h = len(idx)//2
    pool = [insts[i] for i in idx[:h]]; te = [insts[i] for i in idx[h:]]
    sd = lambda x: hash((x["file"], eta, model)) & 0xffff
    best = min(THETA_SC, key=lambda t: st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), t)[2] for x in te))
    best_val = st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), best)[2] for x in te)
    curve = []
    for N in [2, 5, 10, 20, 40]:
        if N > len(pool): break
        excess = []
        for tr in range(20):
            sub = random.Random(tr).sample(pool, N)
            th = min(THETA_SC, key=lambda t: st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), t)[2] for x in sub))
            v = st.mean(sc_arms(x, predict_sc(x, eta, model, sd(x)), th)[2] for x in te)
            excess.append(v - best_val)
        curve.append({"N": N, "excess_loss": round(st.mean(excess), 4)})
    print("SC sample curve (eta=0.2 flip):", curve)
    return {"best_theta": best, "best_loss": round(best_val,4), "curve": curve}

# =============================== VERTEX COVER ===============================
def gen_vc(seed, n=45, p=0.09):
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u+1, n) if rng.random() < p]
    return n, edges, [1.0]*n

def vc_complete(n, edges, chosen):
    chosen = set(chosen); cov = set()
    for (u, v) in edges:
        if u in chosen or v in chosen: cov.add((u, v))
    unc = [e for e in edges if e not in cov]
    # greedily add the vertex covering most uncovered edges
    from collections import Counter
    while unc:
        cnt = Counter()
        for (u, v) in unc: cnt[u]+=1; cnt[v]+=1
        best = cnt.most_common(1)[0][0]; chosen.add(best)
        unc = [(u,v) for (u,v) in unc if u!=best and v!=best]
    return chosen

def prep_vc(seed):
    n, edges, w = gen_vc(seed)
    if not edges: return None
    opt, sol, sto, _ = L.vc_exact(n, edges, w, tlim=10)
    if opt is None or sto != "optimal" or opt <= 0: return None
    lp, xs, P0, P1, Ph = L.vc_lp_halfint(n, edges, w)
    fb_set = vc_complete(n, edges, {v for v in range(n) if xs[v] >= 0.5})  # LP-round fallback (valid cover)
    return {"seed": seed, "n": n, "edges": edges, "opt": opt, "star": set(sol),
            "x": xs, "fb": len(fb_set)}

def vc_predict(rec, eta, seed):
    rng = random.Random(seed); star = rec["star"]; n = rec["n"]
    non = [v for v in range(n) if v not in star]; rng.shuffle(non)
    pred = set(v for v in star if rng.random() > eta)
    pred |= set(non[:int(eta*len(star))])
    return pred

def vc_arms(rec, pred, theta):
    n, edges = rec["n"], rec["edges"]; xs = rec["x"]; opt = rec["opt"]; fb = rec["fb"]
    ca = len(vc_complete(n, edges, pred))
    den = {v for v in pred if xs[v] >= theta - 1e-9}
    cl = len(vc_complete(n, edges, den))
    return fb/opt, min(ca, fb)/opt, min(cl, fb)/opt

def vc_curves(recs):
    idx = list(range(len(recs))); random.Random(0).shuffle(idx); h = len(idx)//2
    tr = [recs[i] for i in idx[:h]]; te = [recs[i] for i in idx[h:]]
    THETA_VC = [0.0, 0.5, 1.0, 1.01]
    rows = []
    for eta in ETAS:
        sd = lambda r: hash((r["seed"], eta)) & 0xffff
        th = min(THETA_VC, key=lambda t: st.mean(vc_arms(r, vc_predict(r, eta, sd(r)), t)[2] for r in tr))
        fb = st.mean(vc_arms(r, vc_predict(r, eta, sd(r)), 0.0)[0] for r in te)
        mc = st.mean(vc_arms(r, vc_predict(r, eta, sd(r)), 0.0)[1] for r in te)
        cf = st.mean(vc_arms(r, vc_predict(r, eta, sd(r)), th)[2] for r in te)
        rows.append({"eta": eta, "fallback": round(fb,4), "mincomb": round(mc,4),
                     "CF": round(cf,4), "dom_margin": round(mc-cf,4), "theta": th})
    print("VC: " + " ".join("η%.2f dom=%.3f" % (r["eta"], r["dom_margin"]) for r in rows))
    return rows

def main():
    # SC
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        sc = [r for r in pool.map(R3.prep, files) if r]
    print("SC prepared %d" % len(sc))
    sc_c = sc_curves(sc)
    sc_s = sc_sample_curve(sc)
    # VC
    with Pool(16) as pool:
        vc = [r for r in pool.map(prep_vc, range(120)) if r]
    print("VC prepared %d" % len(vc))
    vc_c = vc_curves(vc)
    out = {"sc_curves": sc_c, "sc_sample": sc_s, "vc_curves": vc_c,
           "n": {"sc": len(sc), "vc": len(vc)}}
    json.dump(out, open(os.path.join(OUT, "r3_harden.json"), "w"), indent=1)
    print("SAVED r3_harden.json")
    # summary verdict
    sc_dom = [r["dom_margin"] for m in sc_c for r in sc_c[m]]
    vc_dom = [r["dom_margin"] for r in vc_c]
    print("\n===== HARDENING SUMMARY =====")
    print("SC dom margin: min=%.3f max=%.3f (all >=0 required)" % (min(sc_dom), max(sc_dom)))
    print("VC dom margin: min=%.3f max=%.3f" % (min(vc_dom), max(vc_dom)))

if __name__ == "__main__":
    main()
