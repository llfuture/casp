#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-C (the make-or-break baseline, review M3): on the ORIGINAL E12 distributions,
add PREDICTION-FREE arms and ask whether the confidence filter's advantage
comes from the prediction or from the LP alone.

Arms (all end with min-with-fallback, deployment-fair):
  fb      greedy-on-full fallback (SC) / LP-rounding cover (VC)
  mc      min-combiner: commit noisy prediction wholesale, complete, min fb
  CF      confidence filter: commit pred & {x >= theta*}, ERM theta* on train
  LP1     prediction-free: commit {x >= 1-eps}, complete, min fb
  LPt     prediction-free: commit {x >= theta'}, ERM theta' on train, min fb
Key quantity: adv = best_prediction_free - CF  (per eta / noise model),
stratified by LP degeneracy (fraction of fractional x values).
Theory guardrails: CF <= mc always (Thm 15); adv can be ~0 on natural
distributions without hurting Thm 17 (which lives on degenerate faces).
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import route3_mve as R3
import r3_harden as H
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
THETA_SC = H.THETA_SC
THETA_VC = [0.0, 0.5, 1.0, 1.01]
ETAS = H.ETAS


def sc_degeneracy(inst):
    xs = inst["x"]
    return sum(1 for v in xs if 0.01 < v < 0.99) / max(1, len(xs))


def vc_degeneracy(rec):
    xs = rec["x"]
    return sum(1 for v in xs if 0.01 < v < 0.99) / max(1, len(xs))


def sc_pred_free(inst, theta):
    xs = inst["x"]
    com = {i for i in range(inst["m"]) if xs[i] >= theta - 1e-9}
    return min(R3.complete_cost(inst, com), inst["gfull"]) / inst["opt"]


def vc_pred_free(rec, theta):
    xs = rec["x"]
    com = {v for v in range(rec["n"]) if xs[v] >= theta - 1e-9}
    return min(len(H.vc_complete(rec["n"], rec["edges"], com)), rec["fb"]) / rec["opt"]


def learn(grid, lossf, group):
    return min(grid, key=lambda t: st.mean(lossf(x, t) for x in group))


def sc_block(insts):
    idx = list(range(len(insts))); random.Random(0).shuffle(idx); h = len(idx) // 2
    tr, te = [insts[i] for i in idx[:h]], [insts[i] for i in idx[h:]]
    # prediction-free arms are eta-independent: learn theta' once, eval once
    thp = learn(THETA_SC, sc_pred_free, tr)
    lp1_te = [sc_pred_free(x, 1.0) for x in te]
    lpt_te = [sc_pred_free(x, thp) for x in te]
    out = {"theta_pred_free": thp,
           "LP1": round(st.mean(lp1_te), 4), "LPt": round(st.mean(lpt_te), 4)}
    curves = {}
    for model in ["flip", "fp", "drop"]:
        rows = []
        for eta in ETAS:
            sd = lambda x: hash((x["file"], eta, model)) & 0xffff
            th = learn(THETA_SC,
                       lambda x, t: H.sc_arms(x, H.predict_sc(x, eta, model, sd(x)), t)[2], tr)
            fb = st.mean(H.sc_arms(x, H.predict_sc(x, eta, model, sd(x)), 0.0)[0] for x in te)
            mc = st.mean(H.sc_arms(x, H.predict_sc(x, eta, model, sd(x)), 0.0)[1] for x in te)
            cf_te = [H.sc_arms(x, H.predict_sc(x, eta, model, sd(x)), th)[2] for x in te]
            cf = st.mean(cf_te)
            best_pf = min(out["LP1"], out["LPt"])
            rows.append({"eta": eta, "fallback": round(fb, 4), "mincomb": round(mc, 4),
                         "CF": round(cf, 4), "LP1": out["LP1"], "LPt": out["LPt"],
                         "theta": th, "adv_pred": round(best_pf - cf, 4)})
            # degeneracy stratification of per-instance advantage
            degs = [sc_degeneracy(x) for x in te]
            qs = sorted(degs); cut1, cut2 = qs[len(qs)//3], qs[2*len(qs)//3]
            buck = {"lo": [], "mid": [], "hi": []}
            for x, c, l in zip(te, cf_te, lpt_te):
                b = "lo" if sc_degeneracy(x) <= cut1 else ("mid" if sc_degeneracy(x) <= cut2 else "hi")
                buck[b].append(l - c)
            rows[-1]["adv_by_deg"] = {k: round(st.mean(v), 4) if v else None for k, v in buck.items()}
        curves[model] = rows
        print("SC[%s]: " % model + " ".join(
            "e%.2f adv=%.3f" % (r["eta"], r["adv_pred"]) for r in rows), flush=True)
    return {"pred_free": out, "curves": curves,
            "deg_stats": [round(sc_degeneracy(x), 3) for x in te]}


def vc_block(recs):
    idx = list(range(len(recs))); random.Random(0).shuffle(idx); h = len(idx) // 2
    tr, te = [recs[i] for i in idx[:h]], [recs[i] for i in idx[h:]]
    thp = learn(THETA_VC, vc_pred_free, tr)
    lp1_te = [vc_pred_free(r, 1.0) for r in te]
    lpt_te = [vc_pred_free(r, thp) for r in te]
    out = {"theta_pred_free": thp,
           "LP1": round(st.mean(lp1_te), 4), "LPt": round(st.mean(lpt_te), 4)}
    rows = []
    for eta in ETAS:
        sd = lambda r: hash((r["seed"], eta)) & 0xffff
        th = learn(THETA_VC,
                   lambda r, t: H.vc_arms(r, H.vc_predict(r, eta, sd(r)), t)[2], tr)
        fb = st.mean(H.vc_arms(r, H.vc_predict(r, eta, sd(r)), 0.0)[0] for r in te)
        mc = st.mean(H.vc_arms(r, H.vc_predict(r, eta, sd(r)), 0.0)[1] for r in te)
        cf_te = [H.vc_arms(r, H.vc_predict(r, eta, sd(r)), th)[2] for r in te]
        cf = st.mean(cf_te)
        best_pf = min(out["LP1"], out["LPt"])
        row = {"eta": eta, "fallback": round(fb, 4), "mincomb": round(mc, 4),
               "CF": round(cf, 4), "LP1": out["LP1"], "LPt": out["LPt"],
               "theta": th, "adv_pred": round(best_pf - cf, 4)}
        degs = [vc_degeneracy(r) for r in te]
        qs = sorted(degs); cut1, cut2 = qs[len(qs)//3], qs[2*len(qs)//3]
        buck = {"lo": [], "mid": [], "hi": []}
        for r, c, l in zip(te, cf_te, lpt_te):
            b = "lo" if vc_degeneracy(r) <= cut1 else ("mid" if vc_degeneracy(r) <= cut2 else "hi")
            buck[b].append(l - c)
        row["adv_by_deg"] = {k: round(st.mean(v), 4) if v else None for k, v in buck.items()}
        rows.append(row)
    print("VC: " + " ".join("e%.2f adv=%.3f" % (r["eta"], r["adv_pred"]) for r in rows), flush=True)
    return {"pred_free": out, "rows": rows,
            "deg_stats": [round(vc_degeneracy(r), 3) for r in te]}


def main():
    files = sorted(glob.glob(R3.DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    with Pool(16) as pool:
        sc = [r for r in pool.map(R3.prep, files) if r]
    print("SC prepared %d" % len(sc), flush=True)
    sc_res = sc_block(sc)
    with Pool(16) as pool:
        vc = [r for r in pool.map(H.prep_vc, range(120)) if r]
    print("VC prepared %d" % len(vc), flush=True)
    vc_res = vc_block(vc)
    json.dump({"sc": sc_res, "vc": vc_res, "n": {"sc": len(sc), "vc": len(vc)}},
              open(os.path.join(OUT, "exc_lponly.json"), "w"), indent=1)
    print("SAVED exc_lponly.json")
    advs = [r["adv_pred"] for m in sc_res["curves"] for r in sc_res["curves"][m]] + \
           [r["adv_pred"] for r in vc_res["rows"]]
    print("\n===== EX-C VERDICT =====")
    print("adv(best pred-free - CF): min=%.4f max=%.4f mean=%.4f" %
          (min(advs), max(advs), st.mean(advs)))
    print("adv>0 -> prediction adds value beyond the LP on these distributions;")
    print("adv~0 -> value lives on degenerate faces only (Thm 17 / Prop degen).")


if __name__ == "__main__":
    main()
