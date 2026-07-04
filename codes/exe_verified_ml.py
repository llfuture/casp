#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-E (flagship): verified vs unverified ML pruning on Set Cover.

A GBDT predicts per-set "not in any optimum" (prunable). Two deployments of
the SAME predictions:
  unverified : delete all sets with p_i > 0.5, solve the残 instance
               (the ML-problem-reduction template, e.g. Sun-Ernst-Li line)
  verified   : delete only predicted sets that pass the CASP verifier
               (LP-threshold certificate: x*_i < tau = 1/f  =>  f-safe)
Train on distribution A (f5, m=500); test in-distribution and OOD
(f10, m=2000: frequency AND scale shift). Metrics per instance:
feasibility violation, quality gap vs true OPT, fraction pruned/accepted.
Expected: unverified degrades OOD (violations / big gaps); verified never
infeasible, gap bounded, keeps most of the speed-relevant pruning.
"""
import sys, os, glob, json, time, statistics as st
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/set_cover")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
PTHRESH = 0.5


def prep(args):
    path, tlim = args
    try:
        sc = L.load_sc_synth(path)
        U, S, C = sc["universe"], sc["sets"], sc["costs"]
        opt, sol, stt, _ = L.sc_exact(U, S, C, tlim=tlim)
        if opt is None or stt != "optimal" or opt <= 0:
            return None
        lp, xs = L.sc_lp(U, S, C)
        f = L.sc_freq(U, S)
        freq = {}
        for i, s in enumerate(S):
            for e in s:
                freq[e] = freq.get(e, 0) + 1
        return {"file": os.path.basename(path), "U": list(U), "S": S, "C": C,
                "opt": opt, "star": set(sol), "x": xs, "f": f, "freq": freq}
    except Exception:
        return None


def features(inst):
    S, C, xs, freq = inst["S"], inst["C"], inst["x"], inst["freq"]
    m = len(S)
    csort = sorted(C)
    rows = []
    for i in range(m):
        cov = max(len(S[i]), 1)
        fr = [freq[e] for e in S[i]] or [1]
        rows.append([
            xs[i],                                    # LP value
            np.searchsorted(csort, C[i]) / m,         # cost percentile
            cov / max(len(inst["U"]), 1),             # coverage fraction
            C[i] / cov,                               # cost per element
            min(fr), sum(fr) / len(fr),               # element frequency stats
        ])
    y = [0 if i in inst["star"] else 1 for i in range(m)]   # 1 = prunable
    return np.asarray(rows), np.asarray(y)


def solve_restricted(inst, keep_idx, tlim=120):
    opt, sol, stt, _ = L.sc_exact(inst["U"], inst["S"], inst["C"],
                                  restrict=sorted(keep_idx), tlim=tlim)
    return opt, stt


def deploy(inst, p):
    m = len(inst["S"])
    pred_del = {i for i in range(m) if p[i] > PTHRESH}
    tau = 1.0 / inst["f"]
    ver_del = {i for i in pred_del if inst["x"][i] < tau - 1e-9}   # verifier gate
    out = {"file": inst["file"], "m": m, "opt": inst["opt"],
           "pred_del_frac": round(len(pred_del) / m, 4),
           "accept_frac": round(len(ver_del) / max(len(pred_del), 1), 4)}
    for tag, dele in [("unverified", pred_del), ("verified", ver_del)]:
        keep = [i for i in range(m) if i not in dele]
        obj, stt = solve_restricted(inst, keep)
        rec = {"status": stt, "prune_frac": round(len(dele) / m, 4)}
        if obj is None or stt == "infeasible":
            rec["violation"] = "infeasible"
        else:
            gap = 100 * (obj - inst["opt"]) / inst["opt"]
            rec["gap_pct"] = round(gap, 3)
            rec["violation"] = bool(gap > 1e-6)
        out[tag] = rec
    return out


def agg(rows, tag):
    ok = [r[tag] for r in rows]
    if not ok:
        return {"n": 0}
    infeas = sum(1 for r in ok if r.get("violation") == "infeasible")
    gaps = [r["gap_pct"] for r in ok if "gap_pct" in r]
    viol = sum(1 for r in ok if r.get("violation") is True) + infeas
    return {"n": len(ok), "infeasible": infeas,
            "violation_rate": round(viol / max(len(ok), 1), 3),
            "gap_mean_pct": round(st.mean(gaps), 3) if gaps else None,
            "gap_max_pct": round(max(gaps), 3) if gaps else None,
            "prune_frac_mean": round(st.mean(r["prune_frac"] for r in ok), 3)}


def main():
    tr_files = sorted(glob.glob(DATA + "/sc_f5_m500_s1000_*.json.gz")) + \
               sorted(glob.glob(DATA + "/sc_f5_m500_s5000_*.json.gz"))
    ood_files = sorted(glob.glob(DATA + "/sc_f10_m500_s5000_*.json.gz"))[:15] + \
                sorted(glob.glob(DATA + "/sc_f20_m500_s5000_*.json.gz"))[:15]
    with Pool(16) as pool:
        tr_all = [r for r in pool.map(prep, [(p, 60) for p in tr_files]) if r]
    print("prepared f5 (train pool): %d" % len(tr_all), flush=True)
    ntr = int(0.7 * len(tr_all))
    train, test_id = tr_all[:ntr], tr_all[ntr:]
    with Pool(16) as pool:
        test_ood = [r for r in pool.map(prep, [(p, 300) for p in ood_files]) if r]
    print("prepared f10/f20 (OOD): %d" % len(test_ood), flush=True)
    if not test_ood:
        print("WARNING: no OOD instance solved to optimality; reporting ID only")

    X = np.vstack([features(i)[0] for i in train])
    Y = np.concatenate([features(i)[1] for i in train])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X, Y)
    print("GBDT trained on %d sets (prunable rate %.3f)" % (len(Y), Y.mean()), flush=True)

    res = {}
    for split, insts in [("in_distribution", test_id), ("ood_shift", test_ood)]:
        if not insts:
            res[split] = {"rows": [], "unverified": {"n": 0}, "verified": {"n": 0}}
            continue
        rows = []
        for inst in insts:
            p = clf.predict_proba(features(inst)[0])[:, 1]
            rows.append(deploy(inst, p))
            r = rows[-1]
            print("  [%s] %s unv: %s gap=%s | ver: gap=%s acc=%.2f" %
                  (split, r["file"], r["unverified"].get("violation"),
                   r["unverified"].get("gap_pct"), r["verified"].get("gap_pct"),
                   r["accept_frac"]), flush=True)
        res[split] = {"rows": rows,
                      "unverified": agg(rows, "unverified"),
                      "verified": agg(rows, "verified")}
        print("[%s] UNVERIFIED %s" % (split, res[split]["unverified"]), flush=True)
        print("[%s] VERIFIED   %s" % (split, res[split]["verified"]), flush=True)

    json.dump(res, open(os.path.join(OUT, "exe_verified_ml.json"), "w"), indent=1)
    print("SAVED exe_verified_ml.json")
    g = lambda sp, t: res[sp][t].get("violation_rate")
    print("\n===== EX-E VERDICT =====")
    print("unverified violation rate: ID=%s -> OOD=%s ; verified: ID=%s OOD=%s" %
          (g("in_distribution", "unverified"), g("ood_shift", "unverified"),
           g("in_distribution", "verified"), g("ood_shift", "verified")))
    print("story holds if unverified OOD >> ID and verified stays low with bounded gap")


if __name__ == "__main__":
    main()
