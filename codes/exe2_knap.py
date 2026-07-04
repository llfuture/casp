#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-E2 (knapsack line): verified vs unverified ML exclusion on 0/1 Knapsack.

GBDT predicts per-item "not in any optimum" (excludable). Same predictions,
two deployments:
  unverified : force x_i = 0 for all predicted items, solve the rest
  verified   : accept exclusion of i only if the reduced-bound certificate
               fires: U_i (Dantzig bound with x_i=1 forced) < z_low (greedy
               incumbent)  =>  OPT-preserving (Thm knap); rejected predictions
               are ignored.
Train on uncorrelated+weakly (n in {100,200,500}); test ID (same types, n=1000)
and OOD (strongly correlated + subset-sum, all n<=1000). Expected: unverified
loses optimality on correlated types (items nearly interchangeable, ML
over-excludes); verified has ZERO value loss by theorem, honest low accept
rate on correlated types (the certificate is silent, as in E9).
"""
import sys, os, glob, json, statistics as st
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
import knapsack_casp as KC
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/knapsack")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
PTHRESH = 0.5


def prep(args):
    path, tlim = args
    try:
        v, w, W = L.load_knapsack(path)
        n = len(v)
        if n > 1200:
            return None
        opt, sol, stt, _ = KC.exact(v, w, W, tlim)
        if opt is None or stt != "optimal" or opt <= 0:
            return None
        return {"file": os.path.basename(path), "v": v, "w": w, "W": W,
                "opt": opt, "star": set(sol)}
    except Exception:
        return None


def features(inst):
    v, w, W = np.array(inst["v"], float), np.array(inst["w"], float), inst["W"]
    n = len(v)
    dens = v / np.maximum(w, 1e-9)
    order = np.argsort(-dens)
    rank = np.empty(n); rank[order] = np.arange(n) / n
    # Dantzig fractional solution: greedy fill by density
    cap = W; xfrac = np.zeros(n)
    for i in order:
        if w[i] <= cap:
            xfrac[i] = 1.0; cap -= w[i]
        else:
            xfrac[i] = cap / w[i]; break
    X = np.stack([rank, dens / dens.max(), w / W, v / v.max(), xfrac,
                  np.full(n, w.sum() / W)], 1)
    y = np.array([0 if i in inst["star"] else 1 for i in range(n)])
    return X, y


def solve_excl(inst, excl, tlim=120):
    v, w = inst["v"], inst["w"]
    keep = [i for i in range(len(v)) if i not in excl]
    vv = [v[i] for i in keep]; ww = [w[i] for i in keep]
    obj, _, stt, _ = KC.exact(vv, ww, inst["W"], tlim)
    return obj, stt


def deploy(inst, p):
    n = len(inst["v"])
    pred_ex = {i for i in range(n) if p[i] > PTHRESH}
    z_low, _ = KC.greedy_incumbent(inst["v"], inst["w"], inst["W"])
    ver_ex = set()
    for i in pred_ex:                      # verifier: reduced-bound exclusion
        Ui = KC.lp_bound(inst["v"], inst["w"], inst["W"], force1={i})
        if Ui < z_low - 1e-9:
            ver_ex.add(i)
    out = {"file": inst["file"], "n": n, "opt": inst["opt"],
           "pred_ex_frac": round(len(pred_ex) / n, 4),
           "accept_frac": round(len(ver_ex) / max(len(pred_ex), 1), 4)}
    for tag, ex in [("unverified", pred_ex), ("verified", ver_ex)]:
        obj, stt = solve_excl(inst, ex)
        rec = {"status": stt, "excl_frac": round(len(ex) / n, 4)}
        if obj is None:
            rec["violation"] = "no_solution"
        else:
            loss = 100 * (inst["opt"] - obj) / inst["opt"]
            rec["value_loss_pct"] = round(loss, 3)
            rec["violation"] = bool(loss > 1e-6)
        out[tag] = rec
    return out


def agg(rows, tag):
    ok = [r[tag] for r in rows]
    if not ok:
        return {"n": 0}
    losses = [r["value_loss_pct"] for r in ok if "value_loss_pct" in r]
    viol = sum(1 for r in ok if r.get("violation") is True or
               r.get("violation") == "no_solution")
    return {"n": len(ok), "violation_rate": round(viol / len(ok), 3),
            "loss_mean_pct": round(st.mean(losses), 3) if losses else None,
            "loss_max_pct": round(max(losses), 3) if losses else None,
            "excl_frac_mean": round(st.mean(r["excl_frac"] for r in ok), 3)}


def main():
    tr_files = sum([sorted(glob.glob(DATA + "/kp_%s_n%d_*.json.gz" % (t, n)))
                    for t in ["uncorrelated", "weakly_corr"] for n in [100, 200, 500]], [])
    id_files = sum([sorted(glob.glob(DATA + "/kp_%s_n1000_*.json.gz" % t))
                    for t in ["uncorrelated", "weakly_corr"]], [])
    ood_files = sum([sorted(glob.glob(DATA + "/kp_%s_n%d_*.json.gz" % (t, n)))
                     for t in ["strongly_corr", "subset_sum"] for n in [100, 200, 500, 1000]], [])
    with Pool(16) as pool:
        train = [r for r in pool.map(prep, [(p, 60) for p in tr_files]) if r]
        test_id = [r for r in pool.map(prep, [(p, 120) for p in id_files]) if r]
        test_ood = [r for r in pool.map(prep, [(p, 300) for p in ood_files]) if r]
    print("train=%d test_id=%d test_ood=%d" % (len(train), len(test_id), len(test_ood)),
          flush=True)

    X = np.vstack([features(i)[0] for i in train])
    Y = np.concatenate([features(i)[1] for i in train])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X, Y)
    print("GBDT trained on %d items (excludable rate %.3f)" % (len(Y), Y.mean()), flush=True)

    res = {}
    for split, insts in [("in_distribution", test_id), ("ood_correlated", test_ood)]:
        rows = []
        for inst in insts:
            p = clf.predict_proba(features(inst)[0])[:, 1]
            rows.append(deploy(inst, p))
            r = rows[-1]
            print("  [%s] %s unv loss=%s | ver loss=%s acc=%.2f" %
                  (split, r["file"][:30], r["unverified"].get("value_loss_pct"),
                   r["verified"].get("value_loss_pct"), r["accept_frac"]), flush=True)
        res[split] = {"rows": rows, "unverified": agg(rows, "unverified"),
                      "verified": agg(rows, "verified")}
        print("[%s] UNVERIFIED %s" % (split, res[split]["unverified"]), flush=True)
        print("[%s] VERIFIED   %s" % (split, res[split]["verified"]), flush=True)

    json.dump(res, open(os.path.join(OUT, "exe2_knap.json"), "w"), indent=1)
    print("SAVED exe2_knap.json")
    print("\n===== EX-E2 (knapsack) VERDICT =====")
    for sp in res:
        print("%s: unv viol=%s loss_max=%s | ver viol=%s (theorem: must be 0)" %
              (sp, res[sp]["unverified"].get("violation_rate"),
               res[sp]["unverified"].get("loss_max_pct"),
               res[sp]["verified"].get("violation_rate")))


if __name__ == "__main__":
    main()
