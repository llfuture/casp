#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-E3: robustness of the EX-E conclusion to the prediction threshold.
Re-runs the SC verified-vs-unverified deployment at p-thresholds
{0.3, 0.5, 0.7}, reusing exe_verified_ml's prep/features/deploy (preps once).
"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
from multiprocessing import Pool
import exe_verified_ml as E
from sklearn.ensemble import GradientBoostingClassifier

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")


def main():
    DATA = E.DATA
    tr_files = sorted(glob.glob(DATA + "/sc_f5_m500_s1000_*.json.gz")) + \
               sorted(glob.glob(DATA + "/sc_f5_m500_s5000_*.json.gz"))
    ood_files = sorted(glob.glob(DATA + "/sc_f10_m500_s5000_*.json.gz"))[:15] + \
                sorted(glob.glob(DATA + "/sc_f20_m500_s5000_*.json.gz"))[:15]
    with Pool(16) as pool:
        tr_all = [r for r in pool.map(E.prep, [(p, 60) for p in tr_files]) if r]
        test_ood = [r for r in pool.map(E.prep, [(p, 300) for p in ood_files]) if r]
    ntr = int(0.7 * len(tr_all))
    train, test_id = tr_all[:ntr], tr_all[ntr:]
    print("train=%d id=%d ood=%d" % (len(train), len(test_id), len(test_ood)), flush=True)

    X = np.vstack([E.features(i)[0] for i in train])
    Y = np.concatenate([E.features(i)[1] for i in train])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X, Y)

    res = {}
    for th in [0.3, 0.5, 0.7]:
        E.PTHRESH = th
        for split, insts in [("id", test_id), ("ood", test_ood)]:
            rows = [E.deploy(inst, clf.predict_proba(E.features(inst)[0])[:, 1])
                    for inst in insts]
            key = "p%.1f_%s" % (th, split)
            res[key] = {"unverified": E.agg(rows, "unverified"),
                        "verified": E.agg(rows, "verified")}
            u, v = res[key]["unverified"], res[key]["verified"]
            print("[%s] unv: viol=%s infeas=%s gapmax=%s | ver: infeas=%s gapmax=%s prune=%s"
                  % (key, u.get("violation_rate"), u.get("infeasible"),
                     u.get("gap_max_pct"), v.get("infeasible"),
                     v.get("gap_max_pct"), v.get("prune_frac_mean")), flush=True)

    json.dump(res, open(os.path.join(OUT, "exe3_sweep.json"), "w"), indent=1)
    print("SAVED exe3_sweep.json")
    print("\n===== EX-E3 VERDICT =====")
    bad = [k for k, r in res.items()
           if "ood" in k and (r["verified"].get("infeasible") or 0) > 0]
    print("verified infeasible counts across thresholds (must all be 0):",
          "OK" if not bad else ("FAIL at " + ",".join(bad)))


if __name__ == "__main__":
    main()
