#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-G (review M12): does CASP pruning add speedup BEYOND SCIP presolve?
2x2 protocol on OR-Library Set Cover: {presolve on/off} x {CASP prune on/off}.
CASP arm uses the LP-threshold certificate (tau=1/f, f-safe, verifiable).
Reported: wall time per arm, marginal speedup = t(full,pre-on)/t(pruned,pre-on),
objective equality check (pruned obj must be >= full obj; equal when the
certificate happens to be OPT-preserving).
Solver: SCIP 10.0 via PySCIPOpt 6.2.1 (audited).
"""
import sys, os, glob, json, time, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

DATA = os.path.expanduser("~/projects/casp_max/data/benchmarks/orlib")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
TLIM = 300.0


def solve(universe, sets, costs, idx, presolve_on, tlim=TLIM):
    md = Model(); md.hideOutput()
    if not presolve_on:
        md.setPresolve(SCIP_PARAMSETTING.OFF)
    x = {i: md.addVar(vtype="B") for i in idx}
    md.setObjective(quicksum(costs[i] * x[i] for i in idx), "minimize")
    cover = {e: [] for e in universe}
    for i in idx:
        for e in sets[i]:
            if e in cover:
                cover[e].append(i)
    for e in universe:
        if not cover[e]:
            return None, None, "infeasible", 0.0
        md.addCons(quicksum(x[i] for i in cover[e]) >= 1)
    md.setParam("limits/time", tlim)
    t0 = time.time()
    md.optimize()
    el = time.time() - t0
    stat = md.getStatus()
    obj = md.getObjVal() if md.getNSols() > 0 else None
    return obj, stat, stat, el


def run_file(path):
    name = os.path.basename(path)
    d = L.load_scp(path)
    U, S, C = d["universe"], d["sets"], d["costs"]
    if len(S) > 6000 or len(U) > 3000:   # keep rail-scale out of the 2x2 budget
        return {"file": name, "skip": "too large for 2x2 budget", "m": len(S)}
    cert = L.sc_lp_threshold(U, S, C)
    surv = cert["survivors"]
    rec = {"file": name, "m": len(S), "u": len(U), "f": cert["f"],
           "prune_rate": round(cert["prune_rate"], 4)}
    arms = {}
    for tag, idx, pre in [("full_preON", list(range(len(S))), True),
                          ("full_preOFF", list(range(len(S))), False),
                          ("casp_preON", surv, True),
                          ("casp_preOFF", surv, False)]:
        obj, stat, _, el = solve(U, S, C, idx, pre)
        arms[tag] = {"obj": obj, "status": stat, "t": round(el, 2)}
        print("  %s %s: obj=%s status=%s t=%.1fs" % (name, tag, obj, stat, el),
              flush=True)
    rec["arms"] = arms
    fON, cON = arms["full_preON"], arms["casp_preON"]
    if fON["t"] > 0 and cON["t"] > 0:
        rec["marginal_speedup_preON"] = round(fON["t"] / max(cON["t"], 1e-3), 2)
    if arms["full_preOFF"]["t"] > 0:
        rec["presolve_own_speedup"] = round(
            arms["full_preOFF"]["t"] / max(fON["t"], 1e-3), 2)
    if fON["obj"] is not None and cON["obj"] is not None and \
       fON["status"] == "optimal" and cON["status"] == "optimal":
        rec["quality_gap_pct"] = round(100 * (cON["obj"] - fON["obj"]) / fON["obj"], 3)
        rec["opt_preserved"] = bool(abs(cON["obj"] - fON["obj"]) < 1e-6)
    return rec


def main(limit=12):
    files = sorted(glob.glob(DATA + "/scp*.txt"))[:limit]
    print("EX-G on %d OR-Library files, tlim=%ds/arm" % (len(files), TLIM), flush=True)
    rows = []
    for p in files:
        try:
            rows.append(run_file(p))
        except Exception as e:
            print("  SKIP %s: %s" % (os.path.basename(p), e), flush=True)
            rows.append({"file": os.path.basename(p), "skip": str(e)})
    ok = [r for r in rows if "arms" in r]
    spd = [r["marginal_speedup_preON"] for r in ok if "marginal_speedup_preON" in r]
    pres = [r["presolve_own_speedup"] for r in ok if "presolve_own_speedup" in r]
    out = {"tlim": TLIM, "solver": "SCIP 10.0 / PySCIPOpt 6.2.1",
           "rows": rows,
           "summary": {
               "n": len(ok),
               "marginal_speedup_preON": {
                   "mean": round(st.mean(spd), 2) if spd else None,
                   "median": round(st.median(spd), 2) if spd else None,
                   "max": round(max(spd), 2) if spd else None},
               "presolve_own_speedup_mean": round(st.mean(pres), 2) if pres else None,
               "opt_preserved_rate": round(st.mean(
                   1.0 * r.get("opt_preserved", False) for r in ok), 3) if ok else None}}
    json.dump(out, open(os.path.join(OUT, "exg_presolve.json"), "w"), indent=1)
    print("SAVED exg_presolve.json")
    print("\n===== EX-G VERDICT =====")
    print("CASP marginal speedup ON TOP of SCIP presolve:", out["summary"]["marginal_speedup_preON"])
    print("(>1 means pruning is NOT subsumed by presolve; ~1 means it is)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
