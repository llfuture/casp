#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-I: rail instances with a CORRECT loader (Beasley rail format is transposed
relative to scp4x: header "m n", then one line per COLUMN: cost k row_1..row_k).
Questions: (1) does the tau=1/f LP-threshold certificate prune anything when
element frequencies are huge? (2) incumbent quality within a time budget,
full vs pruned. Calibration: rail507 best known optimum = 174.
"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
from pyscipopt import Model, quicksum

DATA = os.path.expanduser("~/projects/casp_max/data/benchmarks/orlib")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
TLIM = 600.0


def load_rail(path):
    """Beasley rail: m rows(elements), n cols(sets); per column: cost k rows."""
    t = open(path).read().split(); it = iter(t)
    m = int(next(it)); n = int(next(it))
    costs, sets = [], []
    for _ in range(n):
        c = float(next(it)); k = int(next(it))
        costs.append(c)
        sets.append([int(next(it)) - 1 for _ in range(k)])
    universe = set(range(m))
    cov = set()
    for s in sets:
        cov |= set(s)
    assert cov == universe, "loader sanity: some row uncovered"
    return universe, sets, costs


def solve(U, S, C, idx, tlim):
    md = Model(); md.hideOutput()
    x = {i: md.addVar(vtype="B") for i in idx}
    md.setObjective(quicksum(C[i] * x[i] for i in idx), "minimize")
    cover = {e: [] for e in U}
    for i in idx:
        for e in S[i]:
            cover[e].append(i)
    for e in U:
        if not cover[e]:
            return None, None, "infeasible", 0.0
        md.addCons(quicksum(x[i] for i in cover[e]) >= 1)
    md.setParam("limits/time", tlim)
    t0 = time.time(); md.optimize(); el = time.time() - t0
    obj = md.getObjVal() if md.getNSols() > 0 else None
    gap = md.getGap() if md.getNSols() > 0 else None
    return obj, gap, md.getStatus(), el


def lp_and_threshold(U, S, C):
    md = Model(); md.hideOutput()
    x = [md.addVar(vtype="C", lb=0, ub=1) for _ in S]
    md.setObjective(quicksum(C[i] * x[i] for i in range(len(S))), "minimize")
    cover = {e: [] for e in U}
    for i, s in enumerate(S):
        for e in s:
            cover[e].append(i)
    for e in U:
        md.addCons(quicksum(x[i] for i in cover[e]) >= 1)
    t0 = time.time(); md.optimize(); t_lp = time.time() - t0
    xs = [md.getVal(v) for v in x]
    f = max(len(cover[e]) for e in U)
    tau = 1.0 / f
    surv = [i for i in range(len(S)) if xs[i] >= tau - 1e-9]
    return {"lp": md.getObjVal(), "t_lp": round(t_lp, 1), "f": f, "tau": tau,
            "survivors": surv, "prune_rate": 1.0 - len(surv) / len(S)}


def run(path):
    name = os.path.basename(path)
    U, S, C = load_rail(path)
    print("%s: rows=%d cols=%d unit-ish costs [%g,%g]" %
          (name, len(U), len(S), min(C), max(C)), flush=True)
    cert = lp_and_threshold(U, S, C)
    print("  LP=%.2f in %.1fs; f=%d tau=%.2e prune_rate=%.4f" %
          (cert["lp"], cert["t_lp"], cert["f"], cert["tau"], cert["prune_rate"]),
          flush=True)
    o_full, g_full, s_full, t_full = solve(U, S, C, list(range(len(S))), TLIM)
    print("  FULL: obj=%s gap=%s status=%s t=%.0fs" % (o_full, g_full, s_full, t_full),
          flush=True)
    rec = {"file": name, "rows": len(U), "cols": len(S), "lp": cert["lp"],
           "t_lp": cert["t_lp"], "f": cert["f"], "prune_rate": round(cert["prune_rate"], 4),
           "full": {"obj": o_full, "gap": g_full, "status": s_full, "t": round(t_full, 1)}}
    if cert["prune_rate"] > 0.01:
        o_r, g_r, s_r, t_r = solve(U, S, C, cert["survivors"], TLIM)
        rec["pruned"] = {"obj": o_r, "gap": g_r, "status": s_r, "t": round(t_r, 1)}
        print("  PRUNED: obj=%s status=%s t=%.0fs" % (o_r, s_r, t_r), flush=True)
    else:
        rec["pruned"] = "skipped (certificate silent: prune_rate <= 1%)"
        print("  PRUNED: skipped, certificate silent", flush=True)
    return rec


def main():
    files = [DATA + "/rail507.txt", DATA + "/rail516.txt"]
    rows = []
    for p in files:
        try:
            rows.append(run(p))
        except Exception as e:
            print("  SKIP %s: %s" % (os.path.basename(p), e), flush=True)
    json.dump({"tlim": TLIM, "known_best": {"rail507": 174, "rail516": 182},
               "rows": rows}, open(os.path.join(OUT, "exi_rail.json"), "w"), indent=1)
    print("SAVED exi_rail.json")
    print("\n===== EX-I VERDICT =====")
    for r in rows:
        print("%s: f=%s prune=%s full_inc=%s (known best 507:174 / 516:182)" %
              (r["file"], r["f"], r["prune_rate"], r["full"]["obj"]))
    print("hypothesis: huge f makes tau=1/f admit ~no pruning -> the honest")
    print("bottleneck on rail is the frequency parameter, not the row count")


if __name__ == "__main__":
    main()
