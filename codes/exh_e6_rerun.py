#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E6 re-run under the audited build (SCIP 10.0, default presolve ON):
wall-clock full-vs-pruned on the HARDER families -- large synthetic SC
(f10, m=2000) -- plus one rail probe (constraint-dominated ceiling).
Timing protocol: tlim per solve; pruning = LP-threshold tau=1/f (verifiable,
f-safe); LP+prune time charged to the CASP arm (net speedup, Prop overhead).
"""
import sys, os, glob, json, time, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from pyscipopt import Model, quicksum

DATA = os.path.expanduser("~/projects/casp_max/data")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
TLIM = 300.0


def solve(U, S, C, idx, tlim=TLIM):
    md = Model(); md.hideOutput()
    x = {i: md.addVar(vtype="B") for i in idx}
    md.setObjective(quicksum(C[i] * x[i] for i in idx), "minimize")
    cover = {e: [] for e in U}
    for i in idx:
        for e in S[i]:
            if e in cover:
                cover[e].append(i)
    for e in U:
        if not cover[e]:
            return None, "infeasible", 0.0
        md.addCons(quicksum(x[i] for i in cover[e]) >= 1)
    md.setParam("limits/time", tlim)
    t0 = time.time(); md.optimize(); el = time.time() - t0
    obj = md.getObjVal() if md.getNSols() > 0 else None
    return obj, md.getStatus(), el


def run_sc(path):
    name = os.path.basename(path)
    sc = L.load_sc_synth(path)
    U, S, C = sc["universe"], sc["sets"], sc["costs"]
    t0 = time.time()
    cert = L.sc_lp_threshold(U, S, C)          # LP + tau=1/f threshold
    t_cert = time.time() - t0
    o_full, s_full, t_full = solve(U, S, C, list(range(len(S))))
    o_red, s_red, t_red = solve(U, S, C, cert["survivors"])
    rec = {"file": name, "m": len(S), "u": len(U), "f": cert["f"],
           "prune_rate": round(cert["prune_rate"], 4), "t_cert": round(t_cert, 2),
           "full": {"obj": o_full, "status": s_full, "t": round(t_full, 2)},
           "casp": {"obj": o_red, "status": s_red, "t": round(t_red, 2)}}
    if t_full > 0:
        rec["net_speedup"] = round(t_full / max(t_red + t_cert, 1e-3), 2)
    if o_full is not None and o_red is not None and s_full == "optimal" and s_red == "optimal":
        rec["gap_pct"] = round(100 * (o_red - o_full) / o_full, 3)
        rec["opt_preserved"] = bool(abs(o_red - o_full) < 1e-6)
    print("  %s: full %.1fs (%s) casp %.1fs+%.1fs prune=%.2f spd=%s gap=%s"
          % (name, t_full, s_full, t_red, t_cert, cert["prune_rate"],
             rec.get("net_speedup"), rec.get("gap_pct")), flush=True)
    return rec


def rail_probe(path, tlim=TLIM):
    name = os.path.basename(path)
    d = L.load_scp(path)
    U, S, C = d["universe"], d["sets"], d["costs"]
    o, s, t = solve(U, S, C, list(range(len(S))), tlim)
    print("  RAIL %s: m=%d obj=%s status=%s t=%.1fs" % (name, len(S), o, s, t), flush=True)
    return {"file": name, "m": len(S), "obj": o, "status": s, "t": round(t, 2)}


def main(limit=40):
    files = sorted(glob.glob(DATA + "/synthetic/set_cover/sc_f10_m2000_*.json.gz"))[:limit]
    print("E6' on %d large synthetic SC (f10, m=2000), tlim=%ds" % (len(files), TLIM), flush=True)
    rows = []
    for p in files:
        try:
            rows.append(run_sc(p))
        except Exception as e:
            print("  SKIP %s: %s" % (os.path.basename(p), e), flush=True)
    rail = []
    for rp in sorted(glob.glob(DATA + "/benchmarks/orlib/rail*.txt"))[:1]:
        try:
            rail.append(rail_probe(rp))
        except Exception as e:
            print("  RAIL SKIP: %s" % e, flush=True)
    ok = [r for r in rows if "net_speedup" in r]
    spd = [r["net_speedup"] for r in ok]
    tf = [r["full"]["t"] for r in rows if r.get("full")]
    out = {"tlim": TLIM, "solver": "SCIP 10.0 / PySCIPOpt 6.2.1",
           "rows": rows, "rail_probe": rail,
           "summary": {"n": len(ok),
                       "full_t": {"median": round(st.median(tf), 2) if tf else None,
                                  "max": round(max(tf), 2) if tf else None},
                       "net_speedup": {"mean": round(st.mean(spd), 2) if spd else None,
                                       "median": round(st.median(spd), 2) if spd else None,
                                       "max": round(max(spd), 2) if spd else None},
                       "opt_preserved_rate": round(st.mean(
                           1.0 * r.get("opt_preserved", False) for r in ok), 3) if ok else None,
                       "gap_max_pct": max((r.get("gap_pct", 0) or 0) for r in ok) if ok else None}}
    json.dump(out, open(os.path.join(OUT, "exh_e6_rerun.json"), "w"), indent=1)
    print("SAVED exh_e6_rerun.json")
    print("\n===== E6' VERDICT =====")
    print("full-solve time median/max:", out["summary"]["full_t"])
    print("net speedup (incl. LP cost):", out["summary"]["net_speedup"])
    print("opt-preserved rate:", out["summary"]["opt_preserved_rate"],
          "max gap %:", out["summary"]["gap_max_pct"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
