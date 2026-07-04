#!/usr/bin/env python3
# D4: speedup robustness across solvers + presolve-off + net gain over pure presolve.
# Answers review C5/D4: "give CASP's net gain relative to pure presolve, multi-solver."
# Solvers: SCIP (default=presolve ON), SCIP (presolve OFF), CBC (2nd independent solver).
# Baseline for NET gain = SCIP-default (presolve ON) from scratch  ==  "pure presolve" solver.
import sys, os, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
import planted as PL
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from multiprocessing import Pool
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
os.makedirs(OUT, exist_ok=True)

def scip_vc(n, edges, w, restrict=None, presolve=True, tlim=300):
    V = restrict if restrict is not None else list(range(n)); Vs = set(V)
    md = Model(); md.hideOutput()
    if not presolve:
        md.setPresolve(SCIP_PARAMSETTING.OFF)
        md.setParam("presolving/maxrounds", 0)
    x = {v: md.addVar(vtype="B") for v in V}
    md.setObjective(quicksum(w[v]*x[v] for v in V), "minimize")
    for (u, v) in edges:
        if u in Vs and v in Vs: md.addCons(x[u]+x[v] >= 1)
    md.setParam("limits/time", tlim); t = time.time(); md.optimize(); el = time.time()-t
    if md.getNSols() > 0:
        return md.getObjVal(), el
    return None, el

def cbc_vc(n, edges, w, tlim=300):
    import pulp
    V = list(range(n))
    p = pulp.LpProblem("vc", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x{v}", cat="Binary") for v in V}
    p += pulp.lpSum(w[v]*x[v] for v in V)
    for (u, v) in edges: p += x[u]+x[v] >= 1
    t = time.time()
    p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=tlim)); el = time.time()-t
    val = pulp.value(p.objective)
    return (val, el) if val is not None else (None, el)

def one(item):
    n, k, s, seed = item
    nt, edges, w, info = PL.gen_planted(n, k, s, seed=seed)
    # CASP: LP half-integral kernel + brute-force the log-core
    t0 = time.time()
    lp, xs, P0, P1, Ph = L.vc_lp_halfint(nt, edges, w)
    core = set(Ph); ce = [(u, v) for (u, v) in edges if u in core and v in core]
    Sc = PL.brute_min_cover(core, ce) or set()
    casp_opt = len(P1) + len(Sc); casp_t = time.time()-t0
    # baselines
    d_opt, d_t = scip_vc(nt, edges, w, presolve=True)      # SCIP default = pure presolve baseline
    np_opt, np_t = scip_vc(nt, edges, w, presolve=False)   # presolve OFF
    try:
        cbc_opt, cbc_t = cbc_vc(nt, edges, w)              # 2nd solver
    except Exception as e:
        cbc_opt, cbc_t = None, None
    def sp(base): return round(base/max(casp_t, 1e-4), 2) if base else None
    mism = any(o is not None and abs(casp_opt-o) > 1e-6 for o in (d_opt, np_opt, cbc_opt))
    return {"n": nt, "core": len(Ph), "casp_opt": casp_opt, "casp_t": round(casp_t, 4),
            "scip_default_t": round(d_t, 4), "scip_nopre_t": round(np_t, 4),
            "cbc_t": round(cbc_t, 4) if cbc_t else None,
            "net_speedup_vs_presolve": sp(d_t),      # <-- the number the review asks for
            "speedup_vs_nopre": sp(np_t), "speedup_vs_cbc": sp(cbc_t),
            "mismatch": mism}

def main(tag, sizes, seeds, procs):
    items = [(n, max(8, n//25), 12, sd) for n in sizes for sd in range(seeds)]
    with Pool(procs) as p:
        res = p.map(one, items)
    def agg(key):
        v = [r[key] for r in res if r[key]]
        return {"mean": round(sum(v)/len(v), 1), "max": max(v), "min": min(v), "n": len(v)} if v else None
    summary = {"n": len(res), "mismatches": sum(int(r["mismatch"]) for r in res),
               "net_speedup_vs_presolve": agg("net_speedup_vs_presolve"),
               "speedup_vs_nopre": agg("speedup_vs_nopre"),
               "speedup_vs_cbc": agg("speedup_vs_cbc")}
    json.dump({"summary": summary, "rows": res}, open(f"{OUT}/D4_{tag}.json", "w"), indent=1)
    print(f"D4-{tag}:", json.dumps(summary))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mve"
    if mode == "mve":
        main("mve", [300, 600], 2, 4)          # Step -1 minimal viable experiment
    else:
        main("full", [400, 800, 1500, 2500, 4000], 6, 12)
    print("DONE")
