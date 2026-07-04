import sys, os, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import steiner_casp as S
from pyscipopt import Model, quicksum
from collections import deque

def solve_rowgen(n, edges, term, tlim=600):
    if len(term) <= 1: return (0.0, "optimal", 0.0, 0)
    r = min(term)
    m = Model(); m.hideOutput()
    y = [m.addVar(vtype="B") for _ in edges]
    m.setObjective(quicksum(w * y[i] for i, (u, v, w) in enumerate(edges)), "minimize")
    inc = [[] for _ in range(n)]
    for i, (u, v, w) in enumerate(edges): inc[u].append(i); inc[v].append(i)
    for t in term:
        if inc[t]: m.addCons(quicksum(y[i] for i in inc[t]) >= 1)
    t0 = time.time(); it = 0
    while True:
        rem = tlim - (time.time() - t0)
        if rem <= 0.5: return (m.getObjVal() if m.getNSols() > 0 else None, "timelimit", time.time()-t0, it)
        m.setParam("limits/time", rem); m.optimize()
        if m.getNSols() == 0: return (None, m.getStatus(), time.time()-t0, it)
        val = m.getObjVal(); status = m.getStatus()
        sel = [(edges[i][0], edges[i][1]) for i in range(len(edges)) if m.getVal(y[i]) > 0.5]
        adj = [[] for _ in range(n)]
        for (u, v) in sel: adj[u].append(v); adj[v].append(u)
        comp = [-1] * n; cid = 0
        for s0 in range(n):
            if comp[s0] == -1 and (adj[s0] or s0 in term):
                comp[s0] = cid; dq = deque([s0])
                while dq:
                    x = dq.popleft()
                    for yy in adj[x]:
                        if comp[yy] == -1: comp[yy] = cid; dq.append(yy)
                cid += 1
        rc = comp[r]
        missing = [t for t in term if comp[t] != rc]
        if not missing:
            return (val, status, time.time()-t0, it)
        if status != "optimal":
            return (val, "timelimit", time.time()-t0, it)
        cuts = set([rc]) | set(comp[t] for t in missing)
        m.freeTransform()
        for c in cuts:
            Sset = set(i for i in range(n) if comp[i] == c)
            cross = [i for i, (u, v, w) in enumerate(edges) if (u in Sset) != (v in Sset)]
            if cross: m.addCons(quicksum(y[i] for i in cross) >= 1)
        it += 1
        if it > 3000: return (val, "maxiter", time.time()-t0, it)

def run(name, tlim=600):
    base = os.path.expanduser("~/projects/casp_max/data/benchmarks/steinlib/B")
    n, edges, term = S.parse_stp(base + "/" + name + ".stp")
    surv, re_, rv = S.reduce_casp(n, edges, term)
    of, sf, tf, itf = solve_rowgen(n, edges, term, tlim)
    orr, sr, tr, itr = solve_rowgen(n, surv, term, tlim)
    return {"inst": name, "n": n, "m": len(edges), "term": len(term),
            "prune": round(1 - len(surv)/max(len(edges), 1), 3),
            "opt_full": of, "st_full": sf, "t_full": round(tf, 1), "it_full": itf,
            "opt_red": orr, "st_red": sr, "t_red": round(tr, 1),
            "both_proven": (sf == "optimal" and sr == "optimal"),
            "match": (of is not None and orr is not None and abs(of-orr) < 1e-6)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        for nm in ["b04", "b05", "b06", "b08"]: print(run(nm, tlim=60))
        sys.exit()
    from multiprocessing import Pool
    INST = ["b%02d" % i for i in range(1, 19)]
    OUT = os.path.expanduser("~/projects/casp_max/outputs/theoremD")
    open(OUT + "/steiner_rowgen.jsonl", "w").close()
    def wk(nm):
        r = run(nm, tlim=1800)
        open(OUT + "/steiner_rowgen.jsonl", "a").write(json.dumps(r) + "\n")
        return r
    with Pool(9) as p: res = p.map(wk, INST)
    PUB = {"b01":82,"b02":83,"b03":138,"b04":59,"b05":61,"b06":122,"b07":111,"b08":104,"b09":220,"b10":86,"b11":88,"b12":174,"b13":165,"b14":235,"b15":318,"b16":127,"b17":131,"b18":218}
    proven = [r for r in res if r["both_proven"]]
    summ = {"n": len(res), "proven": len(proven),
            "mismatches_among_proven": sum(1 for r in proven if not r["match"]),
            "all_match_published": all(abs(r["opt_full"]-PUB[r["inst"]]) < 1e-6 for r in proven),
            "mean_prune": round(sum(r["prune"] for r in res)/len(res), 3)}
    json.dump({"summary": summ, "rows": res}, open(OUT + "/steiner_rowgen.json", "w"), indent=1)
    print("ROWGEN:", json.dumps(summ)); print("ALLDONE")
