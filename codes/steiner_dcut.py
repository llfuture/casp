#!/usr/bin/env python3
# Strong Steiner solver: bidirected-cut formulation + max-flow connectivity-cut separation
# via a pyscipopt constraint handler (lazy cuts at integer + fractional nodes).
import sys, os, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import steiner_casp as S
from pyscipopt import Model, Conshdlr, SCIP_RESULT, quicksum
import networkx as nx

class ConnCuts(Conshdlr):
    def setup(self, model, arcs, xv, r, terms):
        self.m = model; self.arcs = arcs; self.xv = xv; self.r = r
        self.others = [t for t in terms if t != r]
    def _sep(self, sol):
        G = nx.DiGraph()
        for (u, v) in self.arcs:
            cap = self.m.getSolVal(sol, self.xv[(u, v)])
            G.add_edge(u, v, capacity=max(cap, 0.0))
        added = 0
        for t in self.others:
            if not (G.has_node(self.r) and G.has_node(t)):
                # t unreachable at all -> need any cut around t
                pass
            try:
                val, (Sset, _) = nx.minimum_cut(G, self.r, t)
            except Exception:
                Sset = {self.r}; val = 0.0
            if val < 1.0 - 1e-6:
                cross = [(i, j) for (i, j) in self.arcs if i in Sset and j not in Sset]
                if cross:
                    self.m.addCons(quicksum(self.xv[a] for a in cross) >= 1)
                    added += 1
        return added
    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        G = nx.DiGraph()
        for (u, v) in self.arcs:
            G.add_edge(u, v, capacity=max(self.m.getSolVal(solution, self.xv[(u, v)]), 0.0))
        for t in self.others:
            try:
                val, _ = nx.minimum_cut(G, self.r, t)
            except Exception:
                val = 0.0
            if val < 1.0 - 1e-6:
                return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        return {"result": SCIP_RESULT.CONSADDED if self._sep(None) else SCIP_RESULT.FEASIBLE}
    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        return {"result": SCIP_RESULT.CONSADDED if self._sep(None) else SCIP_RESULT.FEASIBLE}
    def conssepalp(self, constraints, nusefulconss):
        return {"result": SCIP_RESULT.CONSADDED if self._sep(None) else SCIP_RESULT.DIDNOTFIND}
    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        for a in self.arcs:
            self.m.addVarLocks(self.xv[a], nlockspos + nlocksneg, nlockspos + nlocksneg)

def solve_dcut(n, edges, term, tlim=600):
    if not edges or len(term) <= 1: return (0.0, "trivial", 0.0)
    T = sorted(term); r = T[0]
    m = Model(); m.hideOutput()
    y = {}; x = {}; arcs = []
    for i, (u, v, w) in enumerate(edges):
        y[i] = m.addVar(vtype="B")
        x[(u, v)] = m.addVar(vtype="B"); x[(v, u)] = m.addVar(vtype="B")
        arcs += [(u, v), (v, u)]
        m.addCons(x[(u, v)] <= y[i]); m.addCons(x[(v, u)] <= y[i])
    m.setObjective(quicksum(w * y[i] for i, (u, v, w) in enumerate(edges)), "minimize")
    ch = ConnCuts()
    m.includeConshdlr(ch, "conn", "connectivity cuts",
                      sepapriority=1, enfopriority=-1, chckpriority=-1,
                      sepafreq=1, propfreq=-1, eagerfreq=-1, needscons=False)
    ch.setup(m, arcs, x, r, term)
    m.setParam("limits/time", tlim)
    t0 = time.time(); m.optimize(); el = time.time() - t0
    if m.getNSols() > 0:
        return (m.getObjVal(), m.getStatus(), el)
    return (None, m.getStatus(), el)

def run(name, tlim=600):
    base = os.path.expanduser("~/projects/casp_max/data/benchmarks/steinlib/B")
    n, edges, term = S.parse_stp(base + "/" + name + ".stp")
    surv, re_, rv = S.reduce_casp(n, edges, term)
    of, sf, tf = solve_dcut(n, edges, term, tlim)
    orr, sr, tr = solve_dcut(n, surv, term, tlim)
    return {"inst": name, "n": n, "m": len(edges), "term": len(term),
            "prune": round(1 - len(surv) / max(len(edges), 1), 3),
            "opt_full": of, "st_full": sf, "t_full": round(tf, 1),
            "opt_red": orr, "st_red": sr, "t_red": round(tr, 1),
            "both_proven": (sf == "optimal" and sr == "optimal"),
            "match": (of is not None and orr is not None and abs(of - orr) < 1e-6)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(run("b01", tlim=120)); sys.exit()
    from multiprocessing import Pool
    INST = ["b%02d" % i for i in range(1, 19)]
    OUT = os.path.expanduser("~/projects/casp_max/outputs/theoremD")
    open(OUT + "/steiner_dcut.jsonl", "w").close()
    def wk(nm):
        r = run(nm, tlim=600)
        open(OUT + "/steiner_dcut.jsonl", "a").write(json.dumps(r) + "\n")
        return r
    with Pool(9) as p:
        res = p.map(wk, INST)
    PUB = {"b01":82,"b02":83,"b03":138,"b04":59,"b05":61,"b06":122,"b07":111,"b08":104,"b09":220,
           "b10":86,"b11":88,"b12":174,"b13":165,"b14":235,"b15":318,"b16":127,"b17":131,"b18":218}
    proven = [r for r in res if r["both_proven"]]
    summ = {"n": len(res), "proven": len(proven),
            "mismatches_among_proven": sum(1 for r in proven if not r["match"]),
            "all_match_published": all(abs(r["opt_full"] - PUB[r["inst"]]) < 1e-6 for r in proven),
            "mean_prune": round(sum(r["prune"] for r in res) / len(res), 3)}
    json.dump({"summary": summ, "rows": res}, open(OUT + "/steiner_dcut.json", "w"), indent=1)
    print("DCUT:", json.dumps(summ)); print("ALLDONE")
