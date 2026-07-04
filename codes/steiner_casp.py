#!/usr/bin/env python3
# E11 (heterogeneous #2): Steiner Tree negative-signal certificate on SteinLib.
# Certificate (OPT-preserving, poly-verifiable):
#   (R1) iterated degree-<=1 non-terminal exclusion: a non-terminal leaf is in no optimal tree.
#   (R2) shortest-path edge domination: edge (u,v,w) excludable if shortest u-v path in G\{e} <= w.
# Both preserve some optimum; applied iteratively. Exact solver: multi-commodity-flow ILP (SCIP).
import sys, os, glob, json, time, heapq
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
from pyscipopt import Model, quicksum
from multiprocessing import Pool

def parse_stp(path):
    n = 0; edges = []; term = set(); sec = None
    for ln in open(path, errors="ignore"):
        p = ln.split()
        if not p: continue
        u = p[0].upper()
        if u == "SECTION": sec = p[1].lower(); continue
        if u == "END": sec = None; continue
        if sec == "graph":
            if u == "NODES": n = int(p[1])
            elif u == "E": edges.append((int(p[1])-1, int(p[2])-1, float(p[3])))
        elif sec == "terminals":
            if u == "T": term.add(int(p[1])-1)
    return n, edges, term

def adj_of(n, edges):
    g = [dict() for _ in range(n)]
    for u, v, w in edges:
        if v not in g[u] or w < g[u][v]:
            g[u][v] = w; g[v][u] = w
    return g

def dijkstra(g, s, t, banned_uv=None):
    dist = {s: 0.0}; pq = [(0.0, s)]
    while pq:
        d, x = heapq.heappop(pq)
        if x == t: return d
        if d > dist.get(x, 1e18): continue
        for y, w in g[x].items():
            if banned_uv and ((x, y) == banned_uv or (y, x) == banned_uv): continue
            nd = d + w
            if nd < dist.get(y, 1e18): dist[y] = nd; heapq.heappush(pq, (nd, y))
    return dist.get(t, 1e18)

def reduce_casp(n, edges, term):
    """Apply R1+R2 iteratively. Returns surviving edge set, removed counts."""
    E = {(min(u, v), max(u, v)): w for u, v, w in edges}
    for u, v, w in edges:  # keep min parallel weight
        k = (min(u, v), max(u, v))
        if w < E[k]: E[k] = w
    removed_e = 0; removed_v = 0
    changed = True
    while changed:
        changed = False
        # degree map
        deg = [0]*n; inc = [[] for _ in range(n)]
        for (a, b) in E: deg[a]+=1; deg[b]+=1; inc[a].append((a,b)); inc[b].append((a,b))
        # R1: degree<=1 non-terminal
        for x in range(n):
            if x not in term and deg[x] <= 1 and inc[x]:
                for k in list(inc[x]):
                    if k in E: del E[k]; removed_e += 1
                removed_v += 1; changed = True
        if changed: continue
        # R2: shortest-path edge domination (one per pass to stay sound)
        g = {}
        for (a, b), w in E.items():
            g.setdefault(a, {})[b] = w; g.setdefault(b, {})[a] = w
        for (a, b), w in list(E.items()):
            gg = [dict() for _ in range(n)]
            for (x, y), ww in E.items(): gg[x][y]=ww; gg[y][x]=ww
            if dijkstra(gg, a, b, banned_uv=(a, b)) <= w + 1e-9:
                del E[(a, b)]; removed_e += 1; changed = True; break
    surv = [(a, b, w) for (a, b), w in E.items()]
    return surv, removed_e, removed_v

def steiner_exact(n, edges, term, tlim=120):
    if not edges or len(term) <= 1: return (0.0, "trivial", 0.0)
    T = sorted(term); r = T[0]; others = T[1:]
    arcs = []
    for u, v, w in edges: arcs += [(u, v, w), (v, u, w)]
    md = Model(); md.hideOutput()
    x = {}  # undirected edge use
    eid = {}
    for i, (u, v, w) in enumerate(edges): eid[(u, v)] = i; x[i] = md.addVar(vtype="B")
    f = {}
    for t in others:
        for (u, v, w) in arcs:
            f[(t, u, v)] = md.addVar(vtype="C", lb=0, ub=1)
    md.setObjective(quicksum(w * x[i] for i, (u, v, w) in enumerate(edges)), "minimize")
    # capacity: flow on arc <= edge use
    for t in others:
        for (u, v, w) in edges:
            md.addCons(f[(t, u, v)] <= x[eid[(u, v)]])
            md.addCons(f[(t, v, u)] <= x[eid[(u, v)]])
    # conservation
    nbr = [[] for _ in range(n)]
    for (u, v, w) in arcs: nbr[u].append((v))
    outarc = {u: [] for u in range(n)}; inarc = {u: [] for u in range(n)}
    for (u, v, w) in arcs: outarc[u].append((u, v)); inarc[v].append((u, v))
    for t in others:
        for node in range(n):
            b = (1.0 if node == r else (-1.0 if node == t else 0.0))
            md.addCons(quicksum(f[(t, a, bb)] for (a, bb) in outarc[node])
                       - quicksum(f[(t, a, bb)] for (a, bb) in inarc[node]) == b)
    md.setParam("limits/time", tlim); t0 = time.time(); md.optimize(); el = time.time()-t0
    if md.getNSols() > 0: return (md.getObjVal(), md.getStatus(), el)
    return (None, md.getStatus(), el)

def run_one(path):
    try:
        n, edges, term = parse_stp(path)
        t0 = time.time(); surv, re_, rv = reduce_casp(n, edges, term); red_t = time.time()-t0
        of, sf, tf = steiner_exact(n, edges, term, tlim=120)
        orr, sr, tr = steiner_exact(n, surv, term, tlim=120)
        mm = (of is not None and orr is not None and abs(of-orr) > 1e-6*max(1, abs(of)))
        return {"inst": os.path.basename(path), "n": n, "m": len(edges), "term": len(term),
                "surv_edges": len(surv), "prune_rate": round(1-len(surv)/max(len(edges), 1), 3),
                "removed_e": re_, "removed_v": rv, "opt": of, "opt_red": orr,
                "mismatch": bool(mm), "t_full": round(tf, 2), "t_red": round(tr, 2),
                "speedup": round(tf/max(tr, 1e-3), 2) if (of and orr) else None}
    except Exception as e:
        return {"inst": os.path.basename(path), "err": str(e)}

if __name__ == "__main__":
    base = os.path.expanduser("~/projects/casp_max/data/benchmarks/steinlib")
    files = sorted(glob.glob(base + "/B/*.stp"))
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    with Pool(12) as p:
        res = [r for r in p.map(run_one, files) if r]
    ok = [r for r in res if r.get("opt") is not None]
    import statistics as st
    pr = [r["prune_rate"] for r in ok]
    summ = {"n_instances": len(res), "solved": len(ok),
            "mismatches": sum(int(r.get("mismatch", False)) for r in ok),
            "mean_prune_rate": round(st.mean(pr), 3) if pr else 0,
            "max_prune_rate": round(max(pr), 3) if pr else 0,
            "n_with_pruning": sum(1 for r in ok if r["prune_rate"] > 0)}
    out = os.path.expanduser("~/projects/casp_max/outputs/theoremD/steiner.json")
    json.dump({"summary": summ, "rows": res}, open(out, "w"), indent=1)
    print("STEINER:", json.dumps(summ))
    print("DONE")
