#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 MVE on FACILITY LOCATION (JMLR path, 2nd problem).
Certificate = per-facility open-threshold: prune (forbid) facility i if LP y_i < theta[bucket_i].
Bucket feature = facility opening cost f_i (observable). Geometry makes y a NOISY prunability
signal: a remote client forces an EXPENSIVE facility open (poison, high-f bucket, low y but in OPT),
while a cheap facility shadowed by a nearer one gets low y and is safe to prune (low-f bucket).
So a single global y-threshold cannot separate prune-safe from must-open; a per-bucket (per f-band)
threshold can. TRUE OPT-preserving reward: re-solve reduced FL, credit prune-rate iff OPT unchanged.
"""
import sys, os, glob, gzip, json, random, math, itertools, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/fl_mve")
OUT  = os.path.expanduser("~/projects/casp_max/outputs/run")
GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
EPS = 1e-9
_C = {}

def gen_instance(rng, m=24, n=32):
    """2D geometric FL. Facilities of 3 f-bands; serving cost = euclidean dist.
    A few 'remote' clients are placed near a single expensive facility (poison)."""
    fac = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(m)]
    # opening costs: 8 cheap, 8 mid, 8 expensive
    f = []
    for i in range(m):
        band = i % 3
        f.append(round(rng.uniform(*[(3, 6), (12, 20), (45, 70)][band]), 2))
    cl = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(n)]
    # make ~3 clients remote, clustered around one expensive facility each
    exp_idx = [i for i in range(m) if i % 3 == 2]
    for r in range(3):
        fi = exp_idx[r % len(exp_idx)]
        cl[r] = (fac[fi][0] + rng.uniform(-0.02, 0.02), fac[fi][1] + rng.uniform(-0.02, 0.02))
    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    c = [[round(3.0 * dist(cl[j], fac[i]), 3) for i in range(m)] for j in range(n)]
    return m, n, f, c

def write(path, m, n, f, c):
    d = {"nF": m, "nC": n, "f": {str(i): f[i] for i in range(m)},
         "d": {str(i): {str(j): c[j][i] for j in range(n)} for i in range(m)}}
    with gzip.open(path, "wt") as fh: json.dump(d, fh)

def gen(dist_name, N, seed):
    d = os.path.join(DATA, dist_name); os.makedirs(d, exist_ok=True)
    rng = random.Random(seed)
    for k in range(N):
        m, n, f, c = gen_instance(rng)
        write(os.path.join(d, "fl_%s_%04d.json.gz" % (dist_name, k)), m, n, f, c)
    print("wrote %d %s" % (N, dist_name))

def fl_solve(m, n, f, c, allow=None, tlim=15):
    """Exact FL restricted to facilities in `allow` (list). Returns (opt, elapsed)."""
    from pyscipopt import Model, quicksum
    idx = allow if allow is not None else list(range(m))
    md = Model(); md.hideOutput()
    y = {i: md.addVar(vtype="B") for i in idx}
    x = {(i, j): md.addVar(vtype="C", lb=0, ub=1) for i in idx for j in range(n)}
    md.setObjective(quicksum(f[i]*y[i] for i in idx) +
                    quicksum(c[j][i]*x[i, j] for i in idx for j in range(n)), "minimize")
    for j in range(n):
        md.addCons(quicksum(x[i, j] for i in idx) >= 1)
        for i in idx: md.addCons(x[i, j] <= y[i])
    import time as _t
    md.setParam("limits/time", tlim); t = _t.time(); md.optimize(); el = _t.time()-t
    if md.getNSols() > 0: return md.getObjVal(), el
    return None, el

def prep(path):
    m, n, f, c = L.load_fl_hard(path)
    opt, sto, tful = L.fl_exact(m, n, f, c, tlim=20)
    if opt is None or opt <= 0: return None
    lpval, yv = L.fl_lp(m, n, f, c)
    return {"file": os.path.basename(path), "m": m, "n": n, "f": f, "c": c,
            "y": [yv[i] for i in range(m)], "opt": opt, "t_full": tful}

def bucket_edges(p):
    # f bands roughly [3,70]; log-ish edges
    lo, hi = 3.0, 70.0
    return [lo * (hi/lo)**(k/p) for k in range(1, p)]

def bucket_of(fi, edges):
    b = 0
    for e in edges:
        if fi >= e: b += 1
    return b

def survivors(inst, thr_fn):
    y = inst["y"]
    return [i for i in range(inst["m"]) if y[i] >= thr_fn(i) - EPS]

def reward(inst, thr_fn):
    surv = survivors(inst, thr_fn); m = inst["m"]
    if len(surv) == m or len(surv) == 0: return 0.0
    pr = 1.0 - len(surv)/m
    key = (inst["file"], hash(frozenset(surv)))
    if key in _C: ok = _C[key]
    else:
        opt_r, _ = fl_solve(inst["m"], inst["n"], inst["f"], inst["c"], allow=surv, tlim=12)
        ok = (opt_r is not None and abs(opt_r - inst["opt"]) < 1e-4)
        _C[key] = ok
    return pr if ok else 0.0

def r_single(inst, tau): return reward(inst, lambda i: tau)
def r_bucket(inst, th, edges): return reward(inst, lambda i: th[bucket_of(inst["f"][i], edges)])
def r_lpdef(inst): return reward(inst, lambda i: 0.5)   # LP-default: prune y<0.5

def erm_single(tr): return max(GRID, key=lambda t: st.mean(r_single(i, t) for i in tr))
def erm_bucket(tr, p, edges):
    cand = [list(t) for t in itertools.product(GRID, repeat=p)] if p <= 2 else \
           [[random.Random(s).choice(GRID) for _ in range(p)] for s in range(300)]
    return max(cand, key=lambda th: st.mean(r_bucket(i, th, edges) for i in tr))

def speedup(te, th, edges, k=12):
    rr = []; pres = 0; nn = 0
    for inst in te[:k]:
        surv = survivors(inst, lambda i: th[bucket_of(inst["f"][i], edges)])
        if len(surv) == inst["m"] or not surv: continue
        opt_r, t_r = fl_solve(inst["m"], inst["n"], inst["f"], inst["c"], allow=surv, tlim=15)
        nn += 1
        if opt_r is not None and abs(opt_r - inst["opt"]) < 1e-4: pres += 1
        if opt_r is not None and t_r > 0: rr.append(inst["t_full"]/max(t_r, 1e-4))
    return {"n": nn, "opt_preserved": pres, "mean_speedup": round(st.mean(rr), 2) if rr else None}

def load(dist, cap=40):
    files = sorted(glob.glob(os.path.join(DATA, dist, "*.json.gz")))[:cap]
    with Pool(16) as pool:
        return [r for r in pool.map(prep, files) if r]

def diagnose(D, p=3):
    import collections
    edges = bucket_edges(p)
    print("  [diag] p=%d edges=%s" % (p, [round(e, 1) for e in edges]))
    by = collections.defaultdict(lambda: [0, 0, []])  # in_open_lowy, total_lowy, all_y
    for inst in D:
        # recompute an exact-open set via full solve proxy: facilities with y>0.5 are "likely open"
        for i in range(inst["m"]):
            b = bucket_of(inst["f"][i], edges); yv = inst["y"][i]
            by[b][2].append(yv)
            if yv < 0.25:
                by[b][1] += 1
    for b in sorted(by):
        ally = sorted(by[b][2]); q = lambda t: ally[min(len(ally)-1, int(t*len(ally)))]
        print("    bucket %d: nfac=%d y q10/50/90=%.2f/%.2f/%.2f lowy_frac=%.2f"
              % (b, len(ally), q(.1), q(.5), q(.9), by[b][1]/max(len(ally), 1)))

def main():
    gen("G2", 40, seed=5)
    D = load("G2")
    print("G2=%d instances" % len(D))
    diagnose(D, 3)
    idx = list(range(len(D))); random.Random(0).shuffle(idx)
    h = len(idx)//2; tr = [D[i] for i in idx[:h]]; te = [D[i] for i in idx[h:]]
    lp = st.mean(r_lpdef(i) for i in te)
    tau = erm_single(tr); s1 = st.mean(r_single(i, tau) for i in te)
    out = {"lp_default": round(lp, 4), "single_tau": round(s1, 4), "tau": tau}
    for p in [2, 3]:
        edges = bucket_edges(p); th = erm_bucket(tr, p, edges)
        bp = st.mean(r_bucket(i, th, edges) for i in te)
        out["perbucket_p%d" % p] = round(bp, 4); out["theta%d" % p] = th
        out["gain_p%d" % p] = round(bp - s1, 4); out["speedup_p%d" % p] = speedup(te, th, edges)
        print("[FL G2] p=%d single=%.3f per-bucket=%.3f gain=%.3f speedup=%s"
              % (p, s1, bp, bp - s1, out["speedup_p%d" % p]))
    print("[FL G2] LP-default(y<0.5)=%.3f single-tau*=%.3f(tau=%.2f)" % (lp, s1, tau))
    json.dump(out, open(os.path.join(OUT, "fl_mve.json"), "w"), indent=1)
    g = out["gain_p3"]
    print("\n===== FL DECISION =====")
    print("FL gain(per-bucket-p3 - single, TRUE OPT-preserving) = %.3f" % g)
    print("GO if >=0.15 & speedup>1 & OPT-preserved high")

if __name__ == "__main__":
    main()
