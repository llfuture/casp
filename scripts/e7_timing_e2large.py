#!/usr/bin/env python3
# (A) E7 wall-clock: CASP (LP + core brute-force) vs SCIP-from-scratch on planted G(n,k,H).
# (B) E2 on large |S| where the proven pruning-rate lower bound is non-trivial.
import sys, os, glob, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
import planted as PL
from multiprocessing import Pool
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")

# ---------- (A) E7 timing ----------
def e7_time(item):
    n, k, s, seed = item
    nt, edges, w, info = PL.gen_planted(n, k, s, seed=seed)
    t0 = time.time()
    lp, xs, P0, P1, Ph = L.vc_lp_halfint(nt, edges, w)
    core = set(Ph); ce = [(u, v) for (u, v) in edges if u in core and v in core]
    Sc = PL.brute_min_cover(core, ce) or set()
    casp_opt = len(P1) + len(Sc)
    casp_t = time.time() - t0
    o, _, st, scip_t = L.vc_exact(nt, edges, w, tlim=300)
    return {"n": nt, "core": len(Ph), "casp_opt": casp_opt, "scip_opt": o,
            "casp_time": round(casp_t, 4), "scip_time": round(scip_t, 4),
            "speedup": round(scip_t / max(casp_t, 1e-4), 2),
            "mismatch": (o is not None and abs(casp_opt - o) > 1e-6)}

def run_e7_timing():
    items = []
    for n in [400, 800, 1500, 2500, 4000]:
        for seed in range(6):
            items.append((n, max(8, n // 25), 12, seed))
    with Pool(12) as p:
        res = p.map(e7_time, items)
    sp = [r["speedup"] for r in res if r["scip_opt"] is not None]
    json.dump({"summary": {"n": len(res), "mean_speedup": round(sum(sp)/len(sp), 1) if sp else None,
               "max_speedup": max(sp) if sp else None,
               "mismatches": sum(int(r["mismatch"]) for r in res)}, "rows": res},
              open(f"{OUT}/e7_timing.json", "w"), indent=1)
    print("E7-timing:", json.load(open(f"{OUT}/e7_timing.json"))["summary"])

# ---------- (B) E2 large |S| ----------
def e2_big(path):
    try:
        sc = L.load_sc_synth(path); U, S, C = sc["universe"], sc["sets"], sc["costs"]
        cert = L.sc_lp_threshold(U, S, C)
        f = cert["f"]; tau = cert["tau"]; lp = cert["lp"]; cmin = min(C) if C else 1.0
        bound = max(0.0, 1.0 - lp / (tau * cmin * max(len(S), 1)))
        return {"file": os.path.basename(path), "f": f, "nS": len(S),
                "emp_rate": round(cert["prune_rate"], 4), "lb": round(bound, 4),
                "gap": round(cert["prune_rate"] - bound, 4), "nontrivial": bound > 1e-6}
    except Exception as e:
        return {"file": os.path.basename(path), "err": str(e)}

def run_e2_big():
    DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/set_cover")
    files = sorted(glob.glob(DATA + "/sc_*_s5000_*.json.gz"))
    with Pool(16) as p:
        res = [r for r in p.map(e2_big, files) if "gap" in r]
    nt = [r for r in res if r["nontrivial"]]
    json.dump({"summary": {"n": len(res), "n_nontrivial_bound": len(nt),
               "violations": sum(1 for r in res if r["gap"] < 0),
               "min_gap_nontrivial": round(min((r["gap"] for r in nt), default=0), 4),
               "mean_lb_nontrivial": round(sum(r["lb"] for r in nt)/len(nt), 4) if nt else 0},
               "rows": res}, open(f"{OUT}/e2_big.json", "w"), indent=1)
    print("E2-big:", json.load(open(f"{OUT}/e2_big.json"))["summary"])

if __name__ == "__main__":
    if sys.argv[1] == "e7t": run_e7_timing()
    elif sys.argv[1] == "e2b": run_e2_big()
    print("DONE")
