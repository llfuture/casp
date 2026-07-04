#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-D (review M7): fair E10 rerun. Three arms on the same VC instances:
  ant_bare : Antoniadis et al. as published (commit noisy prediction + repair)
  ant_comb : the SAME output min-combined with the LP-rounding fallback
             (Thm 10 predicts this flattens near the fallback ratio)
  casp     : NT certificate; exact when the half-integral core is small,
             fallback ratio otherwise (correctness prediction-independent)
"""
import sys, os, glob, json, time, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/baselines"))
import casp_lib as L
import planted as PL
import antoniadis2024 as ANT
from parallel import pmap_chunks

DATA = os.path.expanduser("~/projects/casp_max/data")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CORE_BRUTE = 20


def _one(item):
    path, eta = item
    try:
        n, edges, w = L.load_vc_synth(path)
        oexact, sopt, sto, _ = L.vc_exact(n, edges, w, tlim=60)
        if oexact is None or sto != "optimal" or oexact <= 0:
            return None
        # LP + rounding fallback (valid cover; 2-approx)
        lp, xs, P0, P1, Ph = L.vc_lp_halfint(n, edges, w)
        fb_cost = sum(w[v] for v in range(n) if xs[v] >= 0.5 - 1e-9)
        # positive-signal baseline, as published
        pred = ANT.make_prediction(set(sopt), range(n), eta, seed=7)
        _, ca, _ = ANT.run_vc(n, edges, w, pred)
        # fair variant: min-combiner with the fallback
        comb = min(ca, fb_cost)
        # CASP: NT certificate; exact if core small
        nt = L.vc_nt_certificate(n, edges, w)
        if nt["core_size"] <= CORE_BRUTE:
            core = PL.brute_min_cover(set(nt["Phalf"]), nt["core_edges"]) or set()
            casp = nt["c_fix"] + sum(w[v] for v in core)
            certifies = True
        else:
            casp, certifies = fb_cost, False
        return {"file": os.path.basename(path), "eta": eta, "opt": oexact,
                "fb": round(fb_cost / oexact, 4),
                "ant_bare": round(ca / oexact, 4),
                "ant_comb": round(comb / oexact, 4),
                "casp": round(casp / oexact, 4),
                "casp_certifies": certifies}
    except Exception as e:
        return {"file": os.path.basename(path), "eta": eta, "err": str(e)}


def main(limit=30):
    base = sorted(glob.glob(DATA + "/synthetic/vertex_cover/*.json.gz"))[:limit]
    items = [(p, eta) for p in base for eta in ETAS]
    rows = [r for r in pmap_chunks(_one, items) if r and "err" not in r]
    curves = []
    for eta in ETAS:
        sel = [r for r in rows if r["eta"] == eta]
        if not sel:
            continue
        curves.append({
            "eta": eta, "n": len(sel),
            "fb": round(st.mean(r["fb"] for r in sel), 4),
            "ant_bare": round(st.mean(r["ant_bare"] for r in sel), 4),
            "ant_comb": round(st.mean(r["ant_comb"] for r in sel), 4),
            "casp": round(st.mean(r["casp"] for r in sel), 4),
            "casp_cert_rate": round(st.mean(1.0 * r["casp_certifies"] for r in sel), 3),
            "comb_flat_ok": bool(st.mean(r["ant_comb"] for r in sel)
                                 <= st.mean(r["fb"] for r in sel) + 1e-9)})
        c = curves[-1]
        print("eta=%.1f bare=%.3f comb=%.3f fb=%.3f casp=%.3f cert=%.0f%%" %
              (eta, c["ant_bare"], c["ant_comb"], c["fb"], c["casp"],
               100 * c["casp_cert_rate"]), flush=True)
    json.dump({"curves": curves, "rows": rows, "n_rows": len(rows)},
              open(os.path.join(OUT, "exd_e10_fair.json"), "w"), indent=1)
    print("SAVED exd_e10_fair.json")
    bare_rise = curves[-1]["ant_bare"] - curves[0]["ant_bare"]
    comb_rise = curves[-1]["ant_comb"] - curves[0]["ant_comb"]
    print("\n===== EX-D VERDICT =====")
    print("bare rise over eta: +%.3f ; combined rise: +%.3f (Thm 10: combined stays <= fb)" %
          (bare_rise, comb_rise))
    print("combined <= fb at every eta:", all(c["comb_flat_ok"] for c in curves))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
