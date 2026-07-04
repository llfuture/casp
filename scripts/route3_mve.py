#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route-3 MVE (JMLR gate): noisy-prediction verifiable robust learning vs the MIN-COMBINER.
Core question (does route-3 survive the A3_hardened collapse?):
  Given a NOISY predicted solution S_hat (Set Cover), compare
    A fallback     : greedy on full instance
    B commit-all   : commit S_hat, greedily complete to feasibility
    C min-combiner : min(B, A)                         <- the collapse baseline
    D learned      : commit only predicted sets with LP-confidence x*_i >= theta*
                     (theta* learned per noise level via ERM), complete, then min with fallback
  If D ~= C at every noise eta  -> learning adds nothing beyond the min-combiner -> COLLAPSE (route-3 dead).
  If D <  C by a real margin    -> denoising (learning) beats the min-combiner   -> route-3 has a signal.
Loss = cost/OPT (>=1). Report mean loss per arm vs eta, and gap = C - D.
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from multiprocessing import Pool

DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/set_cover")
OUT  = os.path.expanduser("~/projects/casp_max/outputs/run")
ETAS = [0.1, 0.2, 0.4]
THETA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.01]  # 1.01 => commit nothing (=fallback)

def prep(path):
    sc = L.load_sc_synth(path); U, S, C = sc["universe"], sc["sets"], sc["costs"]
    if len(S) > 1200 or len(U) > 600: return None
    opt, sol, sto, _ = L.sc_exact(U, S, C, tlim=10)
    if opt is None or sto != "optimal" or opt <= 0: return None
    lpval, xs = L.sc_lp(U, S, C)
    _, gfull = L.sc_greedy(U, S, C)
    return {"file": os.path.basename(path), "U": list(U), "S": S, "C": C,
            "opt": opt, "x": xs, "gfull": gfull, "star": set(sol), "m": len(S)}

def predict(inst, eta, seed):
    """Noisy predicted membership: keep OPT sets w.p. 1-eta; add non-OPT sets w.p. eta_fp."""
    rng = random.Random(seed)
    star = inst["star"]; m = inst["m"]
    nfp = max(1, int(eta * len(star)))  # balance false positives ~ eta * |S*|
    non = [i for i in range(m) if i not in star]
    pred = set(i for i in star if rng.random() > eta)
    rng.shuffle(non)
    pred |= set(non[:nfp])
    return pred

def complete_cost(inst, committed):
    """Cost of committing `committed` then greedily covering the rest with all sets."""
    U = set(inst["U"]); S = inst["S"]; C = inst["C"]
    covered = set()
    for i in committed: covered |= set(S[i])
    chosen = set(committed)
    while covered != U:
        rem = U - covered; best, br = None, None
        for i in range(len(S)):
            if i in chosen: continue
            g = len(set(S[i]) & rem)
            if g == 0: continue
            r = C[i] / g
            if br is None or r < br: best, br = i, r
        if best is None: break
        chosen.add(best); covered |= set(S[best])
    return sum(C[i] for i in chosen)

def arms_for(inst, eta, theta, seed):
    pred = predict(inst, eta, seed)
    opt = inst["opt"]; fb = inst["gfull"]; xs = inst["x"]
    cost_commit = complete_cost(inst, pred)                     # B
    denoised = {i for i in pred if xs[i] >= theta - 1e-9}
    cost_learn = complete_cost(inst, denoised)                  # D-commit
    A = fb / opt
    B = cost_commit / opt
    Cc = min(cost_commit, fb) / opt                             # min-combiner
    D = min(cost_learn, fb) / opt                               # learned denoise + min
    return A, B, Cc, D

def evaluate(insts):
    idx = list(range(len(insts))); random.Random(0).shuffle(idx)
    h = len(idx)//2; tr = [insts[i] for i in idx[:h]]; te = [insts[i] for i in idx[h:]]
    res = {}
    for eta in ETAS:
        # learn theta* on train: minimize mean D-loss (denoise+min)
        def Dloss(group, theta):
            return st.mean(arms_for(x, eta, theta, seed=hash((x["file"], eta)) & 0xffff)[3] for x in group)
        theta_star = min(THETA_GRID, key=lambda th: Dloss(tr, th))
        # eval all arms on test with fixed prediction seed per instance
        A = st.mean(arms_for(x, eta, 0.0, seed=hash((x["file"], eta)) & 0xffff)[0] for x in te)
        B = st.mean(arms_for(x, eta, 0.0, seed=hash((x["file"], eta)) & 0xffff)[1] for x in te)
        Cc = st.mean(arms_for(x, eta, 0.0, seed=hash((x["file"], eta)) & 0xffff)[2] for x in te)
        D = st.mean(arms_for(x, eta, theta_star, seed=hash((x["file"], eta)) & 0xffff)[3] for x in te)
        res[str(eta)] = {"fallback": round(A,4), "commit_all": round(B,4),
                         "min_combiner": round(Cc,4), "learned_denoise": round(D,4),
                         "theta_star": theta_star, "gap_C_minus_D": round(Cc - D, 4)}
        print("eta=%.1f: fallback=%.3f commit=%.3f MIN-COMBINER=%.3f LEARNED=%.3f  gap(C-D)=%.4f (theta*=%.2f)"
              % (eta, A, B, Cc, D, Cc - D, theta_star))
    return res

def main():
    files = sorted(glob.glob(DATA + "/sc_f5_m500_s1000_*.json.gz"))[:50]
    if not files:
        files = sorted(glob.glob(DATA + "/*.json.gz"))[:50]
    with Pool(16) as pool:
        insts = [r for r in pool.map(prep, files) if r]
    print("prepared %d instances" % len(insts))
    res = evaluate(insts)
    json.dump({"by_eta": res, "n": len(insts)}, open(os.path.join(OUT, "route3_mve.json"), "w"), indent=1)
    gaps = [res[str(e)]["gap_C_minus_D"] for e in ETAS]
    print("\n===== ROUTE-3 DECISION =====")
    print("gap (min-combiner - learned) per eta:", gaps)
    print("GO if some gap >= ~0.05 (learning beats min-combiner); COLLAPSE if all gaps ~<=0")

if __name__ == "__main__":
    main()
