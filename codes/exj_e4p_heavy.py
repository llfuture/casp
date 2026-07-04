#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-J = E4'' : multi-parameter PAC generalization on a NON-benign distribution.
Same policy class and analysis as e4p_real (per-size-bucket threshold vectors),
but instances have HEAVY-TAILED, bucket-heterogeneous costs, so (a) the best
achievable loss is well above 1, (b) different buckets genuinely need different
thresholds (multi-parameter structure is real), and (c) ERM has room to overfit
-- the generalization gap acquires magnitude, not just p-ordering.
Guardrail (Thm robust): losses must stay bounded by max(f, alpha).
"""
import sys, os, json, time, random, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
import e4p_real as E4
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")


def gen_heavy(seed, n_elem=400, m_sets=400, f_target=5):
    """SC instance with heavy-tailed, size-correlated costs."""
    rng = random.Random(seed)
    # coverage structure: each element in ~f_target sets, mixed set sizes
    sets = [[] for _ in range(m_sets)]
    for e in range(n_elem):
        for s in rng.sample(range(m_sets), f_target):
            sets[s].append(e)
    sets = [sorted(set(s)) for s in sets]
    universe = set(range(n_elem))
    covered = set()
    for s in sets:
        covered |= set(s)
    for e in universe - covered:          # patch rare uncovered elements
        sets[rng.randrange(m_sets)].append(e)
    sets = [sorted(set(s)) for s in sets]
    # heavy-tailed Pareto(alpha=1.1) costs, scale grows with set size bucket
    costs = []
    mx = max(len(s) for s in sets) + 1
    for s in sets:
        pareto = (1.0 - rng.random()) ** (-1.0 / 1.1)     # Pareto(1.1) >= 1
        bucket_scale = 1.0 + 3.0 * (len(s) / mx)          # big sets: pricier tail
        costs.append(min(pareto * bucket_scale, 500.0))
    return {"universe": universe, "sets": sets, "costs": costs}


def prep(seed):
    sc = gen_heavy(seed)
    U, S, C = sc["universe"], sc["sets"], sc["costs"]
    opt, _, sto, _ = L.sc_exact(U, S, C, tlim=60)
    if opt is None or sto != "optimal" or opt <= 0:
        return None
    lpval, xs = L.sc_lp(U, S, C)
    _, gfull = L.sc_greedy(U, S, C)
    sizes = [len(s) for s in S]
    return {"U": list(U), "S": S, "C": C, "opt": opt, "x": xs, "gfull": gfull,
            "sizes": sizes, "mx": max(sizes) + 1, "file": "heavy_%d" % seed,
            "f": L.sc_freq(U, S)}


def main():
    t0 = time.time()
    with Pool(16) as pool:
        insts = [r for r in pool.map(prep, range(70)) if r]
    print("prepared %d heavy-tail instances (%.0fs)" % (len(insts), time.time() - t0),
          flush=True)
    fmax = max(r["f"] for r in insts)
    result = {}
    for p in [2, 4, 8]:
        cand = E4.candidates(p)
        with Pool(16) as pool:
            M = pool.map(E4.loss_vector, [(inst, p, cand) for inst in insts])
        result[str(p)] = E4.analyze(M, cand)
        r = result[str(p)]
        print("p=%d: best_test=%.4f curve=%s maxloss=%.3f" %
              (p, r["best_test_loss"], r["curve"], r["max_observed_loss"]), flush=True)

    bound = fmax  # alpha (greedy fallback ratio) is itself <= observed; f is the binding constant
    ok_bounded = all(result[k]["max_observed_loss"] <= bound + 1e-9 for k in result)
    gaps5 = {k: result[k]["curve"][0]["mean_gap"] for k in result}
    json.dump({"n_instances": len(insts), "f_max": fmax,
               "by_p": result,
               "checks": {"loss_bounded_by_f": ok_bounded,
                          "gap_at_N5_by_p": gaps5}},
              open(os.path.join(OUT, "exj_e4p_heavy.json"), "w"), indent=1)
    print("SAVED exj_e4p_heavy.json")
    print("\n===== E4'' VERDICT =====")
    print("best achievable test loss per p:",
          {k: result[k]["best_test_loss"] for k in result})
    print("gap at N=5 by p (want magnitude >> benign ~0.001 and increasing in p):", gaps5)
    print("losses bounded by f=%d (Thm robust): %s" % (fmax, ok_bounded))


if __name__ == "__main__":
    main()
