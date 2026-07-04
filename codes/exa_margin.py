"""EX-A: validate Theorem 16 (exact asymptotic margin) on D_{beta,C}(n).

Policies (all costs from real component LPs + generic greedy, see casp_lp):
  mc      = min(commit-all-then-complete, LP-rounding fallback)
  cf(th)  = min(commit {predicted & sigma>=th} then complete, fallback)
Closed form under test:
  E[l_mc] -> 1 + min{eta[(C-eta)+beta(1-eta)^2], beta}/(1+2beta),  cf(th>1/2) == 1 surely.
"""
import argparse, json, os, socket, time
import numpy as np
from casp_lp import make_pair, make_triangle

THETAS = [0.0, 0.51, 0.75]


def theory_margin(eta, beta, C):
    return min(eta * ((C - eta) + beta * (1 - eta) ** 2), beta) / (1 + 2 * beta)


def run_config(n, beta, C, etas, seeds):
    pair, tri = make_pair(C), make_triangle()
    g = int(np.floor(beta * n))
    OPT = n * pair.opt + g * tri.opt
    FB = n * pair.round_cost + g * tri.round_cost

    t_mc = {"pair": pair.table_mc(), "tri": tri.table_mc()}
    t_cf = {th: {"pair": pair.table_filter(pair.sigma("unique"), th),
                 "tri": tri.table_filter(tri.sigma("unique"), th)} for th in THETAS}

    # --- deterministic case-level assertions (theory holds per realisation) ---
    assert np.allclose(t_cf[0.0]["pair"], t_mc["pair"]) and \
           np.allclose(t_cf[0.0]["tri"], t_mc["tri"]), "cf(0) != mc (Thm 15(i))"
    for th in (0.51, 0.75):
        assert np.allclose(t_cf[th]["pair"], pair.opt), "cf pair not surely-optimal"
        assert np.allclose(t_cf[th]["tri"], tri.opt), "cf triangle not surely-optimal"

    out = []
    for eta in etas:
        mc_l, cf_l = [], {th: [] for th in THETAS}
        for s in range(seeds):
            rng = np.random.default_rng(hash((n, beta, C, round(eta * 100), s)) % 2**32)
            cp = pair.sample_case_ids(n, eta, rng)
            ct = tri.sample_case_ids(g, eta, rng) if g else np.zeros(0, int)
            commit = t_mc["pair"][cp].sum() + t_mc["tri"][ct].sum()
            mc_l.append(min(commit, FB) / OPT)
            for th in THETAS:
                c = t_cf[th]["pair"][cp].sum() + t_cf[th]["tri"][ct].sum()
                cf_l[th].append(min(c, FB) / OPT)
        mc_l = np.asarray(mc_l)
        cf_best = min(np.mean(cf_l[th]) for th in THETAS if th > 0)
        rec = dict(n=n, beta=beta, C=C, eta=eta, seeds=seeds,
                   mc_mean=float(mc_l.mean()), mc_se=float(mc_l.std(ddof=1) / np.sqrt(seeds)),
                   cf_051_mean=float(np.mean(cf_l[0.51])),
                   cf_best_mean=float(cf_best),
                   margin_meas=float(mc_l.mean() - cf_best),
                   margin_theory=float(theory_margin(eta, beta, C)),
                   mc_theory=float(1 + theory_margin(eta, beta, C)))
        rec["abs_dev"] = abs(rec["margin_meas"] - rec["margin_theory"])
        rec["cf_surely_opt"] = bool(abs(rec["cf_051_mean"] - 1.0) < 1e-12)
        out.append(rec)
    surely = all(r["cf_surely_opt"] for r in out)
    return out, surely


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="../outputs/exa_margin.json")
    args = ap.parse_args()

    if args.full:
        ns, betas, Cs, seeds = [200, 1000, 5000], [0.25, 0.5, 1.0, 2.0], [2, 5, 10], 50
        etas = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
    else:  # MVE
        ns, betas, Cs, seeds = [200], [1.0], [5], 8
        etas = [0.0, 0.1, 0.3, 0.5, 0.7]

    t0 = time.time()
    results, worst = [], 0.0
    for n in ns:
        for beta in betas:
            for C in Cs:
                recs, surely = run_config(n, beta, C, etas, seeds)
                results.extend(recs)
                dev = max(r["abs_dev"] for r in recs)
                worst = max(worst, max(r["abs_dev"] - 3 * (C + 2) / np.sqrt(n)
                                       - 3 * r["mc_se"] for r in recs))
                print(f"[EX-A] n={n} beta={beta} C={C}: max|margin dev|={dev:.4f} "
                      f"cf surely-opt={surely}", flush=True)

    meta = dict(exp="EX-A", host=socket.gethostname(), when=time.strftime("%F %T"),
                elapsed_s=round(time.time() - t0, 1), grid=dict(
                    n=ns, beta=betas, C=Cs, eta=etas, seeds=seeds))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(meta=meta, results=results), open(args.out, "w"), indent=1)
    print(f"[EX-A] wrote {args.out}  ({len(results)} rows, {meta['elapsed_s']}s)")
    print(f"[EX-A] PASS: worst deviation beyond finite-n allowance = {worst:.4f} "
          f"({'OK' if worst <= 0 else 'CHECK'})")


if __name__ == "__main__":
    main()
