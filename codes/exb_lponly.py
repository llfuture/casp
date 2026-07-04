"""EX-B: validate Theorem 17 (prediction breaks LP degeneracy) on H_{m,n}(eps).

Policies:
  lp_commit  = commit {sigma_bar == 1} (no prediction), greedy-complete
  cf(1/2)    = commit {predicted & sigma >= 1/2}, greedy-complete
               (branch cost; the min-with-fallback variant reported separately)
Two sigma canonicalizations: 'ri' (optimal-face midpoint) and 'max'
(per-variable max over the face), cf. Remark rem:canon.
Closed forms under test (per gadget):
  ri : lp_commit = 11/6 surely; E[cf] = (1+eps) + 11/6*eta*(1-eta)
       margin = 11/6*(1-eta(1-eta)) - (1+eps) >= 3/8 - eps > 0  for all eta
  max: lp_commit = 2(1+eps);     margin = (1+eps) - 11/6*eta*(1-eta) > 0
"""
import argparse, json, os, socket, time
import numpy as np
from casp_lp import make_pair, make_gadget


def run_config(n, ratio, eps, C, etas, seeds):
    pair, gad = make_pair(C), make_gadget(eps)
    m = int(round(ratio * n))
    OPT = n * pair.opt + m * gad.opt
    FB = n * pair.greedy_scratch + m * gad.greedy_scratch  # global greedy fallback

    recs = []
    for kind in ("ri", "max"):
        sp, sg = pair.sigma(kind), gad.sigma(kind)
        lp_cost = n * pair.lp_commit_cost(sp) + m * gad.lp_commit_cost(sg)
        tf_p = pair.table_filter(sp, 0.5)
        tf_g = gad.table_filter(sg, 0.5)

        # deterministic sanity: gadget sigma values from the actual LP face
        if kind == "ri":
            assert np.allclose(sg[:2], 0.5, atol=1e-6) and np.all(sg[2:] < 1e-6), \
                f"unexpected ri-sigma on gadget: {sg}"
            assert abs(gad.lp_commit_cost(sg) - gad.greedy_scratch) < 1e-9, \
                "ri LP-commit should commit nothing on the gadget"

        for eta in etas:
            cf_branch, cf_min = [], []
            for s in range(seeds):
                rng = np.random.default_rng(
                    hash((kind, n, ratio, round(eps * 100), round(eta * 100), s)) % 2**32)
                cp = pair.sample_case_ids(n, eta, rng)
                cg = gad.sample_case_ids(m, eta, rng)
                c = tf_p[cp].sum() + tf_g[cg].sum()
                cf_branch.append(c)
                cf_min.append(min(c, FB))
            cf_branch = np.asarray(cf_branch)
            per_gadget_cf = (cf_branch.mean() - n * pair.opt) / m
            rec = dict(sigma=kind, n=n, m=m, eps=eps, C=C, eta=eta, seeds=seeds,
                       lp_commit_cost=float(lp_cost),
                       cf_branch_mean=float(cf_branch.mean()),
                       cf_min_mean=float(np.mean(cf_min)),
                       margin_meas_pg=float((lp_cost - cf_branch.mean()) / m),
                       cf_meas_pg=float(per_gadget_cf),
                       OPT=OPT, FB=FB)
            if kind == "ri":
                rec["cf_theory_pg"] = float((1 + eps) + (11 / 6) * eta * (1 - eta))
                rec["margin_theory_pg"] = float((11 / 6) * (1 - eta * (1 - eta)) - (1 + eps))
                rec["margin_floor"] = float(3 / 8 - eps)
            else:
                rec["cf_theory_pg"] = float((1 + eps) + (11 / 6) * eta * (1 - eta))
                rec["margin_theory_pg"] = float((1 + eps) - (11 / 6) * eta * (1 - eta))
                rec["margin_floor"] = float(1 + eps - 11 / 24)
            rec["abs_dev"] = abs(rec["margin_meas_pg"] - rec["margin_theory_pg"])
            recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="../outputs/exb_lponly.json")
    args = ap.parse_args()

    if args.full:
        ns, ratios, epss, seeds = [200, 1000], [0.5, 1.0, 2.0], [0.1, 0.2, 0.3], 50
        etas = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
    else:  # MVE
        ns, ratios, epss, seeds = [200], [1.0], [0.2], 8
        etas = [0.0, 0.1, 0.3, 0.5, 0.7]
    C = 5.0

    t0 = time.time()
    results, viol = [], 0
    for n in ns:
        for ratio in ratios:
            for eps in epss:
                recs = run_config(n, ratio, eps, C, etas, seeds)
                results.extend(recs)
                dev = max(r["abs_dev"] for r in recs)
                below = sum(r["margin_meas_pg"] < r["margin_floor"] - 3 * 2 / np.sqrt(r["m"] * r["seeds"])
                            for r in recs)
                viol += below
                print(f"[EX-B] n={n} m/n={ratio} eps={eps}: max|margin dev|={dev:.4f} "
                      f"floor violations={below}", flush=True)

    meta = dict(exp="EX-B", host=socket.gethostname(), when=time.strftime("%F %T"),
                elapsed_s=round(time.time() - t0, 1),
                grid=dict(n=ns, ratio=ratios, eps=epss, eta=etas, seeds=seeds, C=C))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(meta=meta, results=results), open(args.out, "w"), indent=1)
    print(f"[EX-B] wrote {args.out} ({len(results)} rows, {meta['elapsed_s']}s)")
    print(f"[EX-B] PASS: margin-floor violations = {viol} ({'OK' if viol == 0 else 'CHECK'})")


if __name__ == "__main__":
    main()
