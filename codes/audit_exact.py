"""Audit: generic whole-instance simulator vs the vectorised fast path.

Builds the FULL instance (no component decomposition), runs the generic
global ratio-greedy over all sets, and compares every policy cost against
the fast path under identical per-set Bernoulli noise. Validates both the
implementation and the disjoint-component decomposition argument.
"""
import numpy as np
from casp_lp import (make_pair, make_triangle, make_gadget,
                     local_greedy, TOL)


def build_full(components):
    """Concatenate component copies into one global set system."""
    cost, sets_elems, star, sigma_u, comp_of = [], [], [], [], []
    e_off = 0
    for comp, sig, copies in components:
        for c in range(copies):
            for i in range(comp.k):
                cost.append(comp.cost[i])
                sets_elems.append(frozenset(e + e_off for e in comp.sets_elems[i]))
                star.append(comp.star[i])
                sigma_u.append(sig[i])
                comp_of.append((id(comp), c))
            e_off += comp.n_elems
    return (np.asarray(cost), sets_elems, np.asarray(star, bool),
            np.asarray(sigma_u), e_off)


def global_policy_costs(cost, sets_elems, n_elems, pred, sigma, theta, fb):
    com_mc = [i for i in range(len(cost)) if pred[i]]
    mc, _ = local_greedy(cost, sets_elems, n_elems, com_mc)
    com_cf = [i for i in com_mc if sigma[i] >= theta - 1e-9]
    cf, _ = local_greedy(cost, sets_elems, n_elems, com_cf)
    return min(mc, fb), min(cf, fb), cf


def audit_exa(n=12, beta=1.0, C=5.0, etas=(0.0, 0.3, 0.6), seeds=3):
    pair, tri = make_pair(C), make_triangle()
    g = int(beta * n)
    cost, se, star, sig, n_elems = build_full(
        [(pair, pair.sigma("unique"), n), (tri, tri.sigma("unique"), g)])
    FB = n * pair.round_cost + g * tri.round_cost
    t_mc_p, t_mc_t = pair.table_mc(), tri.table_mc()
    t_cf_p, t_cf_t = pair.table_filter(pair.sigma("unique"), 0.51), \
                     tri.table_filter(tri.sigma("unique"), 0.51)
    for eta in etas:
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            p = np.where(star, 1 - eta, eta)
            pred = rng.random(len(cost)) < p
            mc_g, cf_g, _ = global_policy_costs(cost, se, n_elems, pred, sig, 0.51, FB)
            # fast path from the SAME prediction bits
            kp = pair.k; kt = tri.k
            bits_p = pred[: n * kp].reshape(n, kp)
            bits_t = pred[n * kp:].reshape(g, kt)
            cp = (bits_p * (1 << np.arange(kp))).sum(1)
            ct = (bits_t * (1 << np.arange(kt))).sum(1)
            mc_f = min(t_mc_p[cp].sum() + t_mc_t[ct].sum(), FB)
            cf_f = min(t_cf_p[cp].sum() + t_cf_t[ct].sum(), FB)
            assert abs(mc_g - mc_f) < 1e-9, (eta, s, mc_g, mc_f)
            assert abs(cf_g - cf_f) < 1e-9, (eta, s, cf_g, cf_f)
    print("[audit] EX-A generic-vs-fast: OK")


def audit_exb(n=10, ratio=1.0, eps=0.2, C=5.0, etas=(0.0, 0.3, 0.6), seeds=3):
    pair, gad = make_pair(C), make_gadget(eps)
    m = int(ratio * n)
    for kind in ("ri", "max"):
        sp, sg = pair.sigma(kind), gad.sigma(kind)
        cost, se, star, sig, n_elems = build_full([(pair, sp, n), (gad, sg, m)])
        FB = n * pair.greedy_scratch + m * gad.greedy_scratch
        # LP-commit globally
        com = [i for i in range(len(cost)) if sig[i] >= 1.0 - TOL]
        lp_g, _ = local_greedy(cost, se, n_elems, com)
        lp_f = n * pair.lp_commit_cost(sp) + m * gad.lp_commit_cost(sg)
        assert abs(lp_g - lp_f) < 1e-9, (kind, lp_g, lp_f)
        tf_p, tf_g = pair.table_filter(sp, 0.5), gad.table_filter(sg, 0.5)
        for eta in etas:
            for s in range(seeds):
                rng = np.random.default_rng(2000 + s)
                p = np.where(star, 1 - eta, eta)
                pred = rng.random(len(cost)) < p
                _, _, cf_g = global_policy_costs(cost, se, n_elems, pred, sig, 0.5, FB)
                kp, kg = pair.k, gad.k
                bits_p = pred[: n * kp].reshape(n, kp)
                bits_g = pred[n * kp:].reshape(m, kg)
                cp = (bits_p * (1 << np.arange(kp))).sum(1)
                cg = (bits_g * (1 << np.arange(kg))).sum(1)
                cf_f = tf_p[cp].sum() + tf_g[cg].sum()
                assert abs(cf_g - cf_f) < 1e-9, (kind, eta, s, cf_g, cf_f)
    print("[audit] EX-B generic-vs-fast: OK")


if __name__ == "__main__":
    audit_exa()
    audit_exb()
    print("[audit] ALL OK")
