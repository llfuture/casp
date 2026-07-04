"""Component LPs and confidence canonicalizations for CASP EX-A / EX-B.

Nothing is hardcoded from theory: every sigma, fallback and completion cost
is derived from an actual HiGHS LP solve / generic greedy on the component.
"""
import numpy as np
from scipy.optimize import linprog

TOL = 1e-7


def solve_cover_lp(cost, elem_rows):
    """min cost.x  s.t.  sum_{i in row} x_i >= 1 for each element row, 0<=x<=1."""
    cost = np.asarray(cost, float)
    n = len(cost)
    A = np.zeros((len(elem_rows), n))
    for r, row in enumerate(elem_rows):
        for i in row:
            A[r, i] = 1.0
    res = linprog(c=cost, A_ub=-A, b_ub=-np.ones(len(elem_rows)),
                  bounds=[(0.0, 1.0)] * n, method="highs")
    assert res.status == 0, res.message
    return np.asarray(res.x), float(res.fun)


def face_range(cost, elem_rows, lp_val):
    """Per-variable [min, max] over the optimal face, via auxiliary LPs."""
    cost = np.asarray(cost, float)
    n = len(cost)
    A = np.zeros((len(elem_rows), n))
    for r, row in enumerate(elem_rows):
        for i in row:
            A[r, i] = 1.0
    A_ub = np.vstack([-A, cost])
    b_ub = np.concatenate([-np.ones(len(elem_rows)),
                           [lp_val + 1e-9 + abs(lp_val) * 1e-9]])
    lo, hi = np.zeros(n), np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        r_min = linprog(c=e,  A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * n, method="highs")
        r_max = linprog(c=-e, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * n, method="highs")
        assert r_min.status == 0 and r_max.status == 0
        lo[i], hi[i] = r_min.x[i], r_max.x[i]
    return lo, hi


def sigma_from_face(lo, hi, kind):
    """'ri' = per-coordinate midpoint of the optimal-face range
    (equals the relative-interior value on the families used here);
    'max' = per-variable maximum over the optimal face."""
    if kind == "ri":
        return (lo + hi) / 2.0
    if kind == "max":
        return hi.copy()
    raise ValueError(kind)


def sets_elems_from_rows(n_sets, elem_rows):
    """elem_rows[e] = sets covering element e  ->  sets_elems[i] = elements of set i."""
    se = [[] for _ in range(n_sets)]
    for e, row in enumerate(elem_rows):
        for i in row:
            se[i].append(e)
    return [frozenset(x) for x in se]


def local_greedy(cost, sets_elems, n_elems, committed=()):
    """Generic ratio-greedy set cover with deterministic (ratio, cost, index)
    tie-breaking. Returns (total cost incl. committed, chosen indices)."""
    covered = set()
    total = 0.0
    chosen = list(committed)
    for i in committed:
        covered |= sets_elems[i]
        total += cost[i]
    while len(covered) < n_elems:
        best_key, best_i = None, None
        for i in range(len(cost)):
            new = len(sets_elems[i] - covered)
            if new == 0:
                continue
            key = (cost[i] / new, cost[i], i)
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        assert best_i is not None, "infeasible completion"
        covered |= sets_elems[best_i]
        total += cost[best_i]
        chosen.append(best_i)
    return total, chosen


def exact_opt(cost, sets_elems, n_elems):
    """Exact component optimum by subset enumeration (components are tiny)."""
    k = len(cost)
    best = np.inf
    universe = set(range(n_elems))
    for mask in range(1 << k):
        cov, c = set(), 0.0
        for i in range(k):
            if mask >> i & 1:
                cov |= sets_elems[i]
                c += cost[i]
        if cov >= universe and c < best:
            best = c
    return best


class Component:
    """A component type: costs, element rows, S*-membership, solved LP data,
    and per-policy case-cost tables over all 2^k prediction cases."""

    def __init__(self, name, cost, elem_rows, star):
        self.name = name
        self.cost = np.asarray(cost, float)
        self.rows = elem_rows
        self.k = len(cost)
        self.n_elems = len(elem_rows)
        self.star = np.asarray(star, bool)          # which sets are in S*
        self.sets_elems = sets_elems_from_rows(self.k, elem_rows)
        self.x, self.lp_val = solve_cover_lp(cost, elem_rows)
        self.lo, self.hi = face_range(cost, elem_rows, self.lp_val)
        self.opt = exact_opt(self.cost, self.sets_elems, self.n_elems)
        self.greedy_scratch, _ = local_greedy(self.cost, self.sets_elems, self.n_elems)
        # LP-rounding cost (EX-A fallback): take every set with x* >= 1/2
        idx = [i for i in range(self.k) if self.x[i] >= 0.5 - TOL]
        cov = set().union(*[self.sets_elems[i] for i in idx]) if idx else set()
        assert cov >= set(range(self.n_elems)), "rounding infeasible on component"
        self.round_cost = float(self.cost[idx].sum())

    def sigma(self, kind):
        if kind == "unique":
            assert np.all(self.hi - self.lo < 1e-6), \
                f"{self.name}: LP optimum not unique; use ri/max"
            return self.x.copy()
        return sigma_from_face(self.lo, self.hi, kind)

    def case_costs(self, commit_rule):
        """cost table over the 2^k prediction cases; commit_rule maps the
        predicted index set to the committed index set."""
        out = np.zeros(1 << self.k)
        for mask in range(1 << self.k):
            pred = [i for i in range(self.k) if mask >> i & 1]
            com = commit_rule(pred)
            total, _ = local_greedy(self.cost, self.sets_elems, self.n_elems, com)
            out[mask] = total
        return out

    def table_mc(self):
        return self.case_costs(lambda pred: list(pred))

    def table_filter(self, sig, theta):
        keep = sig >= theta - 1e-9
        return self.case_costs(lambda pred: [i for i in pred if keep[i]])

    def lp_commit_cost(self, sig):
        com = [i for i in range(self.k) if sig[i] >= 1.0 - TOL]
        total, _ = local_greedy(self.cost, self.sets_elems, self.n_elems, com)
        return total

    def sample_case_ids(self, n_copies, eta, rng):
        """Vectorised flip-noise: bit i set w.p. 1-eta if i in S*, else eta."""
        p = np.where(self.star, 1.0 - eta, eta)
        bits = rng.random((n_copies, self.k)) < p[None, :]
        return (bits * (1 << np.arange(self.k))[None, :]).sum(axis=1)


def make_pair(C):
    # one element, coverable by a (cost 1, in S*) or b (cost C, junk)
    return Component("pair", [1.0, C], [[0, 1]], [True, False])


def make_triangle():
    # unit triangle: elements = 3 edges; S* = designated {x, y}
    return Component("triangle", [1.0, 1.0, 1.0],
                     [[0, 1], [1, 2], [2, 0]], [True, True, False])


def make_gadget(eps):
    # twins T (in S*), T' (junk) cover {p,q,r}; harmonic singletons are junk
    return Component("gadget", [1 + eps, 1 + eps, 1.0, 0.5, 1.0 / 3.0],
                     [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
                     [True, False, False, False, False])
