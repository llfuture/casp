#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heterogeneous Set Cover generator for the P1 MVE (G1 decision gate).

Plants TWO regions with different pruning-safety, distinguishable by an OBSERVABLE
feature (mean element frequency of a set), NOT by any hidden label:

  H-region ("harmless redundancy"): elements covered by many layers (redundancy H_red).
     -> LP spreads mass, many sets get SMALL x*, and pruning a small-x* set is usually
        safe because sibling layers still cover its elements. High safe threshold.
  C-region ("critical / tight"): elements covered by only C_red(=2) layers.
     -> low-x* sets here are fragile: pruning them tends to force expensive re-cover
        or infeasibility (-> fallback). Low safe threshold.

A single global tau cannot separate them (their x* ranges overlap); a per-bucket
threshold keyed on mean element frequency can. That is exactly the learning gain P1 needs.

Two distributions:
  D1: small C-region  (few fragile sets)  -> global aggressive threshold nearly OK
  D2: large C-region  (many fragile sets) -> must protect C; per-bucket wins more
Cross-distribution train(D1)/test(D2) probes generalization; the single-distribution
gain (per-bucket vs single-tau on the SAME distribution) is the primary G1 signal.
"""
import gzip, json, os, random, argparse

def _layers(elems, red, min_sz, max_sz, rng):
    """Cover `elems` with `red` layers; each layer partitions elems into sets of
    random size in [min_sz,max_sz]. Guarantees each element covered exactly `red` times."""
    out = []
    for _ in range(red):
        pool = list(elems); rng.shuffle(pool)
        i = 0
        while i < len(pool):
            sz = rng.randint(min_sz, max_sz)
            out.append(pool[i:i+sz]); i += sz
    return out

def gen_instance(rng, nH, nC, H_red, C_red):
    """Return (sets, costs, num_elements). H elements [0,nH), C elements [nH,nH+nC)."""
    sets, costs = [], []
    H = list(range(nH))
    C = list(range(nH, nH + nC))
    # H-region: high redundancy, cheap, medium-large sets
    for S in _layers(H, H_red, 3, 7, rng):
        if S:
            sets.append(sorted(S)); costs.append(round(rng.uniform(1.0, 1.4), 3))
    # C-region: low redundancy (tight), pricier, small sets
    for S in _layers(C, C_red, 2, 3, rng):
        if S:
            sets.append(sorted(S)); costs.append(round(rng.uniform(1.6, 2.2), 3))
    # a few cross sets covering 1 H + 1 C element to entangle regions slightly (realism)
    for _ in range(max(1, (nH + nC) // 25)):
        a = rng.choice(H); b = rng.choice(C)
        sets.append(sorted({a, b})); costs.append(round(rng.uniform(1.4, 1.9), 3))
    return sets, costs, nH + nC

def write_set(path, n, sets, costs):
    with gzip.open(path, "wt") as f:
        json.dump({"num_elements": n, "sets": sets, "costs": costs}, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/projects/casp_max/data/synthetic/hetero_sc"))
    ap.add_argument("--n", type=int, default=60, help="instances per distribution")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # (name, nH, nC, H_red, C_red)
    specs = {
        "D1": dict(nH=70, nC=20, H_red=6, C_red=2),   # small critical region
        "D2": dict(nH=70, nC=55, H_red=6, C_red=2),   # large critical region
    }
    for name, sp in specs.items():
        d = os.path.join(args.out, name); os.makedirs(d, exist_ok=True)
        for k in range(args.n):
            sets, costs, n = gen_instance(rng, sp["nH"], sp["nC"], sp["H_red"], sp["C_red"])
            write_set(os.path.join(d, "h_%s_%04d.json.gz" % (name, k)), n, sets, costs)
        print("wrote %d %s instances -> %s (nH=%d nC=%d H_red=%d C_red=%d)" %
              (args.n, name, d, sp["nH"], sp["nC"], sp["H_red"], sp["C_red"]))

if __name__ == "__main__":
    main()
