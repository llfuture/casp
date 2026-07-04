#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 MVE iteration 2b: does the per-bucket gain SCALE when the safely-prunable
low-f region is enlarged, while single-tau stays pinned at ~0 (poisoned by a
small high-f region with degenerate in-OPT sets at x*~0)?
Reuses gen + evaluate from mve_freq / mve_p1.
"""
import sys, os, json, statistics as st
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import mve_freq as MF

def main():
    specs = [   # (name, nA_lowf, nB_highf, fA, fB)
        ("F3", 75, 15, 2, 10),
        ("F4", 85, 12, 2, 12),
        ("F5", 90,  8, 2, 14),
    ]
    res = {}
    for name, nA, nB, fA, fB in specs:
        MF.gen(name, 40, nA, nB, fA, fB, seed=20 + len(name))
        D = MF.load_dir(name)
        print("== diagnose %s (n=%d) ==" % (name, len(D)))
        MF.M.diagnose(D, 2)
        res[name] = MF.evaluate(D, name)
    json.dump(res, open(os.path.expanduser("~/projects/casp_max/outputs/run/mve_freq2.json"), "w"), indent=1)
    print("\n===== SCALING CHECK =====")
    for name, *_ in specs:
        r = res[name]
        print("%s: single=%.3f  per-bucket=%.3f  gain=%.3f  speedup=%s"
              % (name, r["single"], r["perbucket_p2"], r["gain"], r["speedup"].get("mean_speedup")))

if __name__ == "__main__":
    main()
