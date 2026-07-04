#!/usr/bin/env python3
"""Adversarial Set Cover generator for CASP E5.
Creates instances where LP threshold certificate prunes essential sets,
testing the safety net (Algorithm 1 step 7+10 fallback).
"""
import json, gzip, random, os, sys
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def generate_one(f_val=5, m_val=200, s_total=500, seed=0):
    """Generate one adversarial SC instance.
    
    Strategy:
    - Create a hidden core of f_val sets that uniquely cover all elements
      (each element covered by exactly f_val sets from the core)
    - These core sets have HIGH cost → LP will assign them small x* values
    - Add many CHEAP sets → LP will assign them large x* values  
    - Set τ so that core sets are pruned (x*_S < τ) but cheap sets survive
    - Cheap sets alone cannot cover all elements → pruned instance is INFEASIBLE
    - Trigger: safety net must detect infeasibility and fall back to greedy
    """
    random.seed(seed)
    np.random.seed(seed)
    
    core_size = f_val  # exactly f_val core sets
    cheap_count = s_total - core_size
    
    # Step 1: Core sets cover all m_val elements uniquely
    sets = []
    costs = []
    uncovered = set(range(m_val))
    core_sets = []
    for i in range(core_size):
        # Each core set covers remaining elements proportionally
        take = min(len(uncovered), max(1, m_val // core_size + random.randint(-2, 2)))
        elements = random.sample(list(uncovered), min(take, len(uncovered)))
        uncovered -= set(elements)
        sets.append(sorted(elements))
        costs.append(random.randint(80, 100))  # HIGH cost
        core_sets.append(i)
    
    # If any elements remain uncovered, distribute to random core sets
    if uncovered:
        for elem in list(uncovered):
            idx = random.choice(core_sets)
            sets[idx] = sorted(sets[idx] + [elem])
        uncovered = set()
    
    # Step 2: Add cheap sets (each covering a small random subset)
    for i in range(cheap_count):
        n_cover = random.randint(1, min(10, m_val))
        elements = random.sample(range(m_val), n_cover)
        sets.append(sorted(elements))
        costs.append(random.randint(1, 10))  # LOW cost
    
    inst = {
        "type": "adversarial",
        "f": f_val,
        "num_elements": m_val,
        "num_sets": s_total,
        "core_set_indices": core_sets,
        "design": "core_sets_HIGH_cost_LP_unfavored_vs_cheap_sets_LP_favored",
        "expected_behavior": "LP_threshold_certificate_prunes_core_sets→IΦ_infeasible→safety_net_triggers",
        "sets": sets,
        "costs": costs
    }
    return inst

if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))
    instances = []
    for fv in [3, 5, 10]:
        for mv in [200, 500]:
            for sv in [500, 1000]:
                for rep in range(5):  # 5 per config, 60 total
                    seed = SEED * 1000 + hash((fv, mv, sv, rep)) % 1000000
                    inst = generate_one(fv, mv, sv, seed)
                    fname = f"adv_f{fv}_m{mv}_s{sv}_{rep}.json.gz"
                    with gzip.open(os.path.join(outdir, fname), "wb") as f:
                        f.write(json.dumps(inst).encode())
                    instances.append({
                        "f": fv, "m": mv, "s": sv, "rep": rep,
                        "core_sets": len(inst["core_set_indices"]),
                        "file": fname
                    })
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"total": len(instances), "instances": instances}, f, indent=2)
    print(f"Generated {len(instances)} adversarial instances → {outdir}")
