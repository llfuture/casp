#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-F2: a LEARNED predictor earns a positive margin on a HIGH-DEGENERACY family
(the learning-side instantiation of Thm lponly; complements EX-F where the
low-degeneracy VC distribution gave no prediction advantage, per Prop degen).

Family: tagged twin-gadget Set Cover (H-family). Each gadget has two LP-identical
twins; the data generator marks the historically chosen twin with an observable
binary TAG (flipped with prob tag_noise) -- side information invisible to the LP
(both twins have identical cost/coverage) but learnable from features. A bipartite
GNN (sets <-> elements, pure torch) predicts membership; predictions feed the
CF/CF+ filters. LP-commit is stuck at the 11/6 harmonic trap per gadget
(Thm lponly); a learned predictor that resolves the degeneracy escapes it.
"""
import sys, os, json, random, statistics as st
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
from casp_lp import make_pair, make_gadget, local_greedy

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TH = [0.0, 0.5, 1.01]


def gen(seed, n_pairs=30, m_gad=30, C=5.0, eps=0.2, tag_noise=0.2):
    """Build one instance: sets list [(cost, elems, tag, in_star)], plus sigma."""
    rng = random.Random(seed)
    pair, gad = make_pair(C), make_gadget(eps)
    sp = pair.sigma("unique")          # A=1, B=0
    sg = gad.sigma("ri")               # twins 1/2, singletons 0
    sets, sigma, star, tags = [], [], [], []
    e_off = 0
    comp_sizes = []
    for _ in range(n_pairs):
        for k, (c, sig) in enumerate(zip(pair.cost, sp)):
            elems = frozenset(e + e_off for e in pair.sets_elems[k])
            in_star = (k == 0)
            tag = 1 if (in_star ^ (rng.random() < tag_noise)) else 0
            sets.append((float(c), elems)); sigma.append(float(sig))
            star.append(in_star); tags.append(tag)
        e_off += pair.n_elems; comp_sizes.append(pair.k)
    for _ in range(m_gad):
        chosen_twin = rng.randint(0, 1)          # historical convention
        for k, (c, sig) in enumerate(zip(gad.cost, sg)):
            elems = frozenset(e + e_off for e in gad.sets_elems[k])
            in_star = (k == chosen_twin)
            tag = 1 if (in_star ^ (rng.random() < tag_noise)) else 0
            sets.append((float(c), elems)); sigma.append(float(sig))
            star.append(in_star); tags.append(tag)
        e_off += gad.n_elems; comp_sizes.append(gad.k)
    opt = n_pairs * pair.opt + m_gad * gad.opt
    return {"sets": sets, "sigma": np.array(sigma), "star": np.array(star, bool),
            "tags": np.array(tags, float), "n_elems": e_off, "opt": opt,
            "n_pairs": n_pairs, "m_gad": m_gad}


def instance_tensors(inst):
    m = len(inst["sets"]); ne = inst["n_elems"]
    costs = np.array([c for c, _ in inst["sets"]])
    cov = np.array([len(e) for _, e in inst["sets"]], float)
    csort = np.sort(costs)
    X = np.stack([np.searchsorted(csort, costs) / m, cov / cov.max(),
                  (costs / cov) / max((costs / cov).max(), 1e-9),
                  inst["tags"]], 1)
    B = np.zeros((m, ne))
    for i, (_, es) in enumerate(inst["sets"]):
        for e in es:
            B[i, e] = 1.0
    Bn = B / np.maximum(B.sum(0, keepdims=True), 1)      # elem <- mean over sets
    Cn = B / np.maximum(B.sum(1, keepdims=True), 1)      # set  <- mean over elems
    y = inst["star"].astype(float)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return t(X), t(Bn), t(Cn), t(y)


class BiGNN(nn.Module):
    def __init__(self, d=4, h=32):
        super().__init__()
        self.up = nn.Linear(d, h)
        self.e1 = nn.Linear(h, h)
        self.s1 = nn.Linear(2 * h, h)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, Bn, Cn):
        hs = torch.relu(self.up(X))                    # set embeddings
        he = torch.relu(self.e1(Bn.T @ hs))            # element embeddings
        hs2 = torch.relu(self.s1(torch.cat([hs, Cn @ he], 1)))
        return self.head(hs2).squeeze(-1)


def policies(inst, pred, th1, th2):
    """CF+ commit, generic greedy completion, min with greedy fallback."""
    sig = inst["sigma"]
    m = len(inst["sets"])
    com = [i for i in range(m) if (pred[i] and sig[i] >= th1 - 1e-9) or sig[i] >= th2 - 1e-9]
    cost = [c for c, _ in inst["sets"]]
    se = [es for _, es in inst["sets"]]
    tot, _ = local_greedy(cost, se, inst["n_elems"], com)
    fb, _ = local_greedy(cost, se, inst["n_elems"], [])
    return min(tot, fb) / inst["opt"]


def evaluate(model, thr_insts, te_insts, name):
    def predset(inst):
        X, Bn, Cn, y = instance_tensors(inst)
        with torch.no_grad():
            p = torch.sigmoid(model(X, Bn, Cn)).cpu().numpy()
        pred = p > 0.5
        tp = (pred & inst["star"]).sum(); fp = (pred & ~inst["star"]).sum()
        fn = (~pred & inst["star"]).sum()
        return pred, 2 * tp / max(2 * tp + fp + fn, 1)
    preds_thr = {id(r): predset(r)[0] for r in thr_insts}
    grid = [(a, b) for a in TH for b in TH]
    b_cf = min(TH, key=lambda a: st.mean(policies(r, preds_thr[id(r)], a, 1.01) for r in thr_insts))
    b_2 = min(grid, key=lambda ab: st.mean(policies(r, preds_thr[id(r)], *ab) for r in thr_insts))
    rows, f1s = [], []
    for r in te_insts:
        pred, f1 = predset(r); f1s.append(float(f1))
        rows.append({
            "mc": policies(r, pred, 0.0, 1.01),
            "cf": policies(r, pred, b_cf, 1.01),
            "cfplus": policies(r, pred, *b_2),
            "lponly": policies(r, np.zeros(len(pred), bool), 1.01, 1.0),
            "oracle": policies(r, r["star"], 0.5, 1.01)})
    out = {"n": len(te_insts), "f1": round(st.mean(f1s), 3),
           "theta_cf": b_cf, "theta_cfplus": list(b_2)}
    for k in ["mc", "cf", "cfplus", "lponly", "oracle"]:
        out[k] = round(st.mean(r[k] for r in rows), 4)
    out["adv_pred"] = round(out["lponly"] - out["cfplus"], 4)
    print("[%s] F1=%.3f mc=%.4f CF=%.4f CF+=%.4f LP-only=%.4f oracle=%.4f adv=%.4f"
          % (name, out["f1"], out["mc"], out["cf"], out["cfplus"],
             out["lponly"], out["oracle"], out["adv_pred"]), flush=True)
    return out


def main():
    print("device:", DEV, flush=True)
    train = [gen(s) for s in range(150)]
    thr = [gen(1000 + s) for s in range(30)]
    test_id = [gen(2000 + s) for s in range(40)]
    test_ood = [gen(3000 + s, m_gad=45, eps=0.3, tag_noise=0.3) for s in range(40)]

    data = [instance_tensors(r) for r in train]
    model = BiGNN().to(DEV)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(120):
        tot = 0.0
        for X, Bn, Cn, y in data:
            optim.zero_grad()
            ls = lossf(model(X, Bn, Cn), y)
            ls.backward(); optim.step(); tot += ls.item()
        if ep % 30 == 0:
            print("  epoch %d loss %.4f" % (ep, tot / len(data)), flush=True)

    res = {"in_distribution": evaluate(model, thr, test_id, "ID"),
           "ood_harder": evaluate(model, thr, test_ood, "OOD")}
    json.dump(res, open(os.path.join(OUT, "exf2_twin_gnn.json"), "w"), indent=1)
    print("SAVED exf2_twin_gnn.json")
    print("\n===== EX-F2 VERDICT =====")
    for k, v in res.items():
        print("%s: learned CF+ %.4f vs LP-only %.4f (adv %.4f, F1 %.3f)"
              % (k, v["cfplus"], v["lponly"], v["adv_pred"], v["f1"]))
    print("story holds if adv > 0: a learned predictor resolves LP degeneracy "
          "that no deterministic LP policy can (Thm lponly, learning-side)")


if __name__ == "__main__":
    main()
