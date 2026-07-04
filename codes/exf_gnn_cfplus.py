#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX-F: a REAL learned predictor (hand-rolled GNN, pure torch) feeding CF/CF+.

Task: per-vertex membership prediction for min Vertex Cover. Features are
purely combinatorial (degree, neighbor stats, triangles) -- deliberately NOT
the LP value, so the predictor is an independent signal from the verifier's
confidence. The predicted set S_hat replaces the synthetic-noise predictions
of E12/E12'; arms: min-combiner, CF(theta*), CF+(theta1,theta2), LP-commit.
Train G(45,0.09); test in-distribution and OOD G(80,0.055).
"""
import sys, os, json, random, statistics as st
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/scripts"))
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
import r3_harden as H
from multiprocessing import Pool

OUT = os.path.expanduser("~/projects/casp_max/outputs/run")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TH = [0.0, 0.5, 1.0, 1.01]


def gen(seed, n, p):
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return n, edges


def prep(args):
    seed, n, p = args
    nn_, edges = gen(seed, n, p)
    if not edges:
        return None
    w = [1.0] * nn_
    opt, sol, stt, _ = L.vc_exact(nn_, edges, w, tlim=30)
    if opt is None or stt != "optimal" or opt <= 0:
        return None
    lp, xs, P0, P1, Ph = L.vc_lp_halfint(nn_, edges, w)
    fb = len(H.vc_complete(nn_, edges, {v for v in range(nn_) if xs[v] >= 0.5}))
    return {"seed": seed, "n": nn_, "edges": edges, "opt": opt,
            "star": set(sol), "x": xs, "fb": fb}


def feats(rec):
    n, edges = rec["n"], rec["edges"]
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = np.array([len(a) for a in adj], float)
    mx = max(deg.max(), 1)
    nbdeg = np.array([np.mean([deg[u] for u in a]) if a else 0 for a in adj])
    tri = np.zeros(n)
    eset = set(edges)
    for u, v in edges:
        for wv in set(adj[u]) & set(adj[v]):
            tri[u] += 1; tri[v] += 1; tri[wv] += 1
    X = np.stack([deg / mx, nbdeg / mx, tri / max(tri.max(), 1),
                  deg / max(n - 1, 1), np.full(n, len(edges) / max(n, 1) / mx)], 1)
    y = np.array([1.0 if v in rec["star"] else 0.0 for v in range(n)])
    # normalized adjacency (mean aggregation, self-loop)
    A = np.eye(n)
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    A = A / A.sum(1, keepdims=True)
    return X, y, A


class GNN(nn.Module):
    def __init__(self, d=5, h=32):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, h), nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, A):
        h1 = torch.relu(self.w1(A @ X))
        h2 = torch.relu(self.w2(A @ h1))
        return self.head(h2).squeeze(-1)


def train_gnn(insts, epochs=150):
    data = [tuple(map(lambda z: torch.tensor(z, dtype=torch.float32, device=DEV),
                      feats(r))) for r in insts]
    model = GNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        tot = 0.0
        for X, y, A in data:
            opt.zero_grad()
            out = model(X, A)
            ls = lossf(out, y)
            ls.backward(); opt.step()
            tot += ls.item()
        if ep % 30 == 0:
            print("  epoch %d loss %.4f" % (ep, tot / len(data)), flush=True)
    return model


def predict(model, rec):
    X, y, A = feats(rec)
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.tensor(X, dtype=torch.float32, device=DEV),
            torch.tensor(A, dtype=torch.float32, device=DEV))).cpu().numpy()
    pred = {v for v in range(rec["n"]) if p[v] > 0.5}
    tp = len(pred & rec["star"]); fp = len(pred - rec["star"]); fn = len(rec["star"] - pred)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return pred, f1


def cfplus_vc(rec, pred, t1, t2):
    xs = rec["x"]
    com = {v for v in pred if xs[v] >= t1 - 1e-9} | \
          {v for v in range(rec["n"]) if xs[v] >= t2 - 1e-9}
    return min(len(H.vc_complete(rec["n"], rec["edges"], com)), rec["fb"]) / rec["opt"]


def eval_split(model, tr_pol, te, name):
    """tr_pol: instances for ERM of thresholds; te: evaluation."""
    preds_tr = {r["seed"]: predict(model, r)[0] for r in tr_pol}
    th_cf = min(TH, key=lambda t: st.mean(
        H.vc_arms(r, preds_tr[r["seed"]], t)[2] for r in tr_pol))
    best2 = min(((a, b) for a in TH for b in TH),
                key=lambda ab: st.mean(cfplus_vc(r, preds_tr[r["seed"]], *ab) for r in tr_pol))
    rows, f1s = [], []
    for r in te:
        pred, f1 = predict(model, r)
        f1s.append(f1)
        fbr, mc, cf = H.vc_arms(r, pred, th_cf)
        lp1 = cfplus_vc(r, set(), 1.01, 1.0)
        cfp = cfplus_vc(r, pred, *best2)
        rows.append({"mc": mc, "cf": cf, "lp1": lp1, "cfplus": cfp, "fb": fbr})
    out = {"n": len(te), "f1_mean": round(st.mean(f1s), 3),
           "theta_cf": th_cf, "theta_cfplus": list(best2)}
    for k in ["fb", "mc", "cf", "lp1", "cfplus"]:
        out[k] = round(st.mean(r[k] for r in rows), 4)
    print("[%s] F1=%.3f fb=%.3f mc=%.3f CF=%.3f LP1=%.3f CF+=%.3f" %
          (name, out["f1_mean"], out["fb"], out["mc"], out["cf"],
           out["lp1"], out["cfplus"]), flush=True)
    return out


def main():
    print("device:", DEV, flush=True)
    with Pool(16) as pool:
        tr = [r for r in pool.map(prep, [(s, 45, 0.09) for s in range(260)]) if r]
    train_insts, thr_insts, test_id = tr[:180], tr[180:210], tr[210:250]
    with Pool(16) as pool:
        test_ood = [r for r in pool.map(prep, [(1000 + s, 80, 0.055) for s in range(60)]) if r]
    print("train=%d thr=%d test_id=%d test_ood=%d" %
          (len(train_insts), len(thr_insts), len(test_id), len(test_ood)), flush=True)

    model = train_gnn(train_insts)
    res = {"in_distribution": eval_split(model, thr_insts, test_id, "ID"),
           "ood_G80": eval_split(model, thr_insts, test_ood, "OOD")}
    json.dump(res, open(os.path.join(OUT, "exf_gnn_cfplus.json"), "w"), indent=1)
    print("SAVED exf_gnn_cfplus.json")
    print("\n===== EX-F VERDICT =====")
    for k, v in res.items():
        print("%s: GNN-fed CF+ %.4f vs LP-only %.4f vs mc %.4f (F1 %.3f)" %
              (k, v["cfplus"], v["lp1"], v["mc"], v["f1_mean"]))
    print("story holds if CF+(GNN) <= LP-only and mc, with F1 well below 1 (imperfect predictor made safe)")


if __name__ == "__main__":
    main()
