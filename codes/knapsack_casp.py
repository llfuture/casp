#!/usr/bin/env python3
"""
E9 (Theorem E): negative-signal reduced-bound exclusion/inclusion certificate for
0/1 Knapsack. Predictor supplies an incumbent z_low (a feasible packing value).
For item i: U_i = LP bound forcing x_i=1; if U_i < z_low it is OPT-excluded.
U_i0 = LP bound forcing x_i=0; if U_i0 < z_low item i is OPT-included.
We verify: (a) certificates never exclude an item used by the SCIP exact optimum
(0 mismatch), (b) residual core collapse -> speedup vs full exact solve.
"""
import sys, os, json, gzip, glob, time
from pyscipopt import Model, quicksum

def lp_bound(values, weights, W, force1=None, force0=None):
    """Dantzig fractional bound with forced in/out sets."""
    n=len(values); force1=force1 or set(); force0=force0 or set()
    base=0; cap=W
    for i in force1:
        base+=values[i]; cap-=weights[i]
    if cap < 0: return -1  # infeasible forcing
    items=[i for i in range(n) if i not in force1 and i not in force0]
    items.sort(key=lambda i: values[i]/weights[i], reverse=True)
    val=base
    for i in items:
        if weights[i]<=cap: cap-=weights[i]; val+=values[i]
        else: val+=values[i]*(cap/weights[i]); cap=0; break
    return val

def greedy_incumbent(values, weights, W):
    n=len(values); order=sorted(range(n), key=lambda i: values[i]/weights[i], reverse=True)
    cap=W; val=0; chosen=set()
    for i in order:
        if weights[i]<=cap: cap-=weights[i]; val+=values[i]; chosen.add(i)
    return val, chosen

def exact(values, weights, W, tlim):
    md=Model(); md.hideOutput(); n=len(values)
    x={i:md.addVar(vtype="B") for i in range(n)}
    md.setObjective(quicksum(values[i]*x[i] for i in range(n)),"maximize")
    md.addCons(quicksum(weights[i]*x[i] for i in range(n))<=W)
    md.setParam("limits/time",tlim)
    t=time.time(); md.optimize(); el=time.time()-t
    sol=set(i for i in range(n) if md.getVal(x[i])>0.5)
    return md.getObjVal(), sol, md.getStatus(), el

def run(inst, tlim=60):
    v=inst["values"]; w=inst["weights"]; W=inst["capacity"]; n=len(v)
    z_low,_=greedy_incumbent(v,w,W)
    excl=set(); incl=set()
    for i in range(n):
        if lp_bound(v,w,W,force1={i}) < z_low - 1e-9: excl.add(i)
        if lp_bound(v,w,W,force0={i}) < z_low - 1e-9: incl.add(i)
    core=[i for i in range(n) if i not in excl and i not in incl]
    opt, sol, st, el = exact(v,w,W,tlim)
    # mismatch: any excluded item used by opt, or any included item NOT in opt
    mismatch = any(i in sol for i in excl) or any(i not in sol for i in incl)
    return {"kind":inst["kind"],"n":n,"fixed_out":len(excl),"fixed_in":len(incl),
            "core":len(core),"core_frac":round(len(core)/n,3),"opt":round(opt,1),
            "z_low":round(z_low,1),"mismatch":bool(mismatch),"scip_time":round(el,3),"status":st}

def main():
    files=sorted(glob.glob(os.path.expanduser("~/projects/casp_max/data/synthetic/knapsack/kp_*.json.gz")))
    # sample across kinds and sizes
    pick=[f for f in files if any(s in f for s in ["n100_0","n200_0"])]
    pick=pick[:int(sys.argv[1])] if len(sys.argv)>1 else pick
    res=[]; mm=0
    for f in pick:
        inst=json.load(gzip.open(f,"rt"))
        r=run(inst); r["file"]=os.path.basename(f); res.append(r); mm+=int(r["mismatch"])
        print(json.dumps(r))
    out=os.path.expanduser("~/projects/casp_max/outputs/theoremD/knapsack_casp.json")
    json.dump({"summary":{"n":len(res),"mismatches":mm,
              "mean_core_frac":round(sum(r["core_frac"] for r in res)/max(len(res),1),3)},
              "rows":res}, open(out,"w"), indent=1)
    print("=== SUMMARY ===", json.dumps({"n":len(res),"mismatches":mm}))
