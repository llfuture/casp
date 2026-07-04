#!/usr/bin/env python3
"""
E1-FL (Theorem D): facility-integral OPT-preserving certificate on REAL UFL.
For each ORLIB cap* instance: solve the UFL LP relaxation; if y* is facility-integral
(checkable), CASP returns nearest-assignment over open facilities as an EXACT optimum
(Theorem D). We verify against an exact SCIP MIP solve: 0 mismatch is required.
Honest reporting: also count how many real instances actually trigger the certificate.
"""
import sys, os, json, time, glob
from pyscipopt import Model, quicksum

def parse_cap(path):
    toks = open(path).read().split()
    it = iter(toks); 
    m = int(float(next(it))); n = int(float(next(it)))
    f = []
    for _ in range(m):
        cap = float(next(it)); opencost = float(next(it)); f.append(opencost)
    c = [[0.0]*m for _ in range(n)]
    for j in range(n):
        dem = float(next(it))
        for i in range(m):
            c[j][i] = float(next(it))
    return m, n, f, c

def build(m, n, f, c, integer):
    md = Model(); md.hideOutput()
    y = {i: md.addVar(vtype=("B" if integer else "C"), lb=0, ub=1) for i in range(m)}
    x = {(i,j): md.addVar(vtype=("B" if integer else "C"), lb=0, ub=1) for i in range(m) for j in range(n)}
    md.setObjective(quicksum(f[i]*y[i] for i in range(m)) +
                    quicksum(c[j][i]*x[i,j] for i in range(m) for j in range(n)), "minimize")
    for j in range(n):
        md.addCons(quicksum(x[i,j] for i in range(m)) >= 1)
        for i in range(m):
            md.addCons(x[i,j] <= y[i])
    return md, y, x

def solve_lp(m, n, f, c):
    md, y, x = build(m, n, f, c, integer=False)
    md.optimize(); val = md.getObjVal()
    yv = {i: md.getVal(y[i]) for i in range(m)}
    return val, yv

def solve_exact(m, n, f, c, tlim):
    md, y, x = build(m, n, f, c, integer=True)
    md.setParam("limits/time", tlim)
    t=time.time(); md.optimize(); el=time.time()-t
    status = md.getStatus()
    return (md.getObjVal() if status in ("optimal","timelimit") else None), status, el

def casp_facint(m, n, f, c, yv, tol=1e-6):
    """If y* facility-integral, return nearest-assignment exact cost (Theorem D)."""
    if not all(v < tol or v > 1-tol for v in yv.values()):
        return None, False
    Fstar = [i for i in range(m) if yv[i] > 0.5]
    if not Fstar: return None, False
    cost = sum(f[i] for i in Fstar) + sum(min(c[j][i] for i in Fstar) for j in range(n))
    return cost, True

def main():
    files = sorted(glob.glob(os.path.expanduser("~/projects/casp_max/data/benchmarks/fl/cap/cap*.txt")))
    files = files[:int(sys.argv[1])] if len(sys.argv)>1 else files
    tlim = float(sys.argv[2]) if len(sys.argv)>2 else 60.0
    res=[]; trig=0; mm=0
    for path in files:
        name=os.path.basename(path)
        try:
            m,n,f,c = parse_cap(path)
            t=time.time(); lpval, yv = solve_lp(m,n,f,c); lp_t=time.time()-t
            casp_cost, triggered = casp_facint(m,n,f,c,yv)
            row={"inst":name,"m":m,"n":n,"lp":round(lpval,2),"facility_integral":triggered,"lp_time":round(lp_t,3)}
            if triggered:
                opt, st, el = solve_exact(m,n,f,c,tlim)
                row["casp_exact"]=round(casp_cost,2); row["scip_opt"]=(round(opt,2) if opt else None)
                row["scip_status"]=st; row["scip_time"]=round(el,3)
                row["mismatch"] = (opt is not None and abs(casp_cost-opt) > 1e-4*max(1,abs(opt)))
                row["speedup"] = round(el/max(lp_t,1e-3),1)
                trig+=1; mm+=int(bool(row["mismatch"]))
            res.append(row); print(json.dumps(row))
        except Exception as e:
            print("ERR",name,e)
    out=os.path.expanduser("~/projects/casp_max/outputs/theoremD/fl_facint.json")
    json.dump({"summary":{"n_instances":len(res),"triggered":trig,"mismatches":mm},"rows":res},
              open(out,"w"), indent=1)
    print("=== SUMMARY ===", json.dumps({"n":len(res),"triggered":trig,"mismatches":mm}))
    print("saved",out)

if __name__=="__main__": main()
