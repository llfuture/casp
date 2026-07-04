#!/usr/bin/env python3
"""CASP core library: loaders + LP certificates + exact solvers.
All four instantiations: Set Cover, Vertex Cover, Facility Location, Knapsack.
Uses pyscipopt for LP relaxations (simplex -> vertex, half-integral for VC) and exact MIPs.
"""
import gzip, json, os, time
from pyscipopt import Model, quicksum

# ----------------------------- LOADERS -----------------------------
def _j(path):
    return json.load(gzip.open(path,"rt")) if path.endswith(".gz") else json.load(open(path))

def load_sc_synth(path):
    d=_j(path); sets=[list(s) for s in d["sets"]]; costs=list(d["costs"])
    universe=set(range(d["num_elements"]))
    return {"universe":universe,"sets":sets,"costs":costs}

def load_scp(path):
    t=open(path).read().split(); it=iter(t)
    m=int(next(it)); n=int(next(it))           # m elements, n sets
    costs=[float(next(it)) for _ in range(n)]
    cover=[[] for _ in range(m)]                # sets covering each element
    for e in range(m):
        k=int(next(it))
        for _ in range(k):
            s=int(next(it))-1; cover[e].append(s)
    sets=[[] for _ in range(n)]
    for e in range(m):
        for s in cover[e]: sets[s].append(e)
    return {"universe":set(range(m)),"sets":sets,"costs":costs}

def load_vc_synth(path):
    d=_j(path); n=d["n"]; edges=[tuple(e) for e in d["edges"]]
    return n, edges, [1.0]*n

def load_dimacs(path):
    n=0; edges=[]
    for ln in open(path):
        p=ln.split()
        if not p: continue
        if p[0]=="p": n=int(p[2])
        elif p[0]=="e": edges.append((int(p[1])-1,int(p[2])-1))
    return n, edges, [1.0]*n

def load_fl_hard(path):
    d=_j(path); m=d["nF"]; n=d["nC"]
    f=[float(d["f"][str(i)]) for i in range(m)]
    c=[[float(d["d"][str(i)][str(j)]) for i in range(m)] for j in range(n)]
    return m,n,f,c

def load_cap(path):
    t=open(path).read().split(); it=iter(t)
    m=int(float(next(it))); n=int(float(next(it))); f=[]
    for _ in range(m): cap=float(next(it)); f.append(float(next(it)))
    c=[[0.0]*m for _ in range(n)]
    for j in range(n):
        dem=float(next(it))
        for i in range(m): c[j][i]=float(next(it))
    return m,n,f,c

def load_knapsack(path):
    d=_j(path); return d["values"], d["weights"], d["capacity"]

# ----------------------------- SET COVER -----------------------------
def sc_lp(universe, sets, costs):
    md=Model(); md.hideOutput()
    x=[md.addVar(vtype="C",lb=0,ub=1) for _ in sets]
    md.setObjective(quicksum(costs[i]*x[i] for i in range(len(sets))),"minimize")
    cover={e:[] for e in universe}
    for i,S in enumerate(sets):
        for e in S: cover[e].append(i)
    for e in universe:
        if cover[e]: md.addCons(quicksum(x[i] for i in cover[e])>=1)
    md.optimize()
    return md.getObjVal(), [md.getVal(v) for v in x]

def sc_freq(universe, sets):
    cnt={e:0 for e in universe}
    for S in sets:
        for e in S: cnt[e]+=1
    return max(cnt.values()) if cnt else 1

def sc_lp_threshold(universe, sets, costs, tau=None):
    """LP-threshold certificate (Thm: f-approx-safe). Returns survivors + pruning rate."""
    f=sc_freq(universe,sets); 
    if tau is None: tau=1.0/f
    lpval, xs = sc_lp(universe,sets,costs)
    survivors=[i for i in range(len(sets)) if xs[i]>=tau-1e-9]
    rate=1.0-len(survivors)/max(len(sets),1)
    return {"survivors":survivors,"prune_rate":rate,"lp":lpval,"f":f,"tau":tau,"x":xs}

def sc_exact(universe, sets, costs, restrict=None, tlim=60):
    idx=restrict if restrict is not None else list(range(len(sets)))
    md=Model(); md.hideOutput()
    x={i:md.addVar(vtype="B") for i in idx}
    md.setObjective(quicksum(costs[i]*x[i] for i in idx),"minimize")
    cover={e:[] for e in universe}
    for i in idx:
        for e in sets[i]: cover[e].append(i)
    feas=True
    for e in universe:
        if cover[e]: md.addCons(quicksum(x[i] for i in cover[e])>=1)
        else: feas=False
    if not feas: return None,None,"infeasible",0.0
    md.setParam("limits/time",tlim); t=time.time(); md.optimize(); el=time.time()-t
    if md.getStatus() in ("optimal","timelimit") and md.getNSols()>0:
        return md.getObjVal(), set(i for i in idx if md.getVal(x[i])>0.5), md.getStatus(), el
    return None,None,md.getStatus(),el

def sc_greedy(universe, sets, costs):
    covered=set(); chosen=set(); U=set(universe)
    while covered!=U:
        rem=U-covered; best,br=None,None
        for i,S in enumerate(sets):
            if i in chosen: continue
            g=len(set(S)&rem)
            if g==0: continue
            r=costs[i]/g
            if br is None or r<br: best,br=i,r
        if best is None: break
        chosen.add(best); covered|=set(sets[best])
    return chosen, sum(costs[i] for i in chosen)

# ----------------------------- VERTEX COVER -----------------------------
def vc_lp_halfint(n, edges, w):
    md=Model(); md.hideOutput()
    x=[md.addVar(vtype="C",lb=0,ub=1) for _ in range(n)]
    md.setObjective(quicksum(w[v]*x[v] for v in range(n)),"minimize")
    for (u,v) in edges: md.addCons(x[u]+x[v]>=1)
    md.optimize(); xs=[md.getVal(v) for v in x]
    P0=[v for v in range(n) if xs[v]<0.25]
    P1=[v for v in range(n) if xs[v]>0.75]
    Ph=[v for v in range(n) if 0.25<=xs[v]<=0.75]
    return md.getObjVal(), xs, P0, P1, Ph

def vc_nt_certificate(n, edges, w):
    """NT persistency certificate (OPT-preserving). Returns core subgraph + fixed cost."""
    lp,xs,P0,P1,Ph=vc_lp_halfint(n,edges,w)
    core=set(Ph); core_edges=[(u,v) for (u,v) in edges if u in core and v in core]
    c_fix=sum(w[v] for v in P1)
    return {"P0":P0,"P1":P1,"Phalf":Ph,"core_edges":core_edges,"c_fix":c_fix,
            "core_size":len(Ph),"prune_rate":1.0-len(Ph)/max(n,1),"lp":lp}

def vc_exact(n, edges, w, restrict=None, tlim=60):
    V=restrict if restrict is not None else list(range(n))
    Vs=set(V)
    md=Model(); md.hideOutput()
    x={v:md.addVar(vtype="B") for v in V}
    md.setObjective(quicksum(w[v]*x[v] for v in V),"minimize")
    for (u,v) in edges:
        if u in Vs and v in Vs: md.addCons(x[u]+x[v]>=1)
    md.setParam("limits/time",tlim); t=time.time(); md.optimize(); el=time.time()-t
    if md.getNSols()>0:
        return md.getObjVal(), set(v for v in V if md.getVal(x[v])>0.5), md.getStatus(), el
    return None,None,md.getStatus(),el

# ----------------------------- FACILITY LOCATION -----------------------------
def fl_lp(m,n,f,c):
    md=Model(); md.hideOutput()
    y={i:md.addVar(vtype="C",lb=0,ub=1) for i in range(m)}
    x={(i,j):md.addVar(vtype="C",lb=0,ub=1) for i in range(m) for j in range(n)}
    md.setObjective(quicksum(f[i]*y[i] for i in range(m))+quicksum(c[j][i]*x[i,j] for i in range(m) for j in range(n)),"minimize")
    for j in range(n):
        md.addCons(quicksum(x[i,j] for i in range(m))>=1)
        for i in range(m): md.addCons(x[i,j]<=y[i])
    md.optimize()
    return md.getObjVal(), {i:md.getVal(y[i]) for i in range(m)}

def fl_facility_integral(m,n,f,c,tol=1e-6):
    """Theorem D: if LP facility-integral, nearest-assignment is exact."""
    lp,yv=fl_lp(m,n,f,c)
    integral=all(v<tol or v>1-tol for v in yv.values())
    if integral:
        Fstar=[i for i in range(m) if yv[i]>0.5]
        cost=sum(f[i] for i in Fstar)+sum(min(c[j][i] for i in Fstar) for j in range(n)) if Fstar else None
        return {"facility_integral":True,"casp_exact":cost,"open":Fstar,"lp":lp}
    return {"facility_integral":False,"lp":lp,"open":[i for i in range(m) if yv[i]>0.5]}

def fl_exact(m,n,f,c,tlim=60):
    md=Model(); md.hideOutput()
    y={i:md.addVar(vtype="B") for i in range(m)}
    x={(i,j):md.addVar(vtype="C",lb=0,ub=1) for i in range(m) for j in range(n)}
    md.setObjective(quicksum(f[i]*y[i] for i in range(m))+quicksum(c[j][i]*x[i,j] for i in range(m) for j in range(n)),"minimize")
    for j in range(n):
        md.addCons(quicksum(x[i,j] for i in range(m))>=1)
        for i in range(m): md.addCons(x[i,j]<=y[i])
    md.setParam("limits/time",tlim); t=time.time(); md.optimize(); el=time.time()-t
    if md.getNSols()>0: return md.getObjVal(), md.getStatus(), el
    return None, md.getStatus(), el

if __name__=="__main__":
    import glob
    base=os.path.expanduser("~/projects/casp_max/data")
    # smoke tests
    sc=load_sc_synth(sorted(glob.glob(base+"/synthetic/set_cover/sc_*.json.gz"))[0])
    cert=sc_lp_threshold(sc["universe"],sc["sets"],sc["costs"])
    print("SC: f=%d prune_rate=%.2f survivors=%d/%d"%(cert["f"],cert["prune_rate"],len(cert["survivors"]),len(sc["sets"])))
    n,edges,w=load_vc_synth(sorted(glob.glob(base+"/synthetic/vertex_cover/*.json.gz"))[0])
    nt=vc_nt_certificate(n,edges,w)
    print("VC: n=%d core=%d prune_rate=%.2f c_fix=%.0f"%(n,nt["core_size"],nt["prune_rate"],nt["c_fix"]))
    m,nn,f,c=load_fl_hard(sorted(glob.glob(base+"/synthetic/fl_hard/*.json.gz"))[0])
    fd=fl_facility_integral(m,nn,f,c)
    print("FL: facility_integral=%s open=%d"%(fd["facility_integral"],len(fd["open"])))
    print("OK")
