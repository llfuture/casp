#!/usr/bin/env python3
"""E7 planted double-pendant model G(n,k,H) + independent Cert optimality checker (Thm A),
and E8 cost-spread Set Cover family (Thm B)."""
import random
from casp_lib import vc_lp_halfint

def gen_planted(n, k, s, seed=0):
    """k centers (each 2 private leaves) + outer independent set + hard kernel H (s/3 triangles).
    Returns n_total, edges, w(unit), info."""
    rng=random.Random(seed)
    edges=[]; idx=0
    centers=list(range(k)); idx=k
    leaves=[]
    for ci in centers:
        l1=idx; l2=idx+1; idx+=2
        edges += [(ci,l1),(ci,l2)]; leaves += [l1,l2]
    # sparse edges among centers
    for i in range(k):
        for j in range(i+1,k):
            if rng.random()<0.1: edges.append((i,j))
    # hard kernel: triangles
    ntri=max(1,s//3); H=[]
    for _ in range(ntri):
        a,b,c=idx,idx+1,idx+2; idx+=3
        edges += [(a,b),(b,c),(a,c)]; H += [a,b,c]
        # attach kernel to a random center (keeps it covered via center? no-edge to center keeps half-int)
    # outer independent set, each joined to >=1 random center
    n_outer=max(0,n-idx)
    outer=[]
    for _ in range(n_outer):
        v=idx; idx+=1; outer.append(v)
        for ci in rng.sample(centers, k=min(k, rng.randint(1,3))):
            edges.append((ci,v))
    ntot=idx
    info={"k":k,"centers":centers,"leaves":leaves,"H":H,"outer":outer,"planted_core":len(H)}
    return ntot, edges, [1.0]*ntot, info

def brute_min_cover(core_vertices, core_edges):
    """Exact min vertex cover on a small core by enumeration (core must be small)."""
    cv=list(core_vertices); m=len(cv); idxmap={v:i for i,v in enumerate(cv)}
    best=None
    for mask in range(1<<m):
        S=set(cv[i] for i in range(m) if mask>>i&1)
        if all(u in S or v in S for (u,v) in core_edges):
            if best is None or len(S)<len(best): best=S
    return best

def cert_checker(n, edges, w, max_core=20):
    """Independent Cert (Thm A): re-solve LP, verify half-integral, brute-force core,
    return (certified, opt_cost, core_size). Sound: only certifies if core small & verified."""
    lp,xs,P0,P1,Ph=vc_lp_halfint(n,edges,w)
    # verify half-integrality
    halfint=all(abs(x)<1e-6 or abs(x-0.5)<1e-6 or abs(x-1)<1e-6 for x in xs)
    if not halfint: return {"certified":False,"reason":"not half-integral","core_size":len(Ph)}
    if len(Ph)>max_core: return {"certified":False,"reason":"core too large","core_size":len(Ph)}
    core=set(Ph); core_edges=[(u,v) for (u,v) in edges if u in core and v in core]
    Score=brute_min_cover(core, core_edges) or set()
    opt=sum(w[v] for v in P1)+sum(w[v] for v in Score)
    return {"certified":True,"opt":opt,"core_size":len(Ph),"c_fix":sum(w[v] for v in P1)}

def gen_costspread_sc(R, n_filler=50, seed=0):
    """E8: one element e with two sets {cost 1} and {cost R}; filler sets keep universe coverable.
    Returns (universe,sets,costs, scorer) where scorer ranks the expensive set first."""
    rng=random.Random(seed)
    universe=set(range(1+n_filler))
    sets=[[0],[0]]; costs=[1.0,float(R)]      # S1 cheap (=OPT for e), S2 expensive
    for j in range(n_filler):
        sets.append([1+j]); costs.append(1.0+rng.random())
    # scorer: popularity-biased -> expensive set scored highest
    score=[0.0]*len(sets); score[1]=1.0; score[0]=0.2
    return {"universe":universe,"sets":sets,"costs":costs,"score":score,"R":R}

if __name__=="__main__":
    n,edges,w,info=gen_planted(200, 20, 9, seed=1)
    print("planted: n=%d centers=%d planted_core=%d"%(n,info["k"],info["planted_core"]))
    chk=cert_checker(n,edges,w)
    print("checker:", chk)
