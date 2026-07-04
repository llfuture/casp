#!/usr/bin/env python3
"""Pisinger-style 0/1 knapsack generator for CASP E9 (heterogeneous problem).
Types: uncorrelated, weakly_corr, strongly_corr, subset_sum, spanner.
Each instance: items {v_i,w_i}, capacity W=floor(0.5*sum w). JSON.gz + manifest."""
import os, json, gzip, random
OUT=os.path.expanduser("~/projects/casp_max/data/synthetic/knapsack")
os.makedirs(OUT, exist_ok=True)
R=1000  # value/weight range
def gen(kind,n,seed):
    rng=random.Random(seed); w=[];v=[]
    if kind=="spanner":
        # spanner(2,10): build from small set of span items, scale
        span=[(rng.randint(1,R),rng.randint(1,R)) for _ in range(10)]
        for i in range(n):
            a,b=span[i%len(span)]; m=rng.randint(1,2)
            w.append(a*m); v.append(b*m)
    else:
        for i in range(n):
            wi=rng.randint(1,R)
            if kind=="uncorrelated": vi=rng.randint(1,R)
            elif kind=="weakly_corr": vi=max(1, wi+rng.randint(-R//10,R//10))
            elif kind=="strongly_corr": vi=wi+R//10
            elif kind=="subset_sum": vi=wi
            else: raise ValueError(kind)
            w.append(wi); v.append(vi)
    W=sum(w)//2
    return {"kind":kind,"n":n,"seed":seed,"weights":w,"values":v,"capacity":W}
manifest=[]
kinds=["uncorrelated","weakly_corr","strongly_corr","subset_sum","spanner"]
ns=[50,100,200,500,1000,2000]
idx=0
for kind in kinds:
    for n in ns:
        for rep in range(6):
            inst=gen(kind,n,1000*idx+rep)
            fn=f"kp_{kind}_n{n}_{rep}.json.gz"
            with gzip.open(os.path.join(OUT,fn),"wt") as f: json.dump(inst,f)
            manifest.append({"file":fn,"kind":kind,"n":n,"rep":rep,"capacity":inst["capacity"]})
        idx+=1
json.dump(manifest, open(os.path.join(OUT,"manifest.json"),"w"), indent=1)
print("generated",len(manifest),"knapsack instances ->",OUT)
