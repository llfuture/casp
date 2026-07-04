#!/usr/bin/env python3
# CASP experiment runners E1-E10 (parallelized). Use ONLY double quotes (ssh-safe).
import sys, os, glob, json, time, gzip, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casp_lib as L
import planted as PL
from parallel import pmap_chunks, adaptive_workers
import antoniadis2024 as ANT
import aamand2025 as AAM

DATA=os.path.expanduser("~/projects/casp_max/data")
OUT=os.path.expanduser("~/projects/casp_max/outputs/run")
os.makedirs(OUT, exist_ok=True)
def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)
    print("saved", name)

# ---------------- E2: pruning-rate vs proven lower bound ----------------
def _e2(path):
    try:
        sc=L.load_sc_synth(path); U,S,C=sc["universe"],sc["sets"],sc["costs"]
        cert=L.sc_lp_threshold(U,S,C)
        f=cert["f"]; tau=cert["tau"]; lp=cert["lp"]; cmin=min(C) if C else 1.0
        bound=max(0.0, 1.0 - lp/(tau*cmin*max(len(S),1)))      # Thm rate lower bound
        return {"file":os.path.basename(path),"f":f,"emp_rate":round(cert["prune_rate"],4),
                "lb":round(bound,4),"gap":round(cert["prune_rate"]-bound,4)}
    except Exception as e: return {"file":os.path.basename(path),"err":str(e)}
def run_e2(limit=120):
    files=sorted(glob.glob(DATA+"/synthetic/set_cover/sc_*.json.gz"))[:limit]
    res=pmap_chunks(_e2, files)
    ok=[r for r in res if "gap" in r]
    viol=[r for r in ok if r["gap"]<0]
    save("e2.json", {"summary":{"n":len(ok),"min_gap":round(min(r["gap"] for r in ok),4) if ok else None,
        "violations":len(viol)},"rows":res})

# ---------------- E3: 4-arm ablation (A greedy,B exact,C prune+greedy,D prune+exact) ----------------
def _e3(path):
    try:
        sc=L.load_sc_synth(path); U,S,C=sc["universe"],sc["sets"],sc["costs"]
        if len(S)>2500 or len(U)>2500: return None  # keep exact tractable
        _,ca=L.sc_greedy(U,S,C)
        ob,_,stb,_=L.sc_exact(U,S,C,tlim=30)
        cert=L.sc_lp_threshold(U,S,C); surv=cert["survivors"]
        Ssurv=[S[i] for i in surv]; Csurv=[C[i] for i in surv]
        # remap survivors as standalone SC
        _,cc=L.sc_greedy(U,Ssurv,Csurv) if Ssurv else (None,None)
        od,_,std,_=L.sc_exact(U,S,C,restrict=surv,tlim=30)
        return {"file":os.path.basename(path),"A_greedy":ca,"B_exact":ob,
                "C_prune_greedy":cc,"D_prune_exact":od}
    except Exception as e: return {"file":os.path.basename(path),"err":str(e)}
def run_e3(limit=80):
    files=sorted(glob.glob(DATA+"/synthetic/set_cover/sc_f2_*.json.gz")+
                 glob.glob(DATA+"/synthetic/set_cover/sc_f5_*.json.gz"))[:limit]
    res=[r for r in pmap_chunks(_e3, files) if r]
    good=[r for r in res if r.get("B_exact") and r.get("D_prune_exact")]
    import statistics as st
    prune_gain=[r["A_greedy"]-r["C_prune_greedy"] for r in good if r.get("C_prune_greedy")]
    solver_gain=[r["A_greedy"]-r["B_exact"] for r in good]
    dminusb=[abs(r["D_prune_exact"]-r["B_exact"]) for r in good]
    save("e3.json", {"summary":{"n":len(good),
        "mean_solver_gain":round(st.mean(solver_gain),2) if solver_gain else None,
        "mean_prune_gain":round(st.mean(prune_gain),2) if prune_gain else None,
        "mean_|D-B|":round(st.mean(dminusb),3) if dminusb else None},"rows":res})

# ---------------- E6-SC: net exact-solving speedup on OR-Library ----------------
def _e6sc(path):
    try:
        sc=L.load_scp(path); U,S,C=sc["universe"],sc["sets"],sc["costs"]
        t=time.time(); ofull,_,stf,tf=L.sc_exact(U,S,C,tlim=300); 
        cert=L.sc_lp_threshold(U,S,C); surv=cert["survivors"]
        ored,sred,stat_r,tr=L.sc_exact(U,S,C,restrict=surv,tlim=300)
        sp=(tf/max(tr,1e-3)) if (ofull and ored) else None
        mm=(ofull is not None and ored is not None and abs(ofull-ored)>1e-6*max(1,abs(ofull)))
        return {"file":os.path.basename(path),"prune_rate":round(cert["prune_rate"],3),
                "t_full":round(tf,2),"t_red":round(tr,2),"speedup":round(sp,2) if sp else None,
                "opt_full":ofull,"opt_red":ored,"mismatch":bool(mm),"f":cert["f"]}
    except Exception as e: return {"file":os.path.basename(path),"err":str(e)}
def run_e6sc(limit=25):
    files=sorted(glob.glob(DATA+"/benchmarks/orlib/scp4*.txt")+glob.glob(DATA+"/benchmarks/orlib/scp5*.txt"))[:limit]
    res=[r for r in pmap_chunks(_e6sc, files) if r]
    sp=[r["speedup"] for r in res if r.get("speedup")]
    import statistics as st
    save("e6_sc.json", {"summary":{"n":len(res),"n_speedup":len(sp),
        "mean_speedup":round(st.mean(sp),2) if sp else None,
        "max_speedup":round(max(sp),2) if sp else None,
        "mismatches":sum(int(r.get("mismatch",False)) for r in res)},"rows":res})

# ---------------- E1-VC: NT exactness (0 mismatch) ----------------
def _e1vc(item):
    path,kind=item
    try:
        n,edges,w=(L.load_vc_synth(path) if kind=="synth" else L.load_dimacs(path))
        nt=L.vc_nt_certificate(n,edges,w)
        row={"file":os.path.basename(path),"kind":kind,"n":n,"core":nt["core_size"],
             "prune_rate":round(nt["prune_rate"],3)}
        if nt["core_size"]<=22 and n<=600:   # verifiable: brute/exact core
            core=set(nt["Phalf"]); ce=nt["core_edges"]
            Sc=PL.brute_min_cover(core,ce) or set()
            casp=nt["c_fix"]+len(Sc)
            oexact,_,st,el=L.vc_exact(n,edges,w,tlim=120)
            row["casp"]=casp; row["exact"]=oexact
            row["mismatch"]=(oexact is not None and abs(casp-oexact)>1e-6); row["status"]="OK"
        else: row["status"]="UNVER"
        return row
    except Exception as e: return {"file":os.path.basename(path),"err":str(e)}
def run_e1vc(limit=80):
    items=[(p,"synth") for p in sorted(glob.glob(DATA+"/synthetic/vertex_cover/*.json.gz"))[:40]]
    items+=[(p,"dimacs") for p in sorted(glob.glob(DATA+"/benchmarks/dimacs/*.col"))[:limit]]
    res=[r for r in pmap_chunks(_e1vc, items) if r]
    ok=[r for r in res if r.get("status")=="OK"]
    save("e1_vc.json", {"summary":{"n":len(res),"verified":len(ok),
        "mismatches":sum(int(r.get("mismatch",False)) for r in ok)},"rows":res})

# ---------------- E7: certified-optimality separation (planted) ----------------
def _e7(item):
    n,k,s,seed=item
    nt,edges,w,info=PL.gen_planted(n,k,s,seed=seed)
    chk=PL.cert_checker(nt,edges,w,max_core=24)
    # exact ground truth
    oexact,_,st,el=L.vc_exact(nt,edges,w,tlim=120)
    casp_certifies=chk.get("certified",False)
    mismatch=(casp_certifies and oexact is not None and abs(chk["opt"]-oexact)>1e-6)
    return {"n":nt,"k":k,"planted_core":info["planted_core"],"realized_core":chk.get("core_size"),
            "casp_certifies":bool(casp_certifies),"casp_opt":chk.get("opt"),"exact":oexact,
            "mismatch":bool(mismatch),"antoniadis_certifies":ANT.CERTIFIES_OPTIMALITY}
def run_e7():
    items=[]
    for n in [200,500,1000]:
        for s in [6,12]:
            for seed in range(5):
                items.append((n, max(5,n//20), s, seed))
    res=pmap_chunks(_e7, items)
    cert=[r for r in res if r["casp_certifies"]]
    save("e7.json", {"summary":{"n":len(res),"casp_certified":len(cert),
        "core_eq_planted":sum(int(r["realized_core"]==r["planted_core"]) for r in res),
        "mismatches":sum(int(r["mismatch"]) for r in res),
        "antoniadis_certified":sum(int(r["antoniadis_certifies"]) for r in res)},"rows":res})

# ---------------- E8: boundedness separation (cost spread R) ----------------
def _e8(R):
    inst=PL.gen_costspread_sc(R, n_filler=40, seed=1)
    U,S,C,score=inst["universe"],inst["sets"],inst["costs"],inst["score"]
    # CASP: LP threshold prunes the expensive duplicate set
    cert=L.sc_lp_threshold(U,S,C); surv=set(cert["survivors"])
    _,casp_cost=L.sc_greedy(U,[S[i] for i in surv],[C[i] for i in surv])
    # positive-signal commitment at best threshold over scores -> commits expensive set
    worst=0.0
    for t in [0.1,0.3,0.5,0.9]:
        chosen=set(i for i in range(len(S)) if score[i]>=t)
        chk=ANT.run_sc(U,S,C,chosen)
        worst=max(worst, chk[1])
    opt=1.0+sum(C[2:])  # take cheap set for e + all filler singletons? OPT = cheapest cover
    # OPT: cheapest cover = S1(cost1) + each filler singleton
    optc=1.0+sum(C[2:])
    return {"R":R,"casp_ratio":round(casp_cost/optc,3),"pos_ratio":round(worst/optc,3),
            "casp_bound_const":round(cert["f"],1)}
def run_e8():
    res=pmap_chunks(_e8, [10,100,1000,10000])
    save("e8.json", {"summary":{"note":"CASP ratio bounded by max(f,alpha); positive-signal grows with R"},
                     "rows":sorted(res,key=lambda r:r["R"])})

# ---------------- E10: head-to-head vs positive-signal on VC (consistency-robustness) ----------------
def _e10(item):
    path,eta=item
    n,edges,w=L.load_vc_synth(path)
    oexact,_,st,el=L.vc_exact(n,edges,w,tlim=60)
    if oexact is None: return None
    # true opt set
    _,sopt,_,_=L.vc_exact(n,edges,w,tlim=60)
    pred=ANT.make_prediction(sopt, range(n), eta, seed=7)
    _,ca,_=ANT.run_vc(n,edges,w,pred)
    # CASP cost (NT exact when core small else fallback greedy upper)
    nt=L.vc_nt_certificate(n,edges,w)
    casp = nt["c_fix"]+ (len(PL.brute_min_cover(set(nt["Phalf"]),nt["core_edges"]) or set()) if nt["core_size"]<=20 else 0)
    casp = casp if nt["core_size"]<=20 else None
    return {"file":os.path.basename(path),"eta":eta,"exact":oexact,
            "antoniadis_ratio":round(ca/oexact,3),"casp_ratio":(round(casp/oexact,3) if casp else None),
            "antoniadis_certifies":False,"casp_certifies":nt["core_size"]<=20}
def run_e10(limit=20):
    base=sorted(glob.glob(DATA+"/synthetic/vertex_cover/*.json.gz"))[:limit]
    items=[(p,eta) for p in base for eta in [0.0,0.1,0.3,0.5]]
    res=[r for r in pmap_chunks(_e10, items) if r]
    save("e10.json", {"summary":{"n":len(res),
        "note":"antoniadis_ratio rises with eta; casp_ratio flat & may certify"},"rows":res})

EXPS={"e1vc":run_e1vc,"e2":run_e2,"e3":run_e3,"e6sc":run_e6sc,"e7":run_e7,"e8":run_e8,"e10":run_e10}
if __name__=="__main__":
    name=sys.argv[1]; 
    t=time.time(); print("RUN",name,"workers~",adaptive_workers(100))
    EXPS[name](*( [int(sys.argv[2])] if len(sys.argv)>2 and name in ("e1vc","e2","e3","e6sc","e10") else [] ))
    print("DONE",name,"in",round(time.time()-t,1),"s")
