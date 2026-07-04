import glob, os, statistics as st, collections, sys, json
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import casp_lib as L
from multiprocessing import Pool
DATA = os.path.expanduser("~/projects/casp_max/data/synthetic/set_cover")
OUT = os.path.expanduser("~/projects/casp_max/outputs/run")

def four_arm(path, tlim=15):
    try:
        sc = L.load_sc_synth(path); U, S, C = sc["universe"], sc["sets"], sc["costs"]
        if len(S) > 1200 or len(U) > 600:
            return None
        _, A = L.sc_greedy(U, S, C)
        B, _, stb, _ = L.sc_exact(U, S, C, tlim=tlim)
        cert = L.sc_lp_threshold(U, S, C); surv = cert["survivors"]
        if not surv:
            return None
        _, Cc = L.sc_greedy(U, [S[i] for i in surv], [C[i] for i in surv])
        D, _, std, _ = L.sc_exact(U, S, C, restrict=surv, tlim=tlim)
        if B is None or D is None or stb != "optimal" or std != "optimal":
            return None
        return dict(file=os.path.basename(path), f=cert["f"], A=A, B=B, C=Cc, D=D,
                    prune_rate=cert["prune_rate"])
    except Exception:
        return None

files = []
for f in [2, 5, 10, 20, 50]:
    files += sorted(glob.glob(DATA + "/sc_f%d_m500_s1000_*.json.gz" % f))[:10]

if __name__ == "__main__":
    with Pool(16) as p:
        res = [r for r in p.map(four_arm, files) if r]
    g = collections.defaultdict(list)
    for r in res:
        g[r["f"]].append(r)
    table = []
    print("f | n | prune_gain(A-C) | solver_gain(A-B) | |D-B| | prune_rate | C==opt")
    for f, v in sorted(g.items()):
        pg = [r["A"] - r["C"] for r in v]; sg = [r["A"] - r["B"] for r in v]
        db = [abs(r["D"] - r["B"]) for r in v]; pr = [r["prune_rate"] for r in v]
        ceqb = sum(1 for r in v if abs(r["C"] - r["B"]) < 1e-6)
        row = dict(f=f, n=len(v), mean_prune_gain=round(st.mean(pg), 1),
                   mean_solver_gain=round(st.mean(sg), 1), mean_abs_D_minus_B=round(st.mean(db), 2),
                   mean_prune_rate=round(st.mean(pr), 2), C_reaches_opt=str(ceqb)+chr(47)+str(len(v)))
        table.append(row)
    import json as J
    J.dump({chr(34)+chr(98)+chr(121)+chr(95)+chr(102)+chr(34): table}, open(OUT+chr(47)+chr(101)+chr(51)+chr(46)+chr(106)+chr(115)+chr(111)+chr(110), chr(119)), indent=1)
    for r in table: print(r)
    print(chr(68)+chr(79)+chr(78)+chr(69))
