import sys, os, json, time
sys.path.insert(0, os.path.expanduser("~/projects/casp_max/codes"))
import steiner_casp as S
from multiprocessing import Pool
base = os.path.expanduser("~/projects/casp_max/data/benchmarks/steinlib/B")
OUT = os.path.expanduser("~/projects/casp_max/outputs/theoremD")
INST = ["b06","b11","b12","b14","b15","b16","b17","b18"]
PUB = {"b06":122,"b11":88,"b12":174,"b14":235,"b15":318,"b16":127,"b17":131,"b18":218}
TLIM = 10800
def work(name):
    path = base + "/" + name + ".stp"
    n, edges, term = S.parse_stp(path)
    surv, re_, rv = S.reduce_casp(n, edges, term)
    of, sf, tf = S.steiner_exact(n, edges, term, tlim=TLIM)
    orr, sr, tr = S.steiner_exact(n, surv, term, tlim=TLIM)
    r = {"inst": name, "n": n, "m": len(edges), "term": len(term), "surv": len(surv),
         "prune": round(1-len(surv)/len(edges), 3),
         "opt_full": of, "st_full": sf, "t_full": round(tf, 1),
         "opt_red": orr, "st_red": sr, "t_red": round(tr, 1), "published": PUB.get(name),
         "both_proven": (sf == "optimal" and sr == "optimal"),
         "match": (of is not None and orr is not None and abs(of-orr) < 1e-6)}
    open(OUT + "/steiner_long.jsonl", "a").write(json.dumps(r) + "\n")
    return r
if __name__ == "__main__":
    open(OUT + "/steiner_long.jsonl", "w").close()
    with Pool(8) as p:
        res = p.map(work, INST)
    json.dump({"rows": res}, open(OUT + "/steiner_long.json", "w"), indent=1)
    print("ALLDONE")
