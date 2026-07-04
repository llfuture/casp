#!/usr/bin/env python3
"""Adaptive parallel map, polite on shared servers.
Worker count adapts to spare capacity = nproc - 1min-loadavg, clamped to [2, min(nproc,n_tasks)].
Override with env CASP_WORKERS."""
import os, multiprocessing as mp

def adaptive_workers(n_tasks):
    env=os.environ.get("CASP_WORKERS")
    if env: return max(1, min(int(env), n_tasks))
    nproc=os.cpu_count() or 4
    try: load1=os.getloadavg()[0]
    except Exception: load1=0.0
    spare=nproc-load1
    w=int(spare*0.8)
    return max(2, min(w, nproc, n_tasks)) if n_tasks>0 else 1

def pmap(fn, items, workers=None):
    items=list(items)
    if not items: return []
    w=workers or adaptive_workers(len(items))
    if w<=1: return [fn(x) for x in items]
    with mp.Pool(processes=w) as pool:
        return pool.map(fn, items)

def pmap_chunks(fn, items, workers=None, chunksize=1):
    items=list(items)
    if not items: return []
    w=workers or adaptive_workers(len(items))
    if w<=1: return [fn(x) for x in items]
    with mp.Pool(processes=w) as pool:
        return list(pool.imap_unordered(fn, items, chunksize=chunksize))
