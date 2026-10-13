"""Shared pieces of the math-ablation study (offline, on the GPU box).

Reuses the rank-optimal study's exports and loaders (experiments/rank_optimal_quantization/roq_common.py):
the paper index's probed candidates per query (LF), its primary codes and list layout (PP), the rotation, the
residual exports and y = Pi x in position order (RQ). Variants of this study write to MA/<tag>/.

evaluate(): one pass over a workload's 1000 query files; every method gives (primary key, D_H) for the probed
candidates of a query; recall@10 against T, T_min at JHQ's recall at T = ck, primary top-T coverage of the
ground truth, pairwise misranking, full-IVF recall (all candidates refined), and the per-query recall curves
(npz) for paired statistics. The test split is only read when asked for.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rank_optimal_quantization"))
import roq_common as C  # noqa: E402,F401

MA = "/root/autodl-tmp/math_ablation"
G = {}


def per_query(path):
    q = C.load_bin(path)
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    isgt = np.isin(q["id"], q["gt"][:C.K])
    gti = np.where(isgt)[0]
    rng = np.random.default_rng(q["qid"])
    rnd = rng.choice(np.where(~isgt)[0], size=min(2000, int((~isgt).sum())), replace=False)
    ck = q["ck"]
    out = dict(qid=q["qid"], R={}, COV={}, MR={}, full={})
    keys = {"JHQ": (q["P"], q["DH"])}
    for name, fn in G["methods"].items():
        keys[name] = fn(q, qr, pos)
    for name, (key, dh) in keys.items():
        out["R"][name], order = C.recall_curve(key, dh, pos, isgt, G["Ts"])
        rk = C.ranks_of(order, gti) if len(gti) else np.array([len(pos)])
        out["COV"][name] = np.array([(rk < t).mean() for t in G["Ts"]], np.float32)
        out["MR"][name] = C.misrank(key, gti, rnd)
        out["full"][name] = int(isgt[np.lexsort((pos, dh))[:C.K]].sum())
    return out


def evaluate(R, label, tag, methods, test=False, procs=24):
    """methods: name -> f(q, qr, pos) -> (primary key, D_H) for the candidates (lower key = closer)."""
    t0 = time.time()
    G.clear()
    G["qrot"] = C.qrot(tag)
    G["methods"] = methods
    files = C.bin_files(tag)
    qids = [int(os.path.basename(f)[4:-4]) for f in files]
    sp = C.splits(qids)
    keep = ("calib", "val", "test") if test else ("calib", "val")
    files = [f for f, x in zip(files, qids) if sp[x] in keep]
    ck = C.load_bin(files[0])["ck"]
    G["Ts"] = Ts = C.t_grid(ck, 4 * ck)
    pool = mp.get_context("fork").Pool(procs)
    res = pool.map(per_query, files, chunksize=4)
    pool.close(); pool.join()
    split = np.array([sp[o["qid"]] for o in res])
    names = ["JHQ"] + list(methods)
    Rr = {m: np.stack([o["R"][m] for o in res]).astype(np.float64) / C.K for m in names}
    Cv = {m: np.stack([o["COV"][m] for o in res]).astype(np.float64) for m in names}
    rows = []
    summ = dict(tag=tag, ck=ck, T_grid=Ts, methods={}, seconds=None)
    for s in keep:
        sel = split == s
        target = float(Rr["JHQ"][sel, Ts.index(ck)].mean())
        for m in names:
            cur = Rr[m][sel].mean(0)
            ok = [t for t, r in zip(Ts, cur) if r >= target - 1e-12]
            row = dict(label=label, tag=tag, method=m, split=s, queries=int(sel.sum()), target_recall=target,
                       T_min=min(ok) if ok else None, recall_at_ck=float(cur[Ts.index(ck)]),
                       coverage_at_ck=float(Cv[m][sel, Ts.index(ck)].mean()),
                       full_ivf_recall=float(np.mean([o["full"][m] for o, f in zip(res, sel) if f]) / C.K),
                       misrank_random=float(np.nanmean([o["MR"][m] for o, f in zip(res, sel) if f])),
                       **{"recall_T%d" % t: float(r) for t, r in zip(Ts, cur)})
            rows.append(row)
            summ["methods"].setdefault(m, {})[s] = {k: row[k] for k in ("T_min", "recall_at_ck", "coverage_at_ck", "full_ivf_recall", "misrank_random")}
    summ["seconds"] = time.time() - t0
    np.savez_compressed(os.path.join(R, "%s_%s_perquery.npz" % (label, tag)), qid=np.array([o["qid"] for o in res]), split=split,
                        Ts=np.array(Ts), **{"R_" + m: Rr[m].astype(np.float32) for m in names},
                        **{"COV_" + m: Cv[m].astype(np.float32) for m in names})
    return rows, summ


def paired(R, label, tag, a, b, splits=("calib", "val"), B=10000):
    """paired recall difference a - b at T = ck/8, ck/4, ck/2, ck (pp, bootstrap 95% CI over queries)."""
    z = np.load(os.path.join(R, "%s_%s_perquery.npz" % (label, tag)))
    Ts = [int(t) for t in z["Ts"]]
    sel = np.isin(z["split"], splits)
    ck = {"arxiv-768_np128_a100": 1000, "vogue-768_np128_a64": 640, "openai3-3072_np128_a4": 40}.get(tag, Ts[-1] // 4)
    rng = np.random.default_rng(C.SEED)
    out = {}
    for t in sorted({max(10, ck // 8), max(10, ck // 4), max(10, ck // 2), ck}):
        if t not in Ts:
            continue
        d = (z["R_" + a][sel, Ts.index(t)] - z["R_" + b][sel, Ts.index(t)]).astype(np.float64)
        idx = rng.integers(0, len(d), (B, len(d)))
        m = d[idx].mean(1)
        out[str(t)] = (100 * float(d.mean()), 100 * float(np.quantile(m, 0.025)), 100 * float(np.quantile(m, 0.975)))
    return out


def write_rows(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def residual_sample_positions(tag):
    """positions of the paper index's residual training sample (100K base rows in id order)."""
    m = C.meta(tag)
    ids = np.fromfile(os.path.join(C.RQ, tag, "list_ids.i32"), dtype=np.int32)
    inv = np.empty(m["N"], np.int64); inv[ids] = np.arange(m["N"])
    return np.sort(inv[C.res_sample_rows(m["N"])])
