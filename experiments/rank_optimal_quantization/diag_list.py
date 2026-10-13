#!/usr/bin/env python3
"""Round-5 diagnosis: where per-list codes lose the ranking. Calibration and validation queries only.

  python3 diag_list.py <results dir> [variant ...]        (default: median list_median)

Per query and code (JHQ's own D_prim and each variant's D_P), against the exact distance D = ||q' - y||^2 of every
probed candidate:
  error e = D_P - D, its variance over the query's candidates split into a between-list part (variance of the
  per-list mean of e, weighted by the number of candidates in the list) and a within-list part
  pairwise misranking of every ground-truth neighbour against the non-neighbour candidates, split by whether the
  negative sits in the neighbour's own IVF list or in another probed list
Output: diag_list.json and a printed table (means over queries).
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402
import eval_variants as EV  # noqa: E402

G = EV.G


def per_query(path):
    q = C.load_bin(path)
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    lp = G["lpos"][pos]
    D = np.empty(len(pos))
    for a in range(0, len(pos), 8192):
        e = qr[None, :] - np.asarray(G["xp"][pos[a:a + 8192]], dtype=np.float64)
        D[a:a + 8192] = np.einsum("ij,ij->i", e, e)
    isgt = np.isin(q["id"], q["gt"][:C.K])
    neg = ~isgt
    _, inv, cnt = np.unique(lp, return_inverse=True, return_counts=True)
    keys = {"JHQ": q["P"]}
    for v in G["variants"]:
        keys[v] = EV.distances(G["var"][v], qr, pos)[0]
    out = dict(qid=q["qid"], n=len(pos), lists=len(cnt), gt_lists=len(set(lp[isgt].tolist())))
    for v, dp in keys.items():
        e = dp - D
        mu = np.bincount(inv, weights=e) / cnt
        ms = ns = mc = nc = 0
        for g in np.where(isgt)[0]:
            same = neg & (lp == lp[g])
            ms += int((dp[same] <= dp[g]).sum()); ns += int(same.sum())
            mc += int((dp[neg & ~same] <= dp[g]).sum()); nc += int((neg & ~same).sum())
        out[v] = dict(var_between=float(np.sum(cnt * (mu - e.mean()) ** 2) / len(e)),
                      var_within=float(np.sum((e - mu[inv]) ** 2) / len(e)),
                      misrank_same_list=ms / max(ns, 1), misrank_other_list=mc / max(nc, 1))
    return out


def main():
    R = sys.argv[1]
    variants = sys.argv[2:] or ["median", "list_median"]
    summ = {}
    for tag in C.TAGS:
        t0 = time.time()
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        G.clear()
        G["qrot"] = C.qrot(tag)
        G["lpos"] = C.list_of_pos(tag)
        G["xp"] = C.xpos(tag)
        G["variants"] = variants
        G["var"] = {}
        for v in variants:
            pz = np.load(os.path.join(C.RQ, tag, "var_" + v, "params.npz"))
            G["var"][v] = dict(per_list=bool(pz["per_list"]), lo32=pz["lo"].astype(np.float32), hi32=pz["hi"].astype(np.float32),
                               res_cb=pz["res_cb"].astype(np.float32),
                               bits=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                               res=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d)))
        files = C.bin_files(tag)
        qids = [int(os.path.basename(f)[4:-4]) for f in files]
        sp = C.splits(qids)
        files = [f for f, x in zip(files, qids) if sp[x] in ("calib", "val")]       # the test split is not read here
        pool = mp.get_context("fork").Pool(24)
        res = pool.map(per_query, files, chunksize=4)
        pool.close(); pool.join()
        assert all(sp[o["qid"]] in ("calib", "val") for o in res)
        off = C.list_offsets(tag)
        ent = dict(queries=len(res), list_size_mean=float(np.diff(off).mean()), list_size_median=float(np.median(np.diff(off))),
                   probed_lists=float(np.mean([o["lists"] for o in res])),
                   lists_holding_the_gt=float(np.mean([o["gt_lists"] for o in res])), codes={})
        for v in ["JHQ"] + variants:
            ent["codes"][v] = {k: float(np.mean([o[v][k] for o in res])) for k in res[0][v]}
        ent["seconds"] = time.time() - t0
        summ[tag] = ent
        print("==", tag, "queries=%d list size mean %.0f median %.0f, the top-10 spread over %.2f lists" % (
            ent["queries"], ent["list_size_mean"], ent["list_size_median"], ent["lists_holding_the_gt"]), flush=True)
        for v, c in ent["codes"].items():
            print("   %-12s error var between lists %.3e  within %.3e  (between share %.2f) | misrank same list %.4f  other lists %.2e" % (
                v, c["var_between"], c["var_within"], c["var_between"] / (c["var_between"] + c["var_within"]),
                c["misrank_same_list"], c["misrank_other_list"]), flush=True)
        json.dump(summ, open(os.path.join(R, "diag_list.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
