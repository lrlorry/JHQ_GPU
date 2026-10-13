#!/usr/bin/env python3
"""Evaluate quantizer variants (built by quantize.py) against the paper index on the same candidates.

  python3 eval_variants.py <results dir> <iteration label> <variant> [<variant> ...] [--test]

For every query and variant V: the primary distance D_P^V = ||q' - y_hat^V||^2 orders the probed candidates;
the primary top-T is refined by V's own D_H^V = ||q' - y_hat^V - r_hat^V||^2 (its residual was trained and
encoded with its primary, quantize.py); recall@10 against the ground truth. JHQ is the paper index itself
(FP64 D_prim and D_H of the candidate files). Also reported: V's primary order refined by JHQ's D_H (the
primary effect alone), pairwise misranking of the ground-truth neighbours against random and near-boundary
negatives (ranks 10..99 by JHQ's D_H), the primary ranks of the ground-truth neighbours, and V's full-IVF
recall (all candidates refined).

Selection discipline: T_min(V) = the smallest grid T whose mean recall reaches JHQ's recall at T = ck on the
same split; decisions use calibration and validation. --test adds the test split, the paired comparison at
each method's VALIDATION-chosen budget, and bootstrap CIs; it is run once, for the final confirmation.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402

G = {}


def per_query(path):
    q = C.load_bin(path)
    n, ck = q["n"], q["ck"]
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    isgt = np.isin(q["id"], q["gt"][:C.K])
    gti = np.where(isgt)[0]
    Ts = G["Ts"]
    rH = np.empty(n, np.int64); rH[np.lexsort((pos, q["DH"]))] = np.arange(n)
    hard = np.where((rH >= C.K) & (rH < 100) & ~isgt)[0]
    rng = np.random.default_rng(q["qid"])
    rnd = rng.choice(np.where(~isgt)[0], size=min(2000, int((~isgt).sum())), replace=False)
    out = dict(qid=q["qid"], n=n, ck=ck, R={}, X={}, MR={}, RK={}, full={})
    keys = {"JHQ": (q["P"], q["DH"])}
    for v in G["variants"]:
        P = G["var"][v]
        bits = np.unpackbits(np.asarray(P["bits"][pos]), axis=1).astype(bool)
        if P["per_list"]:
            l = G["lpos"][pos]
            yh = np.where(bits, P["hi"][l], P["lo"][l]).astype(np.float64)
        else:
            yh = np.where(bits, P["hi"][None, :], P["lo"][None, :]).astype(np.float64)
        rh = C.decode_residual(np.asarray(P["res"][pos]), P["res_cb"]).astype(np.float64)
        a = qr[None, :] - yh
        keys[v] = ((a * a).sum(1), ((a - rh) ** 2).sum(1))
    for v, (dp, dh) in keys.items():
        out["R"][v], order = C.recall_curve(dp, dh, pos, isgt, Ts)
        if v != "JHQ":
            out["X"][v], _ = C.recall_curve(dp, q["DH"], pos, isgt, Ts)
        out["MR"][v] = (C.misrank(dp, gti, rnd), C.misrank(dp, gti, hard))
        rk = C.ranks_of(order, gti) if len(gti) else np.array([n])
        out["RK"][v] = (float(np.median(rk)), float(np.max(rk)), float(np.mean(rk < ck)))
        out["full"][v] = int(isgt[np.lexsort((pos, dh))[:C.K]].sum())
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test = "--test" in sys.argv
    R, label, variants = args[0], args[1], args[2:]
    os.makedirs(R, exist_ok=True)
    rows, summ, pq_rows = [], {}, []
    rng = np.random.default_rng(C.SEED)
    for tag in C.TAGS:
        t0 = time.time()
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        G.clear()
        G["qrot"] = C.qrot(tag)
        G["lpos"] = C.list_of_pos(tag)
        G["variants"] = [v for v in variants if os.path.isdir(os.path.join(C.RQ, tag, "var_" + v))]
        G["var"] = {}
        info = {}
        for v in G["variants"]:
            pz = np.load(os.path.join(C.RQ, tag, "var_" + v, "params.npz"))
            G["var"][v] = dict(per_list=bool(pz["per_list"]), lo=pz["lo"].astype(np.float64), hi=pz["hi"].astype(np.float64), res_cb=pz["res_cb"],
                               bits=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                               res=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d)))
            info[v] = json.load(open(os.path.join(C.RQ, tag, "var_" + v, "info.json")))
        files = C.bin_files(tag)
        ck = C.load_bin(files[0])["ck"]
        G["Ts"] = Ts = C.t_grid(ck, 4 * ck)
        with mp.get_context("fork").Pool(64) as pool:
            res = pool.map(per_query, files, chunksize=4)
        sp = C.splits([o["qid"] for o in res])
        meths = ["JHQ"] + G["variants"]
        splits = ("calib", "val", "test") if test else ("calib", "val")
        Sx = {s: np.array([sp[o["qid"]] == s for o in res]) for s in splits}
        Rr = {v: np.stack([o["R"][v] for o in res]).astype(np.float64) / C.K for v in meths}
        Xr = {v: np.stack([o["X"][v] for o in res]).astype(np.float64) / C.K for v in G["variants"]}
        target = {s: float(Rr["JHQ"][Sx[s], Ts.index(ck)].mean()) for s in splits}

        def tmin(arr, s):
            cur = arr[Sx[s]].mean(0)
            ok = [t for t, r in zip(Ts, cur) if r >= target[s] - 1e-12]
            return min(ok) if ok else None
        summ[tag] = dict(ck=ck, T_grid=Ts, target=target, methods={})
        for v in meths:
            ent = {}
            for s in splits:
                cur = Rr[v][Sx[s]].mean(0)
                tm, tj = tmin(Rr[v], s), tmin(Rr["JHQ"], s)
                row = dict(iteration=label, tag=tag, method=v, split=s, T_min_matched=tm, T_min_JHQ=tj,
                           reduction_vs_JHQ=None if tm is None or tj is None else 1 - tm / tj, target_recall=target[s],
                           T_min_primary_only=(tmin(Xr[v], s) if v in Xr else tj),
                           full_ivf_recall=float(np.mean([o["full"][v] for o, f in zip(res, Sx[s]) if f]) / C.K),
                           misrank_random=float(np.nanmean([o["MR"][v][0] for o, f in zip(res, Sx[s]) if f])),
                           misrank_hard=float(np.nanmean([o["MR"][v][1] for o, f in zip(res, Sx[s]) if f])),
                           gt_rank_median=float(np.mean([o["RK"][v][0] for o, f in zip(res, Sx[s]) if f])),
                           gt_rank_max_mean=float(np.mean([o["RK"][v][1] for o, f in zip(res, Sx[s]) if f])),
                           gt_within_ck=float(np.mean([o["RK"][v][2] for o, f in zip(res, Sx[s]) if f])),
                           primary_mse=(info[v]["primary_mse_per_vector"] if v in info else None),
                           hierarchical_mse=(info[v]["hierarchical_mse_per_vector"] if v in info else None),
                           **{"recall_T%d" % t: float(r) for t, r in zip(Ts, cur)})
                rows.append(row)
                ent[s] = dict(T_min=tm, T_min_JHQ=tj, reduction=row["reduction_vs_JHQ"], recall_at_ck=float(cur[Ts.index(ck)]),
                              full_ivf=row["full_ivf_recall"], misrank_hard=row["misrank_hard"])
            if test and v != "JHQ":
                tv, tj = tmin(Rr[v], "val"), tmin(Rr["JHQ"], "val")
                if tv is not None and tj is not None:
                    dl = Rr[v][Sx["test"], Ts.index(tv)] - Rr["JHQ"][Sx["test"], Ts.index(tj)]
                    same = Rr[v][Sx["test"]] - Rr["JHQ"][Sx["test"]]
                    ent["test_matched"] = dict(T_variant=tv, T_JHQ=tj, reduction=1 - tv / tj, delta=float(dl.mean()),
                                               ci95=list(C.boot_ci(dl, rng)), frac_worse=float((dl < 0).mean()),
                                               worst=float(dl.min()), p1=float(np.sort(dl)[int(0.01 * (len(dl) - 1))]))
                    ent["test_same_T"] = {str(t): dict(delta=float(same[:, i].mean()), ci95=list(C.boot_ci(same[:, i], rng)))
                                          for i, t in enumerate(Ts) if t in (max(C.K, ck // 4), max(C.K, ck // 2), ck)}
                    for qi in np.where(Sx["test"])[0]:
                        pq_rows.append(dict(iteration=label, tag=tag, method=v, qid=res[qi]["qid"], T_variant=tv, T_JHQ=tj,
                                            recall_variant=float(Rr[v][qi, Ts.index(tv)]), recall_JHQ=float(Rr["JHQ"][qi, Ts.index(tj)])))
            summ[tag]["methods"][v] = ent
        summ[tag]["seconds"] = time.time() - t0
        print(tag, json.dumps({v: {s: (e["T_min"], e["T_min_JHQ"], e["recall_at_ck"]) for s, e in ent.items() if isinstance(e, dict) and "T_min" in e}
                               for v, ent in summ[tag]["methods"].items()}), "%.0fs" % summ[tag]["seconds"], flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    suffix = "_test" if test else ""
    with open(os.path.join(R, "eval_%s%s.csv" % (label, suffix)), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(summ, open(os.path.join(R, "eval_%s%s.json" % (label, suffix)), "w"), indent=1)
    if pq_rows:
        with open(os.path.join(R, "eval_%s_perquery_test.csv" % label), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pq_rows[0].keys())); w.writeheader(); w.writerows(pq_rows)


if __name__ == "__main__":
    main()
