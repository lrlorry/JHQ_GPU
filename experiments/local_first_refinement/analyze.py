#!/usr/bin/env python3
"""Local-first refinement pilot, Phases A and B (offline, FP64 values from the
exporter of experiments/residual_aware_bound).

  python3 analyze.py <work dir with <tag>/rb_q*.bin> <results dir>

Groups reproduce v62 exactly: a query's probed candidates in v65's probe order
(nearest list first, each list in storage order), cut at floor(n*g/G). Ranks
use v62's order, (primary distance, position). Selections:
  global(T)        the global primary top-T (v65 is global(ck))
  uniform(G,T)     b_g = T/G (remainder to the nearest groups)
  probe(G,T)       b_g proportional to 1/(g+1): more to the groups holding the
                   nearest probed lists, a weight known at search time
  calib(G,T)       b_g learned on the calibration queries only: the allocation
                   that maximises coverage of their full-IVF top-10 (greedy on
                   the upper concave hull of each group's coverage curve)
  oracle(G,T)      the same allocation computed on the query's own full-IVF
                   top-10: an upper bound no search can reach, labelled as such
Every selection refines its candidates with the full D_H and returns the 10
smallest D_H among them (ties by position). Recall@10 is against the ground
truth. Queries are split by a seeded permutation: 40% calibration, 20%
validation, 40% test (400/200/400 of 1000); budgets and matched-recall points are chosen on
calibration/validation and reported on test.
"""
import csv
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

K = 10
GS = (2, 4, 8, 16)
RATIOS = (1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2, 3 / 4, 1.0, 1.25, 1.5)
POLICIES = ("global", "uniform", "probe", "calib", "oracle")
SEED = 20261013


def load(path):
    with open(path, "rb") as f:
        b = f.read()
    o = 0

    def take(n, dt):
        nonlocal o
        a = np.frombuffer(b, dtype=dt, count=n, offset=o)
        o += a.nbytes
        return a
    qid, n, nprobe, ck, k = (int(x) for x in take(5, np.int32))
    q = dict(qid=qid, n=n, nprobe=nprobe, ck=ck, k=k)
    q["probes"] = take(nprobe, np.int32)
    q["pos"], q["id"], q["rank"] = take(n, np.int32), take(n, np.int32), take(n, np.int32)
    q["P"], q["DH"], _lc, _lg = (take(n, np.float64) for _ in range(4))
    q["qn2"] = float(take(1, np.float64)[0])
    q["v65_ids"], q["v65_d"], q["v65_dh"] = take(k, np.int32), take(k, np.float32), take(k, np.float64)
    q["topck"], q["gt"] = take(ck, np.int32), take(k, np.int32)
    assert o == len(b), path
    return q


def t_grid(ck):
    return sorted(set(max(K, int(np.floor(ck * r))) for r in RATIOS))


def split_budget(T, w):
    """integer budgets summing to T, proportional to w (largest remainders)."""
    w = np.asarray(w, dtype=np.float64)
    x = T * w / w.sum()
    b = np.floor(x).astype(np.int64)
    rem = T - int(b.sum())
    if rem > 0:
        order = np.lexsort((np.arange(len(w)), -(x - b)))
        b[order[:rem]] += 1
    return b


def hull_allocation(ranks_by_group, T, G):
    """maximise sum_g C_g(b_g) s.t. sum b_g = T, C_g(b) = #ranks <= b in group g:
    greedy over the upper concave hull segments of every C_g, by slope."""
    segs = []
    for g in range(G):
        r = np.sort(np.asarray(ranks_by_group[g], dtype=np.int64))
        if len(r) == 0:
            continue
        xs, cs = np.unique(r, return_counts=True)
        pts = [(0, 0.0)] + list(zip(xs.tolist(), np.cumsum(cs).astype(float).tolist()))
        hull = [pts[0]]
        for p in pts[1:]:
            while len(hull) >= 2:
                (x1, y1), (x2, y2) = hull[-2], hull[-1]
                if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                    hull.pop()
                else:
                    break
            hull.append(p)
        for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
            segs.append(((y2 - y1) / (x2 - x1), x2 - x1, g))
    segs.sort(key=lambda s: (-s[0], s[2]))
    b = np.zeros(G, dtype=np.int64)
    left = T
    for slope, length, g in segs:
        if left <= 0:
            break
        take = min(length, left)
        b[g] += take
        left -= take
    if left > 0:      # coverage saturated: spread the rest evenly
        b += split_budget(left, np.ones(G))
    return b


def final_top(mask, dorder):
    """the 10 smallest D_H among the refined (dorder: all candidates by (D_H, pos))."""
    for L in (256, 4096, len(dorder)):
        sel = dorder[:L][mask[dorder[:L]]]
        if len(sel) >= K or L == len(dorder):
            return sel[:K]


def per_query(args):
    path, calib_tables = args
    q = load(path)
    n, ck = q["n"], q["ck"]
    P, DH, pos = q["P"], q["DH"], q["pos"]
    ids = q["id"]
    gt = set(int(x) for x in q["gt"][:K])
    gorder = np.lexsort((pos, P))
    grank = np.empty(n, dtype=np.int64); grank[gorder] = np.arange(1, n + 1)
    dorder = np.lexsort((pos, DH))
    F = dorder[:K]
    id2i = {int(v): i for i, v in enumerate(ids)}
    V = np.array([id2i.get(int(v), -1) for v in q["v65_ids"]])
    V = V[V >= 0]
    rec = lambda sel: len(gt & set(int(ids[i]) for i in sel)) / K
    cov = lambda sel: len(set(sel.tolist()) & set(F.tolist())) / K
    ovl = lambda sel: len(set(sel.tolist()) & set(V.tolist())) / K
    out = dict(qid=q["qid"], n=n, ck=ck, recall_full=rec(F), recall_v65=rec(V) if len(V) else np.nan,
               A=[], B={}, members={})
    Ts = t_grid(ck)
    idx = np.arange(n)
    # global selection (independent of G)
    glob_res = {}
    for T in Ts:
        mask = grank <= T
        sel = final_top(mask, dorder)
        glob_res[T] = (rec(sel), cov(sel), ovl(sel), int(mask.sum()), int(mask.sum()))
    for G in GS:
        cuts = (n * np.arange(G + 1)) // G
        grp = np.searchsorted(cuts, idx, side="right") - 1
        lorder = np.lexsort((pos, P, grp))
        lrank = np.empty(n, dtype=np.int64)
        lrank[lorder] = np.arange(n) - cuts[grp[lorder]] + 1
        sizes = np.diff(cuts)
        # Phase A
        fg, fl, fr = grp[F], lrank[F], grank[F]
        mem_by_group = [fl[fg == g].tolist() for g in range(G)]
        out["members"][G] = mem_by_group
        cnt = np.bincount(fg, minlength=G)
        maxl = np.array([max(m) if m else 0 for m in mem_by_group])
        out["A"].append(dict(G=G, members_per_group="|".join(map(str, cnt.tolist())), group0_share=cnt[0] / K,
                             max_global_rank=int(fr.max()), oracle_local_T=int(maxl.sum()),
                             max_local_rank_group0=int(maxl[0]), max_local_rank_any=int(maxl.max()),
                             v65_max_local_rank=int(lrank[V].max()) if len(V) else -1,
                             v65_max_global_rank=int(grank[V].max()) if len(V) else -1,
                             member_rows=[(int(i), int(g), int(l), int(r)) for i, g, l, r in zip(range(K), fg, fl, fr)],
                             v65_rows=[(int(i), int(grp[v]), int(lrank[v]), int(grank[v])) for i, v in enumerate(V)]))
        # Phase B
        for T in Ts:
            for pol in POLICIES:
                if pol == "global":
                    out["B"][(G, pol, T)] = glob_res[T]
                    continue
                if pol == "uniform":
                    b = split_budget(T, np.ones(G))
                elif pol == "probe":
                    b = split_budget(T, 1.0 / (np.arange(G) + 1.0))
                elif pol == "calib":
                    if calib_tables is None:
                        continue
                    b = np.asarray(calib_tables[str(G)][str(T)], dtype=np.int64)
                else:
                    b = hull_allocation(mem_by_group, T, G)
                b = np.minimum(b, sizes)
                mask = lrank <= b[grp]
                sel = final_top(mask, dorder)
                out["B"][(G, pol, T)] = (rec(sel), cov(sel), ovl(sel), int(mask.sum()), int(b.max()))
    return out


def main():
    W, R = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    tags = sorted(t for t in os.listdir(W) if os.path.isdir(os.path.join(W, t)))
    summary = {}
    A_rows, M_rows, B_rows, GV_rows = [], [], [], []
    for tag in tags:
        files = sorted(glob.glob(os.path.join(W, tag, "rb_q*.bin")))
        qids = sorted(int(os.path.basename(f)[4:-4]) for f in files)
        perm = np.random.default_rng(SEED).permutation(qids)
        c1, c2 = int(0.4 * len(perm)), int(0.6 * len(perm))       # 400 / 200 / 400 of 1000
        split = {int(x): ("calib" if i < c1 else "val" if i < c2 else "test") for i, x in enumerate(perm)}
        with Pool(min(64, os.cpu_count() or 8)) as pool:
            first = pool.map(per_query, [(f, None) for f in files], chunksize=4)
        ck = first[0]["ck"]
        Ts = t_grid(ck)
        # calibrated allocations: calibration queries only
        tables = {}
        for G in GS:
            ranks = [[] for _ in range(G)]
            ncal = 0
            for o in first:
                if split[o["qid"]] != "calib":
                    continue
                ncal += 1
                for g in range(G):
                    ranks[g] += o["members"][G][g]
            tables[str(G)] = {str(T): hull_allocation(ranks, T, G).tolist() for T in Ts}
        with Pool(min(64, os.cpu_count() or 8)) as pool:
            second = pool.map(per_query, [(f, tables) for f in files], chunksize=4)
        res = {o["qid"]: o for o in second}
        # Phase A rows
        for o in second:
            for a in o["A"]:
                A_rows.append(dict(tag=tag, qid=o["qid"], split=split[o["qid"]], n_cand=o["n"], ck=o["ck"],
                                   G=a["G"], members_per_group=a["members_per_group"], group0_share=a["group0_share"],
                                   max_global_rank=a["max_global_rank"], oracle_local_T=a["oracle_local_T"],
                                   max_local_rank_group0=a["max_local_rank_group0"], max_local_rank_any=a["max_local_rank_any"],
                                   v65_max_local_rank=a["v65_max_local_rank"], v65_max_global_rank=a["v65_max_global_rank"],
                                   recall_full=o["recall_full"], recall_v65=o["recall_v65"]))
                for which, rows in (("full_ivf_top10", a["member_rows"]), ("v65_top10", a["v65_rows"])):
                    for i, g, l, r in rows:
                        M_rows.append((tag, o["qid"], a["G"], which, i, g, l, r))
        # Phase B aggregates per split
        def agg(qset, G, pol, T):
            v = [res[qd]["B"].get((G, pol, T)) for qd in qset]
            v = [x for x in v if x is not None]
            if not v:
                return None
            a = np.array(v, dtype=np.float64)
            return dict(recall=a[:, 0].mean(), coverage=a[:, 1].mean(), overlap_v65=a[:, 2].mean(),
                        refined=a[:, 3].mean(), max_group_load=a[:, 4].mean(), per_query_recall=a[:, 0], nq=len(v))
        sets = {s: [qd for qd in res if split[qd] == s] for s in ("calib", "val", "test")}
        ref = {s: agg(sets[s], 8, "global", ck) for s in sets}          # v65 = global(ck)
        for s, qs in sets.items():
            for G in GS:
                for pol in POLICIES:
                    for T in Ts:
                        a = agg(qs, G, pol, T)
                        if a is None:
                            continue
                        B_rows.append(dict(tag=tag, split=s, G=G, policy=pol + (" (offline upper bound)" if pol == "oracle" else ""),
                                           T=T, T_over_ck=T / ck, recall=a["recall"], full_ivf_coverage=a["coverage"],
                                           overlap_with_v65=a["overlap_v65"], refined_mean=a["refined"],
                                           max_group_budget_mean=a["max_group_load"],
                                           recall_v65_same_split=ref[s]["recall"]))
        # paired comparison, test split: local policy vs global at the same T
        rng = np.random.default_rng(SEED)
        tq = sets["test"]
        for G in GS:
            for pol in ("uniform", "probe", "calib", "oracle"):
                for T in Ts:
                    gl, lo = agg(tq, G, "global", T), agg(tq, G, pol, T)
                    if gl is None or lo is None:
                        continue
                    dlt = lo["per_query_recall"] - gl["per_query_recall"]
                    boots = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
                    GV_rows.append(dict(tag=tag, kind="same_T", G=G, policy=pol, T=T, T_over_ck=T / ck,
                                        recall_global=gl["recall"], recall_local=lo["recall"], delta_mean=dlt.mean(),
                                        ci95_lo=float(np.quantile(boots, 0.025)), ci95_hi=float(np.quantile(boots, 0.975)),
                                        frac_queries_local_worse=float((dlt < 0).mean()), frac_queries_local_better=float((dlt > 0).mean()),
                                        worst_query_delta=float(dlt.min()), refined_local=lo["refined"], refined_global=gl["refined"]))
        # matched recall: smallest grid T whose validation recall reaches v65's (validation), reported on test
        target_val = ref["val"]["recall"]
        matched = {}
        for G in GS:
            for pol in POLICIES:
                best = None
                for T in Ts:
                    a = agg(sets["val"], G, pol, T)
                    if a is not None and a["recall"] >= target_val - 1e-9:
                        best = T
                        break
                if best is None:
                    continue
                at = agg(tq, G, pol, best)
                vt = ref["test"]["per_query_recall"]
                dlt = at["per_query_recall"] - vt
                boots = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
                matched["%s/G%d" % (pol, G)] = dict(T=best, T_over_ck=best / ck, test_recall=at["recall"],
                                                    v65_test_recall=ref["test"]["recall"], refined_test=at["refined"])
                GV_rows.append(dict(tag=tag, kind="matched_recall_vs_v65", G=G, policy=pol, T=best, T_over_ck=best / ck,
                                    recall_global=ref["test"]["recall"], recall_local=at["recall"], delta_mean=dlt.mean(),
                                    ci95_lo=float(np.quantile(boots, 0.025)), ci95_hi=float(np.quantile(boots, 0.975)),
                                    frac_queries_local_worse=float((dlt < 0).mean()), frac_queries_local_better=float((dlt > 0).mean()),
                                    worst_query_delta=float(dlt.min()), refined_local=at["refined"], refined_global=ck))
        summary[tag] = dict(ck=ck, T_grid=Ts, n_queries=len(res), split_sizes={s: len(v) for s, v in sets.items()},
                            recall_v65_sim={s: ref[s]["recall"] for s in sets},
                            recall_v65_actual_test=float(np.nanmean([res[qd]["recall_v65"] for qd in tq])),
                            recall_full_ivf_test=float(np.mean([res[qd]["recall_full"] for qd in tq])),
                            matched=matched, calib_tables=tables)
        print(tag, "done:", len(res), "queries; matched:", {k2: (v["T"], round(v["test_recall"], 4)) for k2, v in matched.items()})
    for fn, rows in (("GROUP_RANK_ANALYSIS.csv", A_rows), ("RECALL_VS_BUDGET.csv", B_rows), ("GLOBAL_VS_LOCAL.csv", GV_rows)):
        with open(os.path.join(R, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with gzip.open(os.path.join(R, "GROUP_RANK_MEMBERS.csv.gz"), "wt") as f:
        f.write("tag,qid,G,which,index,group,local_rank,global_rank\n")
        for r in M_rows:
            f.write(",".join(map(str, r)) + "\n")
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
    print("done:", ", ".join(tags))


if __name__ == "__main__":
    main()
