#!/usr/bin/env python3
"""Residual-aware bound pilot: everything after demo_jhq_v66_bound_pilot.

  python3 analyze.py <work dir with <tag>/rb_q*.bin> <results dir>

Per workload tag and query, from the FP64 per-candidate values the demo wrote
(D_prim, D_H, LB_cond, LB_glob for every candidate of the probed lists):

  Pilot A  LB <= D_H re-checked here; LB / D_H and D_H - LB distributions.
  Pilot B  survivors {x : LB(x) <= tau} for
             tau_full  the k-th smallest D_H over all probed candidates (oracle)
             tau_v65   the largest D_H among v65's returned top-k
             tau_seed  the k-th smallest D_H among s seeds: the s candidates of
                       the nearest L probed lists with the smallest D_prim
           and progressive refinement: seeds first, then every other candidate
           in a given order, refined iff LB <= the current k-th refined D_H.
           Orders: probe (list by list, nearest first), position (index
           storage order), lb (ascending LB; an offline ideal that ignores the
           cost of a global sort).
  Pilot D  Recall@10 of v65, of the full-IVF D_H ranking and of the certified
           (progressive) result; overlap with v65.

Writes BOUND_VALIDATION.json, SURVIVOR_ANALYSIS.csv (long format),
RECALL_COMPARISON.csv, survivor_summary.json and figures/*.png.
"""
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

K = 10
SEED_LISTS = (1, 2, 4, 8)
SEED_COUNTS = (32, 64, 128)
ORDERS = ("probe", "position", "lb")


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
    q["P"], q["DH"], q["LC"], q["LG"] = (take(n, np.float64) for _ in range(4))
    q["qn2"] = float(take(1, np.float64)[0])
    q["v65_ids"], q["v65_d"], q["v65_dh"] = take(k, np.int32), take(k, np.float32), take(k, np.float64)
    q["topck"], q["gt"] = take(ck, np.int32), take(k, np.int32)
    assert o == len(b), path
    return q


def progressive(LB, DH, order, seeds, k):
    """Refine seeds, then walk `order` (seeds skipped): refine x iff LB[x] <= tau,
    tau = k-th smallest refined D_H (inf while fewer than k). Returns
    (refined count, final tau, indices of the final k)."""
    best = sorted((float(DH[i]), int(i)) for i in seeds)[:k]
    refined = len(seeds)
    tau = best[k - 1][0] if len(best) >= k else np.inf
    is_seed = np.zeros(len(DH), bool)
    is_seed[seeds] = True
    seq = order[~is_seed[order]]
    lb, dh = LB[seq], DH[seq]
    p, n = 0, len(seq)
    while p < n:
        ok = lb[p:] <= tau
        ev = ok & (dh[p:] < tau)          # refined and lowers tau
        if not ev.any():
            refined += int(ok.sum())
            break
        j = int(np.argmax(ev))
        refined += int(ok[: j + 1].sum())
        best.append((float(dh[p + j]), int(seq[p + j])))
        best.sort()
        best = best[:k]
        tau = best[k - 1][0] if len(best) >= k else np.inf
        p += j + 1
    return refined, tau, [i for _, i in best]


def qstats(args):
    tag, path = args
    q = load(path)
    n, ck, k = q["n"], q["ck"], q["k"]
    DH, P, LC, LG, rank = q["DH"], q["P"], q["LC"], q["LG"], q["rank"]
    rows, rec = [], {}
    base = dict(tag=tag, qid=q["qid"], n_cand=n, ck=ck)
    srt = np.argsort(DH, kind="stable")
    tau_full = float(DH[srt[k - 1]])
    vdh = q["v65_dh"][np.isfinite(q["v65_dh"])]
    tau_v65 = float(vdh.max()) if len(vdh) == k else np.nan
    for bname, LB in (("cond", LC), ("glob", LG)):
        rows.append(dict(base, method="full_oracle", bound=bname, threshold=tau_full, count=int((LB <= tau_full).sum())))
        if np.isfinite(tau_v65):
            rows.append(dict(base, method="v65", bound=bname, threshold=tau_v65, count=int((LB <= tau_v65).sum())))
    orders = {"probe": np.lexsort((q["pos"], rank)), "position": np.argsort(q["pos"], kind="stable"),
              "lb_cond": np.argsort(LC, kind="stable"), "lb_glob": np.argsort(LG, kind="stable")}
    cert = None
    for L in SEED_LISTS:
        pool = np.nonzero(rank < L)[0]
        for s in SEED_COUNTS:
            seeds = pool[np.argsort(P[pool], kind="stable")[:s]]
            if len(seeds) >= k:
                tau_seed = float(np.sort(DH[seeds])[k - 1])
            else:
                tau_seed = np.inf
            for bname, LB in (("cond", LC), ("glob", LG)):
                rows.append(dict(base, method="seed_L%d_s%d" % (L, s), bound=bname, threshold=tau_seed,
                                 count=int((LB <= tau_seed).sum()) if np.isfinite(tau_seed) else n))
                for o in ORDERS:
                    ordr = orders["lb_" + bname] if o == "lb" else orders[o]
                    cnt, tau_end, best = progressive(LB, DH, ordr, seeds, k)
                    rows.append(dict(base, method="prog_%s_L%d_s%d" % (o, L, s), bound=bname, threshold=tau_end, count=cnt))
                    if bname == "cond" and o == "probe" and L == 8 and s == 64:
                        cert = best
    for bname, LB in (("cond", LC), ("glob", LG)):
        for o in ORDERS:
            ordr = orders["lb_" + bname] if o == "lb" else orders[o]
            cnt, tau_end, _ = progressive(LB, DH, ordr, np.zeros(0, dtype=np.int64), k)
            rows.append(dict(base, method="prog_%s_noseed" % o, bound=bname, threshold=tau_end, count=cnt))
    # Pilot D
    gt = set(int(x) for x in q["gt"][:k])
    ids = q["id"]
    full = [int(ids[i]) for i in srt[:k]]
    v65 = [int(x) for x in q["v65_ids"]]
    certified = [int(ids[i]) for i in cert]
    cert_tau = float(np.max(DH[cert]))
    topck_set = set(int(x) for x in q["topck"] if x >= 0)
    full_pos = [int(q["pos"][i]) for i in srt[:k]]
    rec = dict(tag=tag, qid=q["qid"], n_cand=n, ck=ck,
               recall_v65=len(gt & set(v65)) / k, recall_full_ivf=len(gt & set(full)) / k,
               recall_certified=len(gt & set(certified)) / k,
               overlap_v65_full=len(set(v65) & set(full)) / k,
               certified_equals_full=int(set(certified) == set(full) or abs(cert_tau - tau_full) <= 1e-12 * max(1.0, tau_full)),
               full_top_k_in_v65_topck=sum(p in topck_set for p in full_pos) / k,
               tau_full=tau_full, tau_v65=tau_v65)
    # Pilot A
    ok = DH > 1e-12
    a = dict(n=n, viol_cond=int((LC - DH > 1e-12 * np.maximum(1.0, DH)).sum()),
             viol_glob=int((LG - DH > 1e-12 * np.maximum(1.0, DH)).sum()),
             max_viol_cond=float((LC - DH).max()), max_viol_glob=float((LG - DH).max()),
             ratio_cond=(LC[ok] / DH[ok]).astype(np.float32), ratio_glob=(LG[ok] / DH[ok]).astype(np.float32),
             ratio_prim=(P[ok] / DH[ok]).astype(np.float32),
             gap_cond=(DH - LC).astype(np.float32), gap_glob=(DH - LG).astype(np.float32),
             tau_full=tau_full)
    # a few points for scatter plots, normalised by tau_full
    rng = np.random.default_rng(q["qid"])
    pick = rng.choice(n, size=min(n, 4000), replace=False)
    pts = np.stack([DH[pick] / tau_full, LC[pick] / tau_full, LG[pick] / tau_full, P[pick] / tau_full]).astype(np.float32)
    return tag, rows, rec, a, pts


def qtl(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    return {"p10": float(np.quantile(x, 0.1)), "p50": float(np.quantile(x, 0.5)), "p90": float(np.quantile(x, 0.9)),
            "p99": float(np.quantile(x, 0.99)), "max": float(x.max()), "min": float(x.min())}


def main():
    W, R = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    tags = sorted(t for t in os.listdir(W) if not t.startswith("v16_") and os.path.isdir(os.path.join(W, t)))
    jobs = [(t, p) for t in tags for p in sorted(glob.glob(os.path.join(W, t, "rb_q*.bin")))]
    with Pool(min(64, os.cpu_count() or 8)) as pool:
        out = pool.map(qstats, jobs, chunksize=1)
    rows, recs, A, PTS = [], [], {}, {}
    for tag, r, rec, a, pts in out:
        rows += r
        recs.append(rec)
        A.setdefault(tag, []).append(a)
        PTS.setdefault(tag, []).append(pts)
    with open(os.path.join(R, "SURVIVOR_ANALYSIS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "qid", "n_cand", "ck", "method", "bound", "threshold", "count"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(R, "RECALL_COMPARISON.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    # validation
    V = {}
    for tag in tags:
        S = json.load(open(os.path.join(W, tag, "rb_summary.json")))
        a = A[tag]
        V[tag] = dict(audit=S["audit"], envelope=S["envelope"], validation_cpp=S["validation"], levels=S["levels"],
                      shape={k: S[k] for k in ("N", "d", "M", "Ds", "K", "Kr", "Br", "L", "nlist", "nprobe", "alpha", "ck", "seed", "n_queries")},
                      validation_py=dict(candidates=int(sum(x["n"] for x in a)), violations_cond=int(sum(x["viol_cond"] for x in a)),
                                         violations_glob=int(sum(x["viol_glob"] for x in a)),
                                         max_violation_cond=float(max(x["max_viol_cond"] for x in a)),
                                         max_violation_glob=float(max(x["max_viol_glob"] for x in a))),
                      LB_over_DH_cond=qtl(np.concatenate([x["ratio_cond"] for x in a])),
                      LB_over_DH_glob=qtl(np.concatenate([x["ratio_glob"] for x in a])),
                      Dprim_over_DH=qtl(np.concatenate([x["ratio_prim"] for x in a])),
                      Dprim_above_DH_fraction=float(np.mean(np.concatenate([x["ratio_prim"] for x in a]) > 1.0)),
                      DH_minus_LB_cond=qtl(np.concatenate([x["gap_cond"] for x in a])),
                      DH_minus_LB_glob=qtl(np.concatenate([x["gap_glob"] for x in a])),
                      LB_over_DH_cond_per_query_median=qtl([np.median(x["ratio_cond"]) for x in a]),
                      per_subspace_LBcond_over_DH=[x / max(y, 1e-30) for x, y in zip(S["per_subspace"]["sum_LB_cond"], S["per_subspace"]["sum_DH"])],
                      per_subspace_zero_LBcond=S["per_subspace"]["zero_LB_cond_fraction"])
    json.dump(V, open(os.path.join(R, "BOUND_VALIDATION.json"), "w"), indent=1)
    # survivor summary
    SS = {}
    for tag in tags:
        T = [r for r in rows if r["tag"] == tag]
        meth = sorted(set((r["method"], r["bound"]) for r in T))
        SS[tag] = {}
        for m, bnd in meth:
            c = np.array([r["count"] for r in T if r["method"] == m and r["bound"] == bnd], dtype=np.float64)
            n = np.array([r["n_cand"] for r in T if r["method"] == m and r["bound"] == bnd], dtype=np.float64)
            ck = T[0]["ck"]
            SS[tag]["%s/%s" % (m, bnd)] = dict(median=float(np.median(c)), p90=float(np.quantile(c, 0.9)),
                                               p95=float(np.quantile(c, 0.95)), p99=float(np.quantile(c, 0.99)),
                                               max=float(c.max()), mean=float(c.mean()),
                                               over_ck_median=float(np.median(c / ck)),
                                               over_ncand_median=float(np.median(c / n)),
                                               frac_queries_below_ck=float(np.mean(c < ck)))
        nc = np.array([r["n_cand"] for r in T if r["method"] == "full_oracle" and r["bound"] == "cond"], dtype=np.float64)
        SS[tag]["_n_cand"] = dict(median=float(np.median(nc)), p90=float(np.quantile(nc, 0.9)), max=float(nc.max()))
        RR = [x for x in recs if x["tag"] == tag]
        SS[tag]["_recall"] = {k2: float(np.mean([x[k2] for x in RR])) for k2 in
                              ("recall_v65", "recall_full_ivf", "recall_certified", "overlap_v65_full",
                               "certified_equals_full", "full_top_k_in_v65_topck")}
    json.dump(SS, open(os.path.join(R, "survivor_summary.json"), "w"), indent=1)
    try:
        figures(R, tags, rows, recs, V, PTS)
    except Exception as e:  # figures are a convenience; the tables above are the record
        print("figures failed:", e)
    print("tags:", ", ".join(tags), "| queries:", len(recs))


def figures(R, tags, rows, recs, V, PTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = os.path.join(R, "figures")
    main = [t for t in tags if t.endswith("np128_a100") or t.endswith("np128_a64") or t.endswith("np128_a4")]
    name = lambda t: t.replace("_np", " np").replace("_a", " α=")
    # fig1: primary distance is not a bound; the residual-aware bounds are
    fig, ax = plt.subplots(len(main), 3, figsize=(13, 3.6 * len(main)), squeeze=False)
    for i, t in enumerate(main):
        p = np.concatenate(PTS[t][:6], axis=1)
        for j, (row, lab) in enumerate(((3, "D_prim (primary only)"), (2, "LB_glob (global codebook envelope)"), (1, "LB_cond (half-code envelope)"))):
            a = ax[i][j]
            a.scatter(p[0], p[row], s=1, alpha=0.25, color=["#888888", "#d95f02", "#1b9e77"][j], rasterized=True)
            lim = float(np.quantile(p[0], 0.995))
            a.plot([0, lim], [0, lim], "k--", lw=1, label="y = D_H")
            a.axhline(1.0, color="r", lw=0.8, label="τ_full (pruning line)")
            a.set_xlim(0, lim); a.set_ylim(0, lim)
            a.set_xlabel("D_H / τ_full"); a.set_ylabel(lab + " / τ_full")
            a.set_title("%s: %s" % (name(t), lab.split(" (")[0]), fontsize=9)
            if i == 0 and j == 0:
                a.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_bound_vs_DH.png"), dpi=110); plt.close(fig)
    # fig2: per-query survivors against ck and the candidate pool
    sel = [t for t in tags]
    fig, ax = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.8), squeeze=False)
    for i, t in enumerate(sel):
        a = ax[0][i]
        T = [r for r in rows if r["tag"] == t and r["bound"] == "cond"]
        qids = sorted(set(r["qid"] for r in T))
        get = lambda m: np.array([next(r["count"] for r in T if r["qid"] == q and r["method"] == m) for q in qids], dtype=float)
        nc = np.array([next(r["n_cand"] for r in T if r["qid"] == q) for q in qids], dtype=float)
        o = np.argsort(nc)
        for m, lab, col in (("full_oracle", "oracle τ_full", "#1b9e77"), ("v65", "τ_v65", "#7570b3"),
                            ("seed_L8_s64", "seed τ (8 lists, 64)", "#d95f02"),
                            ("prog_probe_L8_s64", "progressive, probe order", "#e7298a")):
            a.plot(get(m)[o], ".", ms=3, label=lab, color=col)
        a.plot(nc[o], "-", color="k", lw=1, label="all probed candidates")
        a.axhline(T[0]["ck"], color="r", ls="--", lw=1, label="ck = αk (v65)")
        a.set_yscale("log"); a.set_xlabel("query (sorted by candidates)"); a.set_title(name(t), fontsize=9)
        if i == 0:
            a.set_ylabel("candidates needing residual refinement"); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_survivors_per_query.png"), dpi=110); plt.close(fig)
    # fig3: ck against survivors, medians with P90
    meths = [("full_oracle", "oracle"), ("v65", "τ_v65"), ("seed_L8_s64", "seed"), ("prog_probe_L8_s64", "prog probe"),
             ("prog_lb_L8_s64", "prog LB-order")]
    fig, ax = plt.subplots(figsize=(1.9 * len(sel) + 3, 4.2))
    width = 0.13
    for j, (m, lab) in enumerate(meths):
        med, p90 = [], []
        for t in sel:
            c = np.array([r["count"] for r in rows if r["tag"] == t and r["bound"] == "cond" and r["method"] == m], dtype=float)
            med.append(np.median(c)); p90.append(np.quantile(c, 0.9))
        x = np.arange(len(sel)) + (j + 1) * width
        ax.bar(x, med, width, label=lab + " (median, bar to P90)")
        ax.errorbar(x, med, yerr=[np.zeros(len(med)), np.array(p90) - np.array(med)], fmt="none", color="k", lw=0.8)
    cks = [next(r["ck"] for r in rows if r["tag"] == t) for t in sel]
    ax.bar(np.arange(len(sel)), cks, width, color="r", label="ck = αk (v65 refines this many)")
    ax.set_xticks(np.arange(len(sel)) + 3 * width); ax.set_xticklabels([name(t) for t in sel], fontsize=7, rotation=15)
    ax.set_yscale("log"); ax.set_ylabel("candidates refined (log)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_ck_vs_survivors.png"), dpi=110); plt.close(fig)
    # fig4: recall
    fig, ax = plt.subplots(figsize=(1.6 * len(sel) + 3, 3.8))
    for j, (k2, lab) in enumerate((("recall_v65", "v65"), ("recall_full_ivf", "full-IVF D_H ranking"),
                                   ("recall_certified", "bound-certified"))):
        v = [np.mean([x[k2] for x in recs if x["tag"] == t]) for t in sel]
        ax.bar(np.arange(len(sel)) + j * 0.27, v, 0.27, label=lab)
        for xx, yy in zip(np.arange(len(sel)) + j * 0.27, v):
            ax.text(xx, yy + 0.002, "%.3f" % yy, ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(sel)) + 0.27); ax.set_xticklabels([name(t) for t in sel], fontsize=7, rotation=15)
    lo = min(np.mean([x["recall_v65"] for x in recs if x["tag"] == t]) for t in sel)
    ax.set_ylim(max(0.0, lo - 0.05), 1.0); ax.set_ylabel("Recall@10 (sampled queries)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig4_recall.png"), dpi=110); plt.close(fig)
    # fig5: per-subspace tightness
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for t in main:
        ax[0].plot(V[t]["per_subspace_LBcond_over_DH"], lw=1, label=name(t))
        ax[1].plot(V[t]["per_subspace_zero_LBcond"], lw=1, label=name(t))
    ax[0].set_xlabel("subspace m"); ax[0].set_ylabel("Σ LB_cond,m / Σ D_H,m"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("subspace m"); ax[1].set_ylabel("fraction of candidates with LB_cond,m = 0")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_per_subspace.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
