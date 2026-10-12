#!/usr/bin/env python3
"""Figures for the local-first pilot, from analyze.py's tables (no new data).

  python3 figures.py <results dir>
"""
import csv
import gzip
import json
import os
import sys

import numpy as np

TAGS = ("arxiv-768_np128_a100", "vogue-768_np128_a64", "openai3-3072_np128_a4")
NAME = {t: t.replace("_np", " np").replace("_a", " α=") for t in TAGS}


def main():
    R = sys.argv[1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = os.path.join(R, "figures")
    os.makedirs(F, exist_ok=True)
    A = list(csv.DictReader(open(os.path.join(R, "GROUP_RANK_ANALYSIS.csv"))))
    B = list(csv.DictReader(open(os.path.join(R, "RECALL_VS_BUDGET.csv"))))
    S = json.load(open(os.path.join(R, "summary.json")))
    M = list(csv.DictReader(gzip.open(os.path.join(R, "GROUP_RANK_MEMBERS.csv.gz"), "rt")))

    # fig1: where the full-IVF top-10 live (G = 8), and their local vs global primary ranks
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, t in enumerate(TAGS):
        rows = [r for r in A if r["tag"] == t and r["G"] == "8"]
        cnt = np.array([[int(x) for x in r["members_per_group"].split("|")] for r in rows], dtype=float)
        a = ax[0][i]
        a.bar(np.arange(8), cnt.mean(axis=0) / 10, color="#1b9e77")
        a.set_xlabel("group (0 = nearest probed lists)"); a.set_ylabel("share of the full-IVF top-10 (mean)")
        a.set_title("%s, G = 8" % NAME[t], fontsize=10); a.set_ylim(0, 1)
        mem = [m for m in M if m["tag"] == t and m["G"] == "8" and m["which"] == "full_ivf_top10"]
        lr = np.array([int(m["local_rank"]) for m in mem]); gr = np.array([int(m["global_rank"]) for m in mem])
        g0 = np.array([m["group"] == "0" for m in mem])
        a = ax[1][i]
        for x, lab, st in ((gr, "global primary rank (all members)", "-"), (lr[g0], "local rank, members in group 0", "--"),
                           (lr[~g0], "local rank, members in groups 1-7", ":")):
            xs = np.sort(x)
            a.plot(xs, np.arange(1, len(xs) + 1) / len(xs), st, lw=1.4, label=lab)
        ck = S[t]["ck"]
        a.axvline(ck, color="r", lw=1, label="ck = %d" % ck)
        a.set_xscale("log"); a.set_xlabel("primary rank of a full-IVF top-10 member"); a.set_ylabel("CDF")
        if i == 0:
            a.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_group_rank.png"), dpi=110); plt.close(fig)

    # fig2: recall against the refinement budget T, test split
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    sty = {"global": ("k", "-", "global primary top-T"), "uniform": ("#d95f02", "--", "local uniform"),
           "probe": ("#7570b3", "--", "local probe-weighted"), "calib": ("#1b9e77", "-.", "local calibrated"),
           "oracle (offline upper bound)": ("#999999", ":", "local oracle (offline, not online)")}
    for a, t in zip(axes, TAGS):
        for G, mk in ((8, "o"), (4, "s")):
            for pol, (c, ls, lab) in sty.items():
                if pol == "global" and G == 4:
                    continue
                rr = sorted((float(r["T_over_ck"]), float(r["recall"])) for r in B
                            if r["tag"] == t and r["split"] == "test" and r["G"] == str(G) and r["policy"] == pol)
                if rr:
                    a.plot([x for x, _ in rr], [y for _, y in rr], ls, color=c, marker=mk, ms=3, lw=1.2 if G == 8 else 0.7,
                           alpha=1.0 if G == 8 else 0.5, label=lab + ("" if pol == "global" else " (G=%d)" % G))
        a.axhline(S[t]["recall_v65_sim"]["test"], color="r", lw=1, label="v65 (global top-ck)")
        a.axhline(S[t]["recall_full_ivf_test"], color="b", lw=0.8, ls=":", label="full-IVF (refine everything)")
        a.set_xscale("log", base=2); a.set_xlabel("refinement budget T / ck"); a.set_ylabel("Recall@10 (test queries)")
        lo = S[t]["recall_v65_sim"]["test"]
        a.set_ylim(lo - 0.08, S[t]["recall_full_ivf_test"] + 0.01); a.set_title(NAME[t], fontsize=10)
        if t == TAGS[0]:
            a.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_recall_vs_budget.png"), dpi=110); plt.close(fig)

    # fig3: budgets per group at T = ck (G = 8) against where the targets are
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for a, t in zip(axes, TAGS):
        ck = S[t]["ck"]
        rows = [r for r in A if r["tag"] == t and r["G"] == "8"]
        share = np.array([[int(x) for x in r["members_per_group"].split("|")] for r in rows], dtype=float).mean(axis=0) / 10
        uni = np.full(8, ck / 8)
        pw = 1.0 / (np.arange(8) + 1); pw = ck * pw / pw.sum()
        cal = np.array(S[t]["calib_tables"]["8"][str(ck)], dtype=float)
        w = 0.2
        for j, (v, lab, c) in enumerate(((uni, "uniform", "#d95f02"), (pw, "probe-weighted", "#7570b3"),
                                         (cal, "calibrated", "#1b9e77"), (share * ck, "full-IVF top-10 share × ck", "#333333"))):
            a.bar(np.arange(8) + (j - 1.5) * w, v / ck, w, color=c, label=lab)
        a.set_xlabel("group (0 = nearest probed lists)"); a.set_ylabel("budget b_g / T  (T = ck = %d)" % ck)
        a.set_title(NAME[t] + ", G = 8", fontsize=10)
        if t == TAGS[0]:
            a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_group_budgets.png"), dpi=110); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
