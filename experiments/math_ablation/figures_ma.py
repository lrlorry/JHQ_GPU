#!/usr/bin/env python3
"""Figures of the math-ablation study from its CSV outputs (local, no data access).

  python3 figures_ma.py <results dir> <out dir>

fig1_recall_T.png          recall@10 against the refinement budget T (validation split): V1 (E8) and V3 (RD) against
                           the paper index and its offline re-implementation, per workload
fig2_evt_frontier.png      V2: recall@10 against the fraction of primary candidates scanned, EVT policies against
                           the nprobe cut and the per-query oracle (validation)
fig3_recall_qps.png        measured recall-QPS: B0 (v65, alpha x nprobe) and the V1 GPU prototype (A: paper index
                           through the full table, B: E8), b = 8 and b = 1000
fig4_stages.png            B0 per-stage GPU time at the paper's configuration (nsys)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAGS = (("arxiv-768_np128_a100", "arxiv-768"), ("vogue-768_np128_a64", "vogue-768"), ("openai3-3072_np128_a4", "openai3-3072"))
COL = {"JHQ": "#000000", "jhq_re": "#666666", "E8": "#d95f02", "E8|JHQ-DH": "#fdae61", "E8g": "#a6611a",
       "RD-uniform": "#1b9e77", "RD-empirical": "#7570b3", "RD-waterfill": "#66a61e", "RD-queryweighted": "#e7298a"}


def rows_of(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    R, O = sys.argv[1], sys.argv[2]
    os.makedirs(O, exist_ok=True)
    e8 = [r for r in rows_of(os.path.join(R, "v1_e8", "eval_e8.csv")) if r["split"] == "val"]
    rd = [r for r in rows_of(os.path.join(R, "v3_rd", "eval_rd.csv")) if r["split"] == "val"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, (tag, ds) in enumerate(TAGS):
        for i, (rows, title) in enumerate(((e8, "V1 E8"), (rd, "V3 RD"))):
            a = axes[i, j]
            for r in rows:
                if r["tag"] != tag:
                    continue
                Ts = sorted(int(k[8:]) for k in r if k.startswith("recall_T") and r[k])
                a.plot(Ts, [float(r["recall_T%d" % t]) for t in Ts], "-o" if r["method"] == "JHQ" else "-", ms=3,
                       color=COL.get(r["method"], None), lw=2 if r["method"] == "JHQ" else 1.2, label=r["method"])
            tg = [float(r["target_recall"]) for r in rows if r["tag"] == tag]
            if tg:
                a.axhline(tg[0], color="red", ls="--", lw=1)
                a.set_ylim(tg[0] - 0.03, tg[0] + 0.008)
            a.set_xscale("log"); a.grid(alpha=0.3); a.set_title("%s — %s (val)" % (title, ds), fontsize=10)
            a.set_xlabel("refinement budget T"); a.set_ylabel("recall@10"); a.legend(fontsize=7)
    fig.suptitle("Recall@10 vs T near the matched-recall target (red: the paper index at T = ck)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(O, "fig1_recall_T.png"), dpi=110); plt.close(fig)
    # V2
    ev = [r for r in rows_of(os.path.join(R, "v2_evt", "EVT_POLICIES.csv")) if r["split"] == "val"]
    if ev:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        for a, (tag, ds) in zip(axes, TAGS):
            rs = [r for r in ev if r["tag"] == tag]
            npc = sorted((float(r["frac_candidates"]), float(r["recall"])) for r in rs if r["policy"] == "nprobe")
            a.plot([x for x, _ in npc], [y for _, y in npc], "-o", color="black", ms=3, label="nprobe cut (B0)")
            for pol, mk in (("global", "s"), ("iid_evt", "^"), ("gev", "v"), ("logistic", "D")):
                pts = [(float(r["frac_candidates"]), float(r["recall"])) for r in rs if r["policy"] == pol]
                a.scatter([x for x, _ in pts], [y for _, y in pts], s=12, marker=mk, label="EVT " + pol, alpha=0.7)
            o = [r for r in rs if r["policy"] == "oracle"]
            if o:
                a.scatter([float(o[0]["frac_candidates"])], [float(o[0]["recall"])], s=80, marker="*", color="red", label="oracle")
            full = max(y for _, y in npc)
            a.set_ylim(full - 0.05, full + 0.005); a.grid(alpha=0.3)
            a.set_xlabel("fraction of primary candidates scanned"); a.set_ylabel("recall@10 (val)"); a.set_title(ds, fontsize=10)
            a.legend(fontsize=7)
        fig.suptitle("V2 EVT list pruning against the nprobe cut (offline simulation on the paper index's candidates)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(O, "fig2_evt_frontier.png"), dpi=110); plt.close(fig)
    # recall-QPS
    fr = rows_of(os.path.join(R, "RECALL_QPS_FRONTIER.csv"))
    if fr:
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        for i, b in enumerate(("8", "1000")):
            for j, (tag, ds) in enumerate(TAGS):
                a = axes[i, j]
                for var, mk in (("B0 v65", "o"), ("V1 GPU A: paper index, full 256 table", "s"), ("V1 GPU B: E8, full 256 table", "^")):
                    for npb in ("128", "512"):
                        pts = sorted((float(r["recall_at_10"]), float(r["qps_median"])) for r in fr
                                     if r["dataset"] == ds and str(r["batch"]) == b and str(r["nprobe"]) == npb and r["variant"] == var)
                        if pts:
                            a.plot([x for x, _ in pts], [y for _, y in pts], "-" + mk, ms=4, label="%s np%s" % (var, npb))
                a.set_yscale("log"); a.grid(alpha=0.3); a.set_title("%s, b = %s" % (ds, b), fontsize=10)
                a.set_xlabel("recall@10 (1000 queries)"); a.set_ylabel("QPS (median of rounds)"); a.legend(fontsize=6)
        fig.suptitle("Measured recall-QPS (RTX 5090): B0 = v65 alpha x nprobe sweep; V1 GPU prototype arms", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(O, "fig3_recall_qps.png"), dpi=110); plt.close(fig)
    st = rows_of(os.path.join(R, "STAGE_BREAKDOWN.csv"))
    if st:
        keys = [k for k in st[0] if k.startswith("us_")]
        lab = ["%s b%s" % (r.get("tag", ""), r.get("batch", "")) for r in st]
        fig, a = plt.subplots(figsize=(14, 5))
        bottom = np.zeros(len(st))
        for k in keys:
            v = np.array([float(r[k] or 0) for r in st])
            a.bar(range(len(st)), v / 1000.0, bottom=bottom / 1000.0, label=k[3:])
            bottom += v
        a.set_xticks(range(len(st))); a.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
        a.set_ylabel("GPU time per search (ms)"); a.legend(fontsize=7, ncol=5); a.grid(alpha=0.3, axis="y")
        a.set_title("B0 (v65) per-stage GPU time at the paper's configuration (nsys)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(O, "fig4_stages.png"), dpi=110); plt.close(fig)
    mem = rows_of(os.path.join(R, "INDEX_MEMORY.csv"))
    if mem:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
        names = [(r["variant"], r["dataset"]) for r in mem]
        x = np.arange(len(mem))
        tot = np.array([float(r["primary_bytes_per_vector"]) + float(r["residual_bytes_per_vector"]) + float(r["corr_bytes_per_vector"]) for r in mem])
        axes[0].bar(x, tot, color=["#333333" if v.startswith("B0") else "#d95f02" if v.startswith("V1") else "#7570b3" if v.startswith("V2") else "#1b9e77" for v, _ in names])
        axes[0].set_ylabel("index bytes per vector (primary + residual + corr)")
        lut = np.array([float(r.get("lut_bytes_per_query_primary") or 0) / 1024 for r in mem])
        axes[1].bar(x, lut, color=["#333333" if v.startswith("B0") else "#d95f02" if v.startswith("V1") else "#7570b3" if v.startswith("V2") else "#1b9e77" for v, _ in names])
        axes[1].set_ylabel("primary lookup table per query (KiB)")
        for a in axes:
            a.set_xticks(x); a.set_xticklabels(["%s\n%s" % (v.replace("V3 RD-JHQ ", "V3 "), d) for v, d in names], rotation=90, fontsize=6)
            a.grid(alpha=0.3, axis="y")
        fig.suptitle("Index size per vector (equal by construction) and per-query primary table (E8 loses the 16+16 factorisation)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(O, "fig5_index_memory.png"), dpi=110); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
