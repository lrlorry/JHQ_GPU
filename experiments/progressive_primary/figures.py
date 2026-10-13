#!/usr/bin/env python3
"""Figures for the progressive primary screening pilot.

  python3 figures.py <results dir>   (after analyze.py and cost_model.py)

fig1_recall_vs_bytes.png       test recall difference against full(T*), per policy's Pareto front, at
                               ideal bytes (top) and at realistic bytes (sector level + metadata + scratch)
fig2_topL_coverage.png         coverage of the full-primary top-ck and loss of v65's final top-10 against L
fig3_gpu_cost_breakdown.png    cost model: v65 at T* against the validation-chosen operating points of A0 and
                               of the best other non-inferior policy, scenarios O and C, b = 8 and b = 1000
fig4_word_importance.png       query energy captured by the words read, and correlation of the stage-1
                               estimate with the full score
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAGS = ("arxiv-768_np128_a100", "vogue-768_np128_a64", "openai3-3072_np128_a4")
POLS = ("A0", "A0c", "A1", "A1c", "A2", "A3")
COL = {"A0": "#444444", "A0c": "#999999", "A1": "#1b9e77", "A1c": "#66c2a5", "A2": "#d95f02", "A3": "#7570b3"}
MK = {"A0": "s", "A0c": "D", "A1": "o", "A1c": "v", "A2": "^", "A3": "P"}


def title(tag):
    return tag.replace("_np", " np").replace("_a", " α=")


def front(pts):
    """upper-left Pareto front: points not beaten by one with fewer bytes and higher delta."""
    pts = sorted(pts)
    out, best = [], -np.inf
    for x, y in pts:
        if y > best + 1e-12:
            out.append((x, y)); best = y
    return out


def main():
    R = sys.argv[1]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    rb = list(csv.DictReader(open(os.path.join(R, "RECALL_VS_BYTES.csv"))))
    cov = list(csv.DictReader(open(os.path.join(R, "COVERAGE_VS_READ_RATIO.csv"))))
    imp = list(csv.DictReader(open(os.path.join(R, "WORD_IMPORTANCE_ANALYSIS.csv"))))
    S = json.load(open(os.path.join(R, "analysis_summary.json")))["workloads"]
    CM = json.load(open(os.path.join(R, "cost_model.json")))

    # fig 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, tag in enumerate(TAGS):
        Ts = S[tag]["T_star"]
        for i, (col, lab) in enumerate((("ideal_bytes_ratio", "ideal primary bytes / B_old"),
                                        ("total_bytes_ratio", "realistic bytes / B_old (32-B sectors + metadata + scratch)"))):
            a = axes[i, j]
            for pol in POLS:
                pts = [(float(r[col]), float(r["delta"])) for r in rb
                       if r["tag"] == tag and r["policy"] == pol and int(r["T_final"]) == Ts and float(r["p"]) < 1.0]
                fr = front(pts)
                a.plot([x for x, _ in fr], [y for _, y in fr], "-", marker=MK[pol], color=COL[pol], ms=5, lw=1.3, label=pol)
                op = S[tag]["operating_points"]["ideal" if i == 0 else "realistic"].get("%s/T%d" % (pol, Ts))
                if op:
                    a.plot([op[col]], [op["delta"]], marker=MK[pol], color=COL[pol], ms=13, mfc="none", mew=2)
            a.axhline(-0.001, color="red", ls="--", lw=1)
            a.axhline(0.0, color="k", lw=0.6)
            if i == 0:
                a.axvline(0.70, color="red", ls=":", lw=1)
            a.set_ylim(-0.012, 0.003); a.set_xlim(0.1, 1.15)
            a.set_xlabel(lab, fontsize=8); a.set_ylabel("test Δ recall@10 vs full(T*=%d)" % Ts, fontsize=8)
            a.set_title(title(tag) + (" — ideal" if i == 0 else " — realistic"), fontsize=10)
            if i == 0 and j == 0:
                a.legend(fontsize=8, ncol=2, title="Pareto front; ring = validation operating point", title_fontsize=7)
    fig.suptitle("Recall vs primary bytes (test split, final top-T*; red dashed = −0.1 pp margin, red dotted = 30% byte gate)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig1_recall_vs_bytes.png"), dpi=110); plt.close(fig)

    # fig 2
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for j, tag in enumerate(TAGS):
        for i, (p, col, lab, logy) in enumerate((("0.25", "coverage_full_primary_topck", "coverage of full-primary top-ck", False),
                                                  ("0.5", "coverage_full_primary_topck", "coverage of full-primary top-ck", False),
                                                  ("0.5", "miss_rate_v65_final10", "share of v65's final top-10 lost in stage 1", True))):
            a = axes[i, j]
            for pol in POLS:
                pts = sorted((int(r["L_over_ck"]), float(r[col])) for r in cov if r["tag"] == tag and r["policy"] == pol and r["p"] == p)
                ys = [max(y, 1e-5) for _, y in pts] if logy else [y for _, y in pts]      # 1e-5 drawn for "none lost"
                a.plot([x for x, _ in pts], ys, "-", marker=MK[pol], color=COL[pol], ms=5, label=pol)
            a.set_xscale("log", base=2)
            if logy:
                a.set_yscale("log")
                a.text(0.98, 0.04, "bottom line = none lost", transform=a.transAxes, ha="right", fontsize=7)
            a.set_xlabel("L / ck", fontsize=8); a.set_ylabel(lab, fontsize=8)
            a.set_title("%s — p = %s" % (title(tag), p), fontsize=10)
            a.grid(alpha=0.3)
            if i == 0 and j == 0:
                a.legend(fontsize=8, ncol=2)
    fig.suptitle("Stage-1 global top-L: what it keeps (test split; λ chosen on calibration)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig2_topL_coverage.png"), dpi=110); plt.close(fig)

    # fig 3
    parts = (("front", "front-end", "#cccccc"), ("scan", "primary scan / stage 1", "#1f78b4"), ("stage1", None, "#1f78b4"),
             ("merge", "group merge (b=8)", "#a6cee3"), ("stage2", "stage 2 (gather + top-T*)", "#e31a1c"),
             ("extra", "μ metadata + top-L scratch", "#fb9a99"), ("refine", "refine", "#33a02c"), ("topk", "final top-k", "#b2df8a"),
             ("host", "host", "#ffff99"))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, tag in enumerate(TAGS):
        for i, b in enumerate(("b8", "b1000")):
            a = axes[i, j]
            D = CM["workloads"][tag][b]
            ops = D["ops"]["realistic"]
            bars = [("v65 @T*\n[M→D]", D["v65_at_Tstar"], D["v65_at_Tstar_total"])]
            # the best other policy that is non-inferior on test, at its validation operating point under
            # either rule; ties go to the simpler policy (A1 before the ones with a mean correction)
            pref = ("A1", "A2", "A0c", "A1c", "A3")
            nov = [(round(v["price"]["C"]["gain"], 9), -pref.index(p), rule, p) for rule in ("realistic", "ideal")
                   for p, v in D["ops"][rule].items() if p != "A0" and v["non_inferior"]]
            picks = ([("realistic", "A0")] if "A0" in ops else []) + ([max(nov)[2:]] if nov else [])
            for rule, pol in picks:
                v = D["ops"][rule][pol]
                for sc in ("O", "C"):
                    bars.append(("%s %s%s\n[%s]" % (pol, v["config"].replace(" ", "\n"), "" if v["non_inferior"] else "\n(not NI)", sc),
                                 v["price"][sc]["stages"], v["price"][sc]["total"]))
            x = np.arange(len(bars))
            bottom = np.zeros(len(bars))
            seen = set()
            for key, lab, c in parts:
                v = np.array([st.get(key, 0.0) for _, st, _ in bars])
                if not v.any():
                    continue
                lab2 = lab if lab and lab not in seen else None
                seen.add(lab)
                a.bar(x, v, 0.65, bottom=bottom, color=c, edgecolor="white", lw=0.4, label=lab2)
                bottom += v
            base = bars[0][2]
            for xi, (_, _, tot) in zip(x, bars):
                a.text(xi, tot * 1.01, "%.0f µs\n%+.0f%%" % (tot, 100 * (base / tot - 1)), ha="center", fontsize=7)
            a.set_xticks(x); a.set_xticklabels([n for n, _, _ in bars], fontsize=7)
            a.set_ylabel("µs per batch of %s" % b[1:], fontsize=8)
            a.set_ylim(0, max(t for _, _, t in bars) * 1.18)
            a.set_title("%s — %s" % (title(tag), b.replace("b", "b=")), fontsize=10)
            if i == 0 and j == 0:
                handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8)
    fig.suptitle("GPU cost model [D/T]: v65 at T* against the validation-chosen operating points (realistic-bytes rule) of A0 and of the best\n"
                 "other non-inferior policy, scenarios O (optimistic) and C (central); % = throughput change vs v65 at T*; no new kernel was run", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(R, "figures", "fig3_gpu_cost_breakdown.png"), dpi=110); plt.close(fig)

    # fig 4
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, tag in enumerate(TAGS):
        for i, (col, lab) in enumerate((("query_energy_share", "share of Σ(c q'_j)² in the words read"),
                                        ("pearson_r_estimate_vs_full", "Pearson r(stage-1 estimate, full score)"))):
            a = axes[i, j]
            for pol, lam in (("A0", "0.0"), ("A1", "0.0"), ("A2", "1.0"), ("A1c", "0.25"), ("A3", "1.0")):
                if i == 0 and pol in ("A1c", "A3"):
                    continue
                pts = sorted((float(r["p"]), float(r[col])) for r in imp
                             if r["tag"] == tag and r["policy"] == pol and r["lam"] == lam and float(r["p"]) < 1)
                a.plot([x for x, _ in pts], [y for _, y in pts], "-", marker=MK[pol], color=COL[pol], label="%s (λ=%s)" % (pol, lam))
            if i == 0:
                a.plot([0.1, 0.8], [0.1, 0.8], "k:", lw=0.8, label="uniform")
            a.set_xlabel("p (fraction of words read in stage 1)", fontsize=8); a.set_ylabel(lab, fontsize=8)
            a.set_title(title(tag), fontsize=10); a.grid(alpha=0.3)
            if j == 0:
                a.legend(fontsize=7)
    fig.suptitle("Word importance (test split)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig4_word_importance.png"), dpi=110); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
