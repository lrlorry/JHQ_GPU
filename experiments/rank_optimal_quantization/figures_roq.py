#!/usr/bin/env python3
"""Figures of the rank-optimal quantization study, from the CSV/JSON outputs (no data access).

  python3 figures_roq.py <results dir> <split> [eval label ...]

fig_recall_T_<split>.png   recall@10 against the refinement budget T (log), JHQ and every evaluated method, with
                           the matched-recall target (JHQ at T = ck) as a dashed line
fig_llr_<split>.png        round 1: T_min against the weight parameter of each likelihood family
fig_bias.png               how far the data is from the paper's Lemma 2 (prep_<tag>.json), and the bit balance of
                           the round-2 codes (var info is copied into eval json when present)
fig_rank_error.png         RANK_ERROR_ANALYSIS.csv: empirical misranking against the model bounds
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
COL = ["#000000", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#1f78b4"]


def title(t):
    return t.replace("_np", " np").replace("_a", " α=")


def rows_of(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def main():
    R, split = sys.argv[1], sys.argv[2]
    labels = sys.argv[3:]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    # recall-T
    ev = []
    for lb in labels:
        ev += [r for r in rows_of(os.path.join(R, "eval_%s.csv" % lb)) if r["split"] == split]
        ev += [r for r in rows_of(os.path.join(R, "eval_%s_test.csv" % lb)) if r["split"] == split]
    it1 = [r for r in rows_of(os.path.join(R, "iter1_llr.csv")) if r["split"] == split]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for a, tag in zip(axes, TAGS):
        seen = {}
        for r in [x for x in it1 if x["tag"] == tag and x["chosen_on_calib"] == "True"] + [x for x in ev if x["tag"] == tag]:
            name = r["method"] + ("(%s)" % r["param"] if r.get("param") not in (None, "", "0.0") and r["method"] != "JHQ" else "")
            if name in seen:
                continue
            Ts = sorted(int(k[8:]) for k in r if k.startswith("recall_T") and r[k] not in ("", None))
            ys = [float(r["recall_T%d" % t]) for t in Ts]
            seen[name] = (Ts, ys)
        for i, (name, (Ts, ys)) in enumerate(seen.items()):
            a.plot(Ts, ys, "-o" if name == "JHQ" else "-", ms=3, color=COL[i % len(COL)], lw=2 if name == "JHQ" else 1.2, label=name)
        tgt = [float(r["target_recall"]) for r in ev + it1 if r["tag"] == tag]
        if tgt:
            a.axhline(tgt[0], color="red", ls="--", lw=1)
        a.set_xscale("log"); a.set_xlabel("refinement budget T (primary top-T)"); a.set_ylabel("recall@10 (%s split)" % split)
        a.set_title(title(tag), fontsize=10); a.grid(alpha=0.3)
        allys = [y for _, (T, ys) in seen.items() for y in ys]
        if allys:
            a.set_ylim(max(0.0, min(allys) - 0.02), min(1.0, max(allys) + 0.01))
        a.legend(fontsize=7)
    fig.suptitle("Recall@10 vs refinement budget, %s split (red dashed = JHQ at T = ck: the matched-recall target)" % split, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig_recall_T_%s.png" % split), dpi=110); plt.close(fig)
    # round 1 parameter scan
    if it1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        for a, tag in zip(axes, TAGS):
            for i, fam in enumerate(("LLR-iso", "LLR-bias", "LMP")):
                pts = sorted((float(r["param"]), int(r["T_min_matched"]) if r["T_min_matched"] else np.nan) for r in it1 if r["tag"] == tag and r["method"] == fam)
                a.plot([p for p, _ in pts], [t for _, t in pts], "-o", ms=4, color=COL[i + 1], label=fam)
            tj = [int(r["T_min_JHQ"]) for r in it1 if r["tag"] == tag and r["method"] == "JHQ" and r["T_min_JHQ"]]
            if tj:
                a.axhline(tj[0], color="black", ls="--", lw=1, label="JHQ")
            a.set_xscale("log"); a.set_xlabel("weight parameter (λσ, 2βσ², κσ)"); a.set_ylabel("T_min at the matched recall (%s)" % split)
            a.set_title(title(tag), fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig_llr_%s.png" % split), dpi=110); plt.close(fig)
    # Lemma 2 diagnostics
    preps = {t: json.load(open(os.path.join(R, "prep_%s.json" % t))) for t in TAGS if os.path.exists(os.path.join(R, "prep_%s.json" % t))}
    if preps:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        keys = (("mean_vector_norm", "norm of the data mean (unit vectors)"), ("abs_mean_over_sd_mean", "mean |μ_j| / σ_j after rotation"),
                ("bit_entropy_mean", "entropy of a JHQ bit (bits)"), ("coord_abs_corr_mean", "mean |corr| between rotated coordinates"))
        for a, (k, lab) in zip(axes, keys):
            v = [preps[t]["diag"][k] for t in preps]
            a.bar(range(len(v)), v, color=COL[1:len(v) + 1])
            a.set_xticks(range(len(v))); a.set_xticklabels([title(t).split(" ")[0] for t in preps], fontsize=9)
            a.set_title(lab, fontsize=9)
            for i, x in enumerate(v):
                a.text(i, x, "%.3f" % x, ha="center", va="bottom", fontsize=8)
        fig.suptitle("How far the data is from the paper's Lemma 2 (rotated coordinates i.i.d. N(0, σ²))", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig_bias.png"), dpi=110); plt.close(fig)
    # rank error
    re = rows_of(os.path.join(R, "RANK_ERROR_ANALYSIS.csv"))
    re = [r for r in re if r["split"] == ("val" if split == "test" else split)]
    if re:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        for a, tag in zip(axes, TAGS):
            rs = [r for r in re if r["tag"] == tag]
            x = np.arange(len(rs))
            a.bar(x - 0.3, [np.log10(max(float(r["empirical_misrank_mean"]), 1e-300)) for r in rs], 0.2, label="empirical misrank")
            a.bar(x - 0.1, [float(r["log10_bhattacharyya_model_median"]) for r in rs], 0.2, label="Bhattacharyya (model, LLR)")
            a.bar(x + 0.1, [float(r["log10_chernoff_own_score_median"]) for r in rs], 0.2, label="Chernoff (model, own score)")
            a.bar(x + 0.3, [np.log10(max(float(r["cantelli_mean"]), 1e-300)) for r in rs], 0.2, label="Cantelli (empirical)")
            a.set_xticks(x); a.set_xticklabels([r["code"] for r in rs])
            a.set_ylabel("log10 probability"); a.set_title(title(tag), fontsize=10); a.grid(alpha=0.3, axis="y")
            if tag == TAGS[0]:
                a.legend(fontsize=7)
        fig.suptitle("Pairwise misranking (ground-truth top-10 vs other candidates): bounds against the data", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig_rank_error.png"), dpi=110); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
