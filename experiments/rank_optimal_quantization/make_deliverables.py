#!/usr/bin/env python3
"""METHOD_COMPARISON.csv and RECALL_VS_BUDGET.csv from the per-round outputs (local, no data).

  python3 make_deliverables.py <results dir>

Rounds and the files they come from:
  round 1   iter1_llr.csv              likelihood-ratio scores on the paper index's codes (calibration, validation)
  round 2   eval_iter2.csv             per-coordinate thresholds, re-encoded
  round 3   eval_iter3.csv             rank-loss thresholds / levels, re-encoded
  round 4   eval_iter4.csv             linear-MMSE scoring on unchanged codes
  round 5   eval_iter5.csv             per-list thresholds, re-encoded
  final     eval_final_test.csv        the test split, read once for the methods carried to the end
Bit budget per vector is the same for every method: d primary bits + 8 d residual bits + one float (corr); the
global/list metadata column counts the floats each method adds to the paper index's.
"""
import csv
import json
import os
import sys

ROUND = {"iter2": 2, "iter3": 3, "iter4": 4, "iter5": 5, "final": "final"}
D = {"arxiv-768_np128_a100": 768, "vogue-768_np128_a64": 768, "openai3-3072_np128_a4": 3072}
NLIST = {"arxiv-768_np128_a100": 8192, "vogue-768_np128_a64": 4096, "openai3-3072_np128_a4": 8192}
DESC = {
    "JHQ": "paper index: zero thresholds, shared levels +-c, its own residual",
    "jhq_re": "offline re-implementation of JHQ (residual retrained the same way as every variant)",
    "lloydmax": "per-coordinate 2-level Lloyd-Max (reconstruction-optimal)",
    "median": "per-coordinate threshold at the median, conditional-mean levels (rank-optimal under the symmetric model)",
    "mean": "per-coordinate threshold at the mean, conditional-mean levels",
    "learned_rl_t_zero": "rank-loss thresholds from t = 0, conditional-mean levels",
    "learned_rl_t_median": "rank-loss thresholds from the median, conditional-mean levels",
    "learned_rl_tl_zero": "rank-loss thresholds and levels from t = 0",
    "learned_rl_tl_median": "rank-loss thresholds and levels from the median",
    "lmmse@JHQ": "linear-MMSE (Bussgang) score on the paper index's codes",
    "lmmse@median": "linear-MMSE (Bussgang) score on the median codes",
    "list_median": "per-IVF-list threshold at the list median, list conditional-mean levels",
    "list_mean": "per-IVF-list threshold at the list mean, list conditional-mean levels",
}


def metadata_floats(m, tag):
    d = D[tag]
    if m in ("JHQ", "jhq_re") or m.startswith("LLR") or m == "LMP":
        return 0 if m != "LLR-bias" else 2 * d
    if m.startswith("lmmse@"):
        return d * d + (3 * d if not m.endswith("JHQ") else 0)
    if m.startswith("list_"):
        return 3 * d * NLIST[tag]
    return 3 * d


def main():
    R = sys.argv[1]
    comp, curves = [], []
    it1 = os.path.join(R, "iter1_llr.csv")
    if os.path.exists(it1):
        for r in csv.DictReader(open(it1)):
            name = r["method"] if r["method"] == "JHQ" else "%s(%s)" % (r["method"], r["param"])
            Ts = sorted(int(k[8:]) for k in r if k.startswith("recall_T") and r[k])
            comp.append(dict(round=1, tag=r["tag"], method=name, chosen_on_calibration=r["chosen_on_calib"], split=r["split"],
                             bits_per_vector="d + 8d + 32", added_metadata_floats=metadata_floats(r["method"], r["tag"]),
                             T_min_matched=r["T_min_matched"], T_min_JHQ=r["T_min_JHQ"], target_recall=r["target_recall"],
                             misrank_random=r["misrank_random"], misrank_hard=r["misrank_hard"], gt_rank_max_mean=r["gt_rank_max_mean"],
                             primary_mse="", hierarchical_mse="", full_ivf_recall="",
                             description="likelihood-ratio / LMP weights on the paper index's codes"))
            for t in Ts:
                curves.append(dict(round=1, tag=r["tag"], method=name, split=r["split"], T=t, recall=r["recall_T%d" % t]))
    for lb, rd in ROUND.items():
        for suffix in ("", "_test"):
            p = os.path.join(R, "eval_%s%s.csv" % (lb, suffix))
            if not os.path.exists(p):
                continue
            for r in csv.DictReader(open(p)):
                if rd != "final" and r["split"] == "test":
                    continue
                Ts = sorted(int(k[8:]) for k in r if k.startswith("recall_T") and r[k])
                comp.append(dict(round=rd, tag=r["tag"], method=r["method"], chosen_on_calibration="", split=r["split"],
                                 bits_per_vector="d + 8d + 32", added_metadata_floats=metadata_floats(r["method"], r["tag"]),
                                 T_min_matched=r["T_min_matched"], T_min_JHQ=r["T_min_JHQ"], target_recall=r["target_recall"],
                                 misrank_random=r["misrank_random"], misrank_hard=r["misrank_hard"], gt_rank_max_mean=r["gt_rank_max_mean"],
                                 primary_mse=r.get("primary_mse", ""), hierarchical_mse=r.get("hierarchical_mse", ""),
                                 full_ivf_recall=r["full_ivf_recall"], description=DESC.get(r["method"], "")))
                for t in Ts:
                    curves.append(dict(round=rd, tag=r["tag"], method=r["method"], split=r["split"], T=t, recall=r["recall_T%d" % t]))
    for fn, rows in (("METHOD_COMPARISON.csv", comp), ("RECALL_VS_BUDGET.csv", curves)):
        with open(os.path.join(R, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("METHOD_COMPARISON.csv %d rows, RECALL_VS_BUDGET.csv %d rows" % (len(comp), len(curves)))


if __name__ == "__main__":
    main()
