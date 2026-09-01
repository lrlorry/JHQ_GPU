#!/usr/bin/env python3
"""QPS-vs-recall front: JHQ v21 against cuVS IVF-PQ, FAISS GPU/CPU and JHQ CPU.

Both GPU systems are measured with the whole 1000-query set in one call and
with the device drained before the clock stops; see
scripts/run_head_to_head.sh and the _timed_search helper in
scripts/bench_cuvs_hnsw.py for why neither was true of the earlier numbers.

Usage:
  plot_v21_vs_cuvs.py --jhq results/v21_front_vogue.csv \
                      --cuvs results/cuvs_ivfpq_vogue_synced.csv \
                      --out results/v21_vs_cuvs_vogue.png
Each CSV needs columns recall,qps and, for cuVS, a `method` column to split by
bytes-per-vector.
"""
import argparse, csv, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path, label_col=None):
    series = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                r, q = float(row["recall"]), float(row["qps"])
            except (KeyError, ValueError):
                continue
            series[row.get(label_col, "") if label_col else ""].append((r, q))
    for pts in series.values():
        pts.sort()
    return series


def pareto(points):
    """Keep only points no other point beats on both axes."""
    best, out = 0.0, []
    for r, q in sorted(points, key=lambda p: -p[0]):
        if q > best:
            best, _ = q, out.append((r, q))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jhq", required=True)
    ap.add_argument("--cuvs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Vogue-768, k=10, batch 1000")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, pts in read(a.cuvs, "method").items():
        if "384" not in label:            # matched footprint only
            continue
        f = pareto(pts)
        ax.plot([r for r, _ in f], [q for _, q in f], "o--", label=label, color="tab:orange")
    for _, pts in read(a.jhq).items():
        f = pareto(pts)
        ax.plot([r for r, _ in f], [q for _, q in f], "s-", label="JHQ v21 (cascade)", color="tab:blue")

    ax.set_xlabel("Recall@10 (vs true top-10)")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.set_title(a.title)
    ax.grid(alpha=.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
