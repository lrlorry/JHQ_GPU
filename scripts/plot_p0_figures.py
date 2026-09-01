#!/usr/bin/env python3
"""Figures from the P0 harness CSVs. Reads results/final/p0_*.csv only.

Nothing here is transcribed: an earlier hand-built CSV put one dataset's
numbers under another's name and the error survived into a figure, so every
value on these axes is parsed from a harness row that carries its own commit,
parameters and run count.
"""
import csv, glob, json, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")

# (label, colour, filename glob). JHQ's two residual settings are separate
# series because they are separate memory budgets, not one curve.
SERIES = [
    ("JHQ Br=4",       "tab:blue",   "p0_{ds}*jhq*Br4*.csv"),
    ("JHQ Br=8",       "tab:cyan",   "p0_{ds}*jhq*Br8*.csv"),
    ("JQ (no residual)","tab:purple","p0_{ds}*_jq.csv"),
    ("cuVS IVF-PQ",    "tab:orange", "p0_{ds}*ivfpq.csv"),
    ("CAGRA fp32",     "tab:green",  "p0_{ds}*_cagra.csv"),
    ("CAGRA int8",     "tab:olive",  "p0_{ds}*cagra_int8.csv"),
]
DATASETS = [("vogue", 768), ("arxiv-768", 768),
            ("openai3-1536", 1536), ("openai3-3072", 3072),
            ("bge-m3", 1024), ("stella-trec24", 1024)]


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(D, pattern))):
        with open(path) as fh:
            body = [l for l in fh if not l.startswith("#")]
        for r in csv.DictReader(body):
            if r["status"] != "ok":
                continue
            try:
                rows.append((float(r["recall"]), float(r["qps_mean"]),
                             float(r["qps_std"] or 0), r["vram_mib"]))
            except ValueError:
                continue
    return rows


def front(pts):
    best, keep = 0.0, []
    for r, q, s, v in sorted(pts, key=lambda p: -p[0]):
        if q > best:
            best = q
            keep.append((r, q, s, v))
    return sorted(keep)


def best_at(pts, target):
    c = [p for p in pts if p[0] >= target]
    return max(c, key=lambda p: p[1]) if c else None


def fig_fronts():
    have = [(ds, d) for ds, d in DATASETS if load(f"p0_{ds}*jhq*.csv")]
    if not have:
        return
    ncol = 2
    nrow = (len(have) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4.2 * nrow), squeeze=False)
    for ax, (ds, dim) in zip([a for row in axes for a in row], have):
        for label, colour, pat in SERIES:
            pts = front(load(pat.format(ds=ds)))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=4, lw=1.7,
                    color=colour, label=label)
        ax.set_yscale("log")
        ax.set_xlim(0.55, 1.0)
        ax.grid(alpha=.3, which="both")
        ax.set_title(f"{ds}  (d={dim})", fontsize=11)
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
    for ax in [a for row in axes for a in row][len(have):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("QPS against recall, batch 1000, five runs per point", y=.997)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "figA_fronts.png"), dpi=160)
    print("figA_fronts.png")


def fig_dimension_trend():
    """CAGRA's lead over JHQ against dimension.

    CAGRA scores against the stored vectors, so its distance work is O(d);
    JHQ's is O(M) with M fixed. If that is what drives the gap, the ratio
    should fall as d grows -- and it is the one claim here that generalises
    beyond the four datasets it is measured on.
    """
    pts = []
    for ds, dim in DATASETS:
        j = [p for pat in ("p0_{ds}*jhq*Br4*.csv", "p0_{ds}*jhq*Br8*.csv")
             for p in load(pat.format(ds=ds))]
        c = load(f"p0_{ds}*_cagra.csv")
        if not (j and c):
            continue
        for target, marker in ((0.95, "o"), (0.98, "s")):
            bj, bc = best_at(j, target), best_at(c, target)
            if bj and bc:
                pts.append((dim, bc[1] / bj[1], target, ds, marker))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for target, colour in ((0.95, "tab:green"), (0.98, "tab:red")):
        sel = sorted((d, r, ds) for d, r, t, ds, _ in pts if t == target)
        if not sel:
            continue
        ax.plot([s[0] for s in sel], [s[1] for s in sel], "o-", lw=2, ms=8,
                color=colour, label=f"at Recall ≥ {target}")
        for d, r, ds in sel:
            ax.annotate(ds, (d, r), textcoords="offset points", xytext=(6, 6),
                        fontsize=7.5, color=colour)
    ax.axhline(1.0, color="grey", ls="--", lw=1.2)
    ax.text(800, 1.04, "parity", fontsize=8, color="grey")
    ax.set_xscale("log", base=2)
    ax.set_xticks([768, 1024, 1536, 3072])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("vector dimension")
    ax.set_ylabel("CAGRA QPS / JHQ QPS  (>1 means CAGRA ahead)")
    ax.set_title("CAGRA's throughput lead narrows as dimension grows")
    ax.grid(alpha=.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(D, "figB_dimension_trend.png"), dpi=170)
    print("figB_dimension_trend.png")


def table():
    """The five-question summary, printed from the CSVs."""
    print(f"\n{'dataset':<15}{'method':<18}{'R>=0.95':>20}{'R>=0.98':>20}{'VRAM MiB':>11}")
    for ds, dim in DATASETS:
        printed = False
        for label, _, pat in SERIES:
            pts = load(pat.format(ds=ds))
            if not pts:
                continue
            printed = True
            a, b = best_at(pts, 0.95), best_at(pts, 0.98)
            fa = f"{a[1]:,.0f} @{a[0]:.4f}" if a else "not reached"
            fb = f"{b[1]:,.0f} @{b[0]:.4f}" if b else "not reached"
            vram = (a or b or pts[0])[3] or "-"
            print(f"{ds:<15}{label:<18}{fa:>20}{fb:>20}{vram:>11}")
        if printed:
            print()


if __name__ == "__main__":
    if not glob.glob(os.path.join(D, "p0_*.csv")):
        sys.exit(f"no p0_*.csv under {D} — run scripts/bench_all.py first")
    fig_fronts(); fig_dimension_trend(); table()
