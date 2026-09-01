#!/usr/bin/env python3
"""Figures for the JHQ-GPU results, from results/final/*.csv.

All GPU numbers are measured the same way: the whole 1000-query set in one
call, and the device drained before the clock stops. Neither was true of the
earlier comparison -- JHQ ran at batch 256 against a cuVS figure taken at 1000,
and cuVS's clock was read before its asynchronous search had finished.
"""
import csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")
OUT = D


def rows(name):
    with open(os.path.join(D, name), newline="") as fh:
        return list(csv.DictReader(fh))


def pareto(pts):
    """Points not beaten on both recall and QPS."""
    best, keep = 0.0, []
    for r, q in sorted(pts, key=lambda p: -p[0]):
        if q > best:
            best = q
            keep.append((r, q))
    return sorted(keep)


def fig_front():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    jhq = defaultdict(list)
    for r in rows("jhq_v22_front_vogue.csv"):
        jhq[r["prefix"]].append((float(r["recall"]), float(r["qps"])))
    allj = [p for v in jhq.values() for p in v]
    f = pareto(allj)
    ax.plot([r for r, _ in f], [q for _, q in f], "o-", lw=2.2, ms=6,
            color="tab:blue", label="JHQ-v22 (484 B/vec)", zorder=3)

    cu = defaultdict(list)
    for r in rows("cuvs_ivfpq_vogue_synced.csv"):
        cu[int(r["bytes"])].append((float(r["recall"]), float(r["qps"])))
    for b, style in ((384, "s--"), (192, "^:")):
        pts = sorted(cu[b])
        ax.plot([r for r, _ in pts], [q for _, q in pts], style, ms=5,
                color="tab:orange" if b == 384 else "tab:red",
                label=f"cuVS IVF-PQ ({b} B/vec)")

    # The 484 B front stops at 0.9508 because that is where the code runs out,
    # not the search: probing all 1024 lists only reaches 0.9528. Spending the
    # bytes on residual bits rather than on M is what passes 0.97.
    big = [(float(r["recall"]), float(r["qps"]), int(r["bytes"]))
           for r in rows("jhq_bytes_vogue.csv") if int(r["Br"]) == 8]
    big.sort()
    ax.plot([r for r, _, _ in big], [q for _, q, _ in big], "o--", ms=7, lw=1.8,
            color="tab:cyan", label="JHQ-v22, Br=8 (868-1156 B/vec)")
    for r, q, b in big:
        ax.annotate(f"{b} B", (r, q), textcoords="offset points", xytext=(4, -11),
                    fontsize=7.5, color="tab:cyan")

    cg = sorted((float(r["recall"]), float(r["qps"]))
                for r in rows("cagra_vogue.csv") if r["method"] == "CAGRA-deg64")
    ax.plot([r for r, _ in cg], [q for _, q in cg], "d-.", ms=6,
            color="tab:green", label="cuVS CAGRA (3328 B/vec)")

    ax.set_xlabel("Recall@10 (set intersection with the true top-10)")
    ax.set_ylabel("QPS (batch 1000)")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 1.0)
    ax.grid(alpha=.3, which="both")
    ax.set_title("Vogue-768, 932k vectors, RTX 5090")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_front_vogue.png"), dpi=170)
    print("fig1_front_vogue.png")


def fig_memory():
    """Bytes needed to reach a recall level. This is the axis JHQ wins on."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    jb = [(int(r["bytes"]), float(r["recall"])) for r in rows("jhq_bytes_vogue.csv")]
    jb.sort()
    ax.plot([b for b, _ in jb], [r for _, r in jb], "o-", lw=2.2, ms=7,
            color="tab:blue", label="JHQ (M, Br sweep)")
    for b, r in jb:
        ax.annotate(f"{r:.3f}", (b, r), textcoords="offset points",
                    xytext=(0, 7), fontsize=7.5, ha="center")

    cu = defaultdict(float)
    for r in rows("cuvs_ivfpq_vogue_synced.csv"):
        b = int(r["bytes"])
        cu[b] = max(cu[b], float(r["recall"]))
    cb = sorted(cu.items())
    ax.plot([b for b, _ in cb], [r for _, r in cb], "s--", lw=2, ms=7,
            color="tab:orange", label="cuVS IVF-PQ (best recall at each size)")

    cg = max(float(r["recall"]) for r in rows("cagra_vogue.csv"))
    ax.plot([3328], [cg], "d", ms=11, color="tab:green",
            label=f"cuVS CAGRA (3328 B, {cg:.4f})")

    ax.axhline(0.98, color="grey", ls=":", lw=1)
    ax.text(60, 0.982, "recall 0.98", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("bytes per vector (index payload)")
    ax.set_ylabel("best Recall@10 reached")
    ax.set_title("What each index costs to reach a recall level — Vogue-768")
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_memory_vogue.png"), dpi=170)
    print("fig2_memory_vogue.png")


def fig_ablation():
    rs = rows("ablation_vogue.csv")
    labels = [r["step"] for r in rs]
    qps = [float(r["qps"]) for r in rs]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.barh(range(len(rs)), qps, color=["#999"] + ["tab:blue"] * (len(rs) - 1))
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    base = qps[0]
    for i, (b, q, r) in enumerate(zip(bars, qps, rs)):
        ax.text(q + base * 0.03, i, f"{q:,.0f}  ({q/base:.2f}x)  R={float(r['recall']):.4f}",
                va="center", fontsize=8.5)
    ax.set_xlim(0, max(qps) * 1.45)
    ax.set_xlabel("QPS at Recall@10 ≈ 0.945, nprobe=128, batch 1000")
    ax.set_title("Where the speedup comes from — Vogue-768")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ablation.png"), dpi=170)
    print("fig3_ablation.png")


def fig_datasets():
    rs = [r for r in rows("datasets.csv") if r["nprobe"] == "128"]
    ds = sorted({r["dataset"] for r in rs})
    v16 = [next(float(r["qps"]) for r in rs if r["dataset"] == d and r["variant"] == "v16") for d in ds]
    v22 = [next(float(r["qps"]) for r in rs if r["dataset"] == d and r["variant"] == "v22") for d in ds]
    rec = [next(float(r["recall"]) for r in rs if r["dataset"] == d and r["variant"] == "v22") for d in ds]

    x = range(len(ds))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar([i - 0.2 for i in x], v16, 0.4, label="v16 (published port)", color="#999")
    ax.bar([i + 0.2 for i in x], v22, 0.4, label="v22 (this work)", color="tab:blue")
    for i, (a, b, r) in enumerate(zip(v16, v22, rec)):
        ax.text(i + 0.2, b * 1.02, f"{b/a:.2f}x", ha="center", fontsize=9)
        ax.text(i, -max(v22) * 0.13, f"R={r:.4f}", ha="center", fontsize=8, color="dimgrey")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ds, fontsize=9)
    ax.set_ylabel("QPS, nprobe=128, batch 1000")
    ax.set_title("Speedup holds across datasets at matched recall")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_datasets.png"), dpi=170)
    print("fig4_datasets.png")


if __name__ == "__main__":
    fig_front(); fig_memory(); fig_ablation(); fig_datasets()
