#!/usr/bin/env python3
"""Four figures, one claim each, against the algorithm baselines.

Replaces an earlier set that compared this work against its own previous
version and packed six datasets into one grid. Every value is read from the
harness CSVs; a method that never reaches a recall gate is drawn as such
rather than left out, because "cannot reach" is the finding on several of
these datasets.
"""
import csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")

# ordered by dimension so the trend reads left to right
DATASETS = [("vogue", "Vogue\n768d 932k", 768), ("arxiv-768", "arXiv\n768d 2.3M", 768),
            ("bge-m3", "BGE-M3\n1024d 10.1M", 1024), ("stella-trec24", "Stella\n1024d 17.8M", 1024),
            ("openai3-1536", "OpenAI3\n1536d 999k", 1536), ("openai3-3072", "OpenAI3\n3072d 999k", 3072)]
METHODS = [("JHQ (this work)", "#1F5F8B", ["p0_{d}*jhq*.csv", "sat_{d}*jhq*.csv"]),
           ("JQ, residual off", "#9B8EC4", ["p0_{d}*_jq.csv"]),
           ("cuVS IVF-PQ", "#E08A3C", ["p0_{d}*ivfpq*.csv", "sat_{d}*ivfpq*.csv"]),
           ("cuVS CAGRA fp32", "#3F8F5B", ["p0_{d}*_cagra.csv", "sat_{d}*_cagra.csv"]),
           ("cuVS CAGRA int8", "#8A9A2B", ["p0_{d}*cagra_int8.csv", "sat_{d}*cagra_int8.csv"])]


def load(pats, ds):
    out = []
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(D, pat.format(d=ds)))):
            with open(f) as fh:
                body = [l for l in fh if not l.startswith("#")]
            for r in csv.DictReader(body):
                if r["status"] != "ok":
                    continue
                try:
                    out.append((float(r["recall"]), float(r["qps_mean"]),
                                float(r["vram_mib"] or 0)))
                except ValueError:
                    pass
    return out


def best_at(rows, t):
    c = [p for p in rows if p[0] >= t]
    return max(c, key=lambda p: p[1]) if c else None


def peak(rows):
    return max(rows, key=lambda p: p[0]) if rows else None


STUB = 30.0   # height of the "tried, fell short" marker on the log axis


def fig_gate(gate, fname, title):
    """Throughput at a recall gate, with 'cannot reach' drawn explicitly."""
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    n = len(METHODS)
    w = 0.8 / n
    for j, (label, colour, pats) in enumerate(METHODS):
        xs, ys, notes = [], [], []
        for i, (ds, _, _) in enumerate(DATASETS):
            rows = load(pats, ds)
            b = best_at(rows, gate)
            x = i - 0.4 + w * (j + .5)
            xs.append(x)
            if b:
                ys.append(b[1]); notes.append(None)
            else:
                ys.append(0)
                pk = peak(rows)
                notes.append(f"max\n{pk[0]:.3f}" if pk else "not run")
        # A method that tried and fell short gets a hatched stub carrying the
        # recall it did reach, so it reads differently from one that never ran.
        drawn = [y if y else 0 for y in ys]
        ax.bar(xs, drawn, w * .92, color=colour, label=label, zorder=3)
        for x, y, note in zip(xs, ys, notes):
            if note and note != "not run":
                ax.bar([x], [STUB], w * .92, color=colour, alpha=.35,
                       hatch="///", edgecolor=colour, linewidth=.6, zorder=3)
                ax.text(x, STUB * 1.5, note.replace("max\n", ""), ha="center",
                        va="bottom", fontsize=6.5, color="#A2452E", rotation=90)
            elif note:
                ax.text(x, STUB * 1.5, "not run", ha="center", va="bottom",
                        fontsize=6.2, color="#AAA", rotation=90)
            elif y:
                ax.text(x, y * 1.15, f"{y/1000:.0f}k" if y >= 1000 else f"{y:.0f}",
                        ha="center", fontsize=7.2)
    ax.set_yscale("log")
    ax.set_ylim(8, 1.6e6)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([l for _, l, _ in DATASETS], fontsize=9)
    ax.set_ylabel("QPS (batch 1000, log scale)")
    ax.set_title(title)
    ax.grid(alpha=.25, axis="y", which="both", zorder=0)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.10))
    ax.text(0.004, 0.965, "hatched stub = the method was swept and never reached "
            "this recall; its ceiling is printed on it",
            transform=ax.transAxes, fontsize=7.6, color="#A2452E")
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=165, bbox_inches="tight")
    print(fname)


def fig_ceiling():
    """The highest recall each method reaches at all, and what it costs in memory."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    n = len(METHODS); w = 0.8 / n
    for j, (label, colour, pats) in enumerate(METHODS):
        xs, ys, vr = [], [], []
        for i, (ds, _, _) in enumerate(DATASETS):
            pk = peak(load(pats, ds))
            xs.append(i - 0.4 + w * (j + .5))
            ys.append(pk[0] if pk else 0)
            vr.append(pk[2] if pk else 0)
        ax.bar(xs, ys, w * .92, color=colour, label=label, zorder=3)
        for x, y, v in zip(xs, ys, vr):
            if y:
                ax.text(x, y + .004, f"{y:.3f}", ha="center", fontsize=6.6, rotation=90)
    ax.axhline(0.98, color="#A2452E", ls="--", lw=1.3, zorder=2)
    ax.text(-0.45, 0.9815, "0.98", fontsize=8, color="#A2452E")
    ax.axhline(0.95, color="#777", ls=":", lw=1, zorder=2)
    ax.text(-0.45, 0.9515, "0.95", fontsize=8, color="#777")
    ax.set_ylim(0.60, 1.02)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([l for _, l, _ in DATASETS], fontsize=9)
    ax.set_ylabel("highest Recall@10 the method reaches")
    ax.set_title("Recall ceiling: which methods can get to 0.98 at all")
    ax.grid(alpha=.25, axis="y", zorder=0)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.09))
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_ceiling.png"), dpi=165, bbox_inches="tight")
    print("fig_ceiling.png")


def fig_memory():
    """Measured device memory against the recall it buys, on one dataset."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for label, colour, pats in METHODS:
        rows = [r for r in load(pats, "vogue") if r[2] > 0]
        if not rows:
            continue
        # the cheapest configuration that reaches each of a few gates
        pts = []
        for g in (0.90, 0.95, 0.97, 0.98, 0.99):
            c = [r for r in rows if r[0] >= g]
            if c:
                pts.append((min(c, key=lambda r: r[2])[2], g))
        if not pts:
            pk = peak(rows); pts = [(pk[2], pk[0])]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colour,
                label=label, ms=6, lw=2)
    ax.axhline(0.98, color="#A2452E", ls="--", lw=1.2)
    ax.set_xlabel("device memory in use, MiB (measured, not derived)")
    ax.set_ylabel("Recall@10 reached")
    ax.set_title("Vogue-768: what each method's recall costs in device memory")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_memory.png"), dpi=165)
    print("fig_memory.png")


if __name__ == "__main__":
    fig_gate(0.95, "fig_gate95.png",
             "Throughput at Recall@10 ≥ 0.95 — a bar is missing only where the method never gets there")
    fig_gate(0.98, "fig_gate98.png",
             "Throughput at Recall@10 ≥ 0.98 — int8 CAGRA reaches it on none of the six")
    fig_ceiling()
    fig_memory()
