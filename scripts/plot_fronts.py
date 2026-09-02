#!/usr/bin/env python3
"""QPS against recall, one panel per dataset — the form the field reads.

Replaces a set of bar charts at fixed recall gates. A bar at a gate throws away
the curve, log-scale bar heights cannot be compared by eye, and "never reaches
this recall" needed a hatched stub to show at all. On a front it is simply
where the line stops, which is also the method's ceiling.

The x range starts at 0.85: below that every method is fast and nothing the
paper claims happens there.
"""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")

DATASETS = [("vogue", "Vogue-768", 768, "932k"),
            ("arxiv-768", "arXiv-768", 768, "2.25M"),
            ("openai3-1536", "OpenAI3-1536", 1536, "999k"),
            ("openai3-3072", "OpenAI3-3072", 3072, "999k"),
            ("bge-m3", "BGE-M3", 1024, "10.1M"),
            ("stella-trec24", "Stella-TREC24", 1024, "17.8M")]

METHODS = [
    ("JHQ (this work)",  "#12507B", "o", "-",   2.6, 10),
    ("cuVS CAGRA fp32",  "#2E8B57", "D", "-",   1.8, 6),
    ("cuVS CAGRA int8",  "#8A9A2B", "^", "--",  1.8, 6),
    ("cuVS IVF-PQ",      "#E07B39", "s", "--",  1.8, 6),
    # JQ tops out at 0.647-0.774 across the six, entirely below this x range;
    # its ceiling is printed in each panel instead of drawing an empty line.
]
PATS = {
    "JHQ (this work)":   ["p0_{d}*jhq*.csv", "sat_{d}*jhq*.csv"],
    "JQ (residual off)": ["p0_{d}*_jq.csv"],   # ceiling only, see note in each panel
    "cuVS IVF-PQ":       ["p0_{d}*ivfpq*.csv", "sat_{d}*ivfpq*.csv"],
    "cuVS CAGRA fp32":   ["p0_{d}*_cagra.csv", "sat_{d}*_cagra.csv"],
    "cuVS CAGRA int8":   ["p0_{d}*cagra_int8.csv", "sat_{d}*cagra_int8.csv"],
}


def load(label, ds):
    out = []
    for pat in PATS[label]:
        for f in sorted(glob.glob(os.path.join(D, pat.format(d=ds)))):
            with open(f) as fh:
                body = [l for l in fh if not l.startswith("#")]
            for r in csv.DictReader(body):
                if r["status"] != "ok":
                    continue
                try:
                    out.append((float(r["recall"]), float(r["qps_mean"])))
                except ValueError:
                    pass
    return out


def front(pts):
    """Points no other point beats on both axes."""
    best, keep = 0.0, []
    for r, q in sorted(pts, key=lambda p: -p[0]):
        if q > best:
            best = q
            keep.append((r, q))
    return sorted(keep)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    axes = [a for row in axes for a in row]
    for ax, (ds, title, dim, n) in zip(axes, DATASETS):
        any_line = False
        for label, colour, mk, ls, lw, ms in METHODS:
            f = front(load(label, ds))
            f = [p for p in f if p[0] >= 0.80]
            if not f:
                continue
            any_line = True
            ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk, ls=ls,
                    color=colour, lw=lw, ms=ms, label=label,
                    zorder=5 if label.startswith("JHQ (") else 3,
                    markeredgecolor="white", markeredgewidth=.6)
            # the ceiling is where the line stops; say so once, on the end point
            rmax, qmax = max(f, key=lambda p: p[0])
            ax.annotate(f"{rmax:.3f}", (rmax, qmax), textcoords="offset points",
                        xytext=(6, -3), fontsize=8, color=colour, weight="bold",
                        annotation_clip=False)
        ax.set_yscale("log")
        ax.set_xlim(0.85, 1.028)
        ax.grid(alpha=.22, which="major")
        ax.grid(alpha=.10, which="minor")
        ax.set_title(f"{title}   ·   d={dim}, N={n}", fontsize=11.5, pad=8)
        ax.set_xlabel("Recall@10", fontsize=10)
        ax.set_ylabel("QPS  (batch 1000)", fontsize=10)
        ax.tick_params(labelsize=9)
        if not any_line:
            ax.text(.5, .5, "no configuration above 0.80", transform=ax.transAxes,
                    ha="center", color="#999", fontsize=10)
        jq = load("JQ (residual off)", ds)
        if jq:
            ax.text(.015, .035, f"JQ, residual level off: ceiling {max(p[0] for p in jq):.3f}",
                    transform=ax.transAxes, fontsize=8.4, color="#7A6BA8")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(.5, -.005))
    fig.suptitle("QPS against Recall@10 — each line ends at that method's ceiling, printed beside it. "
                 "Batch 1000, five runs per point.", fontsize=12.5, y=.985)
    fig.tight_layout(rect=[0, .045, 1, .96])
    fig.savefig(os.path.join(D, "fig1_fronts.png"), dpi=155)
    print("fig1_fronts.png")


if __name__ == "__main__":
    main()
