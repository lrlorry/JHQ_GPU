#!/usr/bin/env python3
"""What a footprint buys: measured bytes per vector against the recall ceiling.

An earlier version plotted device memory against recall per operating point,
which produced vertical stacks -- memory is a property of the index, and the
recall points along a curve come from sweeping nprobe or itopk at fixed
memory. Here each index configuration is one point: the memory it holds,
divided by the number of vectors, against the highest recall it reaches at any
search setting.

Memory is read with cudaMemGetInfo and so includes the search workspace, not
only the code payload. That is the honest number for "will this fit on the
card", which is the question the figure is for.
"""
import csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")

DATASETS = {"vogue": ("Vogue", 932328, "o"), "arxiv-768": ("arXiv", 2253000, "s"),
            "openai3-1536": ("OpenAI3-1536", 999000, "^"),
            "openai3-3072": ("OpenAI3-3072", 999000, "v"),
            "bge-m3": ("BGE-M3", 10091524, "D"),
            "stella-trec24": ("Stella", 17776615, "P")}
METHODS = [("JHQ (this work)", "#12507B", ["p0_{d}*jhq*.csv", "sat_{d}*jhq*.csv"]),
           ("cuVS CAGRA fp32", "#2E8B57", ["p0_{d}*_cagra.csv", "sat_{d}*_cagra.csv"]),
           ("cuVS CAGRA int8", "#8A9A2B", ["p0_{d}*cagra_int8.csv", "sat_{d}*cagra_int8.csv"]),
           ("cuVS IVF-PQ", "#E07B39", ["p0_{d}*ivfpq*.csv", "sat_{d}*ivfpq*.csv"])]


def configs(pats, ds):
    """Group rows by the index configuration; recall varies within one, memory does not."""
    groups = {}
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(D, pat.format(d=ds)))):
            with open(f) as fh:
                body = [l for l in fh if not l.startswith("#")]
            for r in csv.DictReader(body):
                if r["status"] != "ok" or not r["vram_mib"]:
                    continue
                try:
                    p = json.loads(r["params"])
                    rec, vram = float(r["recall"]), float(r["vram_mib"])
                except (ValueError, KeyError):
                    continue
                # the index, not the search: drop the knobs that only move recall
                key = tuple(sorted((k, v) for k, v in p.items()
                                   if k not in ("nprobe", "n_probes", "itopk_size",
                                                "search_width", "k", "bytes_per_vec")))
                cur = groups.get(key)
                if cur is None or rec > cur[0]:
                    groups[key] = (rec, vram)
    return list(groups.values())


GATE = 0.98


def cheapest_reaching(pats, ds, gate):
    """Smallest configuration that reaches the gate, and the best one if none do."""
    cs = configs(pats, ds)
    ok = [c for c in cs if c[0] >= gate]
    if ok:
        return min(ok, key=lambda c: c[1]), True
    return (max(cs, key=lambda c: c[0]), False) if cs else (None, False)


def main():
    """One row per dataset: what reaching Recall 0.98 costs, and who cannot."""
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    order = list(DATASETS.items())
    for y, (ds, (dname, n, _)) in enumerate(order):
        for label, colour, pats in METHODS:
            got, reached = cheapest_reaching(pats, ds, GATE)
            if got is None:
                continue
            rec, vram = got
            bpv = vram * 2**20 / n
            if reached:
                ax.scatter(bpv, y, s=150, color=colour, zorder=4,
                           edgecolor="white", linewidth=1.1)
                ax.annotate(f"{bpv:,.0f} B", (bpv, y), textcoords="offset points",
                            xytext=(0, 11), ha="center", fontsize=7.8, color=colour)
            else:
                ax.scatter(bpv, y, s=110, facecolor="none", edgecolor=colour,
                           linewidth=1.6, zorder=4, alpha=.85)
                ax.annotate(f"max {rec:.3f}", (bpv, y), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=7.4,
                            color=colour, style="italic")
    for y in range(len(order)):
        ax.axhline(y, color="#DDD", lw=1, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{v[0]}\n{v[1]/1e6:.1f}M vectors" for _, v in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("measured device memory per vector, bytes (index + search workspace)",
                  fontsize=10.5)
    ax.set_title("Cost of reaching Recall@10 ≥ 0.98\n"
                 "filled = reaches it, at this footprint   ·   hollow = never reaches it, "
                 "ceiling shown", fontsize=11.5)
    ax.grid(alpha=.25, axis="x", which="both")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=9, label=l)
               for l, c, _ in METHODS]
    ax.legend(handles=handles, fontsize=9.5, loc="lower right", framealpha=.95)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig2_memory.png"), dpi=160)
    print("fig2_memory.png")


if __name__ == "__main__":
    main()
