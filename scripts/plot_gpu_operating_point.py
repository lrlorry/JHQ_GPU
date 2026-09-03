#!/usr/bin/env python3
"""The three things the re-measurement found, one figure each.

Reads results/gpu_operating_point/{fronts_gpu,alpha_mechanism,codebook_across_ds,
residual_sample_size}.csv and writes fig_alpha.png, fig_codebook_ds.png and
fig_fronts_gpu.png beside them.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "gpu_operating_point")
ACC, MUTED, NO, SOFT = "#1F5F8B", "#5C6773", "#A2452E", "#9AA5B1"
SERIES = ["#12507B", "#3D7FA6", "#7FA6C2", "#B3C7D6"]
LABEL = {"vogue-768": "Vogue-768", "arxiv-768": "arXiv-768",
         "openai3-1536": "OpenAI3-1536", "openai3-3072": "OpenAI3-3072",
         "bge-m3": "BGE-M3-1024", "stella-trec24": "Stella-1024",
         "vogue": "Vogue-768", "stella": "Stella-1024", "oai3072": "OpenAI3-3072"}


def read(name):
    with open(os.path.join(D, name)) as fh:
        return [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))]


def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)


def fig_alpha():
    """Where the knee is, and what the machine moves.

    The hypothesis going in was that a narrow block would cap the reachable
    candidate set and so cap recall. It does not: recall saturates at the same
    place in every configuration measured. What the machine moves is the price
    of going there, and it moves it by a factor of ten.
    """
    rows = [r for r in read("alpha_mechanism.csv") if r["recall"] != "FAIL"]

    def series(ds, key, val, other, oval):
        pts = sorted([(int(r["alpha"]), float(r["recall"]), int(r["qps"]))
                      for r in rows if r["set"] == ds
                      and r[key] == val and r[other] == oval])
        return pts

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.6, 4.6), dpi=100)

    # left: every configuration's recall curve, to show they are one curve
    for ds, colour in (("vogue", SERIES[0]), ("stella", SERIES[1])):
        first = True
        for key, vals, other, oval in (("block", ["128", "256", "512", "1024"],
                                        "batch", "1024"),
                                       ("batch", ["1", "16", "256", "4096"],
                                        "block", "1024")):
            for v in vals:
                pts = series(ds, key, v, other, oval)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                        color=colour, ms=3, lw=1, alpha=0.55,
                        label=LABEL[ds] if first else None)
                first = False
    ax.set_xscale("log", base=2); ax.set_xlabel("α", fontsize=10.5)
    ax.set_ylabel("Recall@10", fontsize=10.5)
    ax.axvline(4, color=NO, ls="--", lw=1)
    ax.text(4.3, ax.get_ylim()[0], " the paper's α", color=NO, fontsize=8.6,
            va="bottom")
    ax.set_title("Every block width and batch size, overlaid:\nrecall saturates "
                 "in the same place", fontsize=10.8, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    frame(ax)

    # middle: what alpha costs, across block width
    for i, blk in enumerate(["128", "256", "512", "1024"]):
        pts = series("stella", "block", blk, "batch", "1024")
        if not pts:
            continue
        base = dict((a, q) for a, _, q in pts)[2]
        bx.plot([p[0] for p in pts], [100 * (1 - p[2] / base) for p in pts],
                "o-", color=SERIES[i % 4], ms=4,
                label=f"BLOCK={blk}  ({base // 1000}k QPS at α=2)")
    bx.set_xscale("log", base=2); bx.set_xlabel("α", fontsize=10.5)
    bx.set_ylabel("throughput given up, % of α=2", fontsize=10.5)
    bx.set_title("The faster the scan, the dearer refinement is\n"
                 "Stella-1024, block width", fontsize=10.8, loc="left")
    bx.legend(fontsize=8.4, frameon=False, loc="upper left")
    frame(bx)

    # right: and across queries in flight
    for i, bs in enumerate(["1", "16", "256", "4096"]):
        pts = series("vogue", "batch", bs, "block", "1024")
        if not pts:
            continue
        base = dict((a, q) for a, _, q in pts)[2]
        cx.plot([p[0] for p in pts], [100 * (1 - p[2] / base) for p in pts],
                "o-", color=SERIES[i % 4], ms=4,
                label=f"batch={bs}  ({base:,} QPS at α=2)")
    cx.set_xscale("log", base=2); cx.set_xlabel("α", fontsize=10.5)
    cx.set_ylabel("throughput given up, % of α=2", fontsize=10.5)
    cx.set_title("Refinement hides while the card has room\n"
                 "Vogue-768, queries in flight", fontsize=10.8, loc="left")
    cx.legend(fontsize=8.4, frameon=False, loc="upper left")
    frame(cx)

    fig.suptitle("Where recall stops improving is a property of the quantiser; "
                 "what it costs to get there is a property of the machine — "
                 "2.4% to 29.8% for the same α on one card",
                 fontsize=12, x=0.008, ha="left", y=1.04)
    out = os.path.join(D, "fig_alpha.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(os.path.basename(out))


def fig_codebook():
    """Equation 4 against a trained quantiser as the subspace widens."""
    rows = read("codebook_across_ds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=100)
    for ax, ds in zip(axes, ["oai3072", "stella"]):
        for arm, colour, name in (("eq4", ACC, "equation 4, analytical"),
                                  ("pq", NO, "product quantiser, Lloyd-refined")):
            pts = sorted([(int(r["Ds"]), float(r["recall"]) if r["recall"] else None)
                          for r in rows if r["dataset"] == ds and r["codebook"] == arm])
            xs = [p[0] for p in pts if p[1] is not None]
            ys = [p[1] for p in pts if p[1] is not None]
            ax.plot(xs, ys, "o-", color=colour, ms=5, label=name)
            for dsv, rec in pts:
                if rec is None:
                    ax.plot([dsv], [min(ys) if ys else 0.88], "x", color=NO, ms=9,
                            mew=2)
                    ax.annotate("fails", (dsv, min(ys) if ys else 0.88),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8.6, color=NO)
        ax.axvline(8, color=MUTED, ls=":", lw=1)
        lo, hi = ax.get_ylim()
        ax.annotate("Ds = B — past here the paper's\nequal split is undefined",
                    (8, hi), textcoords="offset points", xytext=(7, -26),
                    fontsize=8.4, color=MUTED, va="top")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("subspace width Ds = d/M   (shorter code to the right)",
                      fontsize=10.5)
        ax.set_ylabel("Recall@10", fontsize=10.5)
        ax.set_title(LABEL.get(ds, ds), fontsize=11.5, loc="left")
        ax.legend(fontsize=9, frameon=False, loc="lower left")
        frame(ax)
    fig.suptitle("The analytical codebook reaches every code length, and past "
                 "Ds = 32 it is the one still standing",
                 fontsize=12, x=0.008, ha="left", y=1.02)
    out = os.path.join(D, "fig_codebook_ds.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(os.path.basename(out))


def fig_fronts():
    """One envelope per dataset, over nlist and nprobe."""
    rows = [r for r in read("fronts_gpu.csv") if r["recall"] not in ("FAIL", "")]
    order = ["vogue-768", "arxiv-768", "openai3-1536", "openai3-3072",
             "bge-m3", "stella-trec24"]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), dpi=100)
    for ax, ds in zip(axes.ravel(), order):
        for i, Br in enumerate(["4", "8"]):
            pts = [(float(r["recall"]), int(r["qps"])) for r in rows
                   if r["dataset"] == ds and r["Br"] == Br]
            if not pts:
                continue
            env, best = [], 0
            for rec, q in sorted(pts, key=lambda p: -p[0]):
                if q > best:
                    env.append((rec, q)); best = q
            env.sort()
            ax.plot([p[0] for p in env], [p[1] for p in env], "o-",
                    color=SERIES[i], ms=4, label=f"JHQ Br={Br}")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ".",
                    color=SERIES[i], ms=3, alpha=0.35)
        ax.set_yscale("log")
        ax.set_xlabel("Recall@10", fontsize=10)
        ax.set_ylabel("QPS", fontsize=10)
        ax.set_title(LABEL.get(ds, ds), fontsize=11, loc="left")
        ax.legend(fontsize=8.8, frameon=False, loc="lower left")
        frame(ax)
    fig.suptitle("The front is the envelope over nlist and nprobe at α = 32; "
                 "faint points are the configurations it is drawn from",
                 fontsize=12, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    out = os.path.join(D, "fig_fronts_gpu.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(os.path.basename(out))


if __name__ == "__main__":
    fig_alpha(); fig_codebook(); fig_fronts()
