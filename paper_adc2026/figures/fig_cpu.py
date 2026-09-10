#!/usr/bin/env python3
"""The CPU baseline Section 6.1 said it did not have, and what it is evidence for.

Panel (a) is the comparison the paper currently declines to make: both sides'
(alpha, nprobe) envelopes at matched recall.  Sweeping alpha on the CPU too is
the whole point -- the first baseline pinned alpha=100 while the GPU chose its
alpha, which compared one side's envelope against the other side's worst point.

Panel (b) is why the comparison is worth a figure rather than a sentence.  The
same alpha knob is worth 2.5x to 3x more on the GPU than on the CPU, at every
nprobe, on the same dataset and the same recall targets.  That says selective
refinement is not a CPU idea carried across: the GPU's scan is fast enough that
refinement occupies a larger share of the total, so the same reduction in
survivors buys more.  It is the measured form of the stage-share result.

Parsing, envelopes and interpolation all come from cpu_gpu_envelope.py, which
documents the three estimator traps that produced wrong numbers here (mixing
GPU binaries, max(qps | recall >= R) across sparse grids, and a raw Pareto
front that lets a 1e-4 recall tie-break win).  Do not re-derive them.
"""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import DS_COLOR, DS_MARK, PRETTY, COL, WIDE, plt
import cpu_gpu_envelope as env

# Only these two datasets have a CPU run; openai3-3072 is the one with a GPU
# alpha sweep on the same nprobe grid, so panel (b) uses it.
DS = ["vogue-768", "openai3-3072"]
CPU_C = "#8f8d86"          # the JQ/secondary grey from style.S
XLO = 0.93                 # the recall range where both envelopes have points


def envelopes():
    cpu_rows = env.read_cpu(env.CPU_LOG)
    gpu = env.read_gpu()
    out = {}
    for ds in DS:
        pts = [(r, q, "a=%g np=%d t%d" % (a, np_, th))
               for (d, th, np_), row in cpu_rows.items() if d == ds
               for a, (r, q) in row.items() if np_ >= env.CPU_NPROBE_FLOOR]
        # The GPU side is the rule arm where paper_fronts measured it, and the
        # v57 alpha sweep on openai3-3072; both are v57, never unioned across
        # builds.
        arm = "RULE" if "RULE" in gpu[ds] else "ENV alpha<=8"
        out[ds] = (env.front(pts),
                   env.front([(r, q, arm) for r, q in gpu[ds][arm]]), arm)
    return out


def alpha_lever():
    """nprobe -> (cpu %, gpu %) throughput given up by pinning alpha=100."""
    cpu_rows = env.read_cpu(env.CPU_LOG)
    cpu = {}
    for (ds, th, np_), row in cpu_rows.items():
        if ds != "openai3-3072" or th != 32 or np_ < env.CPU_NPROBE_FLOOR:
            continue
        if 100.0 not in row:
            continue
        r100, q100 = row[100.0]
        best = max((q for a, (r, q) in row.items() if r >= r100 - 3e-3))
        cpu[np_] = 100.0 * (best / q100 - 1.0)
    grows = collections.defaultdict(dict)
    import re
    for ln in open(style.datafile("openai3072_v57_front.log"), errors="ignore"):
        m = re.search(r"v57 (\S+)\s+np=(\d+)\s+a=([\d.]+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            grows[int(m.group(2))][float(m.group(3))] = (float(m.group(4)), int(m.group(5)))
    gpu = {}
    for np_, row in grows.items():
        if 100.0 not in row:
            continue
        r100, q100 = row[100.0]
        best = max((q for a, (r, q) in row.items() if r >= r100 - 3e-3))
        gpu[np_] = 100.0 * (best / q100 - 1.0)
    return sorted(set(cpu) & set(gpu)), cpu, gpu


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 3.05))

    # ---- (a) both envelopes ------------------------------------------------
    ev = envelopes()
    for si, ds in enumerate(DS):
        cf, gf, arm = ev[ds]
        c, mk = DS_COLOR[ds], DS_MARK[ds]
        a.plot([p[0] for p in gf], [p[1] for p in gf], color=c, marker=mk,
               ls="-")
        a.plot([p[0] for p in cf], [p[1] for p in cf], color=c, marker=mk,
               ls=":", mfc="none")
        # Direct labels: with two datasets on two devices a legend box either
        # covers the CPU curves or the GPU ones.
        # Anchor the name where the curve crosses the left edge, not at a data
        # point: every visible point already carries its matched-recall ratio,
        # and the two labels landed on top of each other.
        y0 = env.interp(gf, XLO) or gf[0][1]
        # Below the curve: above it the text meets the leftmost ratio label,
        # while the band under each GPU curve is empty.
        a.annotate("%s, GPU" % PRETTY[ds], (XLO, y0), textcoords="offset points",
                   xytext=(3, -9), fontsize=7, color=c)
        a.annotate("CPU", (cf[0][0], cf[0][1]), textcoords="offset points",
                   xytext=(2, -9), fontsize=7, color=c)
        # The ratio, quoted only where both envelopes cover the recall.  On
        # vogue the last three GPU points sit within a few thousandths of
        # recall of one another, so their labels stack: alternate the offset
        # above and below the curve rather than letting them overprint.
        # Start the two series on opposite parities: with the same one, the
        # first ratio of the lower curve lands beside the upper curve's name.
        placed = si
        for r, q, _ in gf:
            cq = env.interp(cf, r)
            if cq is None:
                continue
            right = r > XLO + 0.85 * (1.0 - XLO)
            dy = 5 if placed % 2 == 0 else -11
            a.annotate(r"$%.0f\times$" % (q / cq), (r, q),
                       textcoords="offset points",
                       xytext=(-3 if right else 2, dy), fontsize=7, color=c,
                       ha="right" if right else "left")
            placed += 1
        # Say out loud where the CPU stops, so the gap is not read as a ratio.
        a.plot([cf[-1][0]], [cf[-1][1]], color=c, marker="|", ms=7, mew=1.0)
    a.set_yscale("log")
    a.set_xlabel("Recall@10")
    a.set_ylabel("QPS")
    a.set_xlim(XLO, 1.0)
    a.text(0.02, 0.97, "(a)", transform=a.transAxes, fontsize=7,
           va="top", color="#52514e")

    # ---- (b) the alpha lever, CPU against GPU ------------------------------
    nps, cpu, gpu = alpha_lever()
    xs = range(len(nps))
    w = 0.36
    b.bar([x - w / 2 for x in xs], [cpu[n] for n in nps], w,
          color=CPU_C, label="CPU, 32 threads")
    b.bar([x + w / 2 for x in xs], [gpu[n] for n in nps], w,
          color=DS_COLOR["openai3-3072"], label="GPU, RTX 5090")
    for x, n in zip(xs, nps):
        b.text(x, max(cpu[n], gpu[n]) + 3, r"$%.1f\times$" % (gpu[n] / cpu[n]),
               ha="center", fontsize=7, color="#52514e")
    b.set_xticks(list(xs))
    b.set_xticklabels([str(n) for n in nps])
    b.set_xlabel("nprobe")
    b.set_ylabel(r"QPS given up by $\alpha{=}100$ (%)")
    b.set_ylim(0, max(gpu.values()) * 1.28)
    b.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.0, 0.86))
    b.text(0.03, 0.97, "(b) openai3-3072, matched recall",
           transform=b.transAxes, fontsize=7, va="top", color="#52514e")

    style.save(fig, "fig_cpu")

    # The numbers the caption and Section 6 must quote, printed so the text is
    # never typed from memory.
    print("\n  matched-recall ratios actually drawn:")
    for ds in DS:
        cf, gf, arm = ev[ds]
        rs = [(r, q / env.interp(cf, r)) for r, q, _ in gf if env.interp(cf, r)]
        print("    %-14s GPU arm=%-12s %s" % (PRETTY[ds], arm,
              "  ".join("R=%.4f:%.0fx" % t for t in rs)))
        print("    %-14s CPU envelope ends at R=%.4f (%.0f qps, %s)"
              % ("", cf[-1][0], cf[-1][1], cf[-1][2]))
    print("  alpha lever (openai3-3072):")
    for n in nps:
        print("    nprobe=%-5d CPU %+5.1f%%  GPU %+6.1f%%  ratio %.2fx"
              % (n, cpu[n], gpu[n], gpu[n] / cpu[n]))


if __name__ == "__main__":
    main()
