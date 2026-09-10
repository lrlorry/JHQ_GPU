#!/usr/bin/env python3
"""Separating the two costs the factorisation trades between, both measured.

The paper's mechanism claim used to be inferential: six controls were
"consistent with" an issue-sensitive scan, because Nsight counters were
unavailable in the container (ERR_NVGPUCTRPERM). Static instruction counts do
not need a profiler. cuobjdump -sass gives the emitted machine code for
scan_ivf_coalesced_kernel at each granularity, and the counts confirm the
design's arithmetic exactly: generic loads go 42, 84, 168 for G=1, 2, 4 -- one,
two and four lookups across 42 statically unrolled subspace iterations -- and
G=8's 144 is eight lookups over 18 iterations, the compiler having unrolled less
because the body grew. FADD per iteration is G+1 at every G, which is the
independent check that the normalisation is right.

That gives an x axis the paper did not have. Panel (a): with the table resident
(G=2, 4, 8) query time is close to linear in issued work per subspace, and the
slope is the issue cost. G=1 sits above that line, and the gap is the residency
cost -- the one thing the line cannot explain, since G=1 issues the least work
of all four. Panel (b): that gap grows with probe depth and is larger at M=384
than M=96, which is what "a table that misses is paid once per candidate per
subspace" predicts.

So the two terms Section 3.3 trades between are now measured separately rather
than argued from a family of controls. G=2 wins because it is the only point
that pays little of either.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import DS_COLOR, DS_MARK, PRETTY, WIDE, plt

WORK = ("LD", "FADD", "LOP3", "SHF")   # the table lookup and what feeds it
SETS = (("vogue-768", 96), ("openai3-3072", 384))
NPS = (8, 32, 128, 512)


def sass():
    """G -> (lookups per subspace, issued work per subspace)."""
    H = collections.defaultdict(dict)
    g = None
    for ln in open(style.datafile("sass_counts.log")):
        m = re.match(r"### G=(\d)", ln)
        if m:
            g = int(m.group(1))
            continue
        m = re.match(r"\s*(\d+)\s+([A-Z0-9_]+)\s*$", ln)
        if m and g:
            H[g][m.group(2)] = int(m.group(1))
    out = {}
    for g, d in H.items():
        unroll = d["LD"] // g          # LD is exactly g lookups per iteration
        assert d["FADD"] // unroll == g + 1, (g, d["FADD"], unroll)
        out[g] = (g, sum(d.get(k, 0) for k in WORK) / unroll)
    return out


def timing():
    """(dataset, nprobe) -> {G: microseconds per query}, packed layout."""
    D = collections.defaultdict(dict)
    for ln in open(style.datafile("lut_groups.log"), errors="ignore"):
        m = re.search(r"(\S+)\s+M=\d+\s+layout=(\w+)\s+G=(\d)\s+np=(\d+)\s+"
                      r"recall=[\d.]+\s+qps=(\d+)", ln)
        if m and m.group(2) == "word":
            D[(m.group(1), int(m.group(4)))][int(m.group(3))] = 1e6 / int(m.group(5))
    return D


def penalty(work, t):
    """How far G=1 sits above a line fitted to the resident-table points.

    Least squares over all three, not the G=2/G=8 endpoints: the first version
    joined the ends and ignored G=4 while the text called it a three-point fit.

    The result is an extrapolation and should be read as one. G=1 issues 4.4
    instructions per candidate-subspace and the fitted points span 13.0 to 50.4,
    so the prediction sits two thirds of the fitted range outside it. The sign
    and the trend are what the figure carries; the percentages are indicative.
    """
    xs = [work[g][1] for g in (2, 4, 8)]
    ys = [t[g] for g in (2, 4, 8)]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    pred = my + slope * (work[1][1] - mx)
    return pred, 100.0 * (t[1] / pred - 1.0)


def main():
    work, T = sass(), timing()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.4))

    NP_SHOWN = 512
    for ds, M in SETS:
        t = T[(ds, NP_SHOWN)]
        c, mk = DS_COLOR[ds], DS_MARK[ds]
        xs = [work[g][1] for g in (2, 4, 8)]
        a.plot(xs, [t[g] for g in (2, 4, 8)], color=c, marker=mk, ls="-")
        # Direct label: a legend box lands on one curve or the other here.
        a.annotate("%s, $M{=}%d$" % (PRETTY[ds], M), (xs[1], t[4]),
                   textcoords="offset points", xytext=(4, -11), fontsize=6.5,
                   color=c)
        pred, pct = penalty(work, t)
        a.plot([work[1][1]], [t[1]], color=c, marker=mk, mfc="none", ms=5)
        a.plot([work[1][1], work[1][1]], [pred, t[1]], color=c, lw=0.8, ls=":")
        a.annotate(r"$%+.0f\%%$" % pct, (work[1][1], t[1]),
                   textcoords="offset points", xytext=(5, 3), fontsize=7, color=c)
        # the line the resident points define, extended back to G=1's work
        sl = (t[8] - t[2]) / (work[8][1] - work[2][1])
        a.plot([work[1][1], work[8][1]],
               [t[2] - sl * (work[2][1] - work[1][1]), t[8]],
               color=c, lw=0.6, ls="--", alpha=0.55)
    for g in (1, 2, 4, 8):
        a.annotate("$G{=}%d$" % g, (work[g][1], 0.015),
                   xycoords=("data", "axes fraction"), ha="center", va="bottom",
                   fontsize=6.5, color="#898781")
    a.set_xlabel(u"issued instructions per candidate–subspace (SASS)")
    a.set_ylabel(r"$\mu$s per query")
    a.text(0.03, 0.90, "(a) nprobe$=$%d" % NP_SHOWN, transform=a.transAxes,
           ha="left", fontsize=7, color="#52514e")

    for ds, M in SETS:
        ys = [penalty(work, T[(ds, n)])[1] for n in NPS]
        b.plot(range(len(NPS)), ys, color=DS_COLOR[ds], marker=DS_MARK[ds],
               label="%s, $M{=}%d$" % (PRETTY[ds], M))
    b.axhline(0, color="0.35", lw=0.7, ls=":")
    b.set_xticks(range(len(NPS)))
    b.set_xticklabels([str(n) for n in NPS])
    b.set_xlabel("nprobe")
    b.set_ylabel(r"$G{=}1$ above the line (\%)")
    b.legend(loc="upper left", fontsize=6.5)
    b.text(0.97, 0.05, "(b)", transform=b.transAxes, ha="right", fontsize=7,
           color="#52514e")

    style.save(fig, "fig_issue")
    print("\n  numbers for the text:")
    for g in (1, 2, 4, 8):
        print("    G=%d  %d lookup(s)/subspace, %.1f issued instructions"
              % (g, work[g][0], work[g][1]))
    for ds, M in SETS:
        print("    %-14s residency cost %s"
              % (PRETTY[ds],
                 "  ".join("np%d:%+.0f%%" % (n, penalty(work, T[(ds, n)])[1])
                           for n in NPS)))


if __name__ == "__main__":
    main()
