#!/usr/bin/env python3
"""Figure: why the table is split in half, and what the split is worth.

Section 6.4.2. The paper's strongest representation-specific claim is that the
primary table factorises exactly, 256 entries to 16+16, because the codebook
is a Cartesian product of one-dimensional levels. Two questions follow that
the paper could not answer, and both are now measured on one build at one
commit (`data/lut_groups.log`, v59).

## Why halves, and not quarters?

The identity holds for any partition of the code's eight base-2 digits, so
G groups of 8/G bits give G tables of 2^(8/G) entries:

    G=1  1 x 256 = 256 entries, 1 load a candidate a subspace
    G=2  2 x  16 =  32          2                              (shipped)
    G=4  4 x   4 =  16          4
    G=8  8 x   2 =  16          8

Panel (a) is that sweep. All four return identical recall at every point --
they must, the identity is exact -- so the comparison is pure throughput.

**G=2 wins all eight configurations, and the losers are informative.** G=4 and
G=8 have *smaller* tables than G=2 and are monotonically slower, in proportion
to their loads a candidate: 2, 4, 8. So the scan is not table-size-bound. Once
the table is small enough to sit in shared memory without bank conflicts,
shrinking it further buys nothing and costs an instruction per subspace per
candidate. That is the same conclusion as the table-free distance in Section
6.5, which moved less memory and still lost, and as the packed load in Section
6.4.3, which moves the same bytes and wins on instruction count. Three
experiments, one mechanism: this kernel is issue-bound.

## What the factorisation is worth, and when

Panel (b) is the 2x2 the paper had been inferring across version directories:
the full table against the factorised one, on the byte layout and on the
packed one, all from the same source tree.

**The effect is far more M-dependent than `results/v47_split_lut/` reports.**
That note measured 28 cells at M=96 and M=128 and found +6 to +14.5% with none
negative. At M=384 the factorisation is worth **1.23x to 1.53x**; at M=96 it
is worth 1.02x to 1.12x, and one cell (nprobe=32, packed layout) comes out at
0.96 -- slightly negative.

The mechanism is visible in the arithmetic. A 256-entry table is M*256*4
bytes: 96 KiB at M=96, which the carveout can still hold, and 384 KiB at
M=384, which it cannot, so G=1 there falls back to reading the table from
global memory. The factorisation matters exactly where the full table stops
fitting. Reported as an average over datasets it is a modest constant; read
against M it is a scaling property, which is the more useful claim and the
one the paper should make.

**The two payoffs are close to independent.** On openai3-3072 the
factorisation is worth 1.13/1.24/1.36 on the byte layout and 1.10/1.23/1.53 on
the packed one -- roughly the same either way -- so the gains multiply rather
than overlap. That is the evidence the "pays twice" framing needed and did not
have.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# lutgroups6.log is the same sweep on six datasets; lut_groups.log's phase 3
# is the only source for the table x layout square, so both are read.
rows, sec = collections.defaultdict(dict), None
for ln in open(datafile("lut_groups.log")):
    if "phase 2" in ln: sec = "sweep"
    elif "phase 3" in ln: sec = "square"
    elif "phase" in ln: sec = None
    m = re.search(r"^  (\S+)\s+M=(\d+)\s+layout=(\w+)\s+G=(\d+)\s+np=(\d+)\s+"
                  r"recall=([\d.]+)\s+qps=(\d+)", ln)
    if m and sec == "square":
        cell = rows[(sec, m.group(1), m.group(3), int(m.group(5)))]
        cell.setdefault(int(m.group(4)), []).append(
            (float(m.group(6)), int(m.group(7))))
for ln in open(datafile("lutgroups6.log")):
    m = re.search(r"^\s+(\S+)\s+M=(\d+)\s+layout=(\w+)\s+G=(\d+)\s+np=(\d+)\s+"
                  r"recall=([\d.]+)\s+qps=(\d+)", ln)
    if m:
        cell = rows[("sweep", m.group(1), m.group(3), int(m.group(5)))]
        cell.setdefault(int(m.group(4)), []).append(
            (float(m.group(6)), int(m.group(7))))

# the identity is exact, so this must hold before any timing is read
for k, g in rows.items():
    rs = [v[0] for reps in g.values() for v in reps]
    assert max(rs) - min(rs) <= 2e-4, (k, rs)


def med(cell, g):
    """Median QPS over a cell's repeats -- the aggregation Section 6.1 states."""
    return _st.median([v[1] for v in cell[g]])


import statistics as _st

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 3.05))
GS = [1, 2, 4, 8]
SETS = [("vogue-768", 96), ("openai3-3072", 384)]

# ── (a) granularity ────────────────────────────────────────────────────────
NP = [8, 32, 128, 512]
x = np.arange(len(GS))
# Six datasets at four depths is 24 curves, which is not a figure.  Each depth
# is drawn once as the median over datasets, with the full spread as a band --
# the claim is about the shape, and the band is what says how tight it is.
SWEEP_DS = sorted({k[1] for k in rows if k[0] == "sweep" and k[2] == "word"})
for np_ in NP:
    per = []
    for ds in SWEEP_DS:
        g = rows.get(("sweep", ds, "word", np_))
        if g and 2 in g and all(q in g for q in GS):
            base = med(g, 2)
            per.append([med(g, q) / base for q in GS])
    if not per:
        continue
    arr = np.array(per)
    sh = 0.30 + 0.70 * NP.index(np_) / (len(NP) - 1)
    a.fill_between(x, arr.min(0), arr.max(0), color="#1f6fc4", alpha=0.10 * sh,
                   lw=0)
    a.plot(x, np.median(arr, 0), color="#1f6fc4", alpha=sh, lw=1.2,
           marker="o", ms=3.0, mec="white", mew=0.5,
           label=r"$\mathit{nprobe}{=}%d$" % np_)
a.axhline(1.0, color="0.5", lw=0.7, ls=":")
a.set_xticks(x)
a.set_xticklabels(["1\n$256$", "2\n$2{\\times}16$", "4\n$4{\\times}4$",
                   "8\n$8{\\times}2$"])
a.set_xlabel("groups $G$, and the table it gives")
a.set_ylabel(r"QPS $\div$ QPS at $G{=}2$")
a.set_ylim(0.26, 1.13)   # headroom above 1.0 for the tag, below for the note
a.legend(loc="lower left", fontsize=6.6, labelspacing=0.25, borderpad=0.3)
a.text(0.97, 0.975, "median over six datasets;\nband is their range",
       fontsize=6.6, color="#898781", transform=a.transAxes, ha="right",
       va="top", linespacing=1.3)
a.text(0.03, 0.965, "(a)", transform=a.transAxes, fontsize=8, va="top")
# Short enough for the strip right of the legend; the tick labels already
# spell out 4x4 and 8x2, so the note only has to say they are equal.
a.annotate(r"$4{\times}4$ and $8{\times}2$:" "\n" "same table,\nmore lookups",
           xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
           fontsize=7, color="#52514e", linespacing=1.3)
a.annotate("smaller table,\nslower", xy=(2.55, 0.60), xytext=(1.35, 0.47),
           fontsize=7, color="#52514e", linespacing=1.3,
           arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#898781"))

# ── (b) the 2x2 ────────────────────────────────────────────────────────────
NP2 = [32, 128, 512]   # the phase-3 2x2 grid; nprobe=8 lives in phase 1
w, xs = 0.34, np.arange(len(NP2))
for i, (ds, M) in enumerate(SETS):
    off = (i - 0.5) * w
    lut_b, lut_w = [], []
    for np_ in NP2:
        gb = rows.get(("square", ds, "byte", np_), {})
        gw = rows.get(("square", ds, "word", np_), {})
        lut_b.append(med(gb,2)/med(gb,1) if 1 in gb and 2 in gb else np.nan)
        lut_w.append(med(gw,2)/med(gw,1) if 1 in gw and 2 in gw else np.nan)
    b.bar(xs + off, lut_b, width=w * 0.44, color=DS_COLOR[ds], alpha=0.45,
          ec=DS_COLOR[ds], lw=0.6,
          label="%s, byte layout" % PRETTY[ds].split("-")[0])
    b.bar(xs + off + w * 0.46, lut_w, width=w * 0.44, color=DS_COLOR[ds],
          ec="none", label="%s, packed" % PRETTY[ds].split("-")[0])
b.axhline(1.0, color="0.35", lw=0.7, ls=":")
b.set_xticks(xs); b.set_xticklabels([str(v) for v in NP2])
b.set_xlabel("nprobe")
b.set_ylabel("factorised $\\div$ full table")
b.set_ylim(0.9, 1.72)
b.legend(loc="upper left", fontsize=7, ncol=1, columnspacing=0.9,
         handlelength=1.1, bbox_to_anchor=(0.0, 1.02))
# "barely moves" was drawn from five of six cells; the sixth moves 12.6%,
# and it is the cell carrying the largest gain, so the claim now names the
# range instead of averaging it away.
b.text(0.03, 0.78, "(b)  the two layouts agree\n"
       "      within $6\\%$ in five of six\n"
       "      cells, so the gains mostly\n"
       "      multiply rather than overlap", transform=b.transAxes, fontsize=7,
       ha="left", va="top", color="#52514e", linespacing=1.35)

fig.tight_layout(pad=0.3)
# The numbers Table 2 and Section 6.2 quote from this panel, printed so they
# have a generator instead of an ad-hoc calculation. G*2^(8/G) is 16 for both
# G=4 and G=8, so the pair holds the table footprint fixed.
pairs = []
for key in sorted(rows):
    if key[2] != "word" or key[0] == "square":
        continue          # the granularity phases carry G=4 and G=8
    d = rows[key]
    if 4 in d and 8 in d:
        pairs.append((key[1], key[3], 100.0 * (med(d, 8) / med(d, 4) - 1.0)))
if pairs:
    print("  G=8 against G=4, identical footprint, only lookups differ:")
    for ds, np_, pct in pairs:
        print("     %-14s nprobe=%-5d %+.1f%%" % (ds, np_, pct))
    print("     range %+.1f%% to %+.1f%%"
          % (min(p[2] for p in pairs), max(p[2] for p in pairs)))
    # Panel (a) draws the sweep phase only, so the counts the caption and
    # Section 6.2 quote have to come from the sweep phase only.  Folding the
    # square phase into the same range is what put a -4.2% cell in the text
    # that no line in panel (a) can show.
    def g12(phase):
        return [(k[1], k[3], 100.0 * (med(rows[k], 2) / med(rows[k], 1) - 1.0))
                for k in sorted(rows)
                if k[0] == phase and 1 in rows[k] and 2 in rows[k]
                and (phase != "sweep" or k[2] == "word")]
    sw = g12("sweep")
    if sw:
        print("  panel (a), sweep phase -- G=2 against G=1, %d cells:" % len(sw))
        print("     G=2 wins %d of %d" % (sum(1 for p in sw if p[2] > 0), len(sw)))
        for M in (96, 384):
            v = [p for p in sw if (M == 96) == ("vogue" in p[0])]
            if v:
                pct = [p[2] for p in v]
                print("     M=%d: %+.1f%% to %+.1f%%   by nprobe %s"
                      % (M, min(pct), max(pct),
                         ", ".join("np%d %+.1f%%" % (p[1], p[2])
                                   for p in sorted(v, key=lambda q: q[1]))))
    sq = g12("square")
    if sq:
        neg = [p for p in sq if p[2] <= 0]
        print("  square phase (panel (b)'s cells), %d of %d negative:" % (len(neg), len(sq)))
        for ds, np_, pct in neg:
            same = [p for p in sw if p[0] == ds and p[1] == np_]
            print("     %-14s nprobe=%-4d %+.1f%%%s"
                  % (ds, np_, pct,
                     "   same point in the sweep phase: %+.1f%%" % same[0][2]
                     if same else ""))
        print("  panel (b), factorised/full by layout:")
        for ds in ("vogue-768", "openai3-3072"):
            for np_ in (32, 128, 512):
                v = {}
                for lay in ("byte", "word"):
                    d = rows.get(("square", ds, lay, np_))
                    if d and 1 in d and 2 in d:
                        v[lay] = med(d, 2) / med(d, 1)
                if len(v) == 2:
                    print("     %-14s nprobe=%-4d byte %.3f  packed %.3f  "
                          "gap %+.1f%%" % (ds, np_, v["byte"], v["word"],
                                           100 * (v["word"] / v["byte"] - 1)))

save(fig, "fig_lutgroups")
