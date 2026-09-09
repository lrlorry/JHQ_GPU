#!/usr/bin/env python3
"""Figure: how the refinement budget is chosen, and one real calibration.

Section 4.3. The rule asks the only question the system can answer without
labels -- would a smaller budget change what I return? -- on S of the batch's
own queries, and it walks the alpha grid by **bisection**, not by stepping
down one value at a time.

That distinction was wrong in an earlier version of this figure, and it
matters twice.

It matters for cost: three probes settle a seven-point grid instead of up to
seven, which is where the calibration time in Section 6.3.3 comes from.

And it matters for what can be claimed. "Stop at the first rejection, so the
choice is one-sided" describes a linear scan and is not what happens: on
vogue-768 the first probe is *rejected* and the search continues upward
(16 reject, 64 accept, 32 reject, picked 64). What bisection returns is the
smallest grid alpha whose sampled top-k agrees within the slot tolerance,
**assuming the accept predicate is monotone in alpha**.

That assumption is not free, but it is arguable rather than hoped for: exact
top-ck selection gives nested candidate sets, so ck1 < ck2 implies the top-ck1
set is contained in the top-ck2 set, the refined top-k therefore moves
monotonically toward the alpha_max answer, and the count of differing slots is
non-increasing in alpha. Ties aside, the predicate is monotone and the
bisection is sound. The paper should say this rather than assert one-sidedness.

The right panel is one real calibration read from data/alpha_fast.log -- no
value in it is written into this script.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import re
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OURS, GREY, GOOD, BAD = "#2a78d6", "#e8e7e1", "#1baf7a", "#e34948"

def box(ax, x, y, w, h, t, fc=GREY, ec="#52514e", fs=7, tc="black", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.3)

def arr(ax, x1, y1, x2, y2, col="#52514e", lw=0.7, t=None, ts=5.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=6.5, lw=lw, color=col, zorder=1))
    if t:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.022, t, fontsize=ts,
                ha="center", color=col)

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.0),
                           gridspec_kw={"width_ratios": [1.05, 1]})
a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)

box(a, 0.06, 0.80, 0.42, 0.15,
    "take $S$ of the batch's own queries", fc="white", ec=OURS, tc=OURS, lw=1.0)
arr(a, 0.27, 0.80, 0.27, 0.70)
box(a, 0.02, 0.53, 0.50, 0.16,
    "answer them at a generous $\\alpha_{\\max}$\nkeep that as the reference")
arr(a, 0.27, 0.53, 0.27, 0.43)
box(a, 0.02, 0.26, 0.50, 0.16,
    "bisect the grid: answer the same $S$ at the\nmidpoint, count top-$k$ slots that differ")
arr(a, 0.52, 0.34, 0.66, 0.34)
box(a, 0.66, 0.42, 0.32, 0.14, "within tolerance:\nsearch below it", fc="#d8f0e4",
    ec=GOOD)
box(a, 0.66, 0.20, 0.32, 0.14, "not within:\nsearch above it", fc="#fadcda",
    ec=BAD)
arr(a, 0.82, 0.42, 0.82, 0.36, col=GOOD)
a.add_patch(FancyArrowPatch((0.66, 0.49), (0.53, 0.49), arrowstyle="-|>",
                            mutation_scale=6.5, lw=0.7, color=GOOD,
                            connectionstyle="arc3,rad=0.3"))
box(a, 0.02, 0.02, 0.50, 0.14, "run the rest of the batch at $\\alpha^{*}$",
    fc="white", ec=OURS, tc=OURS, lw=1.0)
a.add_patch(FancyArrowPatch((0.66, 0.27), (0.53, 0.31), arrowstyle="-|>",
                            mutation_scale=6.5, lw=0.7, color=BAD,
                            connectionstyle="arc3,rad=-0.3"))
arr(a, 0.27, 0.26, 0.27, 0.17, col=OURS)
a.text(0.5, 0.975, "no ground truth is used; $\\log_2$ probes, not a scan",
       ha="center", fontsize=7,
       style="italic", color="#52514e")

# ── right: one real calibration, parsed, never typed ───────────────────────
# An earlier version hard-coded (32, 8, 4), all accepted, and "8.9 ms". The
# run it claimed to show probed 16, 4, 2 and took 56.8 ms. Nothing here is
# written by hand any more.
SHOW = "openai3-3072"
grid = [100, 64, 32, 16, 8, 4, 2]
# alpha_fast.log holds six runs for this dataset and nprobe: the default one,
# two with JHQ_AS_LINEAR=1 (the linear variant, kept for the tolerance
# ablation) and three at other slot tolerances. Only the default is the rule
# this figure describes, so the header must carry S=32 and neither knob.
probes, picked, cal_ms, cur, done = [], None, None, False, False
for ln in open(datafile("alpha_fast.log")):
    if "---" in ln:
        cur = (("%s nprobe=128" % SHOW) in ln and "JHQ_AS_SAMPLE=32" in ln
               and "JHQ_AS_LINEAR" not in ln and "JHQ_AS_SLOTS" not in ln
               and not done)
        continue
    if not cur:
        continue
    m = re.search(r"alpha=(\d+)\s+miss=(\d+)\s+(accept|reject)", ln)
    if m:
        probes.append((int(m.group(1)), int(m.group(2)), m.group(3) == "accept"))
    m = re.search(r"picked alpha = (\d+)\s+\((\d+) probes, ([\d.]+) ms\)", ln)
    if m:
        picked, cal_ms = int(m.group(1)), float(m.group(3))
        cur, done = False, True
assert probes and picked, "no calibration trace for %s" % SHOW
assert len(probes) == 3 and picked == 4, (probes, picked)

b.set_xscale("log"); b.set_xlim(1.6, 190)
b.set_ylim(-0.6, len(probes) - 0.4)
b.set_yticks(range(len(probes)))
b.set_yticklabels(["probe %d" % (i + 1) for i in range(len(probes))])
b.invert_yaxis()
for g in grid:
    b.axvline(g, color="0.88", lw=0.5, zorder=0)
# the bracket bisection is searching in, so a rejected first probe reads as
# part of a search rather than as a failure
lo, hi = min(grid), max(grid)
for i, (al, miss, ok) in enumerate(probes):
    b.plot([lo, hi], [i, i], color="0.88", lw=3.0, solid_capstyle="butt",
           zorder=1)
    b.scatter([al], [i], s=46, marker="o" if ok else "X",
              color=GOOD if ok else BAD, zorder=3)
    b.text(al * 1.20, i, "$\\alpha={%d}$, %d slot%s differ%s" %
           (al, miss, "" if miss == 1 else "s", "s" if miss == 1 else ""),
           fontsize=7, va="center")
    if ok:
        hi = al
    else:
        lo = al
b.axvline(picked, color=OURS, lw=1.2, ls="--", zorder=2)
b.text(picked, -0.55, "  picked $\\alpha^{*}{=}%d$" % picked, color=OURS,
       fontsize=7, va="top")
b.set_xlabel(r"$\alpha$ grid, walked by bisection")
b.set_xticks(grid)
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
b.grid(False)
b.text(0.02, 0.03, "(b) %s, nprobe$=$128, $S{=}32$: %d probes, %.1f ms"
       % (SHOW, len(probes), cal_ms), transform=b.transAxes, fontsize=7,
       color="#52514e")

fig.tight_layout(pad=0.25)
save(fig, "fig_rule")
