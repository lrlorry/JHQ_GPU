#!/usr/bin/env python3
"""Figure: how the refinement budget is chosen.

Section 4.3. The rule asks the only question the system can answer without
labels -- would a smaller budget change what I return -- and stops at the
first rejection, which is what makes it one-sided: it can pick an alpha larger
than needed, never smaller.

The right panel is one real calibration, openai3-3072 at nprobe=128: three
probes by bisection, alpha=4 accepted, and the batch then runs at twice the
throughput of the fixed 100 at identical recall.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OURS, GREY, GOOD, BAD = "#2a78d6", "#e8e7e1", "#1baf7a", "#e34948"

def box(ax, x, y, w, h, t, fc=GREY, ec="#52514e", fs=6.4, tc="black", lw=0.7):
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
    "answer the same $S$ at a smaller $\\alpha$\ncount top-$k$ slots that differ")
arr(a, 0.52, 0.34, 0.66, 0.34)
box(a, 0.66, 0.42, 0.32, 0.14, "within tolerance:\nkeep going down", fc="#d8f0e4",
    ec=GOOD)
box(a, 0.66, 0.20, 0.32, 0.14, "not within:\nstop, take the last", fc="#fadcda",
    ec=BAD)
arr(a, 0.82, 0.42, 0.82, 0.36, col=GOOD)
a.add_patch(FancyArrowPatch((0.66, 0.49), (0.53, 0.49), arrowstyle="-|>",
                            mutation_scale=6.5, lw=0.7, color=GOOD,
                            connectionstyle="arc3,rad=0.3"))
box(a, 0.02, 0.02, 0.50, 0.14, "run the rest of the batch at $\\alpha^{*}$",
    fc="white", ec=OURS, tc=OURS, lw=1.0)
arr(a, 0.82, 0.20, 0.53, 0.10, col=BAD)
a.text(0.5, 0.975, "no ground truth is used", ha="center", fontsize=6,
       style="italic", color="#52514e")

# ── right: one real calibration ────────────────────────────────────────────
grid = [100, 64, 32, 16, 8, 4, 2]
probes = [(32, 0, True), (8, 0, True), (4, 0, True)]     # alpha, misses, accept
b.set_xscale("log"); b.set_xlim(1.6, 160)
b.set_ylim(-0.6, 2.6)
b.set_yticks(range(3))
b.set_yticklabels(["probe 1", "probe 2", "probe 3"])
b.invert_yaxis()
for i, g in enumerate(grid):
    b.axvline(g, color="0.88", lw=0.5, zorder=0)
for i, (al, miss, ok) in enumerate(probes):
    b.scatter([al], [i], s=46, marker="o" if ok else "X",
              color=GOOD if ok else BAD, zorder=3)
    b.text(al * 1.18, i, f"$\\alpha={al}$, {miss} slots differ",
           fontsize=6, va="center")
b.axvline(4, color=OURS, lw=1.2, ls="--", zorder=2)
b.text(4, -0.5, "  picked $\\alpha^{*}{=}4$", color=OURS, fontsize=6.5, va="top")
b.set_xlabel(r"$\alpha$ grid, walked by bisection")
b.set_xticks(grid)
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.grid(False)
b.text(0.5, 0.06,
       "openai3-3072, nprobe=128, $S{=}32$: 3 probes, 8.9 ms,\n"
       "then $2.0\\times$ the throughput of $\\alpha{=}100$ at identical recall",
       transform=b.transAxes, ha="center", fontsize=6, color="#52514e",
       linespacing=1.4)

fig.tight_layout(pad=0.25)
save(fig, "fig_rule")
