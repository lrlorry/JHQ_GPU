#!/usr/bin/env python3
"""Figure: which GPU design choices pay, and which do not.

Sections 6.4 and 6.5, two panels.

(a) What each change was worth, as a range across the configurations it was
    measured on. The launch-sizing bar is a correction, not a design choice --
    every search ran batch_cap blocks and zero-padded the rest -- and is
    marked as such rather than presented beside the others.
(b) The five changes that did not pay. Four traded a memory saving for shared
    memory or registers; v54's table-free primary distance traded the other
    way, spending instructions to buy occupancy, and lost hardest on the one
    dataset whose occupancy it tripled.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

pos = [  # label, low %, high %, is_correction
    ("Adaptive $\\alpha$",              0,   165, False),
    ("Packed 32-bit code loads",       30,   48,  False),
    ("Per-thread probe cursor",         2.5, 148, False),
    ("Factorised Cartesian LUT",        6,   14.5, False),
    ("Launch sized to the batch",       1,    9,  True),
]
neg = [  # label, low %, high %
    ("Table-free primary distance",   -52, -30),
    ("Scan/refine fusion",            -15,   1),
    ("Larger selection buffer",       -26,   4),
    ("Bounded early exit",             -3,   4),
    ("Residual regrouping",            -1,   3),
]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.1))
for ax, rows, col in ((a, pos, "#1baf7a"), (b, neg, "#e34948")):
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = r[1], r[2]
        corr = len(r) > 3 and r[3]
        ax.barh(i, hi - lo, left=lo, height=0.55,
                color=col, alpha=0.35 if corr else 0.8,
                hatch="//" if corr else None, edgecolor=col, lw=0.6)
        ax.plot([lo, hi], [i, i], color=col, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_xlabel("QPS change at matched recall (\\%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

a.text(0.97, 0.06, "(a) changes that paid", transform=a.transAxes,
       ha="right", fontsize=7.5)
b.text(0.03, 0.06, "(b) changes that did not", transform=b.transAxes,
       ha="left", fontsize=7.5)
a.barh([], [], color="#1baf7a", alpha=0.35, hatch="//", edgecolor="#1baf7a",
       label="a correction, not a design")
a.legend(loc="upper right", fontsize=6.5)
fig.tight_layout(pad=0.3)
save(fig, "fig_ablation")
