#!/usr/bin/env python3
"""Figure: which GPU design choices pay, and which do not.

Sections 6.4 and 6.5, two panels.

(a) What each change was worth. The rows are grouped by what kind of claim
    they support, because an earlier version listed them flat and invited the
    reading that they are equally novel:

      * **JHQ-specific.** The factorised Cartesian LUT is exact only because
        the primary codebook is a Cartesian product of one-dimensional levels.
        No change to the code, the codebook or the distances. Its range is the
        widest here and the width is the finding, not noise: -4% at M=96,
        where the full 256-entry table still fits in shared memory, to +53% at
        M=384, where it does not. fig_lutgroups has that against M; a bar can
        only carry the span.
      * **Representation-enabled.** Four subspaces share a 32-bit load only
        because Ds | B at B=8 forces one bit a dimension. A consequence of the
        admissibility rule rather than an independent idea.
      * **Corrections.** These remove unintended work. They are not designs
        and are drawn hatched so no reader has to take the caption's word for
        which is which.

    Adaptive alpha is **not** on this panel. It is a policy result, its gain
    is measured against fixed alpha=100 rather than at matched recall, and it
    carries a recall cost that a bar cannot show. It belongs to Section 6.3
    with fig_alpha and fig_economics, and putting it here made the largest bar
    in the figure the one thing in it that was not a kernel change.

(b) The five changes that did not pay. Four traded a memory saving for shared
    memory or registers; v54's table-free primary distance traded the other
    way, spending instructions to buy occupancy, and lost hardest on the one
    dataset whose occupancy it tripled.

Every bar is a **range across the configurations it was measured on** --
datasets and nprobe -- not a confidence interval. The runs behind different
rows are not the same set, so the widths are comparable in meaning but the
rows are not a single controlled experiment. The 2x2 matched ablation that
would make them one (full/split LUT x byte/packed codes) is not yet run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# label, low %, high %, kind.  kind: "jhq" the representation makes it exact,
# "repr" the admissibility rule enables it, "corr" it removes unintended work.
pos = [
    ("Factorised Cartesian LUT",       -4,   53,   "jhq"),
    ("Packed 32-bit code loads",       30,   48,   "repr"),
    ("Per-thread probe cursor",         2.5, 148,  "corr"),
    ("Launch sized to the batch",       1,    9,   "corr"),
]
neg = [  # label, low %, high %
    ("Table-free primary distance",   -52, -30),
    ("Scan/refine fusion",            -15,   1),
    ("Larger selection buffer",       -26,   4),
    ("Bounded early exit",             -3,   4),
    ("Residual regrouping",            -1,   3),
]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.0))
for ax, rows, col in ((a, pos, "#1baf7a"), (b, neg, "#e34948")):
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = r[1], r[2]
        corr = len(r) > 3 and r[3] == "corr"
        ax.barh(i, hi - lo, left=lo, height=0.55,
                color=col, alpha=0.35 if corr else 0.8,
                hatch="//" if corr else None, edgecolor=col, lw=0.6)
        ax.plot([lo, hi], [i, i], color=col, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_xlabel("QPS change at identical recall (%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

a.text(0.97, 0.06, "(a) changes that paid", transform=a.transAxes,
       ha="right", fontsize=7.5)
b.text(0.03, 0.06, "(b) changes that did not", transform=b.transAxes,
       ha="left", fontsize=7.5)
# Patch proxies, not empty barh: barh([], []) ignores the colour and the
# legend comes out in the default cycle colour, which here read as blue.
from matplotlib.patches import Patch
a.legend(handles=[Patch(facecolor="#1baf7a", alpha=0.8, edgecolor="#1baf7a",
                        label="design"),
                  Patch(facecolor="#1baf7a", alpha=0.35, hatch="//",
                        edgecolor="#1baf7a", label="correction")],
         loc="upper right", fontsize=7, borderpad=0.3, handlelength=1.2)
# the kind of claim each row supports, said on the figure rather than only in
# the caption
KIND = {"jhq": "JHQ-specific", "repr": "enabled by $D_s\\,|\\,B$", "corr": ""}
for i, r in enumerate(pos):
    if KIND[r[3]]:
        a.text(r[2] + 3, i, KIND[r[3]], fontsize=7, va="center",
               color="#52514e")
a.set_xlim(-6, 200)
fig.tight_layout(pad=0.3)
save(fig, "fig_ablation")
