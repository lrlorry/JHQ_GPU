#!/usr/bin/env python3
"""Figure: the JHQ-GPU pipeline.

Section 3.1. Two rows: what happens once at build time, and what happens per
query. The point of the drawing is which stages are GPU-native redesigns and
which are unchanged JHQ semantics, so the two are shaded differently rather
than described in a caption.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

KEEP = "#e8e7e1"   # JHQ semantics, unchanged
GPU  = "#cde2fb"   # redesigned for the GPU
OURS = "#2a78d6"   # this work's contribution

def box(ax, x, y, w, h, text, fc, ec="#52514e", fs=6.6, tc="black", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.25)

def arrow(ax, x1, y1, x2, y2, style="-|>", col="#52514e", lw=0.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=7, lw=lw, color=col, zorder=1))

fig, ax = plt.subplots(figsize=(WIDE, 2.15))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ── build ───────────────────────────────────────────────────────────────────
ax.text(0.005, 0.93, "index construction", fontsize=7, style="italic", color="#52514e")
bw, bh, by = 0.148, 0.20, 0.66
xs = [0.02, 0.20, 0.38, 0.56, 0.74]
labels = [("JL transform\n$Y = X\\Pi^{\\!\\top}$", GPU),
          ("primary code\nEq. 4, Cartesian", KEEP),
          ("residual code\n1-D $k$-means", KEEP),
          ("IVF assignment", KEEP),
          ("subspace-major\nlayout $[N,M]\\!\\to\\![M,N]$", GPU)]
for x, (t, c) in zip(xs, labels):
    box(ax, x, by, bw, bh, t, c)
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + bw, by + bh / 2, xs[i + 1], by + bh / 2)

# ── query ───────────────────────────────────────────────────────────────────
ax.text(0.005, 0.50, "per query", fontsize=7, style="italic", color="#52514e")
qy = 0.20
qxs = [0.02, 0.175, 0.335, 0.515, 0.695, 0.855]
qw = 0.128
qlabels = [("JL transform\n$q' = \\Pi q$", GPU),
           ("IVF probe\nselection", GPU),
           ("factorised table\n$256 \\to 16{+}16$", GPU),
           ("coalesced scan\nexact top-$\\alpha k$", GPU),
           ("residual\nrefinement", KEEP),
           ("top-$k$", KEEP)]
for x, (t, c) in zip(qxs, qlabels):
    box(ax, x, qy, qw, bh, t, c)
for i in range(len(qxs) - 1):
    arrow(ax, qxs[i] + qw, qy + bh / 2, qxs[i + 1], qy + bh / 2)

# the calibration feeds alpha into the scan stage
box(ax, 0.515, 0.005, 0.128, 0.115,
    "calibration\npicks $\\alpha$", "white", ec=OURS, tc=OURS, fs=6.2, lw=1.1)
arrow(ax, 0.579, 0.12, 0.579, qy, col=OURS, lw=1.1)

# the layout feeds the scan
arrow(ax, 0.814, by, 0.60, qy + bh, style="-|>", col="#898781")
ax.text(0.72, 0.505, "codes read here", fontsize=5.8, color="#898781",
        ha="center")

# legend
for i, (c, t, ec) in enumerate([(KEEP, "JHQ semantics, unchanged", "#52514e"),
                                (GPU, "redesigned for the GPU", "#52514e"),
                                ("white", "this work's policy", OURS)]):
    box(ax, 0.285 + i * 0.245, 0.945, 0.020, 0.045, "", c, ec=ec,
        lw=1.1 if ec == OURS else 0.7)
    ax.text(0.325 + i * 0.245, 0.9675, t, fontsize=6, va="center")

fig.tight_layout(pad=0.15)
save(fig, "fig_pipeline")
