#!/usr/bin/env python3
"""Figure: the physical layout of the primary codes, and what a warp reads.

Section 3.2.3 and 3.3.3. Two changes to the same bytes, neither of which
touches a distance.

(a) Candidate-major to subspace-major. Stored [N, M], the 32 threads of a warp
    scanning subspace m read one byte each, M bytes apart -- 32 separate
    transactions. Stored [M, N] the same 32 bytes are adjacent.

(b) Four subspaces in one 32-bit word. Ds = B = 8 makes the primary code one
    bit a dimension, so four subspaces are 32 dimensions and fit a uint32. A
    warp then asks for 32 consecutive words: one 128-byte transaction with
    nothing discarded, where a byte request filled 32 bytes of a 128-byte line
    and threw three quarters away.

Measured at +30% to +48% (../data/, results/front6/v52.log). No candidate's
distance changes; the same codes are read in a different order.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import Rectangle, FancyArrowPatch

HOT, COLD, WORD = "#2a78d6", "#e8e7e1", "#f6c9b4"

def cell(ax, x, y, w, h, fc, ec="#898781", lw=0.35, t=None, fs=4.6, tc="black"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=2))
    if t:
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.35))
for ax in (a, b):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── (a) the transpose ───────────────────────────────────────────────────────
cw, ch = 0.052, 0.075
a.text(0.22, 0.955, "stored $[N, M]$", ha="center", fontsize=7)
a.text(0.78, 0.955, "stored $[M, N]$", ha="center", fontsize=7)
for r in range(4):                      # 4 candidates x 5 subspaces
    for c in range(5):
        hot = (c == 1)
        cell(a, 0.02 + c * cw, 0.78 - r * ch, cw, ch,
             HOT if hot else COLD, t=f"$c_{{{r}{c}}}$",
             tc="white" if hot else "#52514e")
    a.text(0.02 + 5 * cw + 0.012, 0.78 - r * ch + ch / 2, f"$x_{r}$",
           fontsize=5.6, va="center", color="#52514e")
a.text(0.155, 0.455, "one candidate is a row;\nthe warp strides by $M$",
       ha="center", fontsize=6, color="#52514e", linespacing=1.3)

a.add_patch(FancyArrowPatch((0.34, 0.63), (0.46, 0.63), arrowstyle="-|>",
                            mutation_scale=8, lw=0.9, color="#52514e"))
a.text(0.40, 0.665, "transpose", ha="center", fontsize=6, color="#52514e")

for r in range(5):                      # 5 subspaces x 4 candidates
    for c in range(4):
        hot = (r == 1)
        cell(a, 0.55 + c * cw, 0.80 - r * ch, cw, ch,
             HOT if hot else COLD, t=f"$c_{{{c}{r}}}$",
             tc="white" if hot else "#52514e")
    a.text(0.55 + 4 * cw + 0.012, 0.80 - r * ch + ch / 2, f"$m_{r}$",
           fontsize=5.6, va="center", color="#52514e")
a.text(0.66, 0.355, "one subspace is a row;\nthe warp reads 32 adjacent bytes",
       ha="center", fontsize=6, color="#52514e", linespacing=1.3)
a.text(0.03, 0.045, "(a) candidate-major $\\to$ subspace-major",
       fontsize=7, color="black")

# ── (b) four subspaces in one word ──────────────────────────────────────────
b.text(0.5, 0.955, "$D_s = B = 8$: one bit a dimension", ha="center", fontsize=7)
bw = 0.115
for i in range(4):
    cell(b, 0.14 + i * bw, 0.70, bw * 0.94, 0.11, WORD,
         t=f"$m_{{{i}}}$: 8 bits", fs=5.6)
b.add_patch(Rectangle((0.135, 0.685), 4 * bw + 0.005, 0.14, fill=False,
                      ec="#eb6834", lw=1.1, zorder=4))
b.text(0.60 + 0.02, 0.755, "one $\\mathtt{uint32}$", fontsize=6.5,
       va="center", color="#eb6834")
b.text(0.5, 0.635, "= 32 dimensions", ha="center", fontsize=6, color="#52514e")

for y, lab, n, col, note in [
        (0.40, "byte request", 32, COLD,
         "warp asks 32 bytes; the line is 128,\nso three quarters is discarded"),
        (0.12, "word request", 32, HOT,
         "warp asks 32 words = 128 bytes;\none transaction, nothing discarded")]:
    for j in range(16):
        w = 0.036
        cell(b, 0.09 + j * w, y, w * 0.9, 0.075,
             col if (col is HOT or j < 4) else COLD,
             ec="#898781")
    if col is COLD:
        b.add_patch(Rectangle((0.088, y - 0.008), 4 * 0.036, 0.091, fill=False,
                              ec="#e34948", lw=1.0, zorder=4))
    b.text(0.09, y + 0.10, lab, fontsize=6.2)
    b.text(0.68, y + 0.035, note, fontsize=5.8, color="#52514e",
           va="center", linespacing=1.3)
b.text(0.03, 0.02, "(b) four subspaces in one 32-bit load  ($+30\\%$ to $+48\\%$)",
       fontsize=7, color="black")

fig.tight_layout(pad=0.2)
save(fig, "fig_layout")
