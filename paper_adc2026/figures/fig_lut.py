#!/usr/bin/env python3
"""Figure: the Cartesian factorisation of the primary distance table.

Section 3.3.2. Equation 4 builds the K codewords as a Cartesian product of the
per-dimension levels of equation 3, so the high nibble of a code fixes the
first Ds/2 coordinates of its centroid and the low nibble the rest. The table
splits exactly:

    T_m[c] = T_m^hi[c >> 4] + T_m^lo[c & 15]

256 entries a subspace become 16 + 16. The identity is exact for this
codebook and is verified against the centroids at train time, not assumed --
a Lloyd-refined product quantiser does not have the property and the check
throws rather than returning wrong distances.

The right panel now draws the budget the table actually competes for, because
the manuscript claimed the full table fits at M=96 and it does not. The scan
kernel's own test is

    lut_in_smem = (scan_base + M * SPLIT * 4) <= smem_optin

with scan_base = (2*K_LOCAL*BLOCK + 2*BLOCK)*4 = 40*BLOCK bytes of candidate
scratch that is claimed first. At the device's 101,376 B opt-in limit that
leaves 94.0 KiB for the table at BLOCK=128 and 59.0 KiB at BLOCK=1024. The
smallest full table in this paper is 96 KiB, so it misses the most generous of
those by 2 KiB and every factorised table clears the least generous. The
residency threshold in Section 3.3 is real, but the full table is off-chip at
every M, and what scales with M is the cost of the lookups that then miss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDE, 2.45),
                             gridspec_kw={"width_ratios": [1.15, 1]})
for a in (ax, bx):
    a.axis("off")

# ── left: the code splits, the table splits ────────────────────────────────
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.95, "one 8-bit code", ha="center", fontsize=7, color="#52514e")
for i, (x, lab, col) in enumerate([(0.22, "high 4\nbits", "#9ec5f4"),
                                   (0.50, "low 4\nbits", "#f6c9b4")]):
    ax.add_patch(Rectangle((x, 0.74), 0.28, 0.16, fc=col, ec="#52514e", lw=0.7))
    ax.text(x + 0.14, 0.82, lab, ha="center", va="center", fontsize=7,
            linespacing=1.2)

ax.add_patch(FancyArrowPatch((0.34, 0.735), (0.22, 0.60), arrowstyle="-|>",
                             mutation_scale=7, lw=0.8, color="#2a78d6"))
ax.add_patch(FancyArrowPatch((0.66, 0.735), (0.78, 0.60), arrowstyle="-|>",
                             mutation_scale=7, lw=0.8, color="#eb6834"))

for x, n, col, lab in [(0.06, 16, "#9ec5f4", "$T^{\\mathrm{hi}}_m$"),
                       (0.62, 16, "#f6c9b4", "$T^{\\mathrm{lo}}_m$")]:
    for j in range(n):
        ax.add_patch(Rectangle((x + (j % 8) * 0.042, 0.44 - (j // 8) * 0.075),
                               0.038, 0.068, fc=col, ec="#52514e", lw=0.4))
    ax.text(x + 0.168, 0.53, lab, ha="center", fontsize=7)
    ax.text(x + 0.168, 0.26, "16 entries", ha="center", fontsize=7, color="#52514e")

# mathtext has no \; and no \&; \gg and \wedge it does have.
ax.text(0.5, 0.11,
        r"$T_m[c] = T^{\mathrm{hi}}_m[c \gg 4] + T^{\mathrm{lo}}_m[c \wedge 15]$",
        ha="center", fontsize=7.5)
ax.text(0.5, -0.02, "exact for the Eq. 4 codebook;\nchecked against the centroids at train time",
        ha="center", va="bottom", fontsize=7, color="#52514e", style="italic",
        linespacing=1.25)

# ── right: what it costs per query, at the M this work runs ────────────────
bx.set_xlim(0, 1); bx.set_ylim(0, 1)
rows = [("$M{=}96$\n(d=768)", 96), ("$M{=}128$\n(d=1024)", 128),
        ("$M{=}192$\n(d=1536)", 192), ("$M{=}384$\n(d=3072)", 384)]
bx.text(0.0, 0.99, "resident table per query", ha="left", va="top", fontsize=7,
        color="#52514e")
maxkb = 384 * 256 * 4 / 1024

# The band the table has to land inside, across the four block sizes the scan
# allows: 101,376 B opt-in less 40*BLOCK B of candidate scratch.
OPTIN = 101376
budget = sorted((OPTIN - 40 * b) / 1024 for b in (128, 256, 512, 1024))
bx.add_patch(Rectangle((0.24 + 0.72 * budget[0] / maxkb, 0.12),
                       0.72 * (budget[-1] - budget[0]) / maxkb, 0.78,
                       fc="#f2d9d6", ec="none", zorder=0))
for v in (budget[0], budget[-1]):
    bx.plot([0.24 + 0.72 * v / maxkb] * 2, [0.12, 0.90], color="#c0504d",
            lw=0.6, ls=(0, (2, 1.6)), zorder=1)
# One short line on the title row; the caption carries the derivation.
# One short line, right-aligned on the title row: the two-line version at the
# left overlapped the first M row label.
bx.text(1.0, 0.99, u"budget: $%.0f$–$%.0f$ KiB" % (budget[0], budget[-1]),
        ha="right", va="top", fontsize=7, color="#c0504d")
for i, (lab, M) in enumerate(rows):
    y = 0.76 - i * 0.19
    full = M * 256 * 4 / 1024
    fact = M * 32 * 4 / 1024
    bx.text(0.0, y + 0.035, lab, fontsize=7, va="center", linespacing=1.2)
    bx.add_patch(Rectangle((0.24, y + 0.055), 0.72 * full / maxkb, 0.055,
                           fc="#e1e0d9", ec="#898781", lw=0.5))
    bx.add_patch(Rectangle((0.24, y - 0.015), 0.72 * fact / maxkb, 0.055,
                           fc="#2a78d6", ec="none"))
    bx.text(0.24 + 0.72 * full / maxkb + 0.012, y + 0.082,
            f"{full:.0f} KiB", fontsize=7, va="center", color="#52514e")
    bx.text(0.24 + 0.72 * fact / maxkb + 0.012, y + 0.012,
            f"{fact:.0f}", fontsize=7, va="center", color="#2a78d6")
bx.add_patch(Rectangle((0.24, 0.085), 0.03, 0.035, fc="#e1e0d9", ec="#898781", lw=0.5))
bx.text(0.285, 0.102, "full, $256$/subspace", fontsize=7, va="center")
bx.add_patch(Rectangle((0.24, 0.020), 0.03, 0.035, fc="#2a78d6", ec="none"))
bx.text(0.285, 0.037, "factorised, $16{+}16$", fontsize=7, va="center")

fig.tight_layout(pad=0.2)
save(fig, "fig_lut")
