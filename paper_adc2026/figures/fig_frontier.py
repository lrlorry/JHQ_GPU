#!/usr/bin/env python3
"""System frontier at fixed alpha=100, with same-pool calibration for context.

The solid JHQ curve supports the primary system comparison. The dashed curve
uses S=32 calibration drawn from the evaluated query pool and excludes its
calibration cost; it is not evidence of quality on subsequent unseen queries.
The separate budget experiment measures held-out calibration at S=128.
Endpoints are measured sweep endpoints, not architectural recall ceilings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

fronts, base = load_fronts(), load_baselines()

# 3x2 at 3.85in tall took 55% of a text page and left its page with 32 lines of
# body text where a dense page carries 45.  The same six panels as 2x3 are
# wider and much shorter; nothing about the data changes.
fig, axes = plt.subplots(2, 3, figsize=(WIDE, 2.05), sharex=True, sharey=True)
for ax, ds in zip(axes.flat, DATASETS):
    b = base[ds]

    # The archived IVF-RaBitQ configuration needs validation against the
    # published GPU artifact before inclusion. Its GPU paper refines with
    # additional quantized bits, without raw-vector exact reranking; missing
    # exact reranking is not a valid explanation for withdrawing these rows.
    # A different ordering against CAGRA also does not establish a harness bug.
    drawn = []
    for key, pts in (("cagra", b["cagra"]), ("ivfpq", b["ivfpq"])):
        pts = [p for p in pts if p[0] >= 0.85]
        if pts:
            st = {k: v for k, v in S[key].items() if k not in ("label", "marker")}
            ax.plot(*zip(*pts), **st, zorder=2)
            drawn.append((key, pts))
    f = fronts[ds]
    # Same blue as the solid curve, not a second one.  Colour names the system
    # and line style the configuration; two hues for two settings of one system
    # read as two systems, and #6e8eab was close enough to JHQ's blue to be
    # neither.
    ax.plot(*zip(*f["rule"]), color=S["jhq"]["color"], ls="--", lw=1.1,
            alpha=0.75, zorder=3)
    ax.plot(*zip(*f["fix"]), color=S["jhq"]["color"], lw=1.5, zorder=4)
    for key, pts in drawn + [("jhq", f["fix"])]:
        ax.plot(*zip(*pts), ls="none", marker=S[key]["marker"], ms=4.2,
                color=S[key]["color"], mec="white", mew=0.7, zorder=6)

    ax.set_yscale("log")
    ax.set_xlim(0.85, 1.005)
    ax.set_ylim(2e3, 1.3e6)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    # Dataset name and shape inside the axes: six panels do not have room for
    # six titles above them.
    ax.text(0.03, 0.05, f"{PRETTY[ds]}\n{b['N']/1e6:.3f}M $\\times$ {b['d']}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
            linespacing=1.3)

for ax in axes[-1]:          # bottom row, whatever the grid shape is
    ax.set_xlabel("Recall@10")
for ax in axes[:, 0]:
    ax.set_ylabel("QPS")

# Four entries on one row.  At ncol=2 the second label wrapped into the first
# row's rule and the dashed handle was drawn through its own text; both JHQ
# labels are short enough now that one row fits, and the caption carries what
# "calibrated" means.
handles = [plt.Line2D([], [], color=S["jhq"]["color"], marker="o", lw=1.5,
                     label=r"JHQ-GPU, $\alpha{=}100$"),
           plt.Line2D([], [], color=S["jhq"]["color"], ls="--", lw=1.1,
                     alpha=0.75, label="JHQ-GPU, calibrated")]
handles += [plt.Line2D([], [], color=S[k]["color"], marker=S[k]["marker"],
                      ls=S[k].get("ls", "-"), lw=1.0, label=S[k]["label"])
            for k in ("cagra", "ivfpq")]
fig.legend(handles=handles, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.045), columnspacing=1.0,
           handlelength=1.6, fontsize=7)
fig.tight_layout(pad=0.3)
save(fig, "fig_frontier")
