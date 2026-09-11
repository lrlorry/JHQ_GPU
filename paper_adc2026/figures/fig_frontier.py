#!/usr/bin/env python3
"""Figure: recall-QPS frontier, six datasets, every system on one RTX 5090.

Section 6.2. The dashed grey line is JHQ at the fixed alpha=100 this project
used from its first run to its last; the gap to the solid line is what the
calibration rule is worth.

Each curve ends where its own sweep ended, and nothing here marks a region as
out of reach: a baseline whose highest swept point is below JHQ's has not been
shown to be unable to go higher, only not to have been asked to. Read the
curves where they overlap.

CAGRA fp32 is absent on stella and bge-m3 -- 17.8 M and 10.1 M vectors at 1024
float dimensions do not fit the card -- and IVF-RaBitQ is absent there too,
for the reason given in the paper's setup. Those are the only two claims of
the form "cannot", and both are allocation failures, not sweep endpoints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

fronts, base = load_fronts(), load_baselines()

fig, axes = plt.subplots(3, 2, figsize=(WIDE, 3.85), sharex=True, sharey=True)
for ax, ds in zip(axes.flat, DATASETS):
    b = base[ds]

    # IVF-RaBitQ is not in this figure.  cuVS ships RaBitQ's quantiser without
    # the exact re-ranking its published results depend on -- search_params
    # carries n_probes and mode and nothing else -- and at bits_per_dim=1, the
    # configuration RaBitQ is known for, recall tops out at 0.78 on vogue-768
    # however deep the probe.  Reaching the recalls compared here forces
    # bits_per_dim=8 and a scan eight times more expensive, which is why that
    # baseline measured 2.3x to 4.9x behind cuVS CAGRA-int8 at R=0.90-0.95 --
    # the reverse of RaBitQ's own published ordering.  A baseline losing to
    # the system it reports beating is a configuration error of ours, not a
    # result about RaBitQ, so it is withdrawn rather than reported.
    drawn = []
    for key, pts in (("cagra", b["cagra"]), ("ivfpq", b["ivfpq"])):
        pts = [p for p in pts if p[0] >= 0.85]
        if pts:
            st = {k: v for k, v in S[key].items() if k not in ("label", "marker")}
            ax.plot(*zip(*pts), **st, zorder=2)
            drawn.append((key, pts))
    f = fronts[ds]
    ax.plot(*zip(*f["fix"]), color=S["jhq_fix"]["color"], ls="--", lw=1.0, zorder=3)
    ax.plot(*zip(*f["rule"]), color=S["jhq"]["color"], lw=1.5, zorder=4)
    for key, pts in drawn + [("jhq", f["rule"])]:
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

handles = [plt.Line2D([], [], color=S[k]["color"], marker=S[k].get("marker", ""),
                      ls=S[k].get("ls", "-"),
                      lw=1.5 if k == "jhq" else 1.0, label=S[k]["label"])
           for k in ("jhq", "jhq_fix", "cagra", "ivfpq")]
fig.legend(handles=handles, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.08), columnspacing=1.2)
fig.tight_layout(pad=0.3)
save(fig, "fig_frontier")
