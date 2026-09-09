#!/usr/bin/env python3
"""Figure: where JHQ's resident memory goes, and why it is above IVF-RaBitQ's.

Section 6.7, the companion to fig_cost. IVF-RaBitQ is 22-39% smaller wherever
it builds, and its bits per dimension alone predict only 11% -- 8 against 9.
This is the rest of the gap, and it is not a code-size difference:

* the primary and residual codes are 9 bits a dimension against RaBitQ's 8;
* the coarse centroids are nlist x d floats, the same on both;
* and JHQ carries a per-batch search workspace that IVF-RaBitQ does not need
  at the same size -- the factorised table for every query in the batch, and
  the candidate buffers the exact top-alpha-k selection runs in.

The bars are computed from the index parameters, and the measured total from
cudaMemGetInfo is marked, so the difference between the two is visible rather
than hidden in a residual.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# index shape per dataset: (N, d, M, nlist), Br = 8, batch = 1024, k = 10
SHAPE = {
    "vogue-768":    (932328,   768,  96,  4096),
    "arxiv-768":    (2253000,  768,  96,  8192),
    "bge-m3":       (10091524, 1024, 128, 32768),
    "stella":       (17776615, 1024, 128, 32768),
    "openai3-1536": (999000,   1536, 192, 8192),
    "openai3-3072": (999000,   3072, 384, 4096),
}
BR, BATCH, CK = 8, 1024, 1000

meas = {}
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"^  FIX\s+(\S+)\s+.*vram=([\d.]+)", ln)
    if m:
        meas[m.group(1)] = float(m.group(2))
rq = {}
for ln in open(datafile("vram.log")):
    m = re.search(r"^  (\S+)\s+nlist=\d+\s+vram=([\d.]+)", ln)
    if m:
        rq[m.group(1)] = float(m.group(2))

order = [d for d in DATASETS if d in meas]
parts = [("primary codes", "#2a78d6"), ("residual codes", "#86b6ef"),
         ("corr + ids", "#cde2fb"), ("coarse centroids", "#1baf7a"),
         ("per-batch table", "#eda100"), ("candidate buffers", "#eb6834")]

vals = {p: [] for p, _ in parts}
for ds in order:
    N, d, M, nlist = SHAPE[ds]
    bpv = (d * BR + 7) // 8
    MiB = 1024 ** 2
    vals["primary codes"].append(N * M / MiB)
    vals["residual codes"].append(N * bpv / MiB)
    vals["corr + ids"].append(N * 8 / MiB)
    vals["coarse centroids"].append(nlist * d * 4 / MiB)
    vals["per-batch table"].append(BATCH * M * 32 * 4 / MiB)
    # queries, rotated queries, centroid dots, probe lists, top-ck positions
    # and distances, and the final results -- everything sized by the batch.
    vals["candidate buffers"].append(
        BATCH * (2 * d * 4 + nlist * 4 + 2 * CK * 4 + 10 * 8) / MiB)

y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(WIDE, 2.2))
left = np.zeros(len(order))
for p, c in parts:
    v = np.array(vals[p]) / 1024          # GiB
    ax.barh(y, v, left=left, height=0.5, color=c, label=p, ec="none")
    left += v

for i, ds in enumerate(order):
    ax.scatter([meas[ds] / 1024], [i], marker="|", s=110, color="black",
               zorder=4, lw=1.1)
    if ds in rq:
        ax.scatter([rq[ds] / 1024], [i], marker="|", s=110,
                   color=S["rabitq"]["color"], zorder=4, lw=1.1)

ax.set_yticks(y); ax.set_yticklabels([PRETTY[d] for d in order])
ax.set_xlabel("resident GPU memory (GiB)")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(left) * 1.30)

h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], color="black", marker="|", ls="", ms=8, mew=1.1),
      plt.Line2D([], [], color=S["rabitq"]["color"], marker="|", ls="", ms=8, mew=1.1)]
l += ["JHQ, measured", "IVF-RaBitQ, measured"]
ax.legend(h, l, loc="lower right", fontsize=6, ncol=2, columnspacing=1.0)

fig.tight_layout(pad=0.3)
save(fig, "fig_memory")
