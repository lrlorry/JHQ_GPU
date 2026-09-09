#!/usr/bin/env python3
"""Figure: what the second level buys.

Section 6.7. Everything in the GPU design presupposes that the residual level
earns its place -- selective refinement, the alpha budget, the two-level scan
-- so the ablation that drops it is the section's premise, not an extra.

Primary codes alone top out between Recall 0.51 and 0.86, and adding nprobe
does not fix it: arxiv-768 gains 0.0002 going from nprobe 128 to 1024, because
what is missing is resolution, not candidates.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

jq = collections.defaultdict(dict)
for ln in open(datafile("hierarchy_ablation.log")):
    m = re.search(r"^  JQ\s+(\S+)\s+np=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
    if m:
        jq[m.group(1)][int(m.group(2))] = (float(m.group(3)), int(m.group(4)))
jhq = collections.defaultdict(dict)
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"^  FIX\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+a=100 "
                  r"recall=([\d.]+)\s+qps=(\d+)", ln)
    if m:
        jhq[m.group(1)][int(m.group(2))] = (float(m.group(3)), int(m.group(4)))

order = [d for d in DATASETS if d in jq and d in jhq]
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.28))

# left: the ceiling each reaches, over any nprobe
y = np.arange(len(order))
cj = [max(r for r, _ in jq[d].values()) for d in order]
ch = [max(r for r, _ in jhq[d].values()) for d in order]
a.barh(y - 0.19, cj, height=0.36, color=S["jq"]["color"], label="JQ (primary only)")
a.barh(y + 0.19, ch, height=0.36, color=S["jhq"]["color"], label="JHQ (two levels)")
for i, (u, v) in enumerate(zip(cj, ch)):
    a.text(v + 0.008, i + 0.19, f"+{v-u:.3f}", va="center", fontsize=7)
a.set_yticks(y); a.set_yticklabels([PRETTY[d] for d in order])
a.set_xlabel("highest Recall@10 reached"); a.set_xlim(0, 1.16)
a.invert_yaxis(); a.grid(axis="y", visible=False)
# above the axes: the bars fill the panel from 0, so any in-axes position
# covers either a bar or its gain label (frameon is off globally, so an
# opaque frame is not available as a fallback)
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
         fontsize=7, columnspacing=1.2)
a.text(0.03, 0.05, "(a)", transform=a.transAxes, fontsize=8)

# right: the two curves where the ceiling is worst, to show nprobe cannot fix it
for i, ds in enumerate(["arxiv-768", "stella"]):
    if ds not in jq:
        continue
    c = ["#eb6834", "#2a78d6"][i]
    nps = sorted(jq[ds])
    b.plot([jq[ds][n][0] for n in nps], [jq[ds][n][1] for n in nps],
           color=c, marker="v", ls="--", mfc="none", label=f"{PRETTY[ds]}, JQ")
    b.plot([jhq[ds][n][0] for n in nps], [jhq[ds][n][1] for n in nps],
           color=c, marker="o", label=f"{PRETTY[ds]}, JHQ")
b.set_yscale("log"); b.set_xlabel("Recall@10"); b.set_ylabel("QPS")
b.legend(loc="lower left", fontsize=7, borderpad=0.3)
b.text(0.95, 0.90, "(b)", transform=b.transAxes, fontsize=8, ha="right")

fig.tight_layout(pad=0.3)
save(fig, "fig_hierarchy")
