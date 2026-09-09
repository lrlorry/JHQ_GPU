#!/usr/bin/env python3
"""Figure: the refinement budget, and a rule that finds it.

Section 6.3, two panels.

(a) Ranking loss against alpha. The point where it stops improving spans 25x
    across these datasets -- flat from alpha=4 on openai3-3072, still moving
    at 200 on arxiv-768 -- which is why one constant cannot serve all six.
(b) What the rule is worth: QPS at the alpha it picks, over QPS at the fixed
    100, at equal recall. The gain falls as nprobe rises on every dataset
    because alpha sizes the refinement and the scan takes over.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

# (a) ranking loss vs alpha, from the diagnostic sweep
loss = collections.defaultdict(dict)
for ln in open(datafile("alpha_ds.log")):
    m = re.search(r"(\S+)\s+Br=8\s+nlist=\d+\s+np=(\d+)\s+a=([\d.]+)\s+"
                  r"recall=[\d.]+\s+ivf=[\d.]+\s+rank_lost=([\d.]+)", ln)
    if m and int(m.group(2)) == 128:
        loss[m.group(1)][float(m.group(3))] = float(m.group(4))
if not loss:                      # the six-dataset sweep lives in its own log
    for ln in open(datafile("alpha_ds.log")):
        m = re.search(r"(\S+)\s+M=\d+\s+bpd=1\s+np=(\d+)\s+a=([\d.]+)\s+recall=([\d.]+)", ln)
        if m and int(m.group(2)) == 128:
            loss[m.group(1)][float(m.group(3))] = 1.0 - float(m.group(4))

# (b) rule gain vs nprobe, from the 72-run grid
gain = collections.defaultdict(dict)
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"^  RULE\s+(\S+).*np=(\d+)\s+AF_RESULT .*gain=([\d.]+)", ln)
    if m:
        gain[m.group(1)][int(m.group(2))] = float(m.group(3))

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.0))
cyc = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#008300"]
mk = ["o", "s", "^", "D", "x", "v"]

for i, ds in enumerate([d for d in DATASETS if d in loss]):
    xs = sorted(loss[ds])
    a.plot(xs, [loss[ds][x] for x in xs], color=cyc[i], marker=mk[i],
           label=PRETTY[ds])
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"refinement budget $\alpha$")
a.set_ylabel("ranking loss")
a.set_xticks([4, 8, 16, 32, 64, 100, 200])
a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.text(0.03, 0.06, "(a)", transform=a.transAxes, fontsize=8)

for i, ds in enumerate([d for d in DATASETS if d in gain]):
    xs = sorted(gain[ds])
    b.plot(xs, [gain[ds][x] for x in xs], color=cyc[i], marker=mk[i],
           label=PRETTY[ds])
b.axhline(1.0, color="0.4", lw=0.6, ls=":")
b.set_xscale("log", base=2)
b.set_xlabel("nprobe")
b.set_ylabel(r"QPS, rule $\div$ fixed $\alpha{=}100$")
b.set_xticks([8, 32, 128, 256, 512, 1024])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.text(0.03, 0.06, "(b)", transform=b.transAxes, fontsize=8)

h, l = b.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.10),
           columnspacing=1.0)
fig.tight_layout(pad=0.3)
save(fig, "fig_alpha")
