#!/usr/bin/env python3
"""Figure: the batch size changes who wins.

Section 6.6. Every other number in the paper is one batch on one card, and
IVF-RaBitQ's own paper reports 10^4 on a different card. This is the axis that
answers the objection.

The ratio is not stable in batch and moves in opposite directions on the two
datasets: JHQ's lead over IVF-RaBitQ peaks at batch 128 on openai3-3072 and is
falling by 1024, while on vogue-768 IVF-RaBitQ is twice as fast at batch 32 and
JHQ overtakes only at 1024, still climbing.

The sweep goes down from ~1000 rather than up to 10^4 because the query files
hold about 1000 rows; duplicating them inflates throughput 4-26% through L2
reuse alone (data/qdup.log), so the comparison would be against an artefact.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

d = collections.defaultdict(dict)
for ln in open(datafile("batch_sweep.log")):
    m = re.search(r"^  (JHQ|RaBitQ)\s+(\S+)\s+np=\d+\s+.*batch=(\d+)\s+"
                  r"recall=([\d.]+)\s+qps=(\d+)", ln)
    if m:
        d[(m.group(2), int(m.group(3)))][m.group(1)] = (float(m.group(4)), int(m.group(5)))

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.0))
sets = ["openai3-3072", "vogue-768"]
cyc = {"openai3-3072": "#2a78d6", "vogue-768": "#eb6834"}
mk = {"openai3-3072": "o", "vogue-768": "^"}

# left: absolute throughput, both systems
for ds in sets:
    bs = sorted(k[1] for k in d if k[0] == ds)
    for sysname, ls in (("JHQ", "-"), ("RaBitQ", "--")):
        ys = [d[(ds, x)][sysname][1] for x in bs if sysname in d[(ds, x)]]
        a.plot(bs[:len(ys)], ys, color=cyc[ds], ls=ls, marker=mk[ds],
               lw=1.3 if sysname == "JHQ" else 1.0,
               mfc="none" if sysname == "RaBitQ" else cyc[ds])
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xlabel("batch size"); a.set_ylabel("QPS")
a.set_xticks([32, 64, 128, 256, 512, 1024])
a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.text(0.03, 0.90, "(a)", transform=a.transAxes, fontsize=8)

# right: the ratio, which is the point
for ds in sets:
    bs = sorted(k[1] for k in d if k[0] == ds)
    ys = [d[(ds, x)]["JHQ"][1] / d[(ds, x)]["RaBitQ"][1] for x in bs]
    b.plot(bs, ys, color=cyc[ds], marker=mk[ds], label=PRETTY[ds])
b.axhline(1.0, color="0.35", lw=0.7, ls=":")
b.set_xscale("log", base=2)
b.set_xlabel("batch size")
b.set_ylabel(r"JHQ $\div$ IVF-RaBitQ")
b.set_xticks([32, 64, 128, 256, 512, 1024])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.text(0.03, 0.90, "(b)", transform=b.transAxes, fontsize=8)
b.legend(loc="lower right")

# the solid/dashed convention, stated once
a.plot([], [], color="0.35", ls="-", label="JHQ")
a.plot([], [], color="0.35", ls="--", mfc="none", label="IVF-RaBitQ")
a.legend(loc="upper left")

fig.tight_layout(pad=0.3)
save(fig, "fig_batch")
