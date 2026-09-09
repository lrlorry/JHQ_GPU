#!/usr/bin/env python3
"""Figure: batch sensitivity at fixed nprobe.

Section 6.6. Every other number in the paper is one batch on one card, and
IVF-RaBitQ's own paper reports 10^4 on a different card. This is the axis that
answers the objection.

Both systems run at nprobe=128 throughout; the recall each reaches there is
whatever it reaches, and is reported in panel (b). This is a sensitivity plot,
not an iso-recall comparison -- the ratio is a throughput ratio at one
operating point, and only panel (b)'s recall line says how comparable the two
operating points are.

They are close enough to read on openai3-3072 (JHQ 0.9414, IVF-RaBitQ 0.9405
to 0.9450, so the ratio is if anything slightly generous to JHQ) and on
vogue-768 they favour IVF-RaBitQ (JHQ 0.9645, IVF-RaBitQ 0.9549 to 0.9592, so
JHQ is being timed at a higher recall than the system it is divided by).

The ratio is not stable in batch and moves in opposite directions on the two
datasets: JHQ's lead peaks at batch 128 on openai3-3072 and is falling by
1024, while on vogue-768 IVF-RaBitQ is twice as fast at batch 32 and JHQ
overtakes only at 1024, still climbing.

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

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.15))
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

# What the ratio is a ratio *of*: the recall each side reaches at nprobe=128.
# Without this the panel reads as an iso-recall speedup, which it is not.
notes = []
for ds in sets:
    r_j = [d[k]["JHQ"][0] for k in d if k[0] == ds]
    r_r = [d[k]["RaBitQ"][0] for k in d if k[0] == ds]
    notes.append(u"%s: recall %.3f vs %.3f\u2013%.3f"
                 % (PRETTY[ds].split("-")[0], sum(r_j) / len(r_j),
                    min(r_r), max(r_r)))
b.text(0.97, 0.04, "at nprobe$=$128 throughout\n" + "\n".join(notes),
       transform=b.transAxes, ha="right", va="bottom", fontsize=7,
       color="#52514e", linespacing=1.35)
b.legend(loc="upper right", fontsize=7)

# the solid/dashed convention, stated once
a.plot([], [], color="0.35", ls="-", label="JHQ")
a.plot([], [], color="0.35", ls="--", mfc="none", label="IVF-RaBitQ")
a.legend(loc="lower right", fontsize=7)

fig.tight_layout(pad=0.3)
save(fig, "fig_batch")
