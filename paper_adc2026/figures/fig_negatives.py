#!/usr/bin/env python3
"""Figure: two negative results with a mechanism, in full.

Section 6.5. The ablation figure gives every negative one bar; these two earn
the space because each says something the byte counting does not.

(a) The table-free primary distance. At L=2 the levels are +/-a and
    D_P = ||q'||^2 + d a^2 - 2a<q',s_y> is an identity -- recall agrees to
    four decimals in nine of twelve pairs. It cuts per-query shared residency
    from 4d floats to d, tripling occupancy on openai3-3072, and loses 30-52%
    anyway, most at high nprobe where the scan dominates. IVF-RaBitQ's paper
    reports the opposite verdict for the same two kernel shapes on an L40S and
    says only that it "may vary" with bandwidth; this card has twice the
    bandwidth, and both sides are measured here.

(b) Cross-query reuse, as a controlled proxy. Duplicating a query D times makes
    D queries probe exactly the same lists in exactly the same order, which is
    more agreement than real queries clustered by their coarse assignment ever
    show. The gain saturates at D=2 -- what a 96 MB L2 already holding the
    working set looks like -- so the traffic arithmetic overstates what a
    cluster-centric rewrite has to win from reuse.

    It is a proxy, not an upper bound. Duplication holds the schedule fixed
    and varies only how much the queries have in common; a rewrite would also
    change the schedule -- amortising the table build across a group, reordering
    the scan around the list rather than the query -- and those are not on this
    axis. What this measures is the reuse term alone, and the reuse term is
    small.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# (a) v54: LUT against the table-free form
v54 = collections.defaultdict(dict)
for ln in open(datafile("v54.log")):
    m = re.search(r"demo_jhq_(v54_\w+)\s+(\S+)\s+np=(\d+)\s+recall=[\d.]+\s+qps=(\d+)", ln)
    if m:
        v54[(m.group(2), int(m.group(3)))][m.group(1)] = int(m.group(4))

# (b) qdup: gain against duplication factor
qd = collections.defaultdict(dict)
for f in ("qdup.log", "qdup_stella.log"):
    for ln in open(datafile(f)):
        m = re.search(r"^  (\S+)\s+(?:nlist=\d+\s+)?np=(\d+)\s+dup=(\d+)\s+"
                      r"recall=[\d.]+\s+qps=(\d+)", ln)
        if m:
            qd[(m.group(1), int(m.group(2)))][int(m.group(3))] = int(m.group(4))

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.0))

sets = sorted({k[0] for k in v54})
nps = sorted({k[1] for k in v54})
w = 0.8 / len(nps)
cols = ["#9ec5f4", "#5598e7", "#1c5cab"]
for j, np_ in enumerate(nps):
    vals, xs = [], []
    for i, ds in enumerate(sets):
        v = v54.get((ds, np_))
        if v and "v54_lut" in v and "v54_sign" in v:
            vals.append(100 * (v["v54_sign"] / v["v54_lut"] - 1))
            xs.append(i + (j - (len(nps) - 1) / 2) * w)
    a.bar(xs, vals, width=w * 0.92, color=cols[j % 3], label=f"nprobe={np_}")
a.axhline(0, color="0.3", lw=0.7)
a.set_xticks(range(len(sets)))
a.set_xticklabels([PRETTY.get(s, s) for s in sets], rotation=18, ha="right")
a.set_ylabel("QPS change (\\%)")
a.legend(loc="lower left", fontsize=7, ncol=1)
a.grid(axis="x", visible=False)
a.text(0.97, 0.90, "(a) table-free primary distance", transform=a.transAxes,
       ha="right", fontsize=7)

keys = [k for k in sorted(qd) if len(qd[k]) >= 3]
cyc = [DS_COLOR[d] for d in DATASETS]
for i, k in enumerate(keys[:6]):
    ds, np_ = k
    ds_ = sorted(qd[k])
    base = qd[k][ds_[0]]
    b.plot(ds_, [qd[k][x] / base for x in ds_], color=cyc[i % 6],
           marker="osD^vx"[i % 6], label=f"{PRETTY.get(ds, ds)}, np={np_}")
b.axhline(1.0, color="0.35", lw=0.7, ls=":")
b.set_xscale("log", base=2)
b.set_xlabel("query duplication factor $D$")
b.set_ylabel("QPS, relative to $D{=}1$")
b.set_xticks([1, 2, 4, 8])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.legend(loc="lower right", fontsize=7, ncol=2, columnspacing=0.8)
b.text(0.03, 0.90, "(b) cross-query reuse, controlled proxy",
       transform=b.transAxes, fontsize=7)

fig.tight_layout(pad=0.3)
save(fig, "fig_negatives")
