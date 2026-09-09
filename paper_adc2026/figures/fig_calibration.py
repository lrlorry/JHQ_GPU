#!/usr/bin/env python3
"""Figure: the calibration rule, validated against the sweep it replaces.

Section 6.3, the contribution's own evidence. Three panels.

(a) The alpha the rule picks against the alpha an exhaustive sweep finds. On
    the diagonal means it recovered the swept answer without running the
    sweep; above it means it was conservative, which the construction
    guarantees -- it stops at the first rejection, so it can pick larger than
    needed and never smaller.
(b) Sample size. At S=8 openai3-3072 picks alpha=2 and gives up 0.0058 of
    recall; from S=32 on it is exact. The knee, not a chosen constant.
(c) Tolerance. Zero slots leaves the gain on the table, two slots costs 0.0035
    -- outside the 1e-3 floor three cold-cache runs of one binary establish.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# (a) picked vs swept, S=32, from the v55 run
SWEPT = {"openai3-3072": 4, "stella": 16, "vogue-768": 64, "arxiv-768": 200}
picked = []
cur = None
for ln in open(datafile("alpha_sample.log")):
    m = re.search(r"--- (\S+)\s+nprobe=(\d+)\s+BLOCK=\d+\s+S=(\d+)", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), int(m.group(3)))
        continue
    m = re.search(r"AS_RESULT picked=(\S+) recall_picked=(\S+) qps_picked=\d+ "
                  r"recall_max=(\S+)", ln)
    if m and cur and cur[2] == 32:
        picked.append((cur[0], float(m.group(1)),
                       float(m.group(2)) - float(m.group(3))))

# (b) sample size, openai3-3072 and arxiv-768 at nprobe=128
bys = {}
cur = None
for ln in open(datafile("alpha_sample.log")):
    m = re.search(r"--- (\S+)\s+nprobe=(\d+)\s+BLOCK=\d+\s+S=(\d+)", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), int(m.group(3)))
        continue
    m = re.search(r"AS_RESULT picked=(\S+) recall_picked=(\S+) qps_picked=\d+ "
                  r"recall_max=(\S+)", ln)
    if m and cur and cur[1] == 128:
        bys.setdefault(cur[0], {})[cur[2]] = (float(m.group(1)),
                                              float(m.group(2)) - float(m.group(3)))

# (c) tolerance, from the v56 ablation
tol = {}
cur = None
for ln in open(datafile("alpha_fast.log")):
    m = re.search(r"--- (\S+) nprobe=(\d+) BLOCK=\d+\s+(.*?) ---", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), m.group(3))
        continue
    m = re.search(r"AF_RESULT picked=(\S+) probes=\d+ recall_picked=(\S+) "
                  r"qps_picked=\d+ recall_max=(\S+) qps_max=\d+ gain=(\S+)", ln)
    if m and cur and "SLOTS" in cur[2]:
        s = int(re.search(r"SLOTS=(\d+)", cur[2]).group(1))
        tol.setdefault(cur[0], {})[s] = (float(m.group(1)),
                                         float(m.group(2)) - float(m.group(3)),
                                         float(m.group(4)))

fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, 1.95))

# (a)
a.plot([2, 300], [2, 300], color="0.5", lw=0.7, ls=":", zorder=1)
seen = set()
for ds, p, dr in picked:
    if ds not in SWEPT:
        continue
    a.scatter(SWEPT[ds], p, s=26, color=S["jhq"]["color"],
              marker="o" if abs(p - SWEPT[ds]) < 1e-9 else "^",
              zorder=3, label="_")
    if ds not in seen:
        a.annotate(PRETTY[ds].split("-")[0], (SWEPT[ds], p), fontsize=6,
                   xytext=(3, -6), textcoords="offset points")
        seen.add(ds)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$\alpha$ the sweep finds"); a.set_ylabel(r"$\alpha$ the rule picks")
a.set_xticks([4, 16, 64, 200]); a.set_yticks([4, 16, 64, 100])
for ax in (a,):
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.text(0.03, 0.90, "(a)", transform=a.transAxes, fontsize=8)

# (b)
for i, ds in enumerate([d for d in ("openai3-3072", "arxiv-768") if d in bys]):
    xs = sorted(bys[ds])
    col = ["#2a78d6", "#eb6834"][i]
    b.plot(xs, [-bys[ds][x][1] for x in xs], color=col,
           marker=["o", "^"][i], label=PRETTY[ds])
b.axhline(1e-3, color="0.35", lw=0.7, ls=":")
b.text(0.97, 0.30, "build noise", transform=b.transAxes, ha="right", fontsize=6,
       color="0.35")
# Linear, not log: from S=32 on the loss is exactly zero, and a log axis
# cannot draw that -- the lines would fall off the bottom and read as broken.
b.set_xscale("log", base=2)
b.set_xlabel("calibration sample $S$"); b.set_ylabel("recall given up")
b.set_ylim(-3e-4, 6.6e-3)
b.set_xticks([8, 16, 32, 64, 128])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.legend(loc="upper right", fontsize=6.5)
b.text(0.03, 0.90, "(b)", transform=b.transAxes, fontsize=8)

# (c)
ds = "vogue-768"
if ds in tol:
    xs = sorted(tol[ds])
    c2 = c.twinx()
    c.bar([x - 0.15 for x in xs], [-tol[ds][x][1] for x in xs], width=0.3,
          color="#e34948", label="recall given up")
    c2.bar([x + 0.15 for x in xs], [tol[ds][x][2] for x in xs], width=0.3,
           color=S["jhq"]["color"], label="QPS gain")
    c.axhline(1e-3, color="0.35", lw=0.7, ls=":")
    c.set_ylabel("recall given up"); c2.set_ylabel(r"QPS $\div$ fixed $\alpha$")
    c.set_xlabel("disagreeing slots allowed")
    c.set_xticks(xs); c2.grid(False); c2.set_ylim(0.95, 1.35)
    h1, l1 = c.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.5,
             bbox_to_anchor=(0.55, 1.0))
c.text(0.03, 0.90, "(c)", transform=c.transAxes, fontsize=8)

fig.tight_layout(pad=0.3)
save(fig, "fig_calibration")
