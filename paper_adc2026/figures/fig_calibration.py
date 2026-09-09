#!/usr/bin/env python3
"""Figure: the calibration rule, validated against the sweep it replaces.

Section 6.3, the contribution's own evidence. Three panels.

(a) The alpha the rule picks against the alpha an exhaustive sweep finds, at
    the same (dataset, nprobe). Keyed on both: the saturation point is a
    property of the operating point, not of the dataset, so a sweep run at one
    nprobe is not a reference for a rule run at another.

    That restricts the panel to nprobe=128, where alpha6.log swept the full
    grid {4, 8, 16, 32, 64, 100, 200} on all six datasets. The rule was also
    run at nprobe=512, and those four rows are not drawn, because no sweep was
    run at nprobe=512 on those datasets -- there is nothing to compare them
    against. (ALPHA_RULE.md previously reused the nprobe=128 sweep for them;
    that is the reuse this panel now refuses to make.)

    The swept value is the smallest alpha whose ranking loss is within 3e-4 --
    the build-noise floor -- of the loss at the top of the grid. Ranking loss,
    not recall: alpha cannot recover a neighbour whose list was never opened,
    so routing loss does not belong in the quantity alpha is judged on.

    Three of the four are exact. arxiv-768 is the exception and is not a miss
    in the rule's favour or against it: its loss is still falling at alpha=200
    and the rule's own alpha_max is 100, so the rule cannot reach the
    saturation point at all. It returns 100 and reports no gain, which is the
    right behaviour for a budget that is already too small.

    What the construction guarantees is one-sided only with respect to its own
    criterion and its own alpha_max: it stops at the first alpha whose sampled
    top-k disagrees by more than the tolerance, so it cannot return less than
    the smallest alpha that passes the sampled agreement test. The sample
    estimates the true saturation point; it does not bound it, and arxiv-768
    is what the difference looks like.
(b) Sample size. At S=8 openai3-3072 picks alpha=2 and gives up 0.0058 of
    recall; from S=32 on it is exact. The knee, not a chosen constant.
(c) Tolerance. Zero slots leaves the gain on the table, two slots costs 0.0035
    -- outside the 1e-3 floor three cold-cache runs of one binary establish.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

# (a) picked vs swept, S=32, from the v55 run.
# The swept reference, per (dataset, nprobe): the smallest alpha whose ranking
# loss is within the build-noise floor of the loss at the top of the grid.
# alpha6.log is the only sweep with the full grid and the ivf_recall column;
# it covers nprobe 32 and 128, which is what limits this panel to 128.
import collections as _c
_rec = _c.defaultdict(dict)
for ln in open(datafile("alpha6.log")):
    m = re.search(r"(\S+)\s+Br=8\s+nlist=\d+\s+np=(\d+)\s+a=([\d.]+)\s+"
                  r"recall=[\d.]+\s+ivf=[\d.]+\s+rank_lost=([\d.]+)", ln)
    if m:
        _rec[(m.group(1), int(m.group(2)))][float(m.group(3))] = float(m.group(4))
SWEPT, CAPPED = {}, {}
for key, d in _rec.items():
    top = max(d)
    ref = d[top]
    # in units of 1e-4, so the 3e-4 threshold is not decided by float error
    ok = [a for a in sorted(d) if round((d[a] - ref) * 1e4) <= 3]
    if ok:
        SWEPT[key] = min(ok)
        # the grid never flattened: the saturation point is off the right edge
        CAPPED[key] = (min(ok) == top)

picked, GAIN = [], {}
cur = None
for ln in open(datafile("alpha_sample.log")):
    m = re.search(r"--- (\S+)\s+nprobe=(\d+)\s+BLOCK=\d+\s+S=(\d+)", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), int(m.group(3)))
        continue
    m = re.search(r"AS_RESULT picked=(\S+) recall_picked=(\S+) qps_picked=\d+ "
                  r"recall_max=(\S+) qps_max=\d+ gain=([\d.]+)", ln)
    if m and cur and cur[2] == 32:
        picked.append(((cur[0], cur[1]), float(m.group(1)),
                       float(m.group(2)) - float(m.group(3))))
        GAIN[(cur[0], cur[1])] = float(m.group(4))

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

fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, 2.25))

# (a) One row per configuration, sweep marker to rule marker. A scatter
# against the diagonal cannot carry eight labelled points at this width, and
# the diagonal is not the test anyway (see the docstring) -- what matters is
# how far apart the two alphas are and what the difference bought.
rows = sorted([(k, p) for k, p, _ in picked if k in SWEPT],
              key=lambda t: (SWEPT[t[0]], t[0][1]))
for i, (key, p) in enumerate(rows):
    sw = SWEPT[key]
    if sw != p:
        a.plot([sw, p], [i, i], color="0.75", lw=1.0, zorder=1,
               solid_capstyle="round")
    a.scatter([sw], [i], s=26, marker=">" if CAPPED[key] else "|",
              color="0.35", zorder=3, lw=1.2)
    a.scatter([p], [i], s=26, marker="o", color=S["jhq"]["color"], zorder=3)
a.set_yticks(range(len(rows)))
a.set_yticklabels([PRETTY[k[0]].split("-")[0] for k, _ in rows], fontsize=7)
a.set_ylim(-0.7, len(rows) - 0.3)
a.set_xlim(3, 330)
a.grid(axis="y", visible=False)
a.scatter([], [], marker="|", s=26, color="0.35", lw=1.2, label="sweep")
a.scatter([], [], marker=">", s=26, color="0.35", lw=1.2,
          label="sweep, still falling")
a.scatter([], [], marker="o", s=26, color=S["jhq"]["color"], label="rule")
a.legend(loc="lower right", fontsize=7,
         handletextpad=0.4, borderpad=0.3, labelspacing=0.25, framealpha=0)
a.set_xscale("log")
a.set_xlabel(r"$\alpha$ at nprobe$=$128")
a.set_xticks([4, 16, 64, 200])
a.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# the gain each pick bought, as a right-hand tick column rather than as text
# inside the axes, where it collided with the alpha=100 markers
ag = a.twinx()
ag.set_ylim(a.get_ylim())
ag.set_yticks(range(len(rows)))
ag.set_yticklabels([r"$\times$%.2f" % GAIN[k] for k, _ in rows], fontsize=7,
                   color="#52514e")
ag.tick_params(length=0)
for sp in ag.spines.values():
    sp.set_visible(False)
ag.grid(False)

a.text(0.03, 0.93, "(a)", transform=a.transAxes, fontsize=8, va="top")

# (b)
for i, ds in enumerate([d for d in ("openai3-3072", "arxiv-768") if d in bys]):
    xs = sorted(bys[ds])
    col = ["#2a78d6", "#eb6834"][i]
    b.plot(xs, [-bys[ds][x][1] for x in xs], color=col,
           marker=["o", "^"][i], label=PRETTY[ds])
b.axhline(1e-3, color="0.35", lw=0.7, ls=":")
b.text(0.97, 0.30, "build noise", transform=b.transAxes, ha="right", fontsize=7,
       color="0.35")
# Linear, not log: from S=32 on the loss is exactly zero, and a log axis
# cannot draw that -- the lines would fall off the bottom and read as broken.
b.set_xscale("log", base=2)
b.set_xlabel("calibration sample $S$"); b.set_ylabel("recall given up")
b.set_ylim(-3e-4, 6.6e-3)
b.set_xticks([8, 16, 32, 64, 128])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.legend(loc="upper right", fontsize=7)
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
    c.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7,
             bbox_to_anchor=(0.55, 1.0))
c.text(0.03, 0.90, "(c)", transform=c.transAxes, fontsize=8)

fig.tight_layout(pad=0.3)
save(fig, "fig_calibration")
