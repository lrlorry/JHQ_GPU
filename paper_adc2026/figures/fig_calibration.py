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

    The grid is walked by bisection, so there is no "stop at the first
    rejection" and no one-sidedness to claim -- see fig_rule. What the rule
    returns is the smallest grid alpha its sampled criterion accepts, bounded
    above by alpha_max. The sample estimates the true saturation point; it
    does not bound it, and arxiv-768 is what the difference looks like.
(b) Sample size, resampled. This panel used to plot one calibration draw per
    S and read a knee off it, which is how the paper came to recommend S=32.
    That recommendation does not survive being measured properly.

    With the full alpha grid dumped a query at a time on a frozen index
    (data/alpha_perquery/), every draw can be replayed offline: sample S
    queries, run the rule on exactly what it would have seen, and score the
    alpha it picks **on the queries it did not see**. 2000 draws a cell,
    alpha_resample.py, no GPU.

    The single draw was lucky. On vogue-768 at nprobe=128 it picked alpha=64
    and gave up 0.0002. Across 2000 draws at the same S=32 the modal pick is
    32, not 64; the mean held-out loss is 0.0033, sixteen times larger; the
    95th percentile is 0.0110, a full point of recall; and only 27% of draws
    land inside 1e-3.

    S=32 is enough on openai3-3072, where the saturation point is alpha=4 and
    76% of draws are inside 1e-3, rising to 98% at S=64. It is not enough on
    vogue-768, where the loss curve is still falling at S=128 and only 89% of
    draws are inside 1e-3 there. The knee is a property of the dataset, and
    the paper has to say S per workload with its measured risk rather than
    print one constant.
(c) Tolerance, on vogue-768 at nprobe=512, all three points from the
    bisecting rule. The reference is 1e-4, not 1e-3: both arms of each
    comparison run on one index with one set of codebooks, so the only thing
    that varies besides alpha is the top-ck tie-break. The 1e-3 figure quoted
    elsewhere is a build-to-build floor and does not apply to a paired
    comparison.
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

# (b) sample size, resampled offline -- see alpha_resample.py
import json as _json
RS = _json.load(open(datafile("alpha_resample.json")))

# (c) tolerance.
#
# alpha_fast.log carries two search modes. JHQ_AS_LINEAR=1 steps the grid one
# value at a time; without it the rule bisects, and bisection is the rule the
# paper describes. An earlier version of this panel matched on "SLOTS" alone,
# which put slots=0 and slots=2 on the bisection runs and slots=1 -- the
# default, the one the paper actually uses -- on the linear run. Three bars,
# two algorithms.
#
# The default tolerance is one slot and is not written in the header, so the
# slots=1 point is the run with neither knob set.
TOL_DS, TOL_NP = "vogue-768", 512
tol, cur = {}, None
for ln in open(datafile("alpha_fast.log")):
    m = re.search(r"--- (\S+) nprobe=(\d+) BLOCK=\d+\s+(.*?) ---", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), m.group(3))
        continue
    m = re.search(r"AF_RESULT picked=(\S+) probes=\d+ recall_picked=(\S+) "
                  r"qps_picked=\d+ recall_max=(\S+) qps_max=\d+ gain=(\S+)", ln)
    if not (m and cur):
        continue
    ds_, np_, flags = cur
    if ds_ != TOL_DS or np_ != TOL_NP or "JHQ_AS_LINEAR" in flags:
        continue
    hit = re.search(r"JHQ_AS_SLOTS=(\d+)", flags)
    slots = int(hit.group(1)) if hit else 1          # 1 is the default
    tol[slots] = (float(m.group(1)),
                  float(m.group(2)) - float(m.group(3)), float(m.group(4)))
assert sorted(tol) == [0, 1, 2], sorted(tol)

# Two measures on one x is two stacked axes, not one axes with two y-scales:
# a twin axis lets the reader compare bar heights that share no unit.
fig = plt.figure(figsize=(WIDE, 2.45))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.16,
                      wspace=0.52)
a = fig.add_subplot(gs[:, 0])
b = fig.add_subplot(gs[:, 1])
c = fig.add_subplot(gs[0, 2])
d = fig.add_subplot(gs[1, 2], sharex=c)

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

# (b) held-out loss against S: the band is the spread over 2000 draws, the
# line their mean. A single draw is one sample from this band, which is what
# the earlier version of this panel was plotting without saying so.
SS = [8, 16, 32, 64, 128]
for ds, np_ in [("vogue-768", 128), ("openai3-3072", 128)]:
    cells = [RS.get("%s|%d|%d" % (ds, np_, S)) for S in SS]
    if any(c is None for c in cells):
        continue
    # a mean of exactly zero has no place on a log axis; clamp it to the floor
    # and mark it, rather than let the line fall off the bottom of the panel
    FLOOR = 1e-4
    mean = [max(c["heldout_loss_mean"], FLOOR) for c in cells]
    p95  = [max(c["heldout_loss_p95"], FLOOR) for c in cells]
    zero = [S for S, c in zip(SS, cells) if c["heldout_loss_mean"] < FLOOR]
    col  = DS_COLOR[ds]
    b.fill_between(SS, mean, p95, color=col, alpha=0.16, lw=0)
    b.plot(SS, mean, color=col, marker=DS_MARK[ds], ms=3.2,
           mew=1.0 if DS_MARK[ds] == "x" else 0.5,
           label="%s, nprobe$=$%d" % (PRETTY[ds].split("-")[0], np_))
    b.plot(SS, p95, color=col, lw=0.7, ls=":")
    if zero:
        b.plot(zero, [FLOOR] * len(zero), color=col, marker="_", ls="",
               ms=6, mew=1.2)
b.axhline(1e-3, color="0.35", lw=0.7, ls=":")
b.text(8.4, 1.15e-3, "$10^{-3}$", fontsize=7, color="0.35", va="bottom")
b.set_xscale("log", base=2); b.set_yscale("log")
b.set_xlabel("calibration sample $S$")
b.set_ylabel("held-out recall given up")
b.set_xticks(SS)
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
b.set_ylim(8e-5, 6e-2)
b.legend(loc="lower left", fontsize=7)
b.text(0.97, 0.95, "line: mean of 2000 draws\ndotted: 95th percentile",
       transform=b.transAxes, ha="right", va="top", fontsize=7,
       color="#898781", linespacing=1.3)
b.text(0.97, 0.05, "(b)", transform=b.transAxes, fontsize=8, ha="right")
b.text(8.4, 1.05e-4, "  exactly 0", fontsize=7, color="#898781",
       va="bottom")

# (c) upper: what the tolerance costs.  lower: what it buys.
xs = sorted(tol)
c.bar(xs, [-tol[x][1] for x in xs], width=0.5, color="#e34948")
c.axhline(1e-4, color="0.35", lw=0.7, ls=":")
c.text(2.45, 1.3e-4, "$10^{-4}$", fontsize=7, color="0.35", ha="right",
       va="bottom")
c.set_ylabel("recall\ngiven up", linespacing=1.2)
c.tick_params(labelbottom=False)
c.set_xlim(-0.55, 2.55)
for x in xs:
    c.text(x, -tol[x][1], r"$\alpha^{*}{=}%.0f$" % tol[x][0], fontsize=7,
           ha="center", va="bottom", color="#52514e")
c.set_ylim(0, max(-tol[x][1] for x in xs) * 1.42)
c.text(0.04, 0.90, "(c) %s, nprobe$=$%d" % (TOL_DS, TOL_NP),
       transform=c.transAxes, fontsize=7.5, va="top")

d.bar(xs, [tol[x][2] for x in xs], width=0.5, color=S["jhq"]["color"])
d.axhline(1.0, color="0.35", lw=0.7, ls=":")
d.set_ylabel(r"QPS $\div$" "\n" r"fixed $\alpha$", linespacing=1.2)
d.set_xlabel("disagreeing slots allowed")
d.set_xticks(xs)
d.set_ylim(0.95, max(tol[x][2] for x in xs) * 1.06)

fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.17)
save(fig, "fig_calibration")
