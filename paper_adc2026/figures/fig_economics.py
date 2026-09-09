#!/usr/bin/env python3
"""Figure: what the calibration costs, and when the workload has repaid it.

Section 6.3.3. Every other number for the rule is a steady-state throughput
gain, measured after alpha has been chosen. The choosing is not free: it
answers S=32 of the batch's own queries at several alpha before the batch
runs, and that cost is paid once and recovered over the batches that follow.
Reporting the gain without it would be reporting half a trade.

(a) The relation between the two. Each dataset is one path through its six
    nprobe values, from nprobe=8 at the lower right -- large gain, repaid
    inside the first batch -- to nprobe=1024 at the upper left. The direction
    is the same on all six because both ends of the trade move with nprobe:
    a larger probe list puts more true neighbours in the candidate pool, so
    the saturation budget rises and the gain over a fixed alpha=100 shrinks,
    while the calibration itself gets more expensive (2.0 ms to 47.5 ms) for
    exactly the same reason.

    So the cost is largest where the gain is smallest. That is the honest
    shape of this contribution, and it is better said in a figure than left
    for a reader to derive: the rule is worth running in the regime where
    alpha can fall a long way, and is close to free-and-pointless where it
    cannot.

    arxiv-768 is the case that never repays. At nprobe=128 its gain is 0.996 --
    below one, so there is no saving to recover a cost from -- and at 256, 512
    and 1024 it wants 179, 108 and 58 batches. Its saturation budget is above
    the rule's own alpha_max of 100 (see fig_calibration), so the rule returns
    alpha_max and correctly reports that there is nothing to win.

(b) The other half of the trade: what the chosen alpha costs in recall. The
    band is the +-1e-3 build-to-build floor that three cold-cache runs of one
    binary establish, so anything inside it is not a measured loss. One point
    is outside: bge-m3 at nprobe=1024, at -0.0048. It is reported here rather
    than left in a log.

Both panels read from the 36 RULE rows of paper_fronts.log, which carry
gain, cal_ms and batches_to_repay per configuration. batches_to_repay is
-1 there when the gain is below one; that is a sentinel, not a quantity, and
is drawn in the band above the axis rather than plotted.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

rows = []
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"RULE\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+AF_RESULT "
                  r"picked=(\S+) probes=\d+ recall_picked=(\S+) qps_picked=\d+ "
                  r"recall_max=(\S+) qps_max=\d+ gain=(\S+) cal_ms=(\S+) "
                  r"batches_to_repay=(\S+)", ln)
    if m:
        rows.append(dict(ds=m.group(1), np=int(m.group(2)),
                         alpha=float(m.group(3)),
                         drecall=float(m.group(4)) - float(m.group(5)),
                         gain=float(m.group(6)), cal=float(m.group(7)),
                         repay=float(m.group(8))))
assert len(rows) == 36, len(rows)

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.45))

# ── (a) gain against payback ────────────────────────────────────────────────
# The sentinel lives above a rule, not inside the scale: a run that never
# repays has no payback value, and putting one there would invent a number.
NEVER, TOP = 430.0, 250.0
a.axhline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)
a.axvline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)

for ds in DATASETS:
    pts = sorted([r for r in rows if r["ds"] == ds], key=lambda r: r["np"])
    ok = [r for r in pts if r["repay"] > 0]
    c, mk = DS_COLOR[ds], DS_MARK[ds]
    a.plot([r["gain"] for r in ok], [r["repay"] for r in ok],
           color=c, marker=mk, ms=3.4, lw=0.9, alpha=0.9, zorder=3,
           mew=1.0 if mk == "x" else 0.5)
    for r in pts:
        if r["repay"] <= 0:
            a.plot([r["gain"]], [NEVER], color=c, marker=mk, ms=3.4, ls="",
                   mew=1.0 if mk == "x" else 0.5, zorder=3)
    # Only arxiv-768 is labelled here. The other five converge onto one arc
    # and six labels at the same end collide; identity for all six comes from
    # panel (b)'s legend, which carries the same hue and the same marker --
    # the marker being the secondary encoding the palette check requires for
    # the two openai3 hues, which separate by only dE 7.2 under protanopia.
    if ds == "arxiv-768":
        hi = ok[-1]
        # ink, not the series colour -- the path it sits against carries the
        # identity, and coloured body text is the thing that stops reading as
        # text and starts reading as a mark
        a.annotate("arxiv-768", (hi["gain"], hi["repay"]), fontsize=7,
                   color="#3d3c39", xytext=(7, 1), textcoords="offset points",
                   va="center")

a.set_yscale("log")
a.set_ylim(0.35, 620)
a.set_xlim(0.93, 2.83)
a.axhline(TOP, color="0.45", lw=0.7, ls="-", zorder=2)
a.text(2.78, NEVER, "never repaid: the rule is not faster here,\nso there is no saving to recover a cost from",
       fontsize=7, color="#3d3c39", ha="right", va="center", linespacing=1.3)
a.set_yticks([1, 10, 100])
a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel(r"steady-state throughput, rule $\div$ fixed $\alpha{=}100$")
a.set_ylabel("batches to repay the calibration")
a.text(2.78, 1.42, "repaid within the first batch", fontsize=7,
       color="#898781", ha="right", va="center")
a.text(0.03, 0.14, "(a)", transform=a.transAxes, fontsize=8, va="top")
a.annotate("rising nprobe", xy=(1.18, 26), xytext=(1.62, 30),
           fontsize=7, color="#898781", va="center",
           arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#898781",
                           shrinkA=2, shrinkB=2))

# ── (b) what it cost in recall ──────────────────────────────────────────────
b.axhspan(-1e-3, 1e-3, color="#8f8d86", alpha=0.16, lw=0, zorder=0)
for ds in DATASETS:
    pts = sorted([r for r in rows if r["ds"] == ds], key=lambda r: r["np"])
    b.plot([r["np"] for r in pts], [r["drecall"] for r in pts],
           color=DS_COLOR[ds], marker=DS_MARK[ds], ms=3.4, lw=0.9, alpha=0.9,
           mew=1.0 if DS_MARK[ds] == "x" else 0.5, label=PRETTY[ds])
b.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)
b.set_xscale("log", base=2)
b.set_xticks([8, 32, 128, 256, 512, 1024])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
b.set_xlabel("nprobe")
b.set_ylabel("recall, rule $-$ fixed")
b.set_ylim(-0.0056, 0.0018)
b.text(8.6, 0.00062, "build-to-build floor", fontsize=7, color="#5f5e5a",
       va="center")
bad = min(rows, key=lambda r: r["drecall"])
b.annotate("%s at nprobe=%d gives up\n%.4f of recall: the one real cost"
           % (PRETTY[bad["ds"]], bad["np"], -bad["drecall"]),
           (bad["np"], bad["drecall"]), fontsize=7, color="#3d3c39",
           xytext=(-8, 20), textcoords="offset points", ha="right",
           linespacing=1.3)
b.legend(loc="lower left", fontsize=7, ncol=2, columnspacing=1.0,
         handletextpad=0.5)
b.text(0.03, 0.955, "(b)", transform=b.transAxes, fontsize=8, va="top")

fig.tight_layout(pad=0.3)
save(fig, "fig_economics")
