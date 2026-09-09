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

    Four configurations are not on this panel at all, and the reason matters.
    On arxiv-768 at nprobe 128, 256, 512 and 1024 the rule returns alpha=100,
    which is alpha_max, which is also what the fixed arm runs. The two arms are
    then *the same configuration*: demo_jhq_alpha_fast.cu times full(picked)
    and full(grid[0]) separately, three repeats each, so with picked == grid[0]
    the reported gain is the ratio of two 3-run means of one thing. It comes
    out 0.996, 1.003, 1.006 and 1.012 -- repeat-timing variation of about 1.2%
    -- and batches_to_repay then divides the calibration cost by that noise,
    giving 179, 108 and 58. Those are values of 1/noise, not economics, and
    the recall delta on all four is exactly 0.0000, which is the same fact
    seen from the other side.

    So they are excluded, and the correct statement about them is not "slow to
    repay" but **the budget did not change**: arxiv-768's saturation point is
    above the rule's own alpha_max (see fig_calibration), the rule returns
    alpha_max, and there is no steady-state saving for a calibration cost to
    be recovered from.

(b) The other half of the trade: what the chosen alpha costs in recall.

    The band is +-1e-4, and choosing it correctly matters. The 1e-3 figure
    quoted elsewhere in this project is a *build-to-build* floor -- three cold
    starts of one binary give 0.9852 / 0.9842 / 0.9849 because training is not
    reproducible and the codebooks differ in their last bits. That floor does
    not apply here. Both arms of this comparison run on one `idx` object with
    one set of trained codebooks and one assignment, so the difference is
    paired: nothing varies but alpha. The only residual wobble is the top-ck
    tie-break, which is order 1e-4.

    So most of these deltas are real costs, not noise, and the figure names
    the largest rather than calling it the only one. Once the floor is 1e-4,
    openai3-1536 at -0.0028 and bge-m3 across nprobe >= 128 are costs too. The
    honest summary is a range: the rule gives up between 0.0000 and 0.0048 of
    recall, it is largest where nprobe is largest, and the four runs where the
    budget did not change give up exactly nothing.

Over the 32 configurations where the rule did change the budget, payback runs
from 0.5 to 21.0 batches, median 1.75, and the gain is above one in every one
of them (1.044x to 2.653x). The single sub-one gain in the log, 0.996, was one
of the four self-comparisons; removing them removes it.

One caveat this figure cannot remove: the anticorrelation in (a) is largely
definitional. With g the gain, B* = T_cal / (T_fixed - T_rule) =
(T_cal/T_rule) / (g - 1), so B* diverges as g approaches 1 by construction.
What is measured rather than implied is the level -- how large T_cal/T_rule
actually is, and it is small: a median of 1.75 batches means the calibration
costs less than two batches of search even after the gain is taken into
account. Read the panel for the levels, not for the shape.

Both panels read the 36 RULE rows of paper_fronts.log.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

ALL = []
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"RULE\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+AF_RESULT "
                  r"picked=(\S+) probes=\d+ recall_picked=(\S+) qps_picked=\d+ "
                  r"recall_max=(\S+) qps_max=\d+ gain=(\S+) cal_ms=(\S+) "
                  r"batches_to_repay=(\S+)", ln)
    if m:
        ALL.append(dict(ds=m.group(1), np=int(m.group(2)),
                         alpha=float(m.group(3)),
                         drecall=float(m.group(4)) - float(m.group(5)),
                         gain=float(m.group(6)), cal=float(m.group(7)),
                         repay=float(m.group(8))))
assert len(ALL) == 36, len(ALL)

# alpha_max is grid[0] in the rule, and the fixed arm runs alpha=100. A row
# whose pick equals it compared a configuration with itself, so its gain is
# repeat-timing variation and its payback is 1/that. Those rows say something
# real -- the budget did not change -- but they are not economics.
ALPHA_MAX = 100.0
rows = [r for r in ALL if r["alpha"] != ALPHA_MAX]
same = [r for r in ALL if r["alpha"] == ALPHA_MAX]
assert len(rows) == 32 and len(same) == 4, (len(rows), len(same))
assert all(abs(r["drecall"]) < 1e-9 for r in same)   # same config, same recall

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.45))

# ── (a) gain against payback ────────────────────────────────────────────────
# The sentinel lives above a rule, not inside the scale: a run that never
# repays has no payback value, and putting one there would invent a number.
a.axhline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)

for ds in DATASETS:
    pts = sorted([r for r in rows if r["ds"] == ds], key=lambda r: r["np"])
    ok = pts
    if not ok:
        continue
    c, mk = DS_COLOR[ds], DS_MARK[ds]
    a.plot([r["gain"] for r in ok], [r["repay"] for r in ok],
           color=c, marker=mk, ms=3.4, lw=0.9, alpha=0.9, zorder=3,
           mew=1.0 if mk == "x" else 0.5)
    # Only arxiv-768 is labelled here. The other five converge onto one arc
    # and six labels at the same end collide; identity for all six comes from
    # panel (b)'s legend, which carries the same hue and the same marker --
    # the marker being the secondary encoding the palette check requires for
    # the two openai3 hues, which separate by only dE 7.2 under protanopia.
    if ds == "arxiv-768":
        hi = ok[-1]   # nprobe=32; beyond it the budget stops changing
        # ink, not the series colour -- the path it sits against carries the
        # identity, and coloured body text is the thing that stops reading as
        # text and starts reading as a mark
        a.annotate("arxiv-768 stops here", (hi["gain"], hi["repay"]),
                   fontsize=7, color="#3d3c39", xytext=(8, 9),
                   textcoords="offset points", va="center")

a.set_yscale("log")
a.set_ylim(0.35, 46)
a.set_xlim(1.0, 2.83)
a.text(2.78, 41,
       "4 of the 36 runs are not here: on arxiv-768 at\n"
       "nprobe $\\geq$ 128 the rule returns $\\alpha_{\\max}$, which is what\n"
       "the fixed arm runs, so the two arms are one\n"
       "configuration and there is no budget change to\n"
       "repay a cost with",
       fontsize=7, color="#3d3c39", ha="right", va="top", linespacing=1.35)
a.set_yticks([1, 10])
a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel(r"steady-state throughput, rule $\div$ fixed $\alpha{=}100$")
a.set_ylabel("batches to repay the calibration")
a.text(1.03, 0.72, "repaid within the first batch", fontsize=7,
       color="#898781", ha="left", va="center")
a.text(1.03, 0.46, "median 1.75 batches over the 32 that changed budget",
       fontsize=7, color="#898781", ha="left", va="center")
a.text(0.03, 0.97, "(a)", transform=a.transAxes, fontsize=8, va="top")
a.annotate("rising nprobe", xy=(1.10, 12.5), xytext=(1.52, 5.6),
           fontsize=7, color="#898781", va="center",
           arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#898781",
                           shrinkA=2, shrinkB=2))

# ── (b) what it cost in recall ──────────────────────────────────────────────
b.axhspan(-1e-4, 1e-4, color="#8f8d86", alpha=0.22, lw=0, zorder=0)
for ds in DATASETS:
    pts = sorted([r for r in ALL if r["ds"] == ds], key=lambda r: r["np"])
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
b.set_ylim(-0.0056, 0.0011)
b.text(1024, 0.00062, "band is $\\pm10^{-4}$: the top-$ck$ tie-break, the only\n"
       "thing that varies besides $\\alpha$ -- both arms share one index",
       fontsize=7, color="#5f5e5a", va="center", ha="right", linespacing=1.35)
bad = min(rows, key=lambda r: r["drecall"])
b.annotate("largest: %s at nprobe=%d,\n%.4f of recall"
           % (PRETTY[bad["ds"]], bad["np"], -bad["drecall"]),
           (bad["np"], bad["drecall"]), fontsize=7, color="#3d3c39",
           xytext=(-8, 20), textcoords="offset points", ha="right",
           linespacing=1.3)
b.legend(loc="lower left", fontsize=7, ncol=2, columnspacing=1.0,
         handletextpad=0.5)
b.text(0.03, 0.955, "(b)", transform=b.transAxes, fontsize=8, va="top")

fig.tight_layout(pad=0.3)
save(fig, "fig_economics")
