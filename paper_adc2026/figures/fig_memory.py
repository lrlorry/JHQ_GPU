#!/usr/bin/env python3
"""Figure: where JHQ's resident memory goes.

The IVF-RaBitQ comparison this figure used to carry is withdrawn with the
baseline itself: cuVS ships RaBitQ's quantiser without the re-ranking its
published results depend on, so its footprint here is the footprint of a
configuration we were forced into.

Section 6.7, the companion to fig_cost. IVF-RaBitQ is 22-39% smaller wherever
it builds, and its bits per dimension alone predict only 11% -- 8 against 9.
This is the rest of the gap, and it is not a code-size difference:

* the primary and residual codes are 9 bits a dimension against RaBitQ's 8;
* the coarse centroids are nlist x d floats, the same on both;
* and JHQ carries a per-batch search workspace that IVF-RaBitQ does not need
  at the same size -- the factorised table for every query in the batch, and
  the candidate buffers the exact top-alpha-k selection runs in.

The first six bars are computed from the index parameters. They do not add up
to what the card actually reports, and the gap is real: the CUDA context, the
cuBLAS and cuRAND handles, the allocator's slack, and the training buffers
that are freed but whose pool pages are not returned. Rather than let the
stack quietly disagree with the measurement, the last segment is *defined* as
measured minus modelled, so the bar ends exactly at the cudaMemGetInfo total
and the unexplained part is drawn at its true size instead of being left out.
It is labelled "other / unaccounted" rather than attributed: the CUDA context
and the allocator's slack are what it is *expected* to be, but nothing here
measures them separately, and naming a cause we did not measure would be the
same mistake as leaving the gap out.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np
# Height trimmed from 2.55 to 2.0: Section 6.6 carries two full-width figures
# and about 25 lines of text, so at the original heights neither could sit
# with the paragraph that introduces it and one landed in the bibliography.

# index shape per dataset: (N, d, M, nlist), Br = 8, batch = 1024, k = 10
SHAPE = {
    "vogue-768":    (932328,   768,  96,  4096),
    "arxiv-768":    (2253000,  768,  96,  8192),
    "bge-m3":       (10091524, 1024, 128, 32768),
    "stella":       (17776615, 1024, 128, 32768),
    "openai3-1536": (999000,   1536, 192, 8192),
    "openai3-3072": (999000,   3072, 384, 4096),
}
BR, BATCH, CK = 8, 1024, 1000

meas = {}
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"^  FIX\s+(\S+)\s+.*vram=([\d.]+)", ln)
    if m:
        meas[m.group(1)] = float(m.group(2))
rq = {}
for ln in open(datafile("vram.log")):
    m = re.search(r"^  (\S+)\s+nlist=\d+\s+vram=([\d.]+)", ln)
    if m:
        rq[m.group(1)] = float(m.group(2))

order = [d for d in DATASETS if d in meas]
parts = [("primary codes", "#2a78d6"), ("residual codes", "#86b6ef"),
         ("corr + ids", "#cde2fb"), ("coarse centroids", "#1baf7a"),
         ("per-batch table", "#eda100"), ("candidate buffers", "#eb6834")]
OTHER = "other / unaccounted"

vals = {p: [] for p, _ in parts}
for ds in order:
    N, d, M, nlist = SHAPE[ds]
    bpv = (d * BR + 7) // 8
    MiB = 1024 ** 2
    vals["primary codes"].append(N * M / MiB)
    vals["residual codes"].append(N * bpv / MiB)
    vals["corr + ids"].append(N * 8 / MiB)
    vals["coarse centroids"].append(nlist * d * 4 / MiB)
    vals["per-batch table"].append(BATCH * M * 32 * 4 / MiB)
    # queries, rotated queries, centroid dots, probe lists, top-ck positions
    # and distances, and the final results -- everything sized by the batch.
    vals["candidate buffers"].append(
        BATCH * (2 * d * 4 + nlist * 4 + 2 * CK * 4 + 10 * 8) / MiB)

# Defined, not modelled: whatever cudaMemGetInfo reports above the six terms
# above. Negative would mean the model over-counts, which would be a bug in
# the model rather than a memory segment, so it is reported, not drawn.
vals[OTHER] = []
for i, ds in enumerate(order):
    modelled = sum(vals[p][i] for p, _ in parts)
    gap = meas[ds] - modelled
    if gap < 0:
        print("  WARNING: model exceeds measured on %s by %.0f MiB" % (ds, -gap))
    vals[OTHER].append(max(gap, 0.0))

y = np.arange(len(order))
# Stella is 20 GiB and vogue is 1.4, so on one absolute axis the small
# datasets' workspace and centroid segments are a few pixels wide. The second
# panel is the same stack normalised, which is where the composition is
# actually readable.
# 1.85 rather than the 2.0 this had: at 2.0 the figure pushes two more
# references onto the last page, at 1.85 one, and below 1.85 nothing further
# is recovered.  Both panels stay -- the flattening that came of dropping one
# is what this height is protecting against.
fig, (ax, axp) = plt.subplots(1, 2, figsize=(WIDE, 1.85),
                              gridspec_kw=dict(width_ratios=[1.55, 1]))
left = np.zeros(len(order))
for p, c in parts:
    v = np.array(vals[p]) / 1024          # GiB
    ax.barh(y, v, left=left, height=0.5, color=c, label=p, ec="none")
    left += v
vo = np.array(vals[OTHER]) / 1024
ax.barh(y, vo, left=left, height=0.5, color="#f2f1ec", label=OTHER,
        ec="#898781", lw=0.5, hatch="///")
left += vo

for i, ds in enumerate(order):
    # coincides with the bar end by construction; drawn as the check that it
    # does.
    ax.scatter([meas[ds] / 1024], [i], marker="|", s=110, color="black",
               zorder=4, lw=1.1)

# same stack, as a share of the measured total
leftp = np.zeros(len(order))
tot = np.array([meas[d] for d in order])
for p_, c in parts + [(OTHER, "#f2f1ec")]:
    v = np.array(vals[p_]) / tot * 100
    axp.barh(y, v, left=leftp, height=0.5, color=c, ec="none" if p_ != OTHER else "#898781",
             lw=0 if p_ != OTHER else 0.5, hatch=None if p_ != OTHER else "///")
    leftp += v
axp.set_xlim(0, 100); axp.set_yticks(y); axp.set_yticklabels([])
axp.set_xlabel("share of measured total (%)")
axp.invert_yaxis(); axp.grid(axis="y", visible=False)
axp.text(0.03, 0.03, "(b)", transform=axp.transAxes, fontsize=8)
ax.text(0.97, 0.03, "(a)", transform=ax.transAxes, fontsize=8, ha="right")

ax.set_yticks(y); ax.set_yticklabels([PRETTY[d] for d in order])
ax.set_xlabel("resident GPU memory (GiB)")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(left) * 1.08)

h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], color="black", marker="|", ls="", ms=8, mew=1.1)]
l += ["JHQ, measured"]
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
           fontsize=7, columnspacing=1.2)

fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.80))
fig.subplots_adjust(wspace=0.10)
# Section 6.7's VRAM comparison, printed both ways round.  The two differ by
# their denominator and the body once quoted one while phrasing the other:
# "RaBitQ uses 22-39% less than JHQ" is the JHQ-denominated statement, and
# 22-39 is the RaBitQ-denominated one.
_a = [100 * (1 - rq[d] / meas[d]) for d in order if d in rq]
_b = [100 * (meas[d] / rq[d] - 1) for d in order if d in rq]
if _a:
    print("  VRAM over %d datasets: IVF-RaBitQ uses %.0f%%-%.0f%% less than JHQ; "
          "equivalently JHQ uses %.0f%%-%.0f%% more than IVF-RaBitQ"
          % (len(_a), min(_a), max(_a), min(_b), max(_b)))

save(fig, "fig_memory")
