#!/usr/bin/env python3
"""Figure: what each index costs to build, on one card.

Section 6.7. Until now this project could say how long JHQ takes to build and
nothing about anyone else, which made a section called "Build Time and Memory"
half a section: fig_cost's right panel is JHQ's own train+encode split and its
x-axis literally reads "JHQ index build (s)".

Every number here was already measured and none of it needed a re-run. The
cuVS baselines' build time is the `train_ms` column that `scripts/bench_all.py`
has always written; it simply never reached `fronts.json`, which is the file
the frontier figure reads. IVF-RaBitQ's is the `build_ms` line of
`bench_ivf_rabitq.cu`, and the round-trip through serialize/deserialize is
inside it, because without that round-trip the index is not searchable.

## What a bar is

Build time is a property of the index, not of the search: rows that share a
configuration but sweep nprobe or itopk report the same build. So each bar is
the **median across every build of that method on that dataset**, and the
whisker spans them.

That spread is not a confidence interval. It runs over two things at once --
the configurations the sweep covered (IVF-PQ on vogue goes from 2.2 s at 96 B
a vector to 11.6 s at 768 B) and repeated builds of one configuration (CAGRA
int8 at 896 B, ten builds, 4.9 to 6.0 s). A single number per method would
have to pick a configuration, and for IVF-PQ there is nothing to pick: its
frontier in fig_frontier is a Pareto envelope over several byte sizes, so no
one configuration is "the" IVF-PQ on the plot.

## Two things that would make this flattering if left unsaid

**JHQ's build is a cold build, and the trained-state cache hides it.**
`paper_fronts.log` reports train=990 ms for vogue at nprobe=8 and train=25.8 ms
at nprobe=32 -- the second is a cache hit on the first. Taking a median would
report the cache. The bar is the maximum, which is the build that actually
happened.

**The coarse quantiser is trained at 39 points per centroid, and the build
time includes that.** `/root/_pa.sh` passes `JHQ_N_TRAIN = 39 x nlist` on every
dataset -- 1,277,952 for bge-m3 and stella at nlist=32768, 159,744 for vogue
and openai3-3072 at 4096, 319,488 for arxiv and openai3-1536 at 8192. So this
is not a cheap build bought by under-training: an earlier draft of this
docstring said it was, which was wrong, and the mistake matters in the
generous direction -- the bar already carries the cost of the quantiser the
frontier was measured on.

## A missing bar means one of two different things, and they are drawn apart

**The build fails.** CAGRA fp32 on bge-m3 and stella (the `oom` field of
fronts.json) and IVF-RaBitQ on the same two, where the process aborts with
rc=134 even after `train/list` is cut to 32 (`rabitq_bigsets.log`). At 10.1M
and 17.8M vectors of 1024 dimensions the allocation does not fit this card,
and it is the same reason both are absent from fig_frontier.

**The build was never timed.** No cell is in this state any more. IVF-RaBitQ
on openai3-3072 was in it until the log of `/root/_rqb.sh` was collected into
`data/rabitq_o3072_build.log` -- the number had been measured all along and
had simply never left the box. The mark stays in the legend because the
distinction is worth keeping visible: a gap in our measurements and a
limitation of a baseline are different claims, and drawing them the same way
would assert the second from evidence for the first.
"""
import sys, os, re, csv, glob, collections
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

REPO = os.path.join(HERE, "..", "..")
ALIAS = {"stella-trec24": "stella"}
b = collections.defaultdict(list)

# cuVS baselines: train_ms, written by bench_all.py, dropped by fronts.json
for f in glob.glob(os.path.join(REPO, "results", "**", "*.csv"), recursive=True):
    lines = [l for l in open(f) if not l.startswith("#")]
    if not lines:
        continue
    try:
        rows = list(csv.DictReader(lines))
    except Exception:
        continue
    for x in rows:
        m, t = x.get("method") or "", x.get("train_ms")
        if "cuVS" not in m or not t or t in ("", "None"):
            continue
        b[(m, ALIAS.get(x.get("dataset"), x.get("dataset")))].append(float(t) / 1000)

# JHQ: train + add, and the two phases cache differently, which an earlier
# version of this figure got wrong.
#
#   train  is cached. Only the first row of a dataset (nprobe=8) is cold; the
#          rest report 213-273 ms, which is a cache hit, not a training run.
#   add    is not cached. It re-runs on every row, and varies a lot: on stella
#          9.1 s to 21.8 s across six runs of the same encode.
#
# Taking max(train+add) picked stella's nprobe=1024 row -- a *cached* train of
# 255 ms beside an unusually slow 21.8 s encode -- and reported 22 s for a
# build that actually costs 13.4 s. The bar is now the cold row, and the
# whisker carries the encode's own spread against that cold training.
jhq_train, jhq_add = {}, collections.defaultdict(list)
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"FIX\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+.*"
                  r"train=([\d.]+)\s+add=([\d.]+)", ln)
    if m:
        ds, np_ = m.group(1), int(m.group(2))
        jhq_add[ds].append(float(m.group(4)) / 1000)
        if np_ == 8:                       # the only cold training run
            jhq_train[ds] = float(m.group(3)) / 1000
for ds, t in jhq_train.items():
    adds = jhq_add[ds]
    b[("JHQ-GPU", ds)] = [t + min(adds), t + sorted(adds)[len(adds) // 2],
                          t + max(adds)]

# IVF-RaBitQ: build_ms, serialize/deserialize round-trip included
cur = None
for ln in open(datafile("paper_rabitq.log")):
    m = re.match(r"\s*(\S+)\s+nlist=\d+", ln)
    if m and m.group(1) in DATASETS:
        cur = m.group(1)
    m = re.search(r"build_ms=([\d.]+)", ln)
    if m and cur:
        b[("IVF-RaBitQ", cur)].append(float(m.group(1)) / 1000)

# Failures, each with the log that records it. Anything else that is absent
# was simply not timed, and is marked differently.
FAILS = {("cuVS-CAGRA", "bge-m3"), ("cuVS-CAGRA", "stella"),        # fronts.json oom
         ("IVF-RaBitQ", "bge-m3"), ("IVF-RaBitQ", "stella")}        # rc=134
# openai3-3072's RaBitQ build was timed by /root/_rqb.sh, whose log is
# dedicated to that dataset and so does not repeat the name on each line.
# It was simply never collected into data/ until now.
for ln in open(datafile("rabitq_o3072_build.log")):
    m = re.search(r"nlist=\d+\s+bits=8\s+np=\d+\s+mode=0\s+recall=[\d.]+\s+"
                  r"qps=\d+\s+build_ms=([\d.]+)", ln)
    if m:
        b[("IVF-RaBitQ", "openai3-3072")].append(float(m.group(1)) / 1000)

METHODS = [("JHQ-GPU", "jhq"), ("IVF-RaBitQ", "rabitq"),
           ("cuVS-CAGRA", "cagra"), ("cuVS-CAGRA-int8", "cagra8"),
           ("cuVS-IVFPQ", "ivfpq")]
# "(this work)" claimed JHQ itself, which the paper explicitly does not; the
# port is what is ours, and the title calls it JHQ-GPU.  style.py's S dict was
# corrected and this one was missed.
LAB = {"JHQ-GPU": "JHQ-GPU (ours)", "IVF-RaBitQ": "IVF-RaBitQ",
       "cuVS-CAGRA": "CAGRA fp32", "cuVS-CAGRA-int8": "CAGRA int8",
       "cuVS-IVFPQ": "IVF-PQ"}

# The four ranges Section 6.7 quotes, printed so they have a generator
# instead of being read off the bars by hand.  audit_numbers.py reads this.
# Section 6.7's baseline build ranges, under build_crossover.py's convention:
# the fastest build observed for each baseline on each dataset, then the min
# and max across datasets.  Taking the fastest is the choice least favourable
# to JHQ, and quoting each baseline under a rule of its own -- which is how
# the IVF-PQ and IVF-RaBitQ ranges were first written -- is what let a number
# into the text that no aggregation reproduces.
print("  baseline cold builds, fastest per dataset, range across datasets:")
for meth in ("IVF-RaBitQ", "cuVS-CAGRA", "cuVS-CAGRA-int8", "cuVS-IVFPQ"):
    per = {ds: min(ts) for (m, ds), ts in b.items()
           if m == meth and ts and (m, ds) not in FAILS}
    if per:
        lo = min(per, key=per.get); hi = max(per, key=per.get)
        print("     %-16s %.1f to %.1f s   (%s .. %s, %d datasets)"
              % (meth, per[lo], per[hi], lo, hi, len(per)))

fig, ax = plt.subplots(figsize=(WIDE, 2.35))
W = 0.16
x = np.arange(len(DATASETS))
for j, (m, skey) in enumerate(METHODS):
    off = (j - (len(METHODS) - 1) / 2) * W
    col = S[skey]["color"]
    for i, ds in enumerate(DATASETS):
        v = b.get((m, ds))
        if not v:
            # "fails" and "never measured" are different claims
            # A bare cross says a build failed and not why, which is the
            # part a reader needs: every one of these is an allocation the
            # 32 GiB card could not serve, and the sizes are in the caption.
            mark = "OOM" if (m, ds) in FAILS else "--"
            ax.text(i + off, 1.05, mark, ha="center", va="bottom",
                    fontsize=5.5, color=col if mark == "OOM" else "#898781",
                    rotation=90)
            continue
        med = st.median(v)
        ax.bar(i + off, med, width=W * 0.86, color=col, ec="none",
               label=LAB[m] if i == 0 or (m, DATASETS[0]) not in b else "")
        if len(v) > 1 and max(v) > min(v):
            ax.plot([i + off, i + off], [min(v), max(v)], color="0.25", lw=0.7,
                    solid_capstyle="butt", zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.9, 130)
ax.set_xticks(x)
# Six dataset names do not fit side by side in a 4.8-inch block.
ax.set_xticklabels([PRETTY[d] for d in DATASETS], rotation=18, ha="right",
                   fontsize=7)
ax.set_ylabel("index build (s)")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="x", visible=False)

h, l = ax.get_legend_handles_labels()
seen, hh, ll = set(), [], []
for a_, b_ in zip(h, l):
    if b_ and b_ not in seen:
        seen.add(b_); hh.append(a_); ll.append(b_)
# Every absent cell is an out-of-memory build; nothing here is untimed, so a
# "not timed" key was an entry for a state that does not occur.
if any((m, ds) in FAILS for m, _ in METHODS for ds in DATASETS):
    hh.append(plt.Line2D([], [], color="#898781", marker=r"$\mathrm{OOM}$",
                         ls="", ms=9))
    ll.append("out of memory")
# Seven legend entries across one row is a two-column figure's legend; here it
# is what pushed the axes down to nothing while save() held the file at the
# LNCS text width.  The long note that ran under the axes for the same reason
# is now in the LaTeX caption, where it costs no drawing area.
ax.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
          fontsize=6.5, columnspacing=1.0, handlelength=1.2,
          borderpad=0.3, labelspacing=0.3)

fig.tight_layout(pad=0.3)
save(fig, "fig_build")
