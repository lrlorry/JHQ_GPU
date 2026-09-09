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

**JHQ's coarse quantiser was undertrained** -- about 6 points per centroid
where the usual guidance is ~39. Training it properly costs a few seconds more
here and is worth up to +96% QPS, so this figure and the frontier are *both*
measured on the cheap-to-build, slow-to-search configuration. The bar is
therefore a lower bound on the build time of the index a tuned JHQ would ship,
and the paper must say so rather than bank a fast build it would not keep.

## A missing bar means one of two different things, and they are drawn apart

**The build fails.** CAGRA fp32 on bge-m3 and stella (the `oom` field of
fronts.json) and IVF-RaBitQ on the same two, where the process aborts with
rc=134 even after `train/list` is cut to 32 (`rabitq_bigsets.log`). At 10.1M
and 17.8M vectors of 1024 dimensions the allocation does not fit this card,
and it is the same reason both are absent from fig_frontier.

**The build was never timed.** IVF-RaBitQ on openai3-3072 has search results
throughout this paper but no `build_ms` in any frozen log. That is a gap in
our measurements, not a property of the system, and drawing it the same way as
a failure would be a false claim about a baseline. It gets its own mark.
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

# JHQ: train+add, and the cold build is the largest, not the median -- every
# later row of a dataset is a hit on the trained-state cache
jhq = collections.defaultdict(list)
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"FIX\s+(\S+)\s+.*train=([\d.]+)\s+add=([\d.]+)", ln)
    if m:
        jhq[m.group(1)].append((float(m.group(2)) + float(m.group(3))) / 1000)
for ds, v in jhq.items():
    b[("JHQ-GPU", ds)] = [max(v)]

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
METHODS = [("JHQ-GPU", "jhq"), ("IVF-RaBitQ", "rabitq"),
           ("cuVS-CAGRA", "cagra"), ("cuVS-CAGRA-int8", "cagra8"),
           ("cuVS-IVFPQ", "ivfpq")]
LAB = {"JHQ-GPU": "JHQ-GPU (this work)", "IVF-RaBitQ": "IVF-RaBitQ",
       "cuVS-CAGRA": "CAGRA fp32", "cuVS-CAGRA-int8": "CAGRA int8",
       "cuVS-IVFPQ": "IVF-PQ"}

fig, ax = plt.subplots(figsize=(WIDE, 2.6))
W = 0.16
x = np.arange(len(DATASETS))
for j, (m, skey) in enumerate(METHODS):
    off = (j - (len(METHODS) - 1) / 2) * W
    col = S[skey]["color"]
    for i, ds in enumerate(DATASETS):
        v = b.get((m, ds))
        if not v:
            # "fails" and "never measured" are different claims
            mark = "$\\times$" if (m, ds) in FAILS else "?"
            ax.text(i + off, 1.06, mark, ha="center", va="bottom",
                    fontsize=7.5, color="#898781")
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
ax.set_xticklabels([PRETTY[d] for d in DATASETS])
ax.set_ylabel("index build (s)")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="x", visible=False)

h, l = ax.get_legend_handles_labels()
seen, hh, ll = set(), [], []
for a_, b_ in zip(h, l):
    if b_ and b_ not in seen:
        seen.add(b_); hh.append(a_); ll.append(b_)
hh.append(plt.Line2D([], [], color="#898781", marker=r"$\times$", ls="", ms=5))
ll.append("build fails on this card")
hh.append(plt.Line2D([], [], color="#898781", marker="$?$", ls="", ms=5))
ll.append("not timed")
ax.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=7,
          fontsize=7, columnspacing=1.0, handlelength=1.3)
ax.text(0.5, -0.30, "whisker: the range across swept configurations and repeat "
        "builds, not a confidence interval.  JHQ is a cold build (later rows "
        "hit the trained-state cache)\nand uses the undertrained coarse "
        "quantiser, so its bar is a lower bound on the index a tuned JHQ "
        "would ship.", transform=ax.transAxes, ha="center", va="top",
        fontsize=7, color="#898781", linespacing=1.35)

fig.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1))
save(fig, "fig_build")
