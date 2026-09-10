# A rule for alpha, measured against the sweep it replaces

`alpha` fixes the refinement budget `ck = alpha*k`. The paper does not say how
to choose it, and `alpha_ds.log` measures what that omission costs: the point
where ranking loss stops improving spans 25x across these datasets —
openai3-3072 is flat from alpha=4, arxiv-768 is still improving at 200. A
single constant is therefore either wasteful or lossy, and which one it is
cannot be known without running the sweep.

`jhq_v55_alpha_sample/` + `examples/demo_jhq_alpha_sample.cu` run the rule
instead. Raw in `alpha_sample.log`.

## The rule

Take S of the batch's own queries. Answer them once at a generous `alpha_max`
and hold that answer as the reference. Then """ + BISECT + """

Run the rest of the batch at that alpha.

It needs **no ground truth** — it asks "would a smaller budget change what I
return", which the system can see, not "would it change recall", which needs
labels it does not have.

Two things bound what it can claim. The sample only estimates the agreement
the full batch would show, so the criterion itself is noisy. And the search is
bounded above by `alpha_max`, so where the true saturation point lies beyond
it — as on arxiv-768 below — the rule returns `alpha_max` and cannot do
otherwise.

Agreement is set intersection per query, not position equality: the same
neighbours in a different order are the same answer.

## Against the sweep, S = 32, eps = 0.001

The sweep is `alpha6.log`, grid {4, 8, 16, 32, 64, 100, 200}, and "sweep says"
is the smallest alpha whose ranking loss is within 3e-4 of the loss at the top
of that grid. **It only exists at nprobe 32 and 128.**

| dataset | nprobe | rule picks | sweep says | recall vs alpha_max | QPS gain |
|---|---:|---:|---:|---:|---:|
| openai3-3072 | 128 | **4** | **4** | +0.0000 | **2.01x** |
| stella | 128 | **16** | **16** | −0.0003 | **1.27x** |
| vogue-768 | 128 | **64** | **64** | −0.0002 | **1.26x** |
| arxiv-768 | 128 | 100 | >200 | +0.0000 | 1.00x |

**Three of four land on the value the sweep found.** The fourth is arxiv-768,
and it is not the rule erring in either direction: its ranking loss is still
falling at alpha=200 while the rule's own `alpha_max` is 100, so the
saturation point is outside the range the rule can return. It returns 100 and
reports no gain, which is the correct behaviour when the budget is already too
small. The largest recall deviation is −0.0003, inside the 1e-3 floor that
three cold-cache runs of one binary already established.

### The nprobe=512 rows are withdrawn

An earlier version of this table carried four more rows at nprobe=512 and gave
them a "sweep says" column. **No sweep was run at nprobe=512 on those
datasets** — the column reused the nprobe=128 value. The saturation point is a
property of the operating point, not of the dataset: nprobe changes how many
true neighbours are in the candidate pool at all, which is exactly what the
refinement budget then has to sort. Reusing one nprobe's answer for another
assumes the thing the table was supposed to be testing.

What the nprobe=512 runs do still support, because it needs no sweep, is the
gain: 1.38x on openai3-3072, 1.08x on stella, 1.00x on vogue-768 and
arxiv-768, at recall deviations of −0.0001, −0.0003, +0.0000 and +0.0000. That
is a statement about the rule against fixed alpha=100, which is measured, and
not about the rule against the saturation point, which is not.

## How many samples

> **Superseded 2026-09-10.** The table that stood here reported one
> calibration draw per S, on two datasets, and read a knee off it. That is
> where "S=32 is the knee" came from. It does not survive resampling, and the
> single draw it rested on was a lucky one. The old table is kept at the
> bottom of this section for the record.

`alpha_resample.py` replays the rule offline. `demo_jhq_v36` has always
written per-query ids to `out_prefix`, so one frozen index and the full alpha
grid (`paper_adc2026/data/alpha_perquery/`) is enough to sample S queries, run
the rule on exactly what it would have seen, and score the alpha it picks **on
the queries it did not see**. 2000 draws a cell, no GPU.

Held-out recall given up, mean and 95th percentile over 2000 draws, and the
share of draws inside the 1e-3 floor:

| S | vogue-768, np=128 | | | openai3-3072, np=128 | | |
|---|---|---|---:|---|---|---:|
| | mean | p95 | <1e-3 | mean | p95 | <1e-3 |
| 8 | 0.0165 | 0.0333 | 4% | 0.0049 | 0.0059 | 16% |
| 16 | 0.0073 | 0.0109 | 12% | 0.0033 | 0.0059 | 44% |
| **32** | **0.0033** | **0.0110** | **27%** | **0.0014** | **0.0060** | **76%** |
| 64 | 0.0013 | 0.0026 | 54% | 0.0001 | 0.0000 | 98% |
| 128 | 0.0005 | 0.0026 | 89% | 0.0000 | 0.0000 | 100% |

**The knee is a property of the workload, not a constant.** On openai3-3072,
whose saturation point is alpha=4, S=32 already keeps 76% of draws inside 1e-3
and S=64 keeps 98%. On vogue-768 the curve is still falling at S=128, and even
there 11% of draws fall outside.

**And the single-draw number was luck.** That draw picked alpha=64 on vogue at
nprobe=128 and gave up 0.0002. Across 2000 draws at the same S=32 the modal
pick is 32, not 64; the mean held-out loss is 0.0033, sixteen times larger;
and the 95th percentile is 0.0110, a full point of recall.

So the paper should give S per workload with its measured risk -- "S such that
95% of draws stay inside 1e-3" is a criterion the sweep can answer -- rather
than print one constant. On these two that is S=64 for openai3-3072 and beyond
S=128 for vogue-768.

<details><summary>The superseded single-draw table</summary>

openai3-3072 and arxiv-768 at nprobe=128, one draw each:

| S | openai3-3072 | arxiv-768 |
|---|---|---|
| 8 | alpha=2, recall −0.0058 | alpha=64, recall −0.0014 |
| 16 | alpha=4, +0.0000 | alpha=64, recall −0.0014 |
| 32 | alpha=4, +0.0000 | alpha=100, +0.0000 |
| 64 | alpha=4, +0.0000 | alpha=100, +0.0000 |
| 128 | alpha=4, +0.0000 | alpha=100, +0.0000 |

Read as a knee at S=32. The resampled figures above are what those cells look
like when they are not a single draw, and note that this table also scored the
draw on the whole batch, including the S queries it calibrated on.

</details>

## What it costs

10 to 256 ms, depending on how far down the grid the rule walks and how large
the batch's own searches are. On openai3-3072 at nprobe=128: calibration 98.8
ms, against a batch that falls from 24.5 ms to 12.2 ms — **about eight batches,
or 8,000 queries, to pay back**. Negligible for a resident index answering a
stream; a loss for a single batch answered once.

The cost is also one-sided in the useful direction: it is largest exactly where
the gain is largest, because both scale with how far alpha can fall.

## What this changes about the claim

The 25x spread was an observation. This is a rule that recovers it without the
sweep, priced against what it replaces, on four datasets and two operating
points. It does change Algorithm 1, which fixes `ck = alpha*k` with alpha
given: the budget is now chosen from the data at search time.
