# Assessment of the v3 critique

**Short version: it is a good critique, its central factual correction is right
and was my error, and its overall verdict is calibrated to the wrong venue.**

---

## 0. The two corrections it makes to me, both valid

### 0.1 The payback statistic (its item 3)

Correct, and now fixed in `974fcf0`.

On arxiv-768 at nprobe 128, 256, 512 and 1024 the rule returns alpha=100 --
which is `alpha_max`, which is also what the fixed arm runs.
`demo_jhq_alpha_fast.cu` times `full(picked)` and `full(grid[0])` separately,
three repeats each, so with `picked == grid[0]` the reported gain is the ratio
of two 3-run means of **one configuration**. It comes out 0.996, 1.003, 1.006,
1.012 -- about 1.2% of repeat-timing variation -- and `batches_to_repay` then
divides the calibration cost by that noise: 179, 108, 58. Those are values of
1/noise. The recall delta on all four is exactly 0.0000, which is the same
fact seen from the other side.

I put those numbers in a figure and in two review documents. They are out now.

**The correction improves the result.** Over the 32 runs where the budget
actually changed:

| | |
|---|---|
| steady-state gain | 1.044x - 2.653x, **every one above 1** |
| batches to repay | 0.5 - 21.0, median **1.75** |
| above two batches | 15 of 32 |

The single sub-one gain in the log was one of the four self-comparisons. Once
they are removed, there is no configuration where the rule made anything
slower.

And the four are a **result**, not merely an exclusion: they are the case
where the rule correctly declines to act. arxiv-768's saturation budget is
above `alpha_max`, the rule returns `alpha_max`, and reports no gain. A policy
that invented a saving there would be worse. Report them as a row in the
table, labelled "budget unchanged".

### 0.2 Build noise is not the same as a paired difference (its item 5)

Also correct, also mine, also fixed.

I wrote in `SKELETON_REVIEW_v2.md` item 8 that every recall delta below 1e-3
should be called noise. That is wrong here. The 1e-3 floor is **build to
build** -- three cold starts of one binary give 0.9852 / 0.9842 / 0.9849
because training is not reproducible and the codebooks differ in their last
bits. But both arms of the alpha comparison run on **one `idx` object**, one
set of codebooks, one assignment; nothing varies but alpha. The only residual
is the top-`ck` tie-break, order 1e-4.

So the -0.0003 deltas are real, just small. `fig_economics(b)`'s band is now
+-1e-4 and says why. The paper must keep the two floors separate: 1e-3 for
"is this version better than that version", 1e-4 for "is this alpha worse
than that alpha on the same index".

### 0.3 One thing it says that I had already written into the figure

That the gain-payback anticorrelation is partly definitional. It is:
`B* = T_cal/(T_fixed - T_rule) = (T_cal/T_rule)/(g-1)` diverges as `g -> 1` by
construction. The figure is worth having for the **levels** it reports -- a
median of 1.75 batches -- not as a mechanism finding, and its docstring now
says so.

---

## 1. Its strongest point: FastScan and Quicker ADC (its item 1)

**This is the most serious thing in the critique and it must be answered in
the paper, not in a rebuttal.**

FastScan (in FAISS) and Quicker ADC make the ADC lookup table small enough to
live in SIMD registers and scan it with byte shuffles. That is recognisably
the same *goal* as `256 -> 16+16`, and a reviewer who knows the area will say
so. Not citing them would be the single most damaging omission in Section 3.

The distinction is defensible and should be stated in one sentence:

> FastScan and Quicker ADC obtain a small table by **changing the quantiser** --
> 4-bit codes, a coarser codebook, accuracy traded for a table that fits the
> register file. The factorisation here changes nothing: the code stays 8-bit,
> the codebook stays the paper's, and the distances are identical, because the
> primary codebook is already a Cartesian product and the table was redundant.

Lossless-versus-lossy is a real difference. But it has to be claimed
explicitly, and the related-work section has to name both.

### The granularity experiment is worth running, and it has a real hypothesis

The critique asks why `2 x 16` and not `4 x 4` or `8 x 2`. That is the right
question, and the arithmetic makes it more interesting than it first looks. At
`Ds = B = 8` the primary code is 8 base-2 digits, so a split into `g` groups
gives `g` tables of `2^(8/g)` entries:

| g | tables/subspace | entries/subspace | loads+adds per candidate per subspace |
|---:|---|---:|---:|
| 1 | 1 x 256 | 256 | 1 |
| 2 | 2 x 16 | **32** | 2 |
| 4 | 4 x 4 | **16** | 4 |
| 8 | 8 x 2 | 16 | 8 |

g=8 is dominated -- same table as g=4, twice the work -- so the real question
is **g=2 against g=4**, and it is not obvious. g=4 halves shared memory again:
at M=384 on openai3-3072 that is 48 KiB against 24 KiB per block, which is in
the range where occupancy actually moves on this card. It doubles the
per-candidate work in exchange.

So the experiment has a hypothesis, not just a sweep: *g=4 should lose on
instruction count and win on occupancy, and which dominates should depend on
M.* If M=96 prefers g=2 and M=384 prefers g=4, that is a design principle and
not an implementation detail -- which is exactly the elevation the critique
asks for. If g=2 wins everywhere, that is also worth one sentence and closes
the question.

**Agreed: this is the highest-value remaining Section 3 experiment.** The
`full/split x byte/packed` 2x2 matched ablation is the second, and it also
settles whether the LUT gain and the packing gain are additive or overlapping
-- which the "pays twice" framing in `SKELETON_REVIEW_v3.md` item 1 currently
assumes without evidence.

---

## 2. Its second-strongest point: the rule is validated against the wrong
baseline (its item 2)

Correct. Everything the rule is measured against is fixed alpha=100, which is
nobody's tuned choice -- it is this project's historical default. "Faster than
a badly chosen constant" is a weaker claim than the paper implies.

Three things are missing and two of them are nearly free:

1. **A tuned fixed alpha.** The honest comparison is against the best single
   alpha an oracle would pick per dataset. If the rule is within a few percent
   of that, the claim becomes "recovers the oracle constant without the
   sweep", which is much stronger than "beats 100".
2. **Held-out queries.** The rule calibrates on S=32 of the batch and is then
   scored on the batch *including those 32*. At 3% the contamination is small,
   but it is free to remove and a reviewer will notice it is not.
3. **Resampling.** One draw of 32 tells you nothing about the variance of the
   choice.

Its proposed protocol is right and cheap, and I want to underline how cheap:
**freeze one index, run the full alpha grid once saving per-query top-k, then
do all the resampling offline.** No further GPU time. A single sweep supports
thousands of calibration draws and gives the distribution of the chosen alpha
and of the held-out recall loss. That converts "it worked on eight
configurations" into a statement with error bars.

It also asks for an explicit, falsifiable success criterion -- e.g. *"on
held-out queries the rule loses at most 1e-3 mean recall in at least 95% of
calibration draws"*. It is right that without one, "found a sufficient alpha"
is unfalsifiable. Adopt this.

This overlaps my own review's stationarity point: both are about the rule
being evaluated on the data it was fitted to, one within a batch and one
across batches.

---

## 3. Its item 4: connect Sections 3 and 4 experimentally

The hypothesis is good -- as alpha falls, refinement work falls, so the
primary path's *share* rises and Section 3's optimisations matter more -- and
its caveat is exactly right: at fixed nprobe, lowering alpha reduces
refinement survivors, **not** the number of primary candidates scanned. So the
primary scan cost is roughly constant and only its share changes. Anyone
writing this up must not slide from "share rises" to "scan gets cheaper".

One practical constraint the critique cannot know: **stage timing cannot be
collected in the same run as the headline QPS.** Per-stage timers break CUDA
graph capture, which is what the fast path depends on, so the numbers have to
come from a separate diagnostic build -- the same situation as `alpha6.log`,
whose `rank_lost` column is usable while its QPS column is not. That is fine,
but the paper must say which build each number came from, and must not put a
stage breakdown and a QPS number in the same table without a note.

---

## 4. Where I disagree: the verdict is calibrated to the wrong venue

The critique's closing position -- *"originality boundary and core
experimental proof still insufficient, needs substantial strengthening"* -- is
a fair **SIGMOD/VLDB** review. It is not the ADC bar.

ADC is a solid regional venue whose typical accepted systems paper is: a real
system, honestly measured, on real data, against current baselines, with the
limitations stated. On that standard this work is already above the line, and
comfortably so -- six datasets to 17.8M vectors, three GPU baselines, a
frontier at matched recall, ablations, negative results, memory and build
cost, and a level of self-correction (four self-comparisons found and
removed, two noise floors separated) that most submissions do not have.

So the practical reading of the critique is not "the paper is not ready". It
is **"these three experiments are what would make it strong rather than
sound"**, and they are cheap enough to be worth doing anyway:

| experiment | cost | what it buys |
|---|---|---|
| LUT granularity g in {1,2,4,8} x M | one sweep | turns the main Section 3 claim from a choice into a principle |
| offline resampling on a frozen alpha grid | one sweep, then no GPU | error bars and held-out validation for Section 4 |
| full/split x byte/packed 2x2 | one sweep | separates the two payoffs the "pays twice" framing assumes |

All three reuse an index that is already built. None needs new
infrastructure.

**The one thing that is genuinely blocking and is not on that list is the CPU
baseline**, which is a correctness-of-comparison issue rather than a depth
issue, and which is in progress.
