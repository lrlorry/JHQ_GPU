# Review of the mother skeleton v2 (sections 3, 4, 6)

Read against the repository, the frozen logs in `data/`, and the thirteen
figure scripts as they now stand.

**Verdict: adopt the structure.** v2 fixes the thing that was actually wrong
with v1 — it stops organising section 6 by version number and organises it by
research question, and it stops calling every CUDA change a contribution. The
figure-corrections list at the end matches the ten fixes applied in `b1d307a`
almost item for item, so the skeleton and the drawings now agree, which they
did not before.

What follows is what it still gets wrong or leaves out, in the order that
would cost the most at review.

---

## 1. The calibration cost is a headline, not a bullet — and the number in ALPHA_RULE.md is stale

§6.3.3 lists "calibration cost; amortization" as two of five things to report.
That understates it. The rule buys throughput by spending a fixed cost up
front, so **every gain figure in the paper is conditional on the batch count**,
and a reviewer will ask for the break-even before asking anything else.

The number is already measured on every row. `paper_fronts.log` records
`batches_to_repay` on all 36 RULE rows:

> **Corrected 2026-09-10.** The table below first read "0.6 to 178.6, median
> 2.7, 18 of 36 above two batches". That was wrong, and the error is described
> in `SKELETON_REVIEW_v3.md` item 0: four rows compare a configuration with
> itself, so their gain is repeat-timing noise and their payback is 1/noise.
> Excluding them leaves the figures below.

| | over the 32 runs where the budget actually changed |
|---|---|
| calibration cost | 2.0 – 47.5 ms |
| steady-state gain | 1.044 – 2.653 — **every one above 1** |
| batches to repay | 0.5 – 21.0, median 1.75 |
| rows needing more than 2 batches | 15 of 32 |
| runs where the rule changed nothing | 4 (arxiv-768, nprobe >= 128), reported separately |

The correction improves the result rather than damaging it. The row that
appeared never to repay, at gain 0.996, was one of the four self-comparisons:
across every configuration where the rule actually moved the budget, it is
faster, by 1.044x to 2.653x.

**Report the scatter of gain against batches-to-repay, not the gain range
alone**, and say which runs are not on it and why. A single "1.0x to 2.65x"
with the cost in a footnote reads as concealment even though the data acquits
us. Note also that the anticorrelation is largely definitional --
B* = (T_cal/T_rule)/(g-1) diverges as g approaches 1 by construction -- so the
figure is worth having for the levels it reports, not as a mechanism finding.

Two corrections that follow:

- `results/front6/ALPHA_RULE.md` says "about eight batches, or 8,000 queries,
  to pay back". That is the **v55** number. v57's launch sizing cut
  calibration 11x and the current logs say 0.6 to 1.2 batches at nprobe=8.
  The note needs refreshing before anyone quotes it.
- State plainly whether the frontier's RULE arm QPS includes calibration. If
  it does not — and `AF_RESULT` reports `qps_picked` and `cal_ms` separately,
  so it does not — say so in §6.1 next to the timing boundary, not in §6.3.

---

## 2. The CPU JHQ baseline is not optional

§6.1 lists "original CPU JHQ" under *"Additional if finalized"*, below
IVF-RaBitQ, CAGRA and IVF-PQ.

The paper's own premise is that JHQ is a good algorithm that needs GPU
execution. The first question any reviewer asks a port is **how much faster
than the thing it ports**. Demoting that to optional inverts the paper's
motivation. It belongs in §6.2 as a stated speedup against the authors'
own artifact, on the same data, at matched recall.

It is also the one baseline still unfinished. If it does not land, the paper
must say so explicitly rather than omit it silently — an absent CPU number in
a GPU-port paper is read as an unflattering one.

When it does land, §6.1 must state the thread count. The CPU column of
`build_gpu.csv` does not reproduce: 32 pinned threads beat 208 unpinned by
1.9x on this host. An unpinned 208-thread baseline would flatter us by nearly
2x and is indefensible.

---

## 3. "GPU co-design" is one idea, not three

§6.4 groups the Cartesian-factorised LUT, the physical layout, and packed
loading together as *"structural / co-design ablations"*, equal in status.
They are not equal, and a reviewer will separate them for us:

| | what it is |
|---|---|
| Cartesian-factorised LUT | **genuinely JHQ-specific.** 256 -> 16+16 is exact *because* the primary codebook is a Cartesian product of one-dimensional levels. No other quantiser in the comparison admits it. This is the co-design claim. |
| subspace-major layout | **standard practice.** Transposed code layouts are in FAISS-GPU and cuVS already. Ours is a correct application, not a new idea. |
| packed uint32 loads | **enabled by the representation.** Four subspaces fit one word only because Ds=B=8 makes the primary code one bit a dimension — the admissibility rule. Weaker than the LUT, stronger than the transpose. |

Claim the LUT as the contribution, present the layout as necessary
engineering, and present the packing as a consequence of the admissibility
rule. Claiming three co-designs where there is one and a half invites the
reviewer to find the one and a half themselves, which is worse than saying it.

---

## 4. Section 4 has no related-work position, and needs one

The skeleton asks (§4) for the rule to "read as an independent algorithm /
policy contribution" and then never says what it is independent *of*.
Sample-based parameter selection is not new:

- FAISS's `ParameterSpace` autotune has done grid search on a sample for years;
- adaptive early termination for IVF and graph search is an existing line
  (Li et al., *Improving approximate nearest neighbor search through learned
  adaptive early termination*, SIGMOD 2020, and its successors).

Without a paragraph placing the rule against those, "we sample S queries and
pick a parameter" reads as folklore. What is actually defensible and should be
said in one sentence each:

1. the criterion is **the system's own output stability**, not a proxy for
   recall and not a learned predictor, so it needs no labels and no training;
2. it is **validated against the sweep it replaces**, on the full grid, which
   the autotune literature usually does not report;
3. it targets a parameter JHQ leaves externally specified, so it closes a hole
   in the source algorithm rather than tuning a system knob.

---

## 5. The 25x is the wrong shape of number

§4.1 boxes 25x as the variation in saturation budget. From `alpha6.log` at
nprobe=128, the saturation alpha runs **4 (openai3-3072) to at least 200
(arxiv-768, still falling at the top of the grid)**.

"25x" both understates it and overstates its precision: arxiv's value is a
lower bound, not a measurement. Write the range and say the top is censored.
A censored endpoint is more convincing than a round ratio, because it shows
the grid was not cut where it flattered us.

Also state the nprobe. The spread is measured at 128; alpha6.log covers 32 and
128 only.

---

## 6. §4.2's epsilon does not match the implementation

The skeleton writes the criterion as `Diff(R_alpha, R_max) <= eps` and then in
§4.3 says "one-slot disagreement tolerance". Those are different objects. The
implementation (v56) takes an **integer slot count**, not a fraction, and the
tolerance ablation in `fig_calibration(c)` sweeps 0, 1, 2 slots.

Define eps as a slot count in §4.2 so the formula, the algorithm box, and the
ablation axis are the same quantity. Reviewers do check this.

---

## 7. §6.1 needs a baseline-tuning protocol, and two disclosures

The skeleton's fairness rule is about matched recall only. The more common
kill shot on an ANN paper is *"you under-tuned the baselines"*, and §6.1 as
written has nothing to answer it with. It needs a paragraph saying how nlist,
M, nprobe and the baseline parameters were chosen, and over what grids.

Two things this project already knows and must disclose rather than have found:

- **The IVF coarse quantiser was undertrained** — 6 points per centroid where
  the usual guidance is ~39 — and fixing it was worth up to +96% QPS for five
  seconds of build. Any frontier point measured before that fix is
  self-handicapped. State which configuration the reported runs used.
- **vogue-768's own coarse quantiser has a collapsed list holding 14.5% of the
  data** (19,966 of 20,000 distinct vectors, mean cosine 0.623 to its
  centroid). It is a property of the dataset, not a bug, and it makes vogue's
  numbers pessimistic for JHQ. Say so in the setup. Found by a reviewer
  instead, it looks like cherry-picking in reverse.

---

## 8. Repeatability is not stated anywhere

The skeleton has the paper reporting recall deltas of 0.0000 to 0.0058 and
gains to three digits, with no statement of run-to-run variance.

Three cold-cache runs of one binary on vogue gave 0.9852 / 0.9842 / 0.9849
with an identical candidate set: coarse assignment is deterministic, the
codebooks differ in their last bits. **Recall differences below 1e-3 between
two builds mean nothing.** That sentence belongs in §6.1, and every table
that reports a delta smaller than it must say it is within noise. Without it,
the -0.0003 rows in §6.3 read as measured losses.

---

## 9. Smaller things

- **Section 3 is too fragmented.** Nine headings (3.1, three under 3.2, four
  under 3.3, 3.4) in ~1500 words is ~170 words each. Merge 3.2.1 and 3.2.2 —
  the transform and the representation are one paragraph of setup — and let
  3.3 have the room the skeleton says it should have.
- **k sensitivity is missing entirely.** Everything is Recall@10, and the
  budget is `ck = alpha*k`, so alpha and k are coupled by construction. Either
  show one k sweep or state in §6.1 that k=10 throughout and why.
- **§6.5 as its own RQ is a space luxury.** The negative results are among the
  paper's better material and they defend against "did you try X", but if the
  page limit bites, they fold into §6.4 as a subsection rather than losing
  §6.2 detail.
- **The evidence path is wrong.** §6.2 cites
  `paper_adc2026/figures/fronts.json`; the file is
  `report_adc2026/v47/fronts.json`.
- **fig_hierarchy is already at 6.4**, per the figure README — the skeleton
  and the repo agree here, no action.

---

## What v2 got right and should not be relitigated

- Section 6 by research question, not by version. This was the single biggest
  problem with v1.
- The refusal to promote v57's launch sizing, the probe cursor and the bug
  fixes to contributions. They are corrections; they remove unintended work.
- Rule 4, operating regimes over universal superiority. This matches the data:
  JHQ leads through Recall 0.95 on all four RaBitQ-comparable datasets with
  the margin growing in d, and gives way in the high-recall tail. Claiming
  more would be false and is unnecessary.
- The figure-corrections list. All ten are applied as of `b1d307a`.
- "JHQ residual != IVF centroid residual". Worth the box it is in; it is the
  single most likely misreading of section 3.
