# Review of the mother skeleton v3 (sections 3, 4, 6)

**Verdict: write from it.** All nine points from `SKELETON_REVIEW_v2.md` are
absorbed, and two are improved on rather than copied:

- "no other quantiser can do this" became *"not available to the compared
  baselines in their current representations"*. That is the correct
  formulation and mine was overreaching — we have not surveyed every
  quantiser, only the ones in the comparison.
- Rule 5, separating steady-state gain from amortised benefit, states the
  calibration-cost problem more cleanly than the review that prompted it.

What follows is what v3 still gets wrong, in the order it would cost most.

---

## 1. The novelty hierarchy is inverted relative to the measured effect sizes

§0.2 ranks the components:

| component | v3 status | **measured** |
|---|---|---|
| Cartesian-factorised LUT | **main contribution** | **+6 to +14.5%** (`results/v47_split_lut/`, 28 cells, none negative) |
| packed primary-code loading | representation-*enabled* optimisation | **+30 to +48%** (`results/front6/NEGATIVES.md`) |
| subspace-major layout | necessary engineering | (v13, against a naive baseline) |

A reviewer does this subtraction in ten seconds: **the thing labelled the main
contribution is worth a third of the thing labelled a secondary consequence.**
Presented as v3 has it, that reads either as novelty-shopping or as not having
looked at our own numbers.

The fix is not to demote the LUT. It is to notice that **both come from the
same place**, which is a better claim than either alone:

- the table factorises `256 -> 16+16` exactly *because* the primary codebook
  is a Cartesian product of one-dimensional levels (Eq. 4);
- four subspaces fit one `uint32` *because* the admissibility rule `Ds | B` at
  `B=8` forces `Ds=8`, one bit a dimension.

One structural property, two independent payoffs, in different parts of the
machine — one in the query-side working set, one in the database-side code
word. Proposed Section 3 claim:

> **JHQ's admissible primary codebook pays twice: once in the table, where the
> Cartesian construction makes an exact 16+16 factorisation possible, and once
> in the code word, where one bit a dimension lets four subspaces share a
> 32-bit load.**

Then state both effect sizes side by side in §6.4 and say plainly that the
larger of the two is the simpler idea. Volunteering that is worth more than
the two points it costs.

---

## 2. §4.4's break-even model hides a stationarity assumption

The cost model is

\[ B^{*} = \frac{T_{cal}}{T_{fixed}-T_{rule}} \]

which treats calibration as paid once and \(\alpha\) as valid for all \(B\)
subsequent batches. But the sample is drawn **from the batch's own queries**
(`ALPHA_RULE.md`: "Take S of the batch's own queries", S=32, ~3% of a
1000-query batch). So "calibrate once, reuse for B batches" assumes **the
query distribution does not drift** — and nothing in the experiments tests
that. Every batch after the first is answered at an \(\alpha\) chosen from a
sample of a batch it never saw.

This is the one question in Section 4 that the skeleton has no answer for, and
it is the obvious one to ask. Two honest ways out:

1. **State it as an assumption and a limitation.** "Break-even assumes a
   stationary query distribution; we do not measure drift, and a workload
   whose routing behaviour shifts would need periodic re-calibration."
2. **Define a re-calibration period** \(N\) and amortise \(T_{cal}/N\), which
   costs a constant factor in \(B^{*}\) and removes the assumption. At S=32 on
   a 1000-query batch, re-calibrating every batch is a 3% standing overhead —
   cheap enough that this may be the better story, and it would make the rule
   adaptive in time as well as across workloads.

Either is defensible. Silence is not.

---

## 3. §6.1 has become a compliance document

v3 now requires in §6.1: hardware, datasets, query/metric protocol, k
discussion, mandatory CPU baseline with thread count and pinning policy, GPU
baselines, a full baseline-tuning protocol (grids, nlist/nprobe, memory
matching, frontier-point selection, GT/query/batch/timing identity), two
coarse-quantiser disclosures, a repeatability policy, and a timing/calibration
protocol.

That is more than a third of a 2350-word section spent before a single result
appears. Every item is there because it is genuinely needed — the problem is
placement, not content.

Keep in §6.1 the one-sentence version of each plus a pointer; put the
parameter grids, the repeatability protocol and the hardware table in an
artifact appendix or the repository README. A reviewer who wants the grid will
look it up; one who does not should reach §6.2 on the first page of §6.

---

## 4. "Matched recall" is never defined, and it is an interpolation

Rule 4 requires matched recall for every speed claim, and §6.2's tables report
ratios at R = 0.90 / 0.93 / 0.95 / 0.97 / 0.98. **Neither system was measured
at those recalls.** `figures/style.py:interp` log-interpolates QPS between the
two adjacent swept points and returns `None` outside a method's measured range.

That method is defensible — log-linear in QPS is the right space for these
curves, and refusing to extrapolate is the important half — but it is a
modelling choice and it is currently invisible. One sentence in §6.1:

> QPS at a stated recall is interpolated log-linearly between the two adjacent
> measured sweep points; no ratio is reported at a recall outside a method's
> measured range.

Without it, "1.98x at Recall 0.95" claims a measurement that was not made.

---

## 5. Two evidence pointers are wrong, and two rows need `[TBD]`

- **§4.1 cites `data/alpha_ds.log`** for the 4-to-at-least-200 range. That log
  sweeps `{4, 8, 16, 32, 100}` — it contains no 200 and no `ivf_recall`
  column. The censored range and the ranking-loss curve both come from
  **`data/alpha6.log`**. v3 inherited this citation from v2 unchecked.
- **§7's row "GPU implementation is faster than original JHQ"** has no
  evidence and cannot have any until the CPU baseline lands. Mark it `[TBD]`
  under v3's own Rule 3, or someone will draft prose around a number that does
  not exist.
- **Figure 8 (gain vs `batches_to_repay`) does not exist.** It is listed in the
  figure plan as "new" and is correctly identified as necessary. It needs
  writing before §6.3.3 can be drafted; the data is already in
  `paper_fronts.log` on all 36 RULE rows.

---

## 6. §6.3.1 warns against a mistake that is already fixed

> "If the y-axis is 1-Recall@10, call it recall loss/error, not ranking loss."

`fig_alpha` no longer plots `1 - recall`. Since `97a61a3` it plots
`ivf_recall - recall` from `alpha6.log` — routing loss removed, which is
exactly the condition v3 sets for using the term. The instruction should say
so, otherwise a co-author drafting from the skeleton will rename a correct
axis.

---

## 7. A CPU point cannot share the frontier's axes

§6.2 asks for the CPU-JHQ speedup "either as a compact table or a clearly
visible reference point" on `fig_frontier`. A CPU QPS will sit two to three
decades below the GPU curves on a log axis and flatten every distinction the
figure exists to show.

Report it as one matched-recall number per dataset in the text or a three-row
table. Not a curve in the six-panel figure.

---

## 8. Minor

- **"32,607 MiB"** is a `cudaMemGetInfo` reading (total minus what the driver
  reserves), not a specification. Either say so or write 32 GB.
- §6.5's fold-if-tight instruction is right, but note the negatives are also
  the section that answers "did you just not try X" — folding them loses the
  reviewer-defence value, so prefer cutting §6.1 (item 3 above) first.

---

## Next action from this repository

`fig_economics.py` — gain against `batches_to_repay`, 36 points from
`paper_fronts.log`, log payback axis, the gain=1 vertical and the repay=1
horizontal as reference lines. It is the one required figure that does not
exist, and it is the figure that turns the calibration cost from a liability
into the paper's most candid result.
