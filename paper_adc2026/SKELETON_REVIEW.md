# Review of `ADC_sections_3_4_6_skeleton.md`

Checked against the frozen evidence in `data/` on 2026-09-10. The skeleton's
structure is sound and its writing discipline is the most valuable thing in
it. What follows is only what needs changing.

## Numbers that check out

| skeleton says | verified |
|---|---|
| factorised LUT M=96: 96 KiB → 12 KiB; M=384: 384 KiB → 48 KiB | exact (`M*256*4` vs `M*32*4`) |
| factorised LUT +6% to +14.5% | `results/v47_split_lut/` spans +10.9% to +14.5% |
| packed 32-bit code loads +30% to +48% | `results/front6/v52.log` |
| probe cursor +2.5% to +148% | `results/front6/v51.log` |
| query-sized launch +1% to +9% | `data/v57_launch.log` |
| adaptive alpha 1.0× to 2.65× | `data/paper_fronts.log`, 72 runs |
| saturation span ≈ 25× | `data/alpha_ds.log`, `data/bge_alpha_scansplit.log` |
| calibration 6–29 ms | `data/v57_launch.log` |
| openai3-3072 ratios 1.36 / 1.22 / 1.06 / 0.93 / 0.72 | `data/paper_rabitq.log` + `data/openai3072_v57_front.log` |
| hardware and dataset tables | as recorded |

## What is out of date

**IVF-RaBitQ is no longer one dataset of six.** The skeleton says the other
five are "missing/running" and warns against generalising. Four are now
measured, and the pattern is stronger than a single dataset could show:

| dataset | d | R=0.90 | R=0.95 | R=0.97 | R=0.98 |
|---|---:|---:|---:|---:|---:|
| vogue-768 | 768 | 1.24× | 1.02× | 1.00× | 0.94× |
| arxiv-768 | 768 | 1.29× | 1.02× | 0.89× | 0.81× |
| openai3-1536 | 1536 | 2.35× | 1.98× | 1.62× | 1.36× |
| openai3-3072 | 3072 | 1.82× | 1.36× | 1.06× | 0.93× |

**The margin grows with dimensionality**, which is a claim about where each
design pays rather than a scoreboard. stella and bge-m3 are absent for a
reason that has to be worded precisely (`data/paper_rabitq.log`,
`data/rabitq_bigsets.log`): with default parameters the build throws
`rmm::out_of_memory` on the k-means training set, and with `force_streaming`
and a training set cut 32× it throws on the dataset itself. "IVF-RaBitQ cannot
index these" has been the wrong sentence twice; the right one names the
parameter and the allocation.

## What the skeleton has no place for

**1. The batch size changes who wins.** `data/batch_sweep.log`, nprobe=128:

| batch | openai3-3072 | vogue-768 |
|---:|---:|---:|
| 32 | 1.55× | 0.48× |
| 128 | **2.44×** | 0.84× |
| 1024 | 1.49× | **1.12×** |

The ratio is not stable, and it moves in opposite directions on the two
datasets. The skeleton mentions batch only as a threat — "if batch size
changes, re-run all methods" — but it is an axis, and it is the answer to the
sharpest question this evaluation invites, since everything else runs at one
operating point on one card. **It needs its own research question, not a
footnote.**

**2. The serialize/deserialize finding.** cuVS's `ivf_rabitq` returns ids that
are essentially random until the index is round-tripped through
serialize/deserialize; no public header says so, and `data/selftest.log`
isolates it with an answer that is known in advance — a query that is a row of
the index, every list probed, 0/20 before and 20/20 after. This belongs in
6.1.4 as evidence the baseline was set up correctly, not buried. A reviewer
who has tried the same library will recognise it.

**3. Section 4's failure evidence.** The skeleton says "explain failures, not
just successes" and then allocates no space. The two knees are the argument:
at S=8 openai3-3072 picks alpha=2 and loses 0.0058 of recall; at two slots of
tolerance vogue-768 picks alpha=32 and loses 0.0035. Both are outside the
1e-3 build-noise floor, and they are why S=32 and one slot are the settings.

## Where the framing needs correcting

**The query-sized launch is a bug fix, not a design choice.** The skeleton's
ablation table lists it beside the factorised LUT, and its own warning applies
— "large gains caused by removing an implementation pathology should not be
presented as deep algorithmic novelty". Every search ran `batch_cap` blocks
and zero-padded the rest. It should be labelled a correction, and the honest
consequence stated: every QPS this project reported before it was 1–9% low.

**The one-sentence story understates the contribution.** "Exploits structural
properties and introduces an adaptive policy" is true of many papers. The
sharper claim is that a published algorithm fixes `ck = αk` and leaves α
unspecified, that the value it needs varies 25× across datasets, and that it
can be recovered from 32 of the batch's own queries without labels. That is a
gap in the original formulation, not an engineering knob.

## What still cannot be written

- **CPU speedup.** No trustworthy CPU JHQ number exists. `JHQ_official` reads
  its centroids and assignments from files, so the fair experiment is to hand
  it the ones this index trained (`export_ivf_for_cpu`, written; not yet run).
  Thread count must be pinned: 32 threads beat 208 by 1.9× in an earlier
  measurement, so an unpinned CPU column measures how busy the machine was.
- **IVF-RaBitQ memory.** Its bits per dimension put it about 11% under JHQ,
  but the measured figure is not captured yet, so table 3 stays empty on that
  row.
- **Anything about a second GPU.** `data/v54.log` shows a kernel verdict
  flipping with bandwidth; that is a reason to state the limit explicitly, not
  a licence to generalise from it.
