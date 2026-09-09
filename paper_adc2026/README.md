# ADC 2026 — what is measured, and what it rests on

Every number the paper can use, the file it comes from, and whether it is
finished. `data/` holds frozen copies of the logs so the paper does not cite a
path that a later run overwrites; the live versions stay under `results/`.

**Machine.** One RTX 5090 (32,607 MiB, sm_120, 101,376 B opt-in shared per
block, 1792 GB/s, 170 SMs), 208 host cores, 754 GB RAM. Every system in every
comparison ran on this card. Batch is the query file itself — 999 or 1000
queries — answered in one call, and the timed region is host queries in to
host results out, identical for JHQ, cuVS and IVF-RaBitQ.

**Datasets.** vogue-768 (1.0 M × 768), arxiv-768 (2.25 M × 768), bge-m3
(10.1 M × 1024), stella-trec24 (17.8 M × 1024), openai3-1536 (1.0 M × 1536),
openai3-3072 (1.0 M × 3072). Recall@10 against the true top-10.

---

## 1. The contribution: a rule for α  ▸ finished

α sets the refinement budget `ck = αk`. PVLDB 19(7) defines JHQ and does not
say how to choose it; this project used 100 from its first run to its last.

**The rule.** Answer S of the batch's own queries at a generous α_max, hold
that as the reference, answer the same S at smaller α, and keep the smallest
whose top-k still agrees to within a slot. No ground truth: it asks whether a
smaller budget changes what is returned, not whether it changes recall.
The grid is walked by **bisection**, not stepped down — three probes for a
seven-point grid. An earlier version of this note said the rule "stops at the
first rejection, so it is one-sided". That describes a linear scan and is not
what happens: on vogue-768 the first probe is rejected and the search
continues upward. What bisection returns is the smallest grid α its sampled
criterion accepts, which is sound because exact top-ck selection gives nested
candidate sets and so the differing-slot count is monotone in α. It is bounded
above by α_max, so where the saturation point lies beyond α_max the rule
returns α_max and cannot do otherwise. arxiv-768 is that case.

| claim | number | file |
|---|---|---|
| worth 1.0× to 2.65× at equal recall, six datasets, nprobe 8–1024 | 72 runs, one build | `data/paper_fronts.log` |
| recovers the swept saturation α | 3 of 4 at nprobe=128; the 4th is arxiv-768, still improving past the rule's α_max | `data/alpha_sample.log`, `data/alpha6.log` |
| the nprobe=512 rows are **withdrawn** — no sweep was run there, and the earlier table reused the nprobe=128 answer | — | `results/front6/ALPHA_RULE.md` |
| recall cost | ≤0.003, except bge-m3 at nprobe=1024 (0.0048) | `data/paper_fronts.log` |
| S=32 is the knee | S=8 picks α=2 on openai3-3072 and loses 0.0058 | `data/alpha_sample.log` |
| one slot of tolerance is the knee | 0 slots leaves 26% on the table, 2 slots costs 0.0035 | `data/alpha_fast.log` |
| calibration cost | 6–29 ms, 0.7 batches to repay on openai3-3072 | `data/v57_launch.log` |

**The observation behind it.** α's saturation point spans 25× across the six
datasets — flat from α=4 on openai3-3072, still improving at 200 on
arxiv-768 (`data/alpha_ds.log`, `data/bge_alpha_scansplit.log`). A single
constant is therefore either wasteful or lossy and which it is cannot be known
without the sweep.

**The shape is the mechanism.** The gain falls as nprobe rises, on every
dataset: α sizes the refinement, refinement is most of the work at small
nprobe, the scan takes over at large. arxiv-768 gains nothing past nprobe=128,
and that is the rule correctly declining to cut where ranking loss is still
improving.

This changes Algorithm 1, which fixes `ck = αk` with α given.

## 2. Against IVF-RaBitQ (PVLDB 19(11), 2026)  ▸ finished: four datasets, and two it cannot index

Same card, same 999-query batch, same timed region, same nlist. cuVS
`ivf_rabitq` in QUANT4, its fastest mode here.

JHQ divided by IVF-RaBitQ at matched recall (`data/paper_rabitq.log`,
`data/bench_quant.log`, JHQ from the rule arm of `data/paper_fronts.log`):

| dataset | d | R=0.90 | R=0.93 | R=0.95 | R=0.97 | R=0.98 |
|---|---:|---:|---:|---:|---:|---:|
| vogue-768 | 768 | 1.24× | 1.11× | 1.02× | 1.00× | 0.94× |
| arxiv-768 | 768 | 1.29× | 1.17× | 1.02× | 0.89× | 0.81× |
| **openai3-1536** | 1536 | **2.35×** | **2.26×** | **1.98×** | **1.62×** | **1.36×** |
| openai3-3072 | 3072 | 1.82× | 1.57× | 1.36× | 1.06× | 0.93× |

**JHQ leads on all four through Recall 0.95, and the margin grows with
dimensionality** — near parity at d=768, 1.36× to 2.35× across the whole range
at d=1536. Where it gives way is the high-recall tail, earliest on arxiv-768
(0.89× at 0.97) and latest on openai3-1536 (still 1.36× at 0.98).

Three things the write-up must carry:

- **QUANT4 is its fastest mode.** LUT16 measures 1.8× slower at nprobe=1024
  (`data/bench_lut.log`), and its own paper says why: the LUT kernel is
  shared-memory bound at high dimensionality. Reporting LUT16 would have
  overstated JHQ by that much.
- **Its index is unsearchable as built.** cuVS's own test round-trips through
  serialize/deserialize with the comment "reorganize data for efficient
  search"; no public header says so. `data/selftest.log`: querying with a row
  of the index and probing every list, self-recall is **0/20 built and 20/20
  round-tripped**, at two sizes and through two link paths.
- **The RaBitQ-vs-CAGRA ratio here is not comparable to its paper's.** On this
  card it is 0.77× of CAGRA fp32 at Recall 0.95, against the 3.3× average it
  reports on an L40S at batch 10⁴. The JHQ-vs-RaBitQ column is a direct
  measurement; that one is not.

**stella and bge-m3 are missing because IVF-RaBitQ cannot index them on this
card.** Their raw vectors are 67.81 GB and 38.50 GB, so a device matrix is out
of the question, and the host-input path -- which logs "Using streaming
construction: dataset size exceeds comfortable GPU memory limit" and is meant
for exactly this -- then asks for 34,359,738,368 bytes in one allocation and
throws `rmm::out_of_memory`. Six attempts, both datasets, three nprobe each
(`data/paper_rabitq.log`).

JHQ indexes both on the same card and has room: 20,837 MiB on stella and
12,113 MiB on bge-m3, of 32,607 MiB (`data/paper_fronts.log`). CAGRA fp32
cannot reach them either. **So on the two largest datasets here, JHQ is the
only one of the three that produces a result at all**, and that is a library
limit with its own error message behind it, not a gap in our measurement.

## 3. Against CAGRA and IVF-PQ  ▸ finished for JHQ, baselines are from the v47 sweep

`figures/fronts.json` carries every point. CAGRA fp32 does not fit the card on
stella (72 GB of raw float) or bge-m3 (41 GB), so on those two the comparison
is against int8 only, whose recall ceiling is 0.9780 and 0.9376.

Each CAGRA curve stops where our own parameter sweep stopped, which is not a
limit of CAGRA — no claim rests on where it ends.

## 4. Measured negatives, with mechanism  ▸ finished

Five changes whose reasoning still looks right and which did not pay.
`results/front6/NEGATIVES.md`, `results/front6/V54_SIGN_IP.md`,
`results/front6/QDUP.md`.

| | result |
|---|---|
| v53, a larger selection buffer | −17 to −26%, and +2 to +4% on the one dataset whose occupancy is already pinned |
| v50, regroup before refinement | ±0 |
| v50, refinement fused into the scan | −14% on stella |
| v48/v49, early exit on a bound | −2.8% to +4.2% |
| v54, the table-free primary distance | −30 to −52% |

**v54 is the one to write up.** At L=2 the primary levels are ±a, so
`D_P = ‖q'‖² + da² − 2a⟨q', s_y⟩` is an identity — confirmed to four decimals
of recall in nine of twelve pairs. It halves per-query shared residency and
still loses 30–50% on the dataset whose occupancy it triples. IVF-RaBitQ's
paper reports the opposite verdict for the same two kernel shapes on an L40S
and says only that it "may vary" with bandwidth and shared memory. **We have
both sides measured on one card** (`data/v54.log`, 24 configurations); this
card has twice the L40S's bandwidth, so the instruction-heavy form does not
pay here.

`data/qdup.log` is the other one worth a paragraph: the cross-query reuse a
cluster-centric kernel would capture is worth +4 to +26%, and 87% of it is
already served by 96 MB of L2.

## 5. Execution work  ▸ finished

| | gain | file |
|---|---|---|
| v51, a probe cursor per thread instead of per block | +2.5 to +148% | `results/front6/v51.log` |
| v52, four subspaces read as one 32-bit word | +30 to +48% | `results/front6/v52.log` |
| v57, the launch sized to the queries present | +1 to +9%, rising with nprobe | `data/v57_launch.log` |

v57 is worth stating plainly: every search ran `batch_cap` blocks and
zero-padded the rest, so a 999-query batch computed 25 queries of zeros — and
they cost more than their 2.4% share, because a zero query's table is
degenerate, every candidate ties at the threshold, and the compaction fires
far more often than it does for real queries.

---

## What is not finished

| | why it matters | state |
|---|---|---|
| IVF-RaBitQ VRAM | its bits per dimension say 11% under JHQ; the measured figure is not captured | one run, queued |
| Where the high-recall gap comes from | bytes, α, cache reuse and LUT divergence are each measured and each too small; the selection stage at high nprobe is the one untested candidate | the v42 split predates v47's factorised table and cannot run at this M and BLOCK; needs porting forward |
| CPU baseline protocol | three changes in `JHQ_repro`, listed below | not started |
| Recall ties in the selection | `jhq_compact_topck` compares distance alone, which is the 1e-4 wobble between builds | not started |

---

## 6. The batch size changes who wins  ▸ finished

Every other number here is one batch, and IVF-RaBitQ's paper reports 10⁴ on a
different card. `data/batch_sweep.log` sweeps it on both systems, nprobe=128,
real queries (duplicating them to reach 10⁴ would not do: `data/qdup.log`
measures that it inflates throughput 4–26% through L2 reuse alone).

| batch | openai3-3072 JHQ/RaBitQ | vogue-768 JHQ/RaBitQ |
|---:|---:|---:|
| 32 | 1.55× | 0.48× |
| 64 | 2.04× | 0.68× |
| 128 | **2.44×** | 0.84× |
| 256 | 1.92× | 0.94× |
| 512 | 1.71× | 0.98× |
| 1024 | 1.49× | **1.12×** |

**The ratio is not stable in batch, and it moves in opposite directions on the
two datasets.** On openai3-3072 JHQ's lead peaks at batch 128 and is falling by
1024; on vogue-768 IVF-RaBitQ is twice as fast at batch 32 and JHQ only
overtakes at 1024, still climbing.

Two things follow. Our own per-block fixed cost — building the table — is what
loses vogue-768 at small batch, and it amortises away. And **a single-batch
comparison is incomplete, in either direction**: extrapolating vogue-768's
trend to the 10⁴ that IVF-RaBitQ's paper uses would favour JHQ, while
extrapolating openai3-3072's would not.

This is the answer to the sharpest question available about the evaluation —
everything runs on one card at one operating point, and `data/v54.log` already
shows a kernel verdict flipping with a card's bandwidth. The batch axis is now
measured; the second card is not.

---

## 7. The second level, measured  ▸ finished

Everything in Section 3 presupposes that the residual level earns its place:
selective refinement, the alpha budget, the two-level scan. `data/hierarchy_ablation.log`
drops it -- `JHQ_NO_RESIDUAL`, primary codes only -- on the same build,
datasets and nprobe grid as `data/paper_fronts.log`.

Recall ceiling, over every nprobe:

| dataset | JQ (primary only) | JHQ | gain |
|---|---:|---:|---:|
| arxiv-768 | 0.5137 | 0.9898 | **+0.476** |
| vogue-768 | 0.6371 | 0.9939 | **+0.357** |
| bge-m3 | 0.7273 | 0.9815 | +0.254 |
| stella | 0.7459 | 0.9957 | +0.250 |
| openai3-1536 | 0.8127 | 0.9864 | +0.174 |
| openai3-3072 | 0.8642 | 0.9926 | +0.128 |

**The primary level alone tops out between 0.51 and 0.86.** Adding nprobe does
not fix it -- arxiv-768 gains 0.0002 of recall going from nprobe 128 to 1024 --
because what is missing is resolution, not candidates. The second level is
what makes the method reach the recall regime the paper is about, and it costs
1.3x to 2.4x of throughput to run.

That is the motivation Section 3 needs, and it was asserted rather than
measured until now. The previous version of this ablation
(`results/pre_freeze_v22_s2b1/`) ran on v21, before the factorised table, the
probe cursor, the word layout, the launch fix and the alpha rule, so its
numbers could not sit in the same table as the rest of Section 6.

## 8. Memory, measured  ▸ finished

Same call on both sides: `cudaMemGetInfo` after the index is built and the
search workspace allocated (`data/vram.log`, `data/paper_fronts.log`).

| dataset | IVF-RaBitQ | JHQ | JHQ above |
|---|---:|---:|---:|
| vogue-768 | 1,012 MiB | 1,405 MiB | +39% |
| arxiv-768 | 2,018 MiB | 2,545 MiB | +26% |
| openai3-1536 | 1,836 MiB | 2,353 MiB | +28% |
| openai3-3072 | 3,324 MiB | 4,047 MiB | +22% |
| bge-m3 | does not build | 12,113 MiB | — |
| stella | does not build | 20,837 MiB | — |

**IVF-RaBitQ has the smaller footprint wherever it builds**, by more than its
bits per dimension alone predict: 8 against 9 is 11%, and the measured gap is
22–39% because JHQ also carries the selection buffer and the factorised table
in its workspace.

Write this as the trade it is. **Do not write that JHQ is more memory
efficient** -- it is not, and a reviewer who counts bits will see it in a
minute.

---

## 9. A defect in our own coarse quantiser on vogue-768  ▸ found, not fixed

Exporting the IVF for the CPU reference printed a list histogram, and vogue-768
has one list holding **135,489 of 932,328 vectors — 14.5%**, against a mean of
227.6 and a 99th percentile of 839. The next four hold another 7.5%. No other
dataset is close: the largest list is 0.01% on stella, 0.06% on arxiv-768 and
openai3-1536, 0.11% on openai3-3072, 0.28% on bge-m3.

**It is not duplicate data.** Sampling 20,000 vectors from that list gives
19,966 distinct, the same rate as 20,000 sampled from the whole base. All
vectors are unit norm, and the list's members sit at mean cosine 0.623 to
their own centroid -- a broad cone, not a tight cluster. One centroid has
absorbed a diffuse region, which is what an under-trained or badly initialised
k-means does.

All six datasets train the coarse quantiser at the same ratio, 39 points per
centroid, so the budget alone does not explain it; vogue-768's distribution
and that budget together do.

### Why it matters for the numbers already taken

A query that probes this list scans 135,489 candidates where the expected
total for the whole query is `nprobe * mean = 128 * 227.6 = 29,133`. **One
list is five times the intended candidate budget.**

So vogue-768's throughput figures are pessimistic for JHQ, and the comparison
against IVF-RaBitQ on that dataset is unfair in our own disfavour: cuVS trains
its own coarse quantiser and does not inherit this. That is worth stating
plainly rather than leaving as an unexplained weak column -- vogue-768 is
where JHQ trails IVF-RaBitQ most at small batch (0.48x at batch 32) and where
the alpha rule's gain decays fastest.

### What would settle it

Retrain vogue-768's IVF with more points per centroid and re-measure that
column. That needs the GPU, which was released before this was found. Until
then the honest treatment is a footnote on the vogue-768 row, not a silent
number.

Evidence: `data/export_ivf.log`, and the histogram is reproducible from
`<cpu_baseline>/vogue-768/vogue-768_cluster_id_4096.ivecs`.

---

## 10. The CPU baseline: what it is, and the three changes it needs

`results/cpu_baseline/README.md` settled this before this session began, and
re-deriving it cost hours that should not have been spent. Copied here as
`data/cpu_baseline_artifact_note.md`.

**The authors' artifact cannot be built.** Four defects, each sufficient alone
-- a `.gitmodules` path that does not match the tree and carries no gitlink, so
FAISS can never be fetched; AVX intrinsics in three files without
`<immintrin.h>`; no architecture flags anywhere; examples constructing classes
that are unrelated in the shipped headers. Then, with those worked around, it
needs FAISS headers introduced after 1.9.0 together with an
`InvertedListScanner` interface retired before them. Nothing pins a version and
the API points two ways at once.

**That does not leave the paper without a CPU baseline.** Every CPU number in
`results/jhq_cpu_ivf_*.csv` -- six datasets -- comes from `JHQ_repro`, a
from-scratch C++17 reimplementation that builds, implements equation 4 as the
paper states it, and shares `train_1d_kmeans`'s lineage with `cpu/` here. As a
baseline it is sound and its protocol can at least be stated, which the
authors' cannot.

**What is wrong is only the protocol.** `src/jhq_ivf_index.cpp:96` calls
`train_1d_kmeans` without `max_iter`, taking the header default of **25** --
which `results/v34_lloyd_iterations/` measures as 6-8e-3 of recall short of the
2000 the GPU side uses. The recorded 238.55 s on stella therefore describes a
25-iteration codebook against GPU rows describing a 2000-iteration one.

Three changes, and the comparison is CPU against GPU and nothing else:

1. pass `max_iter=2000` at `src/jhq_ivf_index.cpp:96`;
2. match the residual training-set size to the GPU side;
3. pin the thread count -- `results/` records 32 threads beating 208 by 1.9x on
   this host, so an unpinned column measures how busy the machine was.

`JHQ_repro` builds without FAISS and the datasets are already in place, so this
is an afternoon, not a port. The centroid export written this session
(`examples/export_ivf_for_cpu.cu`) is not needed for it -- that was for the
authors' artifact, which reads its IVF from files. `JHQ_repro` trains its own,
so matching `max_train_n` is what aligns the routing there.
