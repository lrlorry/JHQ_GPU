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
Stopping at the first rejection makes it one-sided — it can pick an α larger
than needed, never smaller.

| claim | number | file |
|---|---|---|
| worth 1.0× to 2.65× at equal recall, six datasets, nprobe 8–1024 | 72 runs, one build | `data/paper_fronts.log` |
| recovers the swept saturation α | 7 of 8 configurations | `data/alpha_sample.log`, `data/alpha_fast.log` |
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
| CPU baseline protocol | `max_iter`, `max_train_n`, thread pinning in `JHQ_repro` | not started; that repo is untouched |
| Recall ties in the selection | `jhq_compact_topck` compares distance alone, which is the 1e-4 wobble between builds | not started |
