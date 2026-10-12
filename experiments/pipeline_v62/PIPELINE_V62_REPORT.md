# v62: list-parallel primary screening with an exact global top-ck

**Decision: GO.** Correctness passes on every query, every batch size and every G (0 non-tie
differences, final top-k bit-identical). At b = 8 every workload gains ×2.06–2.76 end to end, and at
b = 32 ×1.63–2.30, stable over 5 paired rounds (min/max within 0.5%). The stage times explain the
gain. At b ≥ 256 v62 regresses (to ×0.46 at arxiv α=200, b=1000, G=8). It is a small-batch execution
mode, not a replacement for v61's scan. Experiments 2 and 4 of the pipeline campaign
(`experiments/pipeline_campaign/MASTER_REPORT.md`) follow up on both findings.

Everything here is one index per dataset: the arms6 trained-state cache, i.e. the paper's index. All
variants of a configuration run in one process on the same queries. The build is Release,
`CMAKE_CUDA_ARCHITECTURES=80` (compute_80 PTX, JIT-compiled by driver 595.71.05 for the RTX 5090),
CUDA 13.0. Commits: Phase 0 `b3b3432`/`7723059`, v62 `862543a`, run script `e0d0518`. Raw logs, build
flags, per-query check outputs, sanitizer logs and nsys traces (gzipped CSV) are in `raw/`.

## 1. Pipelines

```
v61 (G = 0), one CUDA graph per (B, ck, nprobe, lanes):
  H2D q -> rotate (cuBLAS) -> centroid dots (cuBLAS) -> select_probes [B blocks]
        -> build_byte_lut -> scan_ivf_exact [B blocks: one per query, all its probed lists,
           threshold-compaction top-ck] -> residual_refine L16 -> batched_topk_final [B blocks] -> D2H

v62 (G = 2, 4, 8):
  ... same up to the table ...
        -> scan_ivf_group [B x G blocks: block (q, g) owns candidates [g*T/G, (g+1)*T/G) of query q's
           probed lists laid end to end in probe order; exact local top-ck under (distance, position)]
        -> merge_group_topk [B blocks, log2 G pairwise merge-path rounds, ping-pong in global memory]
        -> residual_refine L16 (v61's, unchanged) -> batched_topk_final -> D2H
  G = 1: one group writes the top-ck arrays directly (no merge): v61's work with the total order.
```

## 2. Thread and block mapping

- **Group scan.** `grid = (B, G)`, `block = JHQ_BLOCK` threads (512, or 1024 for openai3-3072).
  - *Candidate split:* T = `query_total[q]` is read on the device. The cut points `T*g/G` are computed
    there, with no per-query host sync, and a binary search over `probe_offsets` finds each thread's
    first list. Groups are balanced by candidate count (shares differ by at most one candidate), and a
    long list may be split between two groups.
  - *Per thread:* one candidate per chunk. The distance code is v61's: the same word loads and the
    same additions, so distances are bit-identical (max |Δprimary| = 0).
  - *Shared memory:* the split LUT, plus a `cap = pow2 ≥ ck + BLOCK` (distance, position) buffer
    sorted bitonically under the strict total order `key_less = (distance, position)`.
  - Residual codes are not touched.
- **Merge.** One 256-thread block per query.
  - Round r merges list pairs. Each pair gets `256 / pairs` threads, and each thread finds its output
    slice with a merge-path binary search, then merges it sequentially.
  - Every round keeps `min(ck, a + b)`. The lists live in two global buffers, `grp [B][G][ck]` and
    `mrg [B][G/2][ck]`, so shared memory holds only the 2·G counts, whatever ck is.
- **Scratch:** `B·G·ck·8 + B·(G/2)·ck·8` bytes, grown on demand and allocated before capture. At
  batch_cap = 1024 that is 60 MiB (vogue α=64), 94 / 188 MiB (arxiv α=100 / 200) and 3.8 MiB (openai3
  α=4) at G = 8. Device memory in use rises by the same amount (e.g. arxiv α=200: 2653 → 2749 MiB).

## 3. Why the global top-ck is exact

Write ≺ for `key_less`: a strict total order on a query's candidates, because positions are unique.
Let S be the query's global top-ck under ≺.

1. Take x ∈ S in group g. Fewer than ck candidates of the whole query precede x, so fewer than ck of
   group g do, and x is in g's local top-ck. The threshold-compaction buffer keeps exactly the
   local top-ck under ≺: it admits only candidates before the current ck-th key, never overflows
   (`cap ≥ ck + BLOCK`), and its sort is a total-order sort. So ∪_g local(g) ⊇ S.
2. Merging two sorted lists and keeping the first `min(ck, a+b)` keeps the smallest `min(ck, a+b)`
   of their union. Every member of S that is in either list is among them, since fewer than ck
   elements of the union precede it. By induction over the rounds, the final list is S, in order.
3. Empty or short groups carry their count (`grp_cnt`), and padding is (INF, −1) after the valid
   prefix, so nothing is silently truncated. The total order removes every tie, so the result does
   not depend on the grouping, the block size or atomic append order.

v61 orders by distance alone, so among candidates tied exactly at the ck-th distance it keeps
whichever the scan appended first. v62's tie choice is deterministic; v61's is not, and its choice
varied with batch size (below).

## 4. Results

### Phase 0: v61 L16 stage times (nsys, GPU µs per search, median)

| workload | b | latency ms | scan | refine | probe select | top-k | rest + host | scan share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv α100 | 8 | 1.076 | 849 | 111 | 56 | 23 | 38 | 79% |
| arxiv α200 | 8 | 1.277 | 935 | 209 | 57 | 31 | 45 | 73% |
| openai3 α4 | 8 | 0.973 | 723 | 24 | 78 | 16 | 132 | 74% |
| vogue α64 | 8 | 1.248 | 1057 | 72 | 53 | 22 | 45 | 85% |
| arxiv α100 | 1000 | 4.500 | 2914 | 714 | 227 | 64 | 581 | 65% |
| vogue α64 | 1000 | 4.583 | 3377 | 459 | 215 | 57 | 476 | 74% |

The scan was 61–85% of latency at every batch size, far above the 10–15% stop threshold. At b = 8 it
ran 8 blocks on a 170-SM GPU. Full tables: `raw/phase0/stages.md`.

### Phase 2: correctness (every G against G = 0, all queries, b = 8, 32, 256, 1000)

- **Coverage:** 64 comparisons (4 workloads × 4 batch sizes × G = 1, 2, 4, 8).
- **Probes:** identical for every query.
- **Primary top-ck:** NON-TIE 0, max |Δ primary distance| = 0.
- **Final top-k:** ordered ids and distance bits identical for 100% of queries.
- **Recall@10:** identical.
- **Tie-only top-ck differences:** 0–9 queries per comparison (184 in total). They are identical
  across G = 1/2/4/8 and differ across batch sizes, e.g. arxiv α100: 0 at b = 8, 1 at b = 1000.
  That pattern is v61's order-dependent choice at exact ties, not v62's.
- **Sanitizers:** compute-sanitizer memcheck reports 0 errors (arxiv, vogue, openai3; 64 queries).
  racecheck reports 0 hazards and synccheck 0 errors (vogue, openai3), on `scan_ivf_group` and
  `merge_group_topk`. racecheck ran at 8 queries and b = 8; at 64 queries it did not finish within
  10 minutes. Every G still runs, and the batch size changes only grid.x.

### Phase 3: end to end, paired speedup over G = 0 (median [min, max] of 5 rounds)

| workload | b | G0 ms | G1 | G2 | G4 | G8 |
|---|---:|---:|---|---|---|---|
| arxiv α100 | 8 | 1.078 | ×0.98 | ×1.37 | ×1.83 | **×2.32** [2.316, 2.320] |
| arxiv α100 | 32 | 1.159 | ×0.98 | ×1.37 | ×1.79 | **×1.82** |
| arxiv α100 | 256 | 1.719 | ×0.96 | ×0.93 | ×0.87 | ×0.75 |
| arxiv α100 | 1000 | 4.549 | ×0.96 | ×0.86 | ×0.71 | ×0.57 |
| arxiv α200 | 8 | 1.278 | ×0.97 | ×1.30 | ×1.67 | **×2.06** [2.058, 2.063] |
| arxiv α200 | 32 | 1.355 | ×0.97 | ×1.28 | ×1.62 | **×1.63** |
| arxiv α200 | 256 | 2.075 | ×0.93 | ×0.88 | ×0.79 | ×0.69 |
| arxiv α200 | 1000 | 5.778 | ×0.96 | ×0.81 | ×0.66 | ×0.46 |
| openai3 α4 | 8 | 0.974 | ×0.99 | ×1.49 | ×1.99 | **×2.37** [2.372, 2.377] |
| openai3 α4 | 32 | 1.020 | ×0.99 | ×1.46 | **×1.89** | ×1.70 |
| openai3 α4 | 256 | 2.179 | ×0.99 | ×1.00 | ×0.96 | ×0.81 |
| openai3 α4 | 1000 | 6.909 | ×0.99 | ×0.95 | ×0.87 | ×0.71 |
| vogue α64 | 8 | 1.249 | ×0.99 | ×1.53 | ×2.15 | **×2.76** [2.756, 2.761] |
| vogue α64 | 32 | 1.356 | ×1.01 | ×1.57 | ×2.23 | **×2.30** |
| vogue α64 | 256 | 2.172 | ×0.98 | ×1.21 | **×1.23** | ×1.11 |
| vogue α64 | 1000 | 4.584 | ×0.99 | ×0.99 | ×0.89 | ×0.73 |

Recall@10 is identical in every row. Latency is the median batch latency. QPS is that of a
synchronous client issuing batches back to back. Both use the same timing scope for every variant.

### Stage times behind the gain (nsys, GPU µs per search, median)

| workload | b | G | scan | merge | refine | other GPU | host+gaps | kernels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv α100 | 8 | 0 | 849 | – | 111 | 100 | 19 | 8 |
| arxiv α100 | 8 | 8 | 216 | 16.7 | 111 | 100 | 20 | 9 |
| arxiv α200 | 8 | 0 | 939 | – | 210 | 110 | 19 | 8 |
| arxiv α200 | 8 | 8 | 225 | 36.5 | 210 | 111 | 20 | 9 |
| openai3 α4 | 8 | 0 | 728 | – | 25 | 194 | 27 | 9 |
| openai3 α4 | 8 | 8 | 155 | 3.7 | 25 | 194 | 31 | 10 |
| vogue α64 | 8 | 0 | 1059 | – | 72 | 90 | 28 | 8 |
| vogue α64 | 8 | 8 | 246 | 9.1 | 72 | 90 | 22 | 9 |
| arxiv α200 | 1000 | 0 | 3530 | – | 1349 | 632 | 268 | 7 |
| arxiv α200 | 1000 | 8 | 7652 | 2624 | 1361 | 632 | 270 | 8 |

- **Small batch:** at b = 8 the scan shrinks 3.9–4.7× and the merge adds 4–37 µs; nothing else
  moves. The e2e gain is the scan saving minus the merge.
- **Launch count:** one extra kernel for G > 1. The graph's host+gaps barely changes (15–31 µs).
- **Full table:** `raw/perf/stages.md`.

## 5. Small against large batches

- **b = 8 (8 blocks).** Grouping multiplies the block count by G, and it is the only lever. Even at
  G = 8 there are only 64 blocks, so the gain keeps rising with G.
- **b = 32.** G = 4 (128 blocks) and G = 8 (256 blocks) are equal within ~2%, except openai3, whose
  1024-thread blocks fill the device earlier and lose 10% at G = 8. The device is full.
- **b ≥ 256.** v61 already has ≥ 256 blocks of work, and grouping only adds work:
  - every group fills its own buffer from an infinite threshold, so admissions and bitonic sorts
    grow roughly with G;
  - the table is staged into shared memory G times per query;
  - the merge's sequential, uncoalesced per-thread reads reach 2.6 ms at arxiv α200 b=1000 G=8.

  Scan time therefore rises with G (arxiv α200 b=1000: 3530 → 7652 µs).
- **vogue at b = 256** is the exception (G4 ×1.23). vogue's per-query candidate counts vary far more
  (CV 0.51, against 0.08–0.11 elsewhere; the largest query in a batch is 2.2× the mean, against
  1.3×). With one block per query, the batch waits for its largest query.

## 6. Failure cases and bottlenecks

- **Regression at large batch, all workloads but vogue:** ×0.46–0.96 at b ≥ 256. The causes are in
  §5: duplicated selection work, duplicated table staging, and an uncoalesced merge.
- **G = 1 costs 1–4%** against v61 at large batch. That is the price of the total order: the extra
  position comparison in admission and sorting. It buys determinism.
- **The buffer sizing rule is fragile.** `cap = pow2 ≥ ck + BLOCK` leaves `cap − BLOCK − ck` free
  slots after each compaction. At arxiv ck = 500 that is 12 slots, so it compacts nearly every
  chunk, and the scan is 1.36× slower than at ck = 1000. This affects v61 and v62 alike and is not
  among the paper's configurations (found in campaign Experiment 2).
- **What remains at b = 8 / G = 8:**
  - the scan, still 47–56% of e2e;
  - refinement (16–35%);
  - probe selection (56 µs; B blocks, the same parallelism shortage);
  - the final top-k (16–31 µs; B blocks).

## 7. Decision

**GO.** Correctness passes; at b = 8 and 32 every workload exceeds ×1.15 (×1.63–2.76, stable); the
stage times account for it. Only for small batches, and it needs a policy for large ones.

## 8. Proposal at the time: budget-aware dynamic local quotas (not implemented in v62)

The proposal was to give each group a quota b_i < ck plus a certificate:
- let u_i be the group's best unreturned key, and τ the ck-th key of the returned union;
- the certificate holds if no u_i precedes τ;
- otherwise widen the quota or fall back to the full local top-ck.

The aim was to cut the G·ck intermediate results and the merge.

**What happened next** (campaign Experiment 2):
- A pilot measured this before building it. The certificate holds iff n_i ≤ b_i for every group,
  where n_i is group i's count of global top-ck members, and group 0 (the nearest lists) holds 46–81%
  of the top-ck.
- So equal quotas, which equal count-proportional quotas under v62's split, almost never certify.
  Calibrated quotas certify ≥ 99% but cannot shrink group 0, which sets the critical path.
- Any failure stalls the whole batch on a single-query fallback.
- Quotas were therefore dropped. In their place, v63 shares one exact bound among a query's groups,
  with no certificate and no fallback. That was measured GO: ×1.03–1.07 over v62 at b = 8, up to
  ×1.56 at b = 1000, with identical results.
