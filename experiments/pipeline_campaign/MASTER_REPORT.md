# JHQ-GPU pipeline campaign: master report

Branch `exp/pipeline-campaign`. RTX 5090 (170 SMs), driver 595.71.05, CUDA 13.0, Release builds,
`CMAKE_CUDA_ARCHITECTURES=80` unless stated (the build every earlier JHQ-GPU number came from). The
index is the paper's: arms6 trained-state caches, `run_budget_arms6.sh` settings. All variants of a
configuration run in one process on the same index and queries. Timing is the paper's scope (wall clock
around `search()`: H2D, graph launch, D2H, sync), 5 rounds with variant order rotated, and speedups are
paired within a round (median [min, max]). Unattended GPU time was 16:44–22:02 UTC on 2026-10-12,
within the 6-hour budget.

## Bottom line

- **Best pipeline: v65** (`jhq_v65_schedule`, `JHQ_PRIMARY_GROUPS=-1`, `JHQ_REFINE_LANES=16`). It
  combines v61's L16 refinement, v62's list-parallel primary screening with v63's exact shared bound,
  and a per-batch host-side schedule. Results are identical to v61 L16: 0 non-tie primary differences,
  final id sets identical in all 234,320 query comparisons, and Recall@10 identical.
- **End to end over v61 L16**, on 6 datasets × np 128/512 at the paper's sampled-rule α (geometric
  means over 13 rows; 11 at b = 1000):

  | b | v65 | range over rows | vs per-row oracle fixed G |
  |---:|---:|---|---|
  | 8 | **×2.73** | ×2.01–3.78 | within 3.4% |
  | 32 | **×2.36** | ×1.67–3.13 | within 0.4% |
  | 256 | ×1.06 | ×0.997–1.44 | regret up to 22% (bge-m3, stella) |
  | 1000 | ×1.015 | ×0.99–1.10 | within 1% |

  Across the recall–QPS frontier (α ×½ and ×2), b = 8 gains ×1.81–3.18. At b = 1000 the frontier is
  unchanged except vogue (+6–14%).
- **What worked:**
  - **Parallel primary screening:** v62, at small batch.
  - **Exact cross-group pruning:** v63, which makes the list-parallel path cheap enough to use at mid
    batch and removes most of v62's large-batch loss.
  - **A two-term schedule:** v65, which never loses to v61 by more than 0.3% in any row.
- **What failed or was skipped:**
  - quota-based budgets (pilot NO-GO);
  - routing by Πᵀc (v64, NO-GO);
  - a new data layout and selective kernel fusion (both SKIPPED, with measured evidence that their
    preconditions do not hold).
- **Research verdict.** The speedup is real, reproducible and correctness-preserving. But it comes
  from GPU scheduling techniques that are standard in cuVS-style IVF search: list-parallel scan plus
  merge, threshold sharing, occupancy-driven work splitting. They are adapted to the one property
  that distinguishes JHQ here, a large, exact global top-αk. It is a systems/engineering
  contribution, not a new algorithm (§ Novelty).

## Decision table

| Version | Hypothesis | Correctness | Best E2E speedup (paired) | Regression | Decision |
|---|---|---|---|---|---|
| v61 L16 (baseline) | — | (v61 report) | L16 over L0: ×1.01–2.37 (this campaign, all rows) | — | baseline |
| sm_120 native build (Exp 0) | a native build is faster than compute_80 PTX JIT | recall equal within 2e-4, NON-TIE 0 | L0 ×1.10–1.31 (768-d) | v61 pipeline 5–27% slower at b ≤ 32; openai3 ×0.92–0.95 | reported separately, not adopted |
| v62 list groups (Exp 1) | one query's primary screening split over G blocks lowers small-batch latency | 64/64 comparisons NON-TIE 0, final bit-identical; sanitizers clean | ×2.76 (vogue b=8), ×2.06–2.76 at b=8, ×1.63–2.30 at b=32 | ×0.46–0.96 at b ≥ 256 | **GO** |
| v63 dynamic local quotas (Exp 2, pilot) | per-group quotas b_i < ck with a certificate cut intermediate results | (offline) certify iff n_i ≤ b_i | upper bound ≤ ~4% at b = 8 (merge only) | batch-level fallback stalls | **NO-GO** (not built) |
| v63 shared bound (Exp 2) | groups share an exact bound: any group's ck-th key bounds the global ck-th | 48/48 identical to v62 (top-ck and final), NON-TIE 0, sanitizers clean | over v62: ×1.03–1.07 (b=8), ×1.56 (arxiv α200 b=1000 G8) | none over v62; openai3 b ≤ 32 ±0.5% | **GO** |
| v64 routing by Πᵀc (Exp 3) | scoring centroids against Πᵀc lets routing overlap rotation | NON-TIE 0 on same-probe queries; final top-k identical; probes differ for 1–8% (768-d) and 40% (openai3) of queries | ×1.005–1.017 at b ≤ 32; ×1.045 at openai3 b=256 | concurrent kernels 8.5% slower (contention) | **NO-GO** |
| v65 schedule (Exp 4) | a fixed rule picks G per batch close to the best fixed G | identical to v61 L16 (above) | geomean ×2.73 / ×2.36 / ×1.06 / ×1.015 at b = 8/32/256/1000 | worst row ×0.997; regret up to 1.22 at b=256 | **GO** |
| v66 layout (Exp 5) | the primary scan is layout-bound | — | — | — | **SKIPPED** (evidence: §Exp 5) |
| v67 fusion (Exp 6) | merge/refine/top-k boundaries cost sync or traffic | — | — | — | **SKIPPED** (evidence: §Exp 6) |
| v65 = final integration (Exp 7) | the successful modules combine | as v65 | as v65 | as v65 | **final pipeline** (no separate v68: nothing else to integrate) |

## Experiment 0: baseline audit and profiling

- **v61 L16 stage times** (nsys, `experiments/pipeline_v62/raw/phase0/stages.md`).
  - The primary scan is 61–85% of latency at every batch size and every workload.
  - At b = 8 it runs one block per query, 8 blocks on 170 SMs: 723–1057 µs of a 0.97–1.25 ms search.
  - Probe selection (52–78 µs) and the final top-k (16–31 µs) are also one-block-per-query kernels.
  - Bounds: if the scan cost nothing, e2e would be ×2.6–6.5; split 8 ways, ×2.1–3.9.
  - Primary screening is the bottleneck, and the small-batch problem is a lack of parallelism.
- **Native sm_120** (`exp0/EXP0_TABLES.md`). It builds and runs (CUDA 13.0 has compute_120), with
  native SASS confirmed by `cuobjdump`. It is not uniformly faster:
  - v61 L0: ×1.10–1.31 faster on the 768-d sets, ×0.95 on openai3.
  - v61 L16: within ±8%, so L16/L0 shrinks under native (arxiv α200: ×1.74 against ×2.39).
  - The small-batch v61 pipeline is slower native (arxiv b=8: 1.364 against 1.078 ms).
  - v62's relative speedups hold or grow (b = 8: ×2.33–2.78).

  Campaign numbers therefore stay compute_80, and sm_120 is reported only in its own table. No
  speedup mixes the two.
- **Latency against throughput.** The harness reports both:
  - single-batch latency, as the median per call;
  - throughput, as queries over summed time for a synchronous client issuing batches back to back.

  Both use the same scope for every variant. All small-batch claims hold for both.

## Experiment 1: v62, parallel primary screening

Full report: `experiments/pipeline_v62/PIPELINE_V62_REPORT.md`. It covers the diagrams, thread
mapping, exactness proof, per-G stage and e2e tables, small against large batch, failure cases, and
the proposal that led to Experiment 2.
- **What changes:** at b = 8 the scan shrinks 3.9–4.7× (e.g. vogue 1059 → 246 µs) and the merge adds
  4–37 µs.
- **Why b ≥ 256 regresses:** every group refills its buffer from an infinite threshold, the table is
  staged G times, and the merge is uncoalesced (2.6 ms at arxiv α200 b=1000 G=8).
- **v61's tie choice is order-dependent.** It keeps whichever exact tie the scan appended first, and
  its tie-only differences moved with batch size. v62 and later are deterministic under a
  (distance, position) total order.

## Experiment 2: budget-aware global selection

**Pilot** (`exp2/raw/quota_study.md`, `exp2/raw/nsys/stages.md`; nothing built before it). A quota
b_i certifies the returned union iff n_i ≤ b_i for every group, where n_i is group i's count of
global top-ck members. That is proved in `exp2_quota_study.py` and needs no GPU run of the policy.
Probes are sorted nearest-first, so group 0 holds most of the global top-ck: 46–81% at G = 8.
- **Equal quotas** (identical to count-proportional quotas under v62's balanced split) almost never
  certify: at G = 8, slack 2 certifies 0–2.1% of queries, and slack 4 (half of v62's ck) 5–67%.
- **Per-group quotas calibrated** on half the queries certify 99.2–100% of the other half at slack 1.5,
  returning 30–35% of v62's G·ck.
- **But the gain cannot appear at small batch.** Group 0's quota cannot shrink, and with B·G blocks
  under-filling the GPU the slowest block sets the kernel time. Running v62 at ck = b, the first pass
  of any quota scheme with zero certificate cost, bounds the saving to the merge: ≤ 27 µs, ~4% at
  arxiv α200 b = 8, < 2% elsewhere.
- **And every failed certificate stalls the whole batch** on a single-query fallback, with
  probability 1 − (1−p)^B.
- Verdict: **NO-GO for quotas**. No adaptive tuning was attempted.

**The optimisation that survived: v63's shared bound** (`exp2/v63_raw/summary.md`).
- **Mechanism:** each group publishes its ck-th key with `atomicMin` into a per-query 64-bit word
  (distance bits << 32 | position, order-preserving for non-negative distances). It admits only
  candidates at or before the bound, and when done writes only the prefix at or before it.
- **Exact under any timing:** a published ck-th key has ck candidates at or before it, so nothing
  after it can be in the global top-ck, and the bound only falls. No certificate, no fallback.
- **Measured exact:** 48/48 configurations give top-ck and final results identical to v62.
  memcheck reports 0 errors, racecheck 0 hazards, synccheck 0 errors.
- **v63 over v62 at the same G:**

  | b | speedup |
  |---:|---|
  | 8 | ×1.03–1.07 (openai3, ck = 40: ×1.00) |
  | 32 | ×1.04–1.14 |
  | 256 | ×1.06–1.43 |
  | 1000 | ×1.02–1.56 |

  E.g. arxiv α100 b=256 G4: scan 1432 → 986 µs; arxiv α200 b=1000 G8: merge 2629 → 471 µs.
- **Primary-distance computation is unchanged.** Every candidate is still scored; the saving is in
  admissions, sorts and merge traffic. That is why it shows most where the GPU is full.
- **Ablation, Fixed Local ck (v62) vs Dynamic Local Budget:** the quota form is bounded and rejected
  above; the exact shared-bound form is kept. The answer to "is a fixed two-stage quota faster than
  multi-round expansion?" is that neither is worth building here, because the bound needs no rounds.

## Experiment 3: v64, routing by Πᵀc

`exp3/raw/summary.md`.
- **Identity:** ⟨Πq, c⟩ = ⟨q, Πᵀc⟩ holds for any Π, and probe selection reads only these dot products
  and ‖c‖². So no orthogonality is needed for exactness in real arithmetic.
- **Orthogonality, measured:** max |ΠᵀΠ − I| is 6.0e-7 (768-d) and 6.5e-5 (openai3-3072).
- **Concurrency:** a second stream with its own cuBLAS handle; fork/join inside the CUDA graph.
- **Correctness:** NON-TIE 0 on every query whose probes agree; final top-k and recall identical
  everywhere.
- **Probe drift:** float differences (dot error ≤ 8e-5 relative) change the probe set for 1–8% of
  queries on the 768-d sets and 40% on openai3. That is a semantic deviation at near-ties, harmless
  here.
- **Mode 1 vs 0:** the coordinate change alone is free (×1.000).
- **Mode 2 vs 0:** overlap saves 5 µs (768-d, b = 8), 16 µs (openai3, b = 8) and 171 µs (openai3,
  b = 1000) of front-end span. End to end that is ×1.003–1.017 at b ≤ 32 and at best ×1.045 (openai3,
  b = 256). Concurrent kernels slowed 8.5% (openai3, b = 1000: kernel sum 1403 → 1522 µs).
- **Verdict: NO-GO.** The gain is small, and the probe drift is real.
- The identity is textbook. This codebase already uses Πᵀc for assignment (`ensure_assign_centroids`).

## Experiment 4: v65, adaptive execution schedule

- **Why it runs:** v62/v63 and v61 each win in different regimes. G = 8 wins at b ≤ 32, v61's scan at
  b ≥ 256, and vogue keeps G > 1 up to b = 1000.
- **The cause of vogue's exception:** its per-query candidate count has CV 0.51 and a batch-max/mean
  of 2.2, against 0.08–0.11 and 1.3 elsewhere. Under one block per query, a batch waits for its
  largest query.
- **Rule** (two constants, fitted on the 4 pipeline_v62 workloads before any other configuration ran):
  - *fill* is the largest G with B·G·BLOCK ≤ 0.6 × (SMs × max threads per SM);
  - *skew* is 4 if the per-query candidate-count CV ≥ 0.3, else 1;
  - G = max(fill, skew), and G = 1 runs v61's scan.
- **The CV is measured once per index** on 512 base vectors as pseudo-queries, never on evaluation
  queries. It agrees with the real queries (vogue 0.495 vs 0.509).
- **No host–device synchronisation per search.** The schedule cost is inside the measured e2e time.
- **Result** (`exp47/BENCH_TABLES.md`; training 4 rows, held-out 9 rows per b):

  | b | v65 geomean (training / held-out) | oracle | worst regret |
  |---:|---|---|---|
  | 8 | ×2.32 / ×2.93 | ×2.30 / ×2.94 | 1.034 |
  | 32 | ×1.96 / ×2.56 | ×1.96 / ×2.56 | 1.004 |
  | 256 | ×1.10 / ×1.04 | ×1.12 / ×1.15 | **1.22** |
  | 1000 | ×1.02 / ×1.01 | ×1.02 / ×1.02 | 1.010 |

- **The misses** are bge-m3 and stella at b = 256 (nlist 32768). There G = 8 gives ×1.11–1.22, but
  their CV is low (0.08–0.14), so the rule picks G0. Per-query totals are not the whole story; a
  list-size statistic would be the next feature to try. The schedule never falls below ×0.997 of v61.
- **The oracle** (best fixed G per row, read off the test queries) is reported only as a bound.
- **Fixed G = 8** alone reaches ×2.73 / ×2.34 at b = 8 / 32 but ×0.89 at b = 1000. The schedule keeps
  the small-batch gain without the large-batch loss.

## Experiment 5: stage-aware data layout — SKIPPED

The precondition, profiling evidence that layout is a bottleneck, does not hold.
- **Large batch is near bandwidth.** At b = 1000 v61's scan already streams primary codes at a
  no-reuse demand of 1.34–1.64 TB/s against the RTX 5090's 1.79 TB/s: vogue 48,000 candidates ×
  96 B in 3.44 ms, arxiv 47,143 × 96 B in 2.92 ms, openai3 18,316 × 384 B in 4.30 ms. The word layout
  is already coalesced. Moving fewer bytes would mean changing the codes, which is forbidden.
- **Small batch is parallelism-bound,** and v62/v63 address that.
- **No counter-level proof:** Nsight Compute counters are unavailable (ERR_NVGPUCTRPERM), so this
  rests on demand estimates only.
- Existing layout work (word layout, v50 `JHQ_REGROUP` / `JHQ_FUSE_REFINE`) was not repackaged.

## Experiment 6: selective kernel fusion — SKIPPED

The precondition, measurable synchronisation or traffic at the merge/refine/top-k boundaries, does not
hold. From the v62 traces:
- **Idle time:** the GPU is idle inside the captured graph for 0.4–3.4 µs per search in total.
- **Boundary gaps:** scan→merge, merge→refine and refine→top-k are 0.06–0.13 µs, at nsys resolution.
- **Intermediate traffic:** the composite distances are ck floats per query (64 KB at b = 8, α = 200).
- **What fusion would not remove:** the top-k kernel's 16–31 µs at b = 8 is its own one-block-per-query
  selection, which fusion keeps. A parallel final selection is the open item, not fusion.

## Experiment 7: final integration and evaluation

- **Final version:** v65 (v61 L16 + v62 groups + v63 shared bound + v65 schedule). v64 is not
  included (NO-GO); nothing else to integrate, so no separate v68.
- **Ablation, paired over v61 L16** (geomean, all rows; b = 8 / 32 / 256 / 1000):

  | variant | b=8 | b=32 | b=256 | b=1000 |
  |---|---:|---:|---:|---:|
  | v61 L0 | ×0.89 | ×0.90 | ×0.82 | ×0.78 |
  | v63 G2 | ×1.32 | ×1.32 | ×1.00 | ×0.96 |
  | v63 G4 | ×2.01 | ×1.97 | ×1.09 | ×0.95 |
  | v63 G8 | ×2.73 | ×2.34 | ×1.10 | ×0.89 |
  | **v65** | **×2.73** | **×2.36** | **×1.06** | **×1.015** |

  v62 without the bound is in Experiment 1's tables.
- **v65 at b = 8, per row:** vogue/512 ×3.78, openai3-3072/512 ×3.57, stella ×3.10–3.12,
  openai3-1536/512 ×2.97, vogue/128 ×2.83, arxiv/512 ×2.67, bge-m3 ×2.54–2.55, openai3-1536/128
  ×2.37, openai3-3072/128 ×2.24, arxiv/128 ×2.01–2.27.
- **Recall–QPS frontier** (np 128, α ×½ and ×2, b = 8 and 1000; `exp47/raw/frontier`): at equal
  recall, b = 8 gains ×1.81–3.18; at b = 1000 ×0.99–1.01, except vogue ×1.06–1.14.
- **Calibration:** none on evaluation queries. The schedule's statistic comes from base vectors. α is
  the paper's sampled-rule value, not re-calibrated.

## Answers

**Q1. Can higher GPU parallelism solve the post-v61 primary-screening bottleneck?** At small and
medium batch, yes. Splitting each query's candidates over G blocks with an exact merge cuts the
screening 4–5× at b = 8 and the whole search ×2.0–3.8, with recall identical. At large batch, no:
the GPU is already full, and splitting only adds selection and merge work. The fix there is not to
split, which is what the schedule does.

**Q2. Does the global top-αk allow more effective budget-aware selection?** Not as quotas. The global
top-αk concentrates in the nearest probed lists, so the critical-path group's quota cannot shrink, and
certification failures stall whole batches. It does allow an exact shared bound: any group's αk-th key
bounds the query's. That prunes far groups' admissions and merge input with no fallback, giving
×1.03–1.07 at b = 8 and up to ×1.56 at b = 1000 over independent groups. It saves selection work and
traffic, not distance computation.

**Q3. Does the new pipeline exploit JHQ's representation, rather than copy RaBitQ/cuVS task
parallelism?**
- **Mostly no.** The decomposition (query, list-group) blocks plus a merge is the same task
  parallelism cuVS IVF-PQ and GPU IVF-RaBitQ use.
- **What is JHQ-specific:** the requirement of an exact, large global top-αk (ck up to 2000) feeding
  a residual refinement. That makes both the exact deterministic merge and the shared bound
  worthwhile, and the bound's effect comes from how αk interacts with probe order.
- **What is not exploited:** the hierarchical code (primary/residual). Refinement is v61's, unchanged.

**Q4. Which optimisations improve end-to-end recall–QPS, and which only kernels?**
- **End to end:** v61 L16 refinement, v62/v63 small-batch screening (×2–3.8), and the v65 schedule.
  They move the frontier at small batch at every α measured.
- **Kernel or front-end only:**
  - v64's overlap: up to 171 µs saved, at most ×1.045 e2e.
  - v63's large-batch scan savings (up to −45%): measured against v62, and against v61 they mostly
    add ≤ 10%.
  - sm_120 native's faster L0 kernel: it did not carry to the small-batch pipeline.

**Q5. What could go into a paper?**
1. A measured characterisation: after warp refinement, JHQ-GPU's small-batch latency is a
   one-block-per-query primary scan (61–85%), with 8 blocks on 170 SMs at b = 8.
2. Exact list-parallel primary screening for a large global top-αk: a total-order merge with a proof,
   deterministic ties, and a ×2.0–3.8 small-batch speedup on six datasets at identical recall.
3. The exact shared bound across groups and why it works for αk-sized selection (probe-order
   concentration); quotas measured as the negative.
4. A two-term batch schedule that keeps the gain without large-batch loss, with its held-out regret.

All four should be framed as GPU-systems adaptation and measurement for JHQ, not as algorithmic
firsts.

## Novelty audit

| element | prior art (not exhaustive) | what is new here |
|---|---|---|
| list-parallel scan + merge (v62) | cuVS IVF-PQ / IVF-Flat search parallelise over (query, probe) and select across probes; FAISS GPU likewise; GPU IVF-RaBitQ fuses scanning and filtering per probe | an exact, deterministic merge for ck up to 2000 feeding JHQ's residual refinement; measured on JHQ's index |
| shared bound (v63) | threshold sharing among parallel top-k workers is standard (Fagin-style threshold algorithms, GPU kNN with a shared k-th distance) | applying it to JHQ's αk-sized exact selection; the measured link to probe-order concentration (group 0 holds 46–81% of the top-ck) |
| schedule (v65) | occupancy-driven work splitting and split-K heuristics (e.g. cuBLAS) | a two-feature rule (fill, per-query work skew) with held-out evaluation |
| Πᵀc routing (v64) | linear algebra; already in this codebase for assignment | none, and NO-GO |
| quotas (Exp 2 pilot) | budgeted top-k with certificates | a negative result with its mechanism (critical-path group, batch-level fallback) |

Not supported by experiments, and so not claimed:
- any statement about other GPUs, about native sm_120 for the full pipeline, or about batches with
  concurrent streams;
- the schedule's generality beyond these 12 configurations;
- any advantage over cuVS or GPU IVF-RaBitQ at matched recall. They were not run in this campaign.

## Reproducibility and caveats

- **Commits** (branch `exp/pipeline-campaign`):

  | scope | commits |
  |---|---|
  | Phase 0 | `d4cd5d2`, `b3b3432`, `7723059` |
  | v62 | `862543a` |
  | v62 run fix | `e0d0518` |
  | Exp 0 script | `a8edba7` |
  | Exp 2 pilot | `41f6522` |
  | v64 | `eac8ff8` |
  | v63 | `e2ed15b` |
  | v65 + benchmark | `d425ab8` |
  | reports | this commit |

- **Scripts:** `experiments/pipeline_v62/run_phase0.sh`, `run_v62.sh`, and
  `experiments/pipeline_campaign/{exp0_sm120,exp2_v63_pilot,exp2_v63,exp3_v64,exp47_bench}.sh`.
  Each records its build flags (`raw/**/build_env.txt`) and aborts on a failed build.
- **Raw data:** `raw/` beside each report, with nsys traces as gzipped CSV. `.nsys-rep` files and the
  quota dumps stay on the box (`/root/autodl-tmp/{pipeline_v62_nsys,campaign_*}`).
- **Where the runs happened:** in a detached worktree (`/root/autodl-tmp/jhq_pipeline_wt`). The box's
  main tree `/root/JHQ_GPU` has another session's uncommitted edits (CMakeLists.txt,
  jhq_v21_cascade/jhq_gpu_index.cu), which were not touched.
- **Deviations from plan:**
  - racecheck ran at 8 queries, b = 8, since 64 queries did not finish in 10 minutes;
  - the first Phase 3 nsys export was blocked by stale Phase 0 SQLite files and was re-exported from
    the same reports;
  - the GPU sat idle about an hour while v63 and v65 were written;
  - stella has 999 queries, so it has no b = 1000 rows.
- **Not modified:** v61, the ADC paper, audited history, and the trained-state caches.
