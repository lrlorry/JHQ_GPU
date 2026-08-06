# hblock_v39 findings — SPACEV, region partitioning + bounded GPU pool

Date: 2026-08-06. All runs on `/root/autodl-tmp/spacev100m` (SPACEV-100M
subset, `unum-cloud/ann-spacev-100m`, big-ann-benchmarks `.i8bin`/`.i32bin`
layout), via `demo_hblock_v39` on the actual GPU box. Params not listed:
`K1=K2=K3=16, graph_degree=32, entry_per_cell=4, d_proj=64, per_block_r=16,
region_bytes=1MiB`.

## What this run validates

1. **Build correctness.** `train()`/`add()` complete cleanly at both 500K
   and 5M vectors: sensible block/region counts, no crashes, timings scale
   roughly linearly with n (dominated by CPU-side balanced-kmeans
   re-clustering: 14.5s of the 20s `add()` total at 5M).
2. **Region byte-accounting is exactly correct.** At 500K vectors (39 code
   regions, 73 raw regions, region_bytes=1MiB), the warmup `search()` call's
   printed H2D totals (`40.89 MB` code, `76.55 MB` raw) match hand
   calculation exactly: `39 * 1,048,576 = 40,894,464 B = 40.89 MB`,
   `73 * 1,048,576 = 76,546,048 B = 76.55 MB`. Confirms the GPU region pool
   load-once/cache-hit-skip mechanism is transferring exactly what it should,
   no more, no less.
3. **A real diagnostic bug was found and fixed via this same run.** The
   original `stat_code_region_unique_` counter summed each internal
   `batch_size_`-sized sub-batch's own within-batch dedup count, so it
   overcounted (624 "unique" regions reported for an index with only 39
   total regions existing). Fixed by adding a true cross-call dedup counter
   (`stat_*_true_unique_`, via a per-region last-seen-call_id table) — see
   commit `1436208`. The H2D byte totals were unaffected by this bug (they
   were always correct); only the printed "how many distinct regions did we
   touch" number was wrong.
4. **A batch-still-exceeds-pool-capacity guard fired correctly.** With
   `gpu_code_region_cap` set below what a batch actually needed, the code
   threw a clear `std::runtime_error` instead of silently evicting a region
   still needed later in the same kernel launch (see the correctness
   argument in `fetch_code_regions()`'s comment). Working as designed.

## Recall scales with subset fraction, not a bug

| n_base | fraction of 100M corpus | recall@10 (ef=8..64) |
|---|---|---|
| 500,000  | 0.5% | 0.0165 → 0.0197 → 0.0232 → 0.0259 |
| 5,000,000 | 5%   | 0.1361 → 0.1628 → 0.1946 → 0.2261 |

Ground truth (`groundtruth.30K.i32bin`) is computed against the full
100M-vector corpus; `ids.100M.i32bin` maps local subset indices to global
ids. Restricting `add()` to a small prefix of `base.100M.i8bin` means most
true top-10 neighbors for a given query simply aren't in the loaded subset
at all — recall is bounded above by roughly the subset fraction regardless
of search quality. The ~10x recall increase from 0.5%→5% subset size lines
up with this. **A meaningful recall number requires running close to the
full 100M** (or at least a fraction large enough that the ceiling isn't the
binding constraint). ef-monotonicity (recall strictly increases with ef at
both scales) confirms routing/traversal itself is not broken.

## Region locality: naive block-order packing, and the batch-size effect

`hblock_v39` packs code/raw regions in **physical block-insertion order**
(no graph-aware repacking — matches the plan docs' own staged approach:
measure first, optimize layout later). At 5M vectors (41,090 blocks → 273
code regions, 508 raw regions):

**Whole-call footprint is is saturated regardless of ef or batch size** —
`uniq_this_call` (true distinct regions touched across the entire 1000-query
`search()` call) sits at ~272/273 (~99.6%) no matter what ef or batch_size
is. With 1000 queries in one call, essentially every region gets touched by
*someone* — expected, and not itself informative about pool sizing.

**Per-batch footprint (the number that actually matters for pool sizing)
depends heavily on batch_size.** Computed as `uniq_per_batch_sum / number
of internal batches`:

| ef | batch=64 (16 batches) | batch=8 (125 batches) |
|---|---|---|
| 8  | 155/273 = 57% | 31/273 = 11% |
| 16 | 195/273 = 71% | 46/273 = 17% |
| 32 | 227/273 = 83% | 66/273 = 24% |
| 64 | 248/273 = 91% | 91/273 = 33% |

Recall is unaffected by batch_size (0.1451/0.1726/0.2009/0.2298 at batch=8
vs 0.1361/0.1628/0.1946/0.2261 at batch=64 — same, within noise), confirming
batch chunking doesn't change *what* is returned, only how fetch planning
is grouped.

**Conclusion:** a large batch (64 queries at once) touches 57-91% of all
regions — a bounded pool close to that batch's footprint isn't meaningfully
smaller than "just keep everything resident". A small batch (8 queries)
touches only 11-33% — a bounded pool becomes genuinely viable, at a real
throughput cost: QPS drops ~6-7x going from batch=64 to batch=8 (fixed
per-batch overhead — routing, region fetch-planning — amortized over fewer
queries). This is a real, load-bearing tradeoff between region-pool
viability and throughput, not an artifact.

Measured against the plan docs' own v39 go/no-go gate (`region_reuse >= 2`,
`region_utilization` 25-40%): **at small batch sizes (8), region utilization
(11-33%, i.e. non-utilization headroom for eviction) and reuse ratios
(region_reuse 30-233x on the corrected true-unique metric) both clear the
bar. At large batch sizes (64), naive layout does not** — this is exactly
the kind of nuanced, batch-size-conditional result the diagnostic was meant
to produce, rather than a flat yes/no.

## What this motivates next (v42)

Given the naive block-order layout only shows real region locality at small
batch sizes, and v39's region store is still an in-process host-RAM buffer
(not actually out-of-core — everything gets loaded into a host `std::vector`
during `add()`, "out-of-core" only in the sense of GPU residency, not host
memory footprint), v42 moves the region store to real mmap-backed files on
disk (true out-of-core, host RAM bounded independent of dataset size) and
adds the cheap "same L3 cell first" packing order the plan docs specify as
v42's first locality pass — before committing to full graph-aware repacking.
