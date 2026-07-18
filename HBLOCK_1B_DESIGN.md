# HBlock 1B Scale Design: Two-Phase Batch Loading

## Overview

HBlock targets 1B-scale approximate nearest neighbor search on GPU.
At 1B scale (d=100, int8), base vectors are 100 GB and compressed codes
are 25 GB — neither fits in 32 GB GPU VRAM. This document describes the
storage layout, query-time loading strategy, and GPU cache design that
enable 1B search without sacrificing QPS.

The key property of HBlock that makes this tractable: beam search
operates on **block centroids** (2 GB, fits in VRAM permanently) and only
accesses actual vector data after beam search completes. This means the
H2D transfer window is known before the first byte is transferred.

---

## Current Architecture (v36, up to 100M)

```
GPU VRAM (permanent):
  routing centroids L1/L2/L3     < 100 MB
  block graph (n_blocks × degree)  ~1 GB   (7.8M blocks at 1B)
  block centroids (d_proj=64)      ~2 GB
  base vectors (float32 or int8)   2.86 GB (Vogue) / 10 GB (100M int8)

Query pipeline:
  route → beam search → PQ scan → exact rerank
  (all GPU, no H2D during search)
```

At 1B, base vectors (100 GB int8) and compressed codes (25 GB) do not
fit. The architecture must change to load them on demand.

---

## Storage Layout

### Separate Code Store and Raw Store

Code pages and raw pages must be stored separately. Mixing them causes
every coarse PQ scan to transfer raw vectors that exact rerank never uses.

```
code_store[]:
  unit: code page
  content: compressed PQ codes for P blocks (P=8 default)
  page payload: P * (128 vectors * 25 bytes) = P * 3200 bytes
  page size: 8 * 3200 = 25600 bytes, aligned to 32 KB

raw_store[]:
  unit: raw page
  content: int8 vectors for P blocks
  page payload: P * (128 vectors * 100 bytes) = P * 12800 bytes
  page size: 8 * 12800 = 102400 bytes, aligned to 128 KB
```

At 1B with P=8 blocks per page:

```
num_blocks     = 1B / 128           ~= 7.8M blocks
num_code_pages = 7.8M / 8           ~= 976K code pages
num_raw_pages  = 7.8M / 8           ~= 976K raw pages
total_code_storage = 976K * 32 KB   ~= 31 GB
total_raw_storage  = 976K * 128 KB  ~= 125 GB
```

### Block-to-Page Mapping Table

A lookup table maps block_id to page_id and offset within the page.
This table lives permanently on GPU.

```
struct BlockPageEntry {
    uint32_t code_page_id;
    uint32_t raw_page_id;
    uint16_t slot_in_page;   // 0 to P-1
    uint16_t vector_count;   // actual vectors in this block (<=128)
};

BlockPageEntry block_page_map[num_blocks];   // 7.8M * 12 bytes = ~93 MB
```

### Block Physical Layout

Blocks within the same L3 cell are laid out contiguously (current
behavior from sort-by-cell-code). Within each cell, blocks are ordered
by projected centroid locality so that adjacent blocks in the array tend
to be graph neighbors. This is the first version; graph-partitioned
cross-cell packing is a later optimization.

---

## GPU Cache

After accounting for permanent residents, ~27 GB VRAM is available
for caching.

```
Total VRAM:            32 GB
  block graph:          1 GB
  block centroids:      2 GB
  routing + workspace:  2 GB
  ──────────────────────────
  available for cache: ~27 GB
```

### Code Cache

Prioritize caching code pages because they are accessed on every search
(beam search visits ef blocks → PQ scan needs their codes).

```
code_cache_size = 20 GB
code_page_size  = 32 KB
cached_pages    = 20 GB / 32 KB = ~640K pages
total_pages     = 976K
coverage        = 640K / 976K = 65%
```

With Zipfian query distribution and 65% page coverage, expected cache
hit rate is 80-90% in practice.

### Raw Cache

Use remaining ~7 GB for raw pages. Raw access is only for exact rerank
of the final top-R candidates (R << ef), so the working set is much
smaller than for code pages.

```
raw_cache_size  = 7 GB
raw_page_size   = 128 KB
cached_pages    = 7 GB / 128 KB = ~56K pages
```

### Cache Replacement Policy

LRU per cache tier. Cache is shared across all queries in a batch;
a page loaded for one query is immediately available to all others.

---

## Query Pipeline (1B)

### Phase 0: Route + Beam Search (pure GPU, no H2D)

```
input:  query batch (B queries)
L1/L2/L3 routing  →  entry cells  →  entry blocks
beam search on block graph using block centroids
output: for each query, visited_block_ids[]  (up to ef block IDs)
```

All data for this phase is permanently resident in VRAM. No H2D.
Beam search result is fully known before any transfer begins.

### Phase 1: Code Page Loading + PQ Scan

```
step 1: map visited_block_ids → code_page_ids  (GPU, using block_page_map)
step 2: deduplicate code_page_ids across entire batch  (GPU sort+unique)
step 3: split into cache_hits and cache_misses
step 4: async H2D for cache_miss pages (cudaMemcpyAsync, multiple streams)
step 5: update code cache (evict LRU if needed)
step 6: GPU PQ scan over visited blocks in loaded pages
output: per query, top-R candidates (vector_id, approx_dist)  R ~= 200
```

Deduplication is the key efficiency gain. A batch of 1024 queries with
ef=256 each could produce up to 262K block requests, but after
deduplication the unique page count is much smaller due to graph
locality (hot blocks are shared across queries).

### Phase 2: Raw Block Loading + Exact Rerank

```
step 1: map top-R vector_ids → raw_block_ids → raw_page_ids  (GPU)
step 2: deduplicate raw_page_ids across batch  (GPU)
step 3: split into cache_hits and cache_misses
step 4: async H2D for cache_miss raw pages
step 5: GPU exact L2 rerank over top-R candidates
output: final top-k per query
```

The working set for Phase 2 is much smaller than Phase 1 (R=200
candidates vs ef=256 blocks, and R candidates come from a smaller
number of raw pages after deduplication).

### Pipeline Overlap

Phase 0 produces block_ids at completion. H2D for Phase 1 can be
issued immediately. If Phase 1 H2D is overlapped with CPU-side
deduplication for Phase 2, total transfer time is closer to
max(code_transfer, raw_transfer) than their sum.

```
timeline:
  [beam search GPU]
                   [dedup code pages | H2D code pages  ]
                                     [PQ scan GPU       ]
                                                [dedup raw | H2D raw | rerank]
```

---

## Transfer Volume Estimate (1B, batch=1024, ef=256)

### Worst case (cold cache)

```
unique code pages after dedup:  ~50K  (estimated, depends on graph locality)
code transfer:  50K * 32 KB  = 1.6 GB
unique raw pages after dedup:   ~5K   (R=200 candidates, fewer unique blocks)
raw transfer:   5K * 128 KB  = 640 MB
total transfer: ~2.2 GB per batch
PCIe 4.0 x16 (32 GB/s): ~69 ms transfer time
```

### Warm cache (65% code hit rate, 50% raw hit rate)

```
code transfer:  50K * 35% * 32 KB  = 560 MB
raw transfer:   5K  * 50% * 128 KB = 320 MB
total transfer: ~880 MB per batch
PCIe time: ~28 ms
```

Beam search itself runs in ~5-10 ms for batch=1024. Transfer dominates
until cache warms up; after warm-up, transfer and compute can overlap.

---

## Build Pipeline Changes Required

### 1. Separate Code and Raw packing

Current `add()` packs vectors and codes into a single layout. For 1B,
`add()` must write two separate stores:

```
write_code_page(page_id, slot, pq_codes[128 * M])
write_raw_page(page_id, slot, int8_vecs[128 * d])
```

Both stores reside in CPU pinned memory during build, then optionally
flushed to NVMe for out-of-core indices.

### 2. block_page_map construction

After block assignment is finalized, assign page_id = block_id / P
and slot = block_id % P, then upload block_page_map to GPU.

### 3. int8 input support

`add()` must accept int8 input (SPACEV format) in addition to float32
(Vogue/arXiv format). A flag in Params selects the path.

```cpp
struct Params {
    bool int8_input = false;   // true for SPACEV-style int8 datasets
    int  page_size  = 8;       // blocks per page
    ...
};
```

For float32 input, vectors are cast to int8 before packing if
int8_storage is enabled, or stored as float32 for smaller datasets
where VRAM permits full residency.

### 4. GPU migration order (from HBLOCK_VNEXT plan)

Priority order for moving build steps to GPU:

```
done:   CPU stable_sort → GPU radix sort (v36)
next:   GPU packing and block centroid computation
then:   GPU per-cell balanced k-means
later:  GPU CSR construction for graph candidates
```

---

## Physical Block Ordering Within Cell

The first implementation orders blocks within each L3 cell by projected
centroid distance from the cell centroid. This is a simple sort and
gives reasonable locality for nearby blocks.

Later optimization: after building the block graph, reorder blocks
within each cell using greedy graph-neighbor packing — assign the next
block_id to the unassigned graph neighbor with the highest edge weight.
This increases the probability that a fetched code page contains multiple
blocks that will be visited in the same beam expansion step.

Cross-cell graph-aware packing (grouping blocks from different L3 cells
that are frequently co-visited) is deferred. It requires replacing
contiguous cell-to-block ranges with explicit block lists and is a more
invasive change.

---

## Key Metrics to Measure

```
page_utilization = unique_blocks_accessed / (unique_pages_loaded * P)
  target: > 0.5  (if << 0.5, page size is too large)

code_cache_hit_rate = cache_hits / (cache_hits + cache_misses)
  target: > 0.7 in steady state

raw_cache_hit_rate
  target: > 0.5

unique_code_pages_per_batch
  expected: 30K-80K for batch=1024, ef=256

code_transfer_ms, raw_transfer_ms, beam_search_ms, pq_scan_ms, rerank_ms
  goal: transfer_ms < beam_search_ms + pq_scan_ms (transfer overlaps compute)
```

Page size sweep to calibrate:

```
P = 4  →  code page 16 KB,  raw page 64 KB
P = 8  →  code page 32 KB,  raw page 128 KB   (recommended starting point)
P = 16 →  code page 64 KB,  raw page 256 KB
```

If page_utilization < 0.3, reduce P. If transfer count is too high
(too many small H2D calls), increase P.

---

## Version Roadmap

```
v36  (done):   GPU radix sort; Vogue/arXiv float32; full GPU residency
v37  (next):   int8 input; SPACEV-100M; full GPU residency (fits at 100M)
               code/raw stored separately in CPU pinned memory
               block_page_map on GPU (unused at 100M, ready for v38)
v38:           H2D two-phase loading; GPU code cache; batch dedup
               benchmark page_utilization and cache hit rate on 100M first
               scale to 1B when layout is validated
v39+:          graph-aware intra-cell block ordering
               cross-cell graph-partitioned page packing
               async H2D overlap with beam search
```

---

## Summary

The strategy that supports 1B batch loading rests on three properties:

1. **Beam search is data-free after build**: block graph and centroids
   (~3 GB) fit permanently. H2D is never triggered during beam search
   itself; all page_ids are known at beam search completion.

2. **Two separate H2D phases with batch deduplication**: code pages
   first (for PQ scan), raw pages second (for exact rerank). Batch
   deduplication across 1024 queries collapses the working set
   significantly due to graph locality.

3. **27 GB GPU cache with 65% code page coverage**: under Zipfian
   query distribution this yields 80-90% cache hit rate in steady
   state, reducing effective transfer to a few hundred MB per batch.
