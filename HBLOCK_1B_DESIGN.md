# HBlock 1B Scale Design: Two-Phase Batch Loading

## Overview

HBlock targets 1B-scale ANN search on GPU. At 1B scale (d=100, int8,
Br=4), fine codes are ~50 GB and raw vectors are 100 GB — neither fits
in 32 GB VRAM. This document describes the storage layout, query-time
loading strategy, and known open problems for 1B operation.

The key property enabling this: beam search operates only on block
centroids (~2 GB, permanently resident). All page IDs needed for PQ scan
are known the moment beam search completes, before any H2D begins.

---

## Code Size (SPACEV d=100, Br=4)

```
bytes_per_vector = (d * Br + 7) / 8 = (100 * 4 + 7) / 8 = 50 bytes
fine codes per block  = 128 * 50    = 6400 bytes
vector IDs per block  = 128 * 4     =  512 bytes
total per block                     ≈ 6912 bytes ≈ 6.9 KB
```

At 1B with 8 blocks per 64 KB code page:

```
num_blocks     = 1B / 128          ≈ 7.8M
num_code_pages = 7.8M / 8          ≈ 976K
code store     = 976K * 64 KB      ≈ 62 GB    (not 31 GB)
raw store      = 976K * 128 KB     ≈ 125 GB
```

---

## GPU Permanent Residents

```
routing centroids L1/L2/L3         < 100 MB
block graph (7.8M × 32 × 4B)       ~1 GB
block centroids (d_proj=64)         ~2 GB
block_page_map (7.8M × 12B)        ~93 MB
visited hash table (see §Known Issues)
─────────────────────────────────────────
total permanent                    ~3.2 GB
available for staging + cache      ~28 GB
```

---

## Storage Layout

### Separation of Code and Raw

Code pages and raw pages are stored separately. PQ scan (Phase 1)
must not trigger raw vector transfer; raw transfer (Phase 2) is only
for the exact rerank candidates.

```
code_store:  fine PQ codes + vector IDs,  page size 64 KB  (8 blocks)
raw_store:   int8 base vectors,            page size 128 KB (8 blocks)
```

Both stores reside in regular (non-pinned) host memory or are mmap'd
from NVMe. Two small pinned staging buffers (~512 MB each) are used
for H2D transfers — not the full 62 GB + 125 GB.

### Block-to-Page Mapping

```c
struct BlockPageEntry {
    uint32_t code_page_id;
    uint32_t raw_page_id;
    uint16_t slot_in_page;    // 0 to 7
    uint16_t vector_count;    // actual vectors <= 128
};
// 7.8M * 12B = ~93 MB, permanently on GPU
BlockPageEntry block_page_map[num_blocks];
```

---

## Query Pipeline

### Phase 0: Route + Beam Search (pure GPU)

```
L1/L2/L3 routing  →  entry blocks per selected cell
beam search on block graph using projected block centroids
output: per query, visited_block_ids[]  (up to ef IDs)
```

All data permanently resident. No H2D. Block IDs for Phase 1 are
fully determined here.

**Open problem (§ entry selection)**: at 1B each L3 cell has ~1907
blocks. Entry selection currently scans all blocks in selected cells,
which is too slow. Must pre-store a small set of entry block indices
per cell.

### Phase 1: Code Page Loading + Per-Block PQ Top-p

```
map visited_block_ids → code_page_ids         (GPU, block_page_map)
deduplicate page IDs across batch              (GPU sort + unique)
check code cache → issue H2D for misses        (cudaMemcpyAsync)
GPU: per-block PQ scan → top-p candidates per block
output: compact list of (query_id, vector_id, approx_dist)
```

**Critical**: PQ distances from different L3 cells are not directly
comparable because each cell has a different residual baseline. The
current v36 pipeline does **per-block top-p**, not a global top-R
across all visited blocks. Keeping per-block top-p is required to
preserve recall semantics.

### Phase 2: Raw Block Loading + Exact Rerank

```
map vector_ids from Phase 1 → raw_block_ids → raw_page_ids  (GPU)
deduplicate raw page IDs across batch          (GPU)
check raw cache → issue H2D for misses
GPU: exact L2 for all candidates
per-query global top-k merge
output: final top-k per query
```

Phase 2 cannot begin until Phase 1 PQ scan completes, because raw
candidate IDs are not known until then.

### Pipeline Overlap (wave-based)

The two phases cannot overlap within a single batch (Phase 2 depends
on Phase 1 output). Overlap is achieved by pipelining across waves:

```
batch i:   [beam] [code H2D | PQ scan] [raw H2D | exact rerank]
batch i+1: [beam] [code H2D | PQ scan] [raw H2D | exact rerank]
                   ↑ overlaps with batch i raw H2D where possible

within Phase 1, if pages split into sub-waves:
  wave k+1 code H2D  ↔  wave k PQ scan
within Phase 2:
  wave k+1 raw H2D   ↔  wave k exact rerank
```

---

## GPU Cache

With ~28 GB available after permanent residents:

```
code cache: 20 GB  →  20 GB / 64 KB   = ~312K pages
coverage:   312K / 976K               = ~32%   (not 65%)

raw cache:  8 GB   →  8 GB / 128 KB   = ~64K pages
coverage:   64K / 976K                = ~7%
```

32% code page coverage is modest. Effective hit rate depends heavily
on query distribution (Zipfian workloads will see higher hit rates for
hot pages) and on block physical ordering (graph-neighbor-packed blocks
increase page reuse per batch). These numbers must be measured, not
assumed.

Key metric to track:

```
page_utilization = blocks_accessed / (pages_loaded * 8)
unique_code_pages_per_batch
cache_hit_rate (measured, not estimated)
```

---

## Known Open Problems

### 1. Visited State at 1B Scale

Current implementation uses a per-query bitmap over block IDs.

```
7.8M blocks → 7.8M bits = 0.975 MB per query
batch = 1024 → ~1 GB just for visited bitmaps
```

Must replace with a fixed-capacity structure proportional to
`ef × degree` (e.g., a compact hash set or Bloom filter per query),
independent of total block count.

### 2. Entry Selection per L3 Cell

At 1B, each L3 cell contains ~1907 blocks. Current code scans all
blocks in selected cells to pick entry points, which becomes a
bottleneck.

Fix: pre-store a small set (e.g., 4–8) of designated entry block
indices per L3 cell at build time. Entry selection becomes an O(1)
table lookup rather than a linear scan.

### 3. Streaming Build for 1B

Current `add()` materializes several host arrays simultaneously:

```
h_proj1_all:  1B * 64 * 4B  = 256 GB
fine codes:   1B * 50B       = 50 GB
sort keys:    1B * 8B        = 8 GB
order:        1B * 4B        = 4 GB
```

This is infeasible on any single machine. Build must be restructured
as a streaming / external-sort pipeline:

```
1. Stream encode in batches of ~10M vectors → write cell-partitioned
   code shards to disk (external partitioning by (c1,c2,c3) key)
2. For each L3 cell: load its shard, run balanced k-means, pack blocks,
   write code_page and raw_page entries
3. Build block graph incrementally, cell by cell
4. Write final block_page_map
```

GPU packing and centroid GPU migration (from HBLOCK_VNEXT plan) are
necessary but not sufficient — the memory layout must change first.

### 4. Cross-Cell PQ LUT Accuracy

Beam search frequently crosses L3 cell boundaries (especially at high
ef). The current single LUT built from the best-route residual is
applied to all visited blocks including those in other cells. This
introduces a systematic distance bias for cross-cell blocks.

At current recall levels (0.997) this is not the bottleneck, but
at 1B with more cross-cell traversal it may become one. Mitigation:
build one LUT per visited cell (at most ck1*ck2*ck3 LUTs) rather than
one global LUT.

---

## Correct Query Flow Summary

```
[permanent GPU] routing tree + block graph + block centroids
        ↓
Phase 0: beam search  →  visited_block_ids (no H2D)
        ↓
Phase 1: map → code_page_ids → dedup → H2D missing pages
         per-block PQ top-p scan
         output: (query_id, vector_id) pairs
        ↓
Phase 2: map → raw_page_ids → dedup → H2D missing pages
         exact L2 rerank
         per-query top-k merge
```

---

## Version Roadmap

```
v36  (done):  GPU radix sort; float32; full GPU residency
v37  (next):  int8 input; SPACEV-100M; full GPU residency (10 GB fits)
              code/raw stored separately; block_page_map built but unused
              fix entry selection (pre-store entry blocks per cell)
v38:          two-phase H2D; wave pipelining; code/raw cache
              fix visited bitmap (hash set proportional to ef*degree)
              measure page_utilization and real cache hit rate on 100M
v39:          streaming build pipeline for 1B
              graph-aware intra-cell block ordering
              multi-LUT PQ scan for cross-cell accuracy
v40+:         cross-cell graph-partitioned page packing
              scale to 1B with validated layout
```

---

## Summary of Corrections vs Prior Draft

| Item | Prior (wrong) | Corrected |
|------|--------------|-----------|
| code bytes/vector | 25 B | 50 B (Br=4, d=100) |
| code store size | ~31 GB | ~62 GB |
| code cache coverage | 65% | ~32% |
| Phase 1 rerank | global PQ top-R | per-block top-p (preserves recall) |
| Phase 1/2 overlap | both H2D concurrent | wave pipeline within each phase |
| host storage | 150 GB pinned | regular RAM/mmap + ~1 GB pinned staging |
| visited state | bitmap OK | 1 GB at 1B, must use hash/Bloom |
| cache hit rate | 80-90% (assumed) | unknown, must measure |
