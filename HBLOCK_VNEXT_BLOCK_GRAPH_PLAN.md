# HBlock vNext: Residual-Aware Block Graph Plan

This note is for the next version after `hblock_v17` / `hblock_v22`.

The important point:

```text
L1/L2/L3 routing stays unchanged.
The new component is only after L3: block-level graph traversal.
```

There is no extra tree layer and no multi-level graph. The final graph is still:

```text
node = physical leaf block
edge = block -> block
```

## Current Code Meaning

In `hblock_v17`:

```text
(c1, c2, c3) = routing cell
leaf block   = physical scan block, usually 128 vectors
```

The current v17 search does:

```text
L1 beam -> L2 beam -> L3 beam
selected L3 cells -> scan all physical blocks inside those cells
```

The problem is that one L3 cell can contain many physical blocks at large scale.

In `hblock_v22`, this was changed to:

```text
selected L3 cells -> first block of each cell -> CPU block graph traversal
```

But the current v22 graph is still too rough:

```text
same cell: previous/next block
neighbor cell: first block of that neighboring cell
```

This is only a prototype. The next version should build real block-level edges.

## Recommended Design

Use the existing HBlock residual structure:

```text
x ~= C1[c1] + C2[c1,c2] + C3[c1,c2,c3] + r3
```

For each physical leaf block:

```text
cell_id(block) = c1*K2*K3 + c2*K3 + c3
block_mean_r3  = mean residual of vectors inside this block
block_centroid = C1 + C2 + C3 + block_mean_r3
```

So the block summary is not independent of the tree. It reuses the L1/L2/L3 residual decomposition.

Recommended stored fields:

```text
block_id
cell_id
leaf_codes[128]
leaf_ids[128]
block_residual_centroid_proj[64]  // fp16 or float
neighbors[graph_degree]
```

For scoring, either use:

```text
score(q, block) = || q - (C1+C2+C3+block_mean_r3) ||^2
```

or the projected version:

```text
score(q, block) ~= || P * q - P * block_centroid ||^2
```

For a first implementation, full float centroids are acceptable. For 1B scale, store 64D projected centroids, preferably fp16.

## How To Build The Graph

Do not do global block kNN:

```text
for each block:
    search all 7.8M blocks
```

That is too expensive.

Use the existing tree/cell structure only to generate candidate cells.

For a block `b` in cell `c`:

```text
candidate_cells =
    same L3 cell c
    nearby L3 cells under the same (c1,c2)
    nearby cells under nearby c2 within the same c1
    a few nearby cells under nearby c1
```

The nearby cells can be generated cheaply from existing centroids:

```text
same (c1,c2): compare C3 centroids
same c1:      compare C2 centroids
cross c1:     compare C1 centroids
```

This does not mean the graph is cell-level. It only means cells are used to narrow the candidate set.

Final edge selection must still be block-level:

```text
for each block b:
    candidates = blocks inside candidate_cells
    rank candidates by block centroid distance
    keep top graph_degree blocks
```

So the logic is:

```text
C1/C2/C3 centroids: choose which cells to look inside
block centroids:    choose the actual block-to-block edges
```

## Entry Block Selection

Do not use the first block of a selected L3 cell.

Current v22 does:

```text
entry = pair_start[cell_id]
```

This is weak because the first block is just a storage-order artifact.

Instead:

```text
for each selected L3 cell:
    choose top-s nearest blocks by block centroid score
```

Recommended initial values:

```text
entry_per_cell = 2 or 4
graph_degree   = 32
graph_budget   = 32, 64, 128
rerank_r       = 128
```

Do not enqueue every block in the selected cell. At 1B scale, a cell may contain thousands of blocks.

## Search Flow

The search flow should be:

```text
1. GPU routing:
   L1 beam -> L2 beam -> L3 beam

2. Entry generation:
   selected L3 cells -> top-s nearest physical blocks per cell

3. Block graph traversal:
   priority queue contains block_ids
   pop best block
   scan its 128 PQ codes
   update top-R candidates
   push neighbor blocks
   stop when visited_blocks reaches graph_budget

4. Exact rerank:
   gather original vectors for top rerank_r ids
   exact L2 rerank
```

The cost model becomes:

```text
tree routing + visited_blocks * 128
```

instead of:

```text
selected_cells * blocks_per_cell * 128
```

## Why Block Centroid Is Needed

The existing L1/L2/L3 centroids can tell us:

```text
cell A is near cell B
```

But they cannot tell us:

```text
which block inside cell B is near the query
which block inside cell B should be connected to block A
```

That is why block centroid / block residual centroid is necessary.

Without block centroid, the graph degenerates into:

```text
cell graph + arbitrary block chain
```

With block centroid, edges become:

```text
block A -> block B17
block A -> block B83
block A -> block C5
```

This is the real block-level graph.

## Implementation Notes For vNext

Start from `hblock_v22`.

Replace:

```text
cell kNN + previous/next block + first block of neighbor cell
```

with:

```text
cell-neighbor candidate generation + block-centroid edge selection
```

Concrete code changes:

```text
1. During add(), compute cell_id for each block.
2. During/after block construction, compute block residual centroid.
3. Build neighbor_cells using C1/C2/C3 centroid distances.
4. For each block, search blocks in candidate cells and keep nearest graph_degree blocks.
5. During search, replace first-block entry with top-s nearest block entries.
6. Keep the existing PQ scan and exact rerank pipeline.
```

The CPU version is acceptable for the next prototype. The final high-throughput version should move:

```text
block adjacency
block centroid summaries
frontier / visited state
block_id grouping
```

onto GPU.

## Current v23 Status And Build-Side Improvements

Current `hblock_v23` has already implemented the important search-side change:

```text
tree routing -> entry blocks -> GPU block graph traversal -> PQ scan -> exact rerank
```

So the current v23 is:

```text
GPU search traversal: yes
GPU graph build:      no
```

The build path is still mostly CPU:

```text
1. block centroid mean is computed on CPU
2. block centroid projection is computed on CPU
3. block graph edges are selected on CPU with std::sort / std::partial_sort
4. the finished adjacency and block projections are copied to GPU
```

This is acceptable for 10M-scale prototyping, but not for billion-scale indexing.

Recommended next improvements:

```text
1. GPU block centroid projection
   Current code copies Pi1 back to CPU and projects block centroids with CPU loops.
   Replace this with cuBLAS GEMM:

       block_cent_proj = Pi1^T * block_cent

   This removes O(num_blocks * d * d_proj) CPU work.

2. GPU edge distance + top-degree selection
   Current build_block_graph() ranks candidate blocks on CPU.
   Move the expensive part to GPU:

       for each block b:
           load candidate block ids
           compute projected 64D distances to candidate blocks
           keep top graph_degree neighbors

   Complexity stays O(num_blocks * candidate_blocks * d_proj),
   but the work becomes massively parallel.

3. Fix candidate cap quality
   Current candidate blocks are deduplicated, sorted by block id, then truncated.
   This can drop useful nearby blocks arbitrarily.

   Better:
       rank candidate cells by C1/C2/C3 centroid distance first
       add blocks from closer cells first
       only then apply max_cand_blocks

4. Move entry selection to GPU
   Current cpu_select_entries() projects each query on CPU and scans blocks
   inside selected L3 cells.

   For 10M this is small, but for 1B each L3 cell can contain many blocks.
   Entry selection should become a GPU kernel:

       selected L3 cells + q_proj + block_cent_proj -> top entry_per_cell blocks

   This also removes the D2H copy of top L1/L2/L3 beams before graph traversal.

5. Avoid full block centroids for large scale
   Current v23 stores h_block_cent_ as full d-dimensional float centroids.
   For 1B, this becomes too large.

   Long term storage should be:

       block_cent_proj[block_id, 64]  // fp16 preferred
       block_norm[block_id]
       block_adj[block_id, degree]

   Full d-dimensional block centroids should only be temporary during build,
   or avoided by directly accumulating projected centroids.

6. Do not keep all base vectors on GPU for 1B
   Current v23 uploads d_base_vecs_ for exact rerank.
   This is fine for small/medium experiments, but impossible at billion scale.

   Billion-scale version should keep only compressed codes and graph metadata
   resident on GPU, then fetch top-rerank_r original vectors separately.
```

The most important short-term change is:

```text
keep v23 search logic,
but replace CPU build_block_graph() distance ranking with GPU distance + top-degree selection.
```

The most important billion-scale change is:

```text
GPU memory should hold block graph + projected block centroids + compressed codes,
not full original base vectors.
```

## Batch Execution And Block Reuse

For the first v23/vNext implementation, each query can run its own block traversal.
For billion-scale throughput, the stronger execution model is a global block task queue.

Logical search state is still per query:

```text
frontier_q
visited_q
topR_q
```

But execution should be grouped by physical block:

```text
1. each query emits block tasks: (qid, block_id)
2. sort or bucket tasks by block_id
3. load each unique block once
4. scan that block for all requesting queries
5. update each query's topR and frontier
```

This is the main systems advantage over vector-level random graph traversal:

```text
same block requested by many queries -> load once, serve many queries
```

The important extra metric is:

```text
block reuse = total block requests / unique block requests
```

If reuse is high, block-level graph search can gain throughput from sequential block reads and one-load-many-query scanning.

## Research Positioning

The individual ingredients are not new by themselves:

```text
tree routing
graph traversal
candidate queue / best-first search
compressed-code scan
exact rerank
```

The useful angle is their combination at the physical GPU block level:

```text
navigation graph over GPU-friendly physical leaf blocks,
with block_id-grouped batch execution and sequential compressed-code scan.
```

Compared with vector-level graph methods:

```text
vector graph: finer navigation, more random access
block graph: coarser navigation, more sequential scan and batch reuse
```

So the core tradeoff is:

```text
trade fine-grained vector navigation for hardware-efficient block-level execution
```

## Metrics To Check

The main metrics are not just recall and QPS.

Also measure:

```text
visited_blocks per query
scanned vectors per query = visited_blocks * 128
entry quality: recall from entry blocks only
graph expansion gain over entries
unique blocks per batch
block reuse = total block requests / unique block requests
```

The design is promising if:

```text
visited_blocks stays within tens to a few hundreds
recall improves with graph_budget
QPS remains controlled because scan unit is fixed-size blocks
```

## One-Sentence Summary

Use L1/L2/L3 routing to find promising L3 cells, use existing centroids to find neighboring cells, but use residual-aware block centroids to choose actual block entries and block-to-block graph edges.
