# HBlock v23 Analysis For Claude

This note summarizes the current `hblock_v23` results and code-level issues.

## Current Results

Files:

```text
results/hblock_v23_vogue.csv
results/hblock_v23_arxiv.csv
results/hblock_v23_comparison.png
```

Vogue:

```text
budget  recall@10  latency_ms  QPS
8       0.5690     2.37        421844
16      0.7341     2.97        336572
32      0.8634     4.23        236545
64      0.8624     4.92        203290
```

Arxiv:

```text
budget  recall@10  latency_ms  QPS
8       0.6074     2.34        427608
16      0.7569     3.40        293907
32      0.8625     4.28        233529
64      0.8704     4.58        218120
```

Main observation:

```text
v23 has very high QPS, but recall saturates around 0.86-0.87.
Increasing graph_budget from 32 to 64 barely improves recall.
```

This suggests the main issue is not simply "not enough visited blocks".
It is more likely one or more of:

```text
block quality
entry quality
graph edge quality
PQ candidate truncation
residual/LUT mismatch
beam/frontier over-pruning
```

## Current v23 Code Facts

v23 is already more advanced than the original CPU prototype.

Search path:

```text
L1/L2/L3 GPU routing
-> fused GPU entry selection + block graph traversal
-> PQ scan on visited blocks
-> PQ top-r merge
-> exact rerank
```

Build path:

```text
block centroid projection: GPU cuBLAS GEMM
block adjacency top-k:    GPU kernel
candidate CSR generation: CPU
block packing/centroid:   CPU
```

Important files:

```text
hblock_v23/jhq_gpu_index.cu
hblock_v23/search.cu
hblock_v23/search.cuh
examples/demo_hblock_v23.cu
scripts/run_hblock_v23.sh
```

## Key Code-Level Issues

### 1. Physical blocks are not semantic blocks

Current code sorts vectors only by routing cell:

```cpp
std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    long long ka = c1[a] * K2K3 + c2[a] * K3 + c3[a];
    long long kb = c1[b] * K2K3 + c2[b] * K3 + c3[b];
    return ka < kb;
});
```

Then it packs every 128 vectors into a block.

This means:

```text
block = storage chunk inside an L3 cell
not necessarily a local semantic cluster
```

The block centroid may be weak because the 128 vectors inside a block are not
guaranteed to be close to each other beyond sharing the same `(c1,c2,c3)`.

Likely impact:

```text
block graph edges are noisy
entry block selection is noisy
graph traversal cannot reliably reach high-recall regions
```

Suggested fix:

```text
Within each L3 cell, reorder vectors before packing blocks.
Possible first versions:

1. sort by projected residual along 1-2 random/JL dimensions
2. sort by residual norm or block score
3. small local k-means / mini-clustering inside each L3 cell
4. group vectors by compact PQ-code prefix

Goal: each physical 128-vector block should be a real local cluster.
```

This is probably the most important structural fix.

### 2. PQ LUT uses only the best routing path

Current code builds `d_q_r3` using only the best `(c1,c2,c3)` path:

```text
extract_best_r3_kernel(...)
build_fine_lut_kernel(...)
```

Then the same fine LUT is used to scan all visited graph blocks.

But graph traversal can visit blocks from other L3 cells. For a block in another
cell, the correct residual should be:

```text
q - C1[cell] - C2[cell] - C3[cell]
```

not necessarily the residual from the best entry cell.

This can make PQ coarse distances inaccurate for blocks outside the best cell,
so true neighbors may not enter rerank.

Suggested diagnostic:

```text
After graph traversal, exact-scan all vectors inside visited blocks.
Compare:

1. graph visited blocks + exact scan
2. graph visited blocks + current PQ scan + rerank

If exact scan recall is high but PQ recall is low, the issue is PQ/LUT/rerank.
If exact scan recall is also low, the issue is entry/graph/block quality.
```

Possible fixes:

```text
1. use per-cell residual LUT for each visited block's cell
2. use a coarser block-level score for traversal, but recompute PQ LUT per cell
3. keep current LUT for speed, but increase TOP_P and rerank_r
```

### 3. TOP_P=4 is too narrow for high recall

Current constant:

```cpp
static constexpr int TOP_P = 4;
```

With `graph_budget=32`, the PQ stage emits at most:

```text
32 blocks * 4 candidates/block = 128 candidates
```

This exactly matches `rerank_r=128`.

At high recall, this is a very narrow candidate pipe. A true neighbor can be
inside a visited block but still be dropped before exact rerank because only
top-4 per block survive PQ scan.

Suggested sweep:

```text
TOP_P = 8, 16
rerank_r = 128, 256
K_MAX = 256 if rerank_r=256 is needed
```

Expected tradeoff:

```text
recall should improve
QPS will drop, mostly in PQ merge/gather/rerank
```

### 4. Entry selection is top-k per 32-block chunk, not true top-k per cell

In `block_search_fused_v23`, entry selection scans blocks inside each selected
L3 cell in chunks of 32:

```cpp
for (int start = 0; start < bc; start += 32) {
    ...
    for (int ep = 0; ep < entry_per_cell; ep++) {
        ...
    }
}
```

This means it selects up to `entry_per_cell` blocks per chunk, not per cell.

For small cells this is almost fine. For large cells, this can insert many
chunk-local entries and rely on the 32-slot beam to filter them.

Potential issue:

```text
entry_per_cell does not mean what the parameter name says
many mediocre chunk-local entries can fill or disturb the beam
```

Suggested fix:

```text
Implement true top-entry_per_cell per selected cell.
Or explicitly rename/understand it as entry_per_chunk.
```

### 5. Beam capacity is fixed at 32 slots

The fused search uses one warp per query:

```text
beam = 32 register slots
degree <= 32
```

Even when `graph_budget` increases from 32 to 64, the active frontier capacity
remains 32.

This may explain why recall barely improves from budget 32 to 64.

Suggested variants:

```text
1. 2 warps/query, 64-slot beam
2. block-level shared-memory beam, 64/128 slots
3. separate entry queue from expansion queue
```

Also check the visited logic:

```text
try_visit happens before insertion into the beam.
If the candidate is marked visited but then not accepted into the 32-slot beam,
it cannot be reconsidered later.
```

This is common in approximate graph search, but it can over-prune when the beam
is small.

### 6. Route still copies beam results to CPU

`route_gpu_v23()` still does:

```cpp
cudaMemcpyAsync(ws.h_top1_ids,  ws.d_top1_ids,  ...)
cudaMemcpyAsync(ws.h_top2_beam, ws.d_top2_beam, ...)
cudaMemcpyAsync(ws.h_top3_beam, ws.d_top3_beam, ...)
cudaStreamSynchronize(s)
```

But the fused graph search uses the GPU buffers directly:

```text
ws.d_top1_ids
ws.d_top2_beam
ws.d_top3_beam
```

So these D2H copies are no longer needed for search.

Suggested fix:

```text
Remove the top1/top2/top3 D2H copies and the synchronize from route_gpu_v23().
```

This should improve QPS but will not fix recall.

### 7. Candidate CSR generation is still CPU-side

Graph edge top-k is GPU, but candidate block lists are still built on CPU:

```text
same (c1,c2): all c3 siblings
near c2 groups: all c3 cells
near c1 groups: top 2 c2 groups, all c3 cells
```

This is acceptable for current experiments, but for larger scale it can become
offline build overhead.

Also, the candidate cap uses `blk_base` as a proxy for cell centroid:

```text
cap by projected distance from first block of the cell
```

This is better than truncating by block id, but it is still a proxy.

Suggested improvements:

```text
1. use actual cell centroid projection for candidate cap
2. include nearest C3 cells by C3 centroid distance instead of all siblings
3. evaluate max_cand_blocks = 512, 1024, 2048
```

## Highest-Priority Diagnostics

### Diagnostic A: exact oracle over visited blocks

Purpose:

```text
separate graph/entry quality from PQ/rerank truncation
```

Procedure:

```text
For each query:
    run current graph traversal
    collect visited blocks
    exact scan all vectors in visited blocks
    compute recall@10
```

Interpretation:

```text
oracle high, current low:
    PQ LUT / TOP_P / rerank bottleneck

oracle low:
    block construction / graph edges / entry selection bottleneck
```

### Diagnostic B: entry-only recall

Measure recall if only entry blocks are scanned, before graph expansion.

Interpretation:

```text
entry high:
    graph expansion or PQ/rerank issue

entry low:
    tree routing or entry block selection issue
```

### Diagnostic C: graph expansion gain

Measure recall for:

```text
entry only
budget 8
budget 16
budget 32
budget 64
budget 128
```

If recall does not improve after budget 32, the graph is not adding useful
blocks or the beam/frontier is over-pruning.

## Recommended Next Implementation Order

1. Remove unnecessary route D2H copies.

```text
Easy QPS win, no algorithm risk.
```

2. Add exact oracle over visited blocks.

```text
This tells whether to work on graph quality or PQ/rerank.
```

3. Sweep `TOP_P` and `rerank_r`.

```text
Try TOP_P=8/16 and rerank_r=128/256.
```

4. Improve physical block construction.

```text
Within each L3 cell, reorder vectors by projected residual / PQ prefix /
mini-kmeans before packing 128-vector blocks.
```

5. Increase beam capacity.

```text
Try 64-slot beam or 2-warps/query.
```

6. Improve per-cell/per-block distance consistency.

```text
Use per-cell residual LUT or a better PQ scoring path for blocks outside the
best routing cell.
```

## Short Conclusion

v23 already proves the throughput potential of block-level graph traversal:

```text
200K-400K QPS range
```

But recall currently saturates around:

```text
0.86-0.87 recall@10
```

The likely root cause is not graph budget alone. The current physical blocks are
not semantic enough, and the PQ/rerank pipe is too narrow for high recall.

The most important next step is:

```text
run exact oracle over visited blocks
```

Then decide whether to fix:

```text
block/graph quality
```

or:

```text
PQ candidate truncation / residual LUT mismatch
```

