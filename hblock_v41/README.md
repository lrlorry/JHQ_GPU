# HBlock v41

v41 preserves v38's search semantics while bounding the GPU-resident vector
payload with block-major logical regions.

```text
resident tree + block graph navigation
-> sorted (query, block) pairs
-> unified-region waves
-> fused per-block PQ top-r -> exact int8 L2
-> global segmented top-k merge
```

Each physical block is one complete record containing PQ codes, vector IDs,
and raw vectors. A bounded GPU staging pool holds one region wave at a time.
Every `(query, block)` pair still executes the same fused operation as v38 and
writes to its original result slot, so region capacity changes transfer and
launch counts but not the candidate set or final merge.

Queries are aggregated before region planning. A region is loaded once per
query batch and serves every query that requests a block in that region.

## Run SIFT100M

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target demo_hblock_v41_sift100m -j8

SIFT100M_DIR=/root/data/sift100m \
REGION_MIB=1 REGION_CAP=512 BATCH=10000 \
bash scripts/run_hblock_v41_sift100m.sh
```

Expected files:

```text
bigann_base.bvecs
bigann_query.bvecs
gnd/idx_100M.ivecs
```

The GPU payload budget is approximately:

```text
REGION_CAP * REGION_MIB MiB
```

Tree centroids, block centroids, graph adjacency, block-to-region metadata,
and query workspaces remain resident in addition to that budget.

## Current Boundary

v41 bounds GPU memory but keeps the region store in host RAM. It uses one
CUDA stream, so region copies and kernels are wave-ordered rather than
double-buffered. File-backed `mmap` stores and copy/compute overlap belong to
the next step.
