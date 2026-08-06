# HBlock v41

v41 is the first bounded-GPU-payload SIFT100M implementation based on v40's
split PQ/exact search path.

```text
resident tree + block graph navigation
-> sorted (query, block) pairs
-> code-region waves -> per-block PQ top-r
-> raw-region waves -> exact int8 L2
-> global segmented top-k merge
```

PQ codes, vector IDs, and raw vectors are packed into block-major logical
regions held in host memory. Only a configurable number of code and raw
regions can reside in their GPU pools. A query batch that touches more regions
than a pool can hold is split into contiguous waves, so pool capacity affects
transfer volume and latency but not which visited blocks are evaluated.

The code and raw stages are separate. Kernel A also emits each candidate's
position inside its physical block, avoiding an `O(N)` vector-location table
on the GPU.

## Run SIFT100M

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target demo_hblock_v41_sift100m -j8

SIFT100M_DIR=/root/data/sift100m \
REGION_MIB=1 CODE_REGION_CAP=256 RAW_REGION_CAP=256 \
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
(CODE_REGION_CAP + RAW_REGION_CAP) * REGION_MIB MiB
```

Tree centroids, block centroids, graph adjacency, block-to-region metadata,
and query workspaces remain resident in addition to that budget.

## Current Boundary

v41 bounds GPU memory but keeps the two region stores in host RAM. It uses one
CUDA stream, so region copies and kernels are wave-ordered rather than
double-buffered. File-backed `mmap` stores and copy/compute overlap belong to
the next step.
