# HBlock v40

v40 is the full-resident staged baseline for SIFT `.bvecs` data. It does not
implement region paging yet.

## Search path

```text
tree routing
-> resident block-graph traversal
-> Kernel A: PQ top-p IDs per (query, block)
-> Kernel B: exact int8 L2 and per-block top-k
-> Kernel C: segmented final top-k per query
```

The explicit `d_pq_candidate_ids` array is the boundary between code scanning
and raw-vector reranking. A later out-of-core version can map those IDs to raw
regions without changing tree or graph traversal.

SIFT `uint8` coordinates are shifted by 128 when read. Applying the same shift
to base and query vectors preserves squared L2 exactly while allowing the
exact kernel to use `int8_t` storage.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target demo_hblock_v40 -j8

SIFT100M_DIR=/root/data/sift100m \
  bash scripts/run_hblock_v40_sift100m.sh
```

Expected files under `SIFT100M_DIR` are:

```text
bigann_base.bvecs
bigann_query.bvecs
gnd/idx_100M.ivecs
```

The build streams `.bvecs` batches and never materializes a float32 copy of
the full base. v40 still keeps raw int8 vectors and PQ payload resident on the
GPU; actual logical-region loading belongs to the next versions.

For large L3 cells, block packing uses the GPU radix order
`(cell_id, projected_coordinate)` instead of quadratic balanced mini-kmeans.
Small cells keep the existing balanced mini-kmeans path.
