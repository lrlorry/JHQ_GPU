# JHQ_GPU

GPU implementation of JHQ (Johnson-Lindenstrauss Enhanced Hierarchical Quantization).

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run (same datasets as JHQ_repro)

```bash
./build/demo_v1_plain \
    ../JHQ_repro/datasets/openai-1024-100k_base.fvecs \
    ../JHQ_repro/datasets/openai-1024-100k_query.fvecs \
    ../JHQ_repro/datasets/openai-1024-100k_groundtruth.ivecs \
    128   # M
    8     # B
    4     # Br
    4.0   # alpha
    10    # k
```

## Directory layout

```
common/              shared utilities (never versioned)
  cuda_utils.cuh
  fvecs_io.cuh

common/recall.cuh    Recall@k evaluator -- shared, never versioned.
                     Replaces the per-demo copy that scanned the whole
                     ground-truth row instead of its first k entries.

cpu/                 shared CPU train code (never versioned)
  erfinv.h
  codebook.h/.cpp    Lloyd-Max analytical codebook + 1D k-means
  jl_transform.h/.cpp  QR rotation matrix (LAPACK)

v1_plain/            version 1 — correctness baseline
  encode.cuh/.cu       Kernel: primary encode, residual encode+correction
  search.cuh/.cu       Kernel: LUT build, ADC scan, top-k, residual refine
                       SearchWorkspace struct (preallocated, no hot-path malloc)
  jhq_gpu_index.cuh/.cu  JHQGpuIndex class (train / add / search)

examples/
  demo_v1_plain.cu   → binary: build/demo_v1_plain
```

## Evaluation

`Recall@k` comes from `common/recall.cuh` for every `demo_jhq_v*`. It is the
standard set intersection against the true top-k:

```
Recall@k = |{returned top-k} ∩ {true top-k}| / k
```

Before `v15_eval_fix` each demo carried its own copy that looped over the full
`.ivecs` row width (20 or 100 here, depending on which generator built the
dataset) rather than the first k, so it reported
`|top-k ∩ true top-gt_k| / k` -- always at least Recall@k, saturating at 1.0000
well before the search is exact, and not comparable across datasets whose
ground truth was generated at different widths. Every number in `results/`
predating v15 is that old score. The demos still print it, labelled
`Pre-v15 score`, so a re-run can be lined up against those CSVs.

```bash
./build/test_recall     # unit tests for the evaluator
```

`demo_jhq_v15_eval_fix` additionally takes an `out_prefix` and writes the
returned ids (`<prefix>.ivecs`) plus a full run record (`<prefix>.json`,
including `gt_width` and `eval_gt_k`), so changing the metric again costs a
re-parse rather than a re-search.

## Design documentation

- `HBLOCK_CURRENT_DESIGN.md`: current architecture, evidence, open problems,
  1B design, and execution roadmap.
- `VERSIONS.md`: historical version-by-version implementation record.
- Version-local `README.md` files: build, run, and implementation boundaries.

## Adding a new version

```bash
cp -r v1_plain v2_my_change
cp examples/demo_v1_plain.cu examples/demo_v2_my_change.cu
# edit v2_my_change/ and examples/demo_v2_my_change.cu
# update includes: s/v1_plain/v2_my_change/g
# CMakeLists.txt: uncomment add_jhq_version(v2_my_change)
make demo_v2_my_change
```

## v1 design choices (ablation targets)

| Component | v1 | next candidates |
|---|---|---|
| Phase 1 top-k | thrust::sort O(N log N) | CUB radix select O(N) |
| Query batching | one query at a time | batch 32–64 |
| Residual codebook | one global Kr-entry codebook | M per-subspace codebooks (paper §4.2) |
| LUT layout | factored M×Ds×K1D in shared mem | flat M×256 |
| Residual k-means | CPU (D2H training residuals) | GPU k-means |
