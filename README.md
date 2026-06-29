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
