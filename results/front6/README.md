# All six fronts, re-measured: 1.31x to 2.32x, and the build cost is five seconds

`scripts/run_front6.sh`, raw in `front6.log`. 48 rows, none failed. v47 (the
head), BLOCK=1024, alpha=100, Br=8, k=10, 1000 queries, three repeats,
`JHQ_DIAG=1`. Cold caches, so `train` and `add` are real for the first row of
each group -- every build timing in `results/` before this was a cache hit
reporting 35 ms.

## Against the published v39 fronts

Matched-recall QPS, interpolated on the overlap of the two fronts:

| | low | | | **high recall** |
|---|---|---|---|---|
| vogue-768 | 1.31x | 1.42x | 1.45x | **1.46x** |
| arxiv-768 | 1.49x | 1.77x | 1.87x | **1.71x** |
| bge-m3 | 1.82x | 1.98x | 1.87x | **1.63x** |
| stella | 1.89x | 1.95x | 2.32x | **2.19x** |
| openai3-1536 | 1.40x | 1.63x | 1.60x | **1.58x** |
| openai3-3072 | 1.59x | 1.88x | 1.91x | **1.88x** |

Two independent sources, both measured here:

- **code** -- v45's compaction predicate (+11 to +38%) and v47's factorised LUT
  (+6 to +14.5%), with v46's sigma fix underneath them.
- **configuration** -- `nlist` raised and `JHQ_N_TRAIN` set to about 39
  training points per centroid instead of the 6 to 98 the published fronts had.

The configuration half alone, both arms measured on v47 so the code is held
fixed:

| | lowest recall measured | **highest** |
|---|---|---|
| vogue-768 | +5% | **+16%** |
| arxiv-768 | +14% | **+41%** |
| bge-m3 | +36% | **+36%** |
| stella | +32% | **+96%** |
| openai3-1536 | +8% | **+23%** |
| openai3-3072 | +11% | **+30%** |

**It grows with recall on all six.** The mechanism `results/v46_sigma/`
identified holds everywhere: better-trained centroids partition more evenly, so
probing a fixed number of lists touches fewer vectors, and the saving is
largest where nprobe is largest.

## The build cost, which was the open question

| dataset | N | nlist | n_train | train | add | VRAM |
|---|---|---|---|---|---|---|
| vogue-768 | 1.0M | 1024 -> 4096 | 100K -> 160K | 0.2 -> **0.3 s** | 0.3 -> 0.3 s | 1379 -> 1405 MiB |
| arxiv-768 | 2.25M | 2048 -> 8192 | 100K -> 319K | 0.2 -> **0.5 s** | 0.5 -> 0.5 s | 2489 -> 2545 |
| openai3-1536 | 999K | 1024 -> 4096 | 100K -> 160K | 0.2 -> **0.4 s** | 0.5 -> 0.9 s | 2251 -> 2293 |
| openai3-3072 | 999K | 1024 -> 4096 | 100K -> 160K | 0.4 -> **0.7 s** | 1.0 -> 0.9 s | 3973 -> 4047 |
| bge-m3 | 10.1M | 8192 -> 16384 | 100K -> 639K | 0.2 -> **1.4 s** | 1.8 -> 3.3 s | 11849 -> 11937 |
| **stella** | **17.8M** | 16384 -> 32768 | 100K -> **1.28M** | 0.3 -> **3.6 s** | 5.7 -> 7.5 s | 20547 -> 20837 |

The largest case is stella: **6.0 s to build becomes 11.1 s, for 2.32x the
query throughput at matched recall**, and VRAM moves 1.4%. Thirteen times the
training set and twice the lists cost five seconds on 17.8M vectors.

That answers the concern this project raised against itself -- that a
throughput number bought with a bigger index is not free. Here it very nearly
is, and the reason is that k-means over `n_train x nlist x d` is a GEMM this
card finishes in seconds while `add()` -- encoding all N vectors, which does
not change -- dominates the build.

## What the numbers are not

- **Recall has a 1e-3 noise floor across builds.** Training is not
  reproducible: the same binary on three cold caches returned 0.9852 / 0.9842 /
  0.9849 on vogue while `cand` was identical in all three, so the IVF
  assignment is deterministic and the codebooks differ in their last bits. Any
  recall difference below 1e-3 between two builds means nothing.
- **`train` is only the first row of each group.** The three that follow read
  the cache the first one wrote.
- **Data loading is not counted**, only `train` and `add`.
- **One alpha and one Br.** alpha=100, Br=8. `results/v43_diag/` showed alpha
  saturating at 20 on stella and 50 on vogue, so a per-dataset alpha is still
  unmeasured on four of these.
