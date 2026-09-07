# What has been swept, and what has only been assumed

Scanned every `results/**/*.csv`: **201 JHQ/JQ rows**. This is what those rows
say was varied, plus the knobs that never appear in a row at all.

## Swept, and recorded in the harness

| | values seen | note |
|---|---|---|
| `nprobe` | 1, 4, 8, 16, 32, 128, 256, 512, 1024 | the one parameter properly swept |
| `Br` | 4, 8 | both of the paper's residual widths |
| `prefix` | 1/1, 1/2 | plus 1/4, 1/8, 1/16, 1/32 in `results/v39_cascade_price/` |
| `M` | 96, 128, 192, 384 | **but see below — one or two per dataset** |

## Fixed at a single value in every row ever recorded

| | the only value | what the paper uses |
|---|---|---|
| **`alpha`** | **100.0** | **{2, 4, 8}** |
| **`B`** | **8** | **{8, 4}** — `[B,Br] = [4,8]` never run |
| `k` | 10 | 10 |
| `ivf_iters` | 8 | — |
| `kmeans_iters` | 5 | — |
| query `batch` | 1024 | — |
| `JHQ_BLOCK` | 1024 | — |

`alpha` is the largest of these. At `alpha=100, k=10` the refinement set is
`ck = 1000`; the paper's `alpha=8` gives 80. Several results in this repository
depend on `ck` — the compaction's 44% share of vogue's search time, where the
cascade's throughput optimum sits — and none of them has been checked at the
paper's range.

## `M` is swept far less than the table above suggests

Equation 4 admits `M ∈ {d, d/2, d/4, d/8}` at B=8. Per dataset:

| dataset | d | admissible M | actually run |
|---|---|---|---|
| vogue-768 | 768 | 96, 192, 384, 768 | **96, 192** |
| arxiv-768 | 768 | 96, 192, 384, 768 | **96** |
| bge-m3 | 1024 | 128, 256, 512, 1024 | **128** |
| stella | 1024 | 128, 256, 512, 1024 | **128** |
| openai3-1536 | 1536 | 192, 384, 768, 1536 | **192** (96 also run, inadmissible) |
| openai3-3072 | 3072 | 384, 768, 1536, 3072 | **384** (96 also run, inadmissible) |

`nlist` likewise takes four values across the six datasets but one per dataset,
so it is a per-dataset constant rather than a swept parameter.

M is not a free knob: it sets `Ds = d/M`, which sets `L = 2^(B/Ds)`, the scalar
levels per dimension. M = d/8 gives **two levels per dimension**; M = d/4 gives
four; M = d/2 gives sixteen. **The measured price of equation 4 -- -0.027 on
vogue's primary level -- was measured only at Ds=8, where the Cartesian product
is at its weakest.** It should narrow as Ds falls, and that has not been checked.

## Swept, but only into standalone logs

These were varied and the results written up, but through one-off scripts
rather than `bench_all.py`, so they carry no provenance header and do not
appear in any CSV row:

`JHQ_RES_MAX_ITER` (5 → 40000), `JHQ_RES_SEED` (0 → 5), `JHQ_RES_TRAIN_N`
(100K vs all of Y), `JHQ_RES_TOL`, `JHQ_LUT32`, `JHQ_PREFIX_KEEP`,
`JHQ_REFINE_*`, `JHQ_SCAN_NO_*`.

## Never swept, and never recorded

A run cannot say what it used for any of these, because nothing writes them
down:

| | default | where |
|---|---|---|
| `add_batch` | **65536** | `jhq_gpu_index.cuh:42` |
| `JHQ_ASSIGN_BATCH` | **32768** (nlist ≥ 8192) / 8192 | four call sites |
| residual streamed batch | **200000**, hard-coded | `train_residual_codebook_streamed` |
| `JHQ_RES_CHUNK` | derived from a device budget since v36 | |
| `JHQ_RES_HOST_GB` | 3/4 of the cgroup limit since v37 | |

`add_batch` is the one worth measuring first. `add()` is **95% of stella's
build** — 5.5-6.5 s against 0.3 s of training — so a 20% change there is a 19%
change in the whole build, against the 0.35% that the residual iteration count
is worth on the same dataset. Its 65536 was reasoned from memory footprint
("well under 1 GB even at d=3072") and never from throughput.

## Suggested order when this is picked up

1. **M**, per dataset, across the admissible set. Fills the largest algorithmic
   gap and tests whether equation 4's price is an artefact of Ds=8.
2. **alpha** at the paper's {2, 4, 8}. Several conclusions here depend on ck.
3. **`add_batch`** and `JHQ_ASSIGN_BATCH`. Cheap, and aimed at 95% of the build.
4. **B=4**, for the paper's `[4,8]` configuration.

Everything above should go through `bench_all.py` so the rows record what
produced them; the standalone-log category above is how a measurement becomes
unciteable a month later.
