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

---

## Why M matters more than it looks: L, and equation 3's placement

`M` is not a free knob. It sets `Ds = d/M`, which sets

```
L = K^(1/Ds) = 2^(B/Ds)      scalar levels per dimension
```

At B=8 the four admissible M give four very different quantisers:

| M | Ds | **L** | primary code |
|---|---|---|---|
| d/8 | 8 | **2** | d/8 bytes — 1 bit/dim |
| d/4 | 4 | **4** | d/4 bytes — 2 bit/dim |
| d/2 | 2 | **16** | d/2 bytes — 4 bit/dim |
| d | 1 | **256** | d bytes — 8 bit/dim, the product structure gone |

Every measurement in this repository is at **L = 2**, the coarsest of the four.

There is a second reason that matters. Equation 3 places the levels at

```
c_i = sigma * sqrt(2) * erfinv(2*q_i - 1),   q_i = (i - 0.5)/L
```

which is the **(i-0.5)/L quantile** — the median of each equiprobable cell. The
paper calls these Lloyd-Max codewords, but the Lloyd-Max optimum is the
conditional *mean* of each cell. At L=2 on a standard normal:

| | level | MSE |
|---|---|---|
| equation 3, quantile | ±0.6745 sigma | 0.3786 sigma^2 |
| true Lloyd-Max, conditional mean | ±0.7979 sigma | 0.3634 sigma^2 |
| | | **+4.19%** |

The two coincide asymptotically in L and diverge most at L=2, which is exactly
where everything here was measured. So the -0.027 attributed to equation 4 has
two components that have not been separated: the Cartesian product's restricted
codeword set, and the quantile-rather-than-centroid placement of the levels
inside it. Both should shrink as L grows.

Sweeping M therefore tests three things at once: the recall/code-size trade,
whether equation 4's price is an artefact of L=2, and whether the placement gap
matters at the settings the paper actually reports.

---

## Equation 4 puts a floor under the primary code length

`Ds | B` is usually stated as a restriction on which M are admissible. It is
also a restriction on **code length**, and that is the more consequential
reading.

At B=8 the primary code is M bytes, and `Ds | B` forces `Ds <= 8`, so

```
M >= d/8   ->   primary code >= d/8 bytes = d bits = 1 bit per dimension
```

**The primary level of JHQ cannot be made shorter than one bit per dimension**,
whatever M is chosen. A freely trained product quantiser has no such floor: at
d=1024, M=32, Ds=32 it produces a 32-byte code, 0.25 bit/dim, four times
shorter than anything equation 4 can reach.

This is where the paper's own grid runs into its own equation. For 1024-d
datasets it lists `M ∈ {64, 128, 256}`; M=64 is 0.5 bit/dim, and equation 4
rejects it. The reference implementation's documented "refusal of M < d/B" is
the same rule seen from the implementation side.

### Three encoders, and which configurations can reach them

| Ds \| B ? | Ds | encoder | per-subspace cost |
|---|---|---|---|
| yes | ≤ 8 | **separable** — per-dimension binary search | `Ds` |
| no | < 32 | hand loop over K codewords | `K·Ds` |
| no | ≥ 32 | GEMM, `‖c‖² − 2yᵀc` via cuBLAS | `K·Ds` at tensor-core rate |

The branches are mutually exclusive. Every configuration measured in this
repository takes the first, so the other two are unexercised — which is why
they are marked "not validated" in the source.

The crossover between the two general encoders was measured in `832c84d`
(primary encode, milliseconds, loop against GEMM): Ds=8 vogue 98 vs 247, Ds=16
openai3-1536 219 vs 263, Ds=32 openai3-3072 439 vs 272. The hand loop keeps a
subspace's Ds floats in registers per thread, and at Ds=32 that register
pressure costs the occupancy hiding its shared-memory traffic — the same wall
the scan kernel hit in `results/v40_scan_lut/`. **Those numbers compare two
general encoders on a free codebook; neither was ever compared against the
separable path, because at those Ds the separable path does not exist.**

### The untested region

`M < d/8` is not a gap in the sweep, it is a region equation 4 structurally
cannot enter. Reaching it means turning equation 4 off, which also gives up the
separable encoder and the O(MK) construction. The question that has never been
asked is the one that matters for short codes:

> at a fixed bit budget below 1 bit/dim, how does a freely trained product
> quantiser compare with JHQ's shortest admissible configuration?

Nothing here answers it. The 98/219/439-against-247/263/272 numbers are encoder
wall-times, not a recall-against-code-length trade.
