# How much of search is the residual refine stage?

Step 1 of three, and it decides whether the other two are worth running.

`JHQ_NO_RESIDUAL` already existed in `residual_refine_fused_kernel`: it leaves
`d_res` unused, so the per-candidate loop over d dimensions and every
`list_res` read fold away at compile time. Two binaries from the same sources,
same index, same queries; the latency difference is the whole cost of residual
refinement. No instrumentation, so nothing is distorted the way `STEP_TIMING`
distorts by breaking graph capture.

RTX 5090, 1000 queries, alpha=100, k=10, 100K residual training sample.
Raw: `step1.log`.

| dataset | full | JHQ_NO_RESIDUAL | refine | share | repeat noise |
|---|---|---|---|---|---|
| vogue-768 | 46.78 ms | 43.74 ms | 3.04 ms | **6.5%** | 0.42 ms |
| bge-m3 | 122.76 ms | 116.37 ms | 6.39 ms | **5.2%** | 0.26 ms |
| stella | 87.48 ms | 82.42 ms | 5.06 ms | **5.8%** | 0.21 ms |

The gap is 12-25x the spread between repeats, so the measurement is not in
doubt. The `JHQ_NO_RESIDUAL` recall column (0.6342 / 0.7159 / 0.7445) is the
primary level alone and is not a result.

## So the cache work is capped at 6%

The three variants in `jhq_v38_refine_cache/` were written to stop the residual
codebook being evicted from L1. Even eliminating the refine stage entirely buys
5-6% of throughput, so a cache tweak that captures half of it is worth 3%.

Of the three, only `ldcs` is worth measuring: one line, marking the
read-once `list_res` loads evict-first, no occupancy cost. `smem` moves the
codebook into 99 KB of opt-in shared, which on a 100 KB SM leaves one block
resident and takes away the occupancy that hides L2 latency -- a real risk of
being slower, for a stage worth 6%. `l1` is the same trade with less upside.

## Where the other 94% is

Not measured here. The same compile-out trick would size the primary scan and
the top-ck selection without instrumentation, and that is where any further
search work belongs: at alpha=100 and nprobe=8, stella scans about 8700
candidates a query to refine 1000 of them, so the scan sees 8.7x the traffic
the refine stage does.

---

# Steps 2 and 3

## The three cache variants, three repeats each (`step23.log`)

Latency for 1000 queries, mean of three, against `v38_base`:

| | vogue-768 | bge-m3 | stella |
|---|---|---|---|
| base | 46.91 ms | 122.99 ms | 87.10 ms |
| `__ldcs` on `list_res` | 53.65 ms **+14.4%** | 133.87 ms **+8.8%** | 99.84 ms **+14.6%** |
| codebook into opt-in shared | 45.47 ms **-3.1%** | 123.04 ms 0 | 87.45 ms 0 |
| carveout biased to L1 | 45.84 ms **-2.3%** | 120.29 ms **-2.2%** | 85.21 ms **-2.2%** |

Recall is identical across all four (0.984x / 0.958x / 0.991x), as it must be.

**`__ldcs` was reasoned backwards.** The argument was that a candidate's
residual code is read once and never reused, so marking it evict-first would
leave L1 to the codebook. But the reuse is inside a *cache line*, not across
candidates: byte j=0 fetches the 32-byte line that then serves j=0..31, and
evict-first throws it away between those uses. The line is refetched up to 32
times, which is the 9-15% it costs.

Shared memory helps only where the codebook fits. `3072 + 98304 = 101,376` is
exactly this card's `MaxSharedMemoryPerBlockOptin` at M=96; M=128 wants 135,168
and falls back to the same path as base, which is why bge-m3 and stella show
nothing.

The carveout is the same finding from the other side, and the useful one: the
codebook *was* being evicted, and giving L1 more of the unified block fixes it
without the occupancy cost of putting 99 KB of shared on a 100 KB SM.

## Where search time actually goes (`prof.log`)

`nsys profile -t cuda --cuda-graph-trace=node`, which reports kernels inside a
captured graph -- the thing `STEP_TIMING` could not do. The profile covers the
whole process, so search kernels are the ones with 6 instances (5 timed repeats
plus a warm-up) against the hundreds the build runs.

| kernel | per repeat, vogue | per repeat, stella | share of search |
|---|---|---|---|
| **`scan_ivf_exact_kernel`** | 42.49 ms | 79.50 ms | **91%** |
| `residual_refine_fused_kernel` | 3.49 ms | 5.15 ms | 7% |
| `select_probes_kernel` | -- | 1.02 ms | 1% |
| sum | 45.98 ms | 85.67 ms | vs 46.9 / 87.1 measured |

The 7% agrees with step 1's 5.8% from an independent method.

**The primary scan is search.** The large entries in the raw profile --
`assign_from_dots8_kernel` at 34% of stella's whole process,
`cutlass_80_tensorop_i16832gemm_s8` at 25%, `scatter_list_storage_kernel` at
6.5% -- are index construction, at 232 to 575 instances.

At alpha=100 and nprobe=8 stella scans about 8700 candidates a query at M=128
lookups each: 1.11e9 lookups in 79.5 ms, or 14 G/s. Against 170 SMs at ~2.4 GHz
that is roughly 29 cycles a lookup, which is a lot for a table that should be
in shared memory. Any further search work belongs here.

## v39 verified (`v39.log`)

Interleaved with `v38_base`, three repeats each, so drift cannot read as a win:

| | base | v39 | gain | repeat spread |
|---|---|---|---|---|
| vogue-768 | 46.60 ms | 45.84 ms | **1.6%** | 0.41 ms |
| bge-m3 | 122.66 ms | 119.96 ms | **2.2%** | 0.67 ms |
| stella | 86.82 ms | 85.26 ms | **1.8%** | 1.13 ms |

Recall unchanged. v39 carries one policy and no flag: shared where the codebook
fits, the L1 carveout where it does not.

---

## At BLOCK=1024 the ceiling is 9%, not 6%

Everything above ran at `JHQ_BLOCK`'s default of 256. The fronts run at 1024,
where the scan gains 39-46% and `residual_refine_fused_kernel` -- launched with
a hard-coded 256 threads -- does not keep up:

| | BLOCK=256 | BLOCK=1024 |
|---|---|---|
| vogue-768 | 3.08 ms = 6.6% | 2.45 ms = **9.0%** |
| bge-m3 | 4.10 ms = 3.4% | 3.15 ms = **4.9%** |
| stella | 3.59 ms = 4.2% | 3.25 ms = **6.3%** |

So the cache work is capped at 9% on vogue rather than 6%, and the hard-coded
`<<<B, 256, ...>>>` in the refine launch is itself now the more interesting
target: it is the only kernel in the search path that does not scale with
`JHQ_BLOCK`. Raw and method in `results/block_sweep/`.
