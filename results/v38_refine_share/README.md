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
