# What the scan kernel is actually spending time on

`ncu` cannot run on this box: the container inherits `RmProfilingAdminOnly=1`
from a driver it cannot reload, so every counter returns `ERR_NVGPUCTRPERM`.
The split is done by removing parts of the kernel and timing what is left --
the same method that sized the refine stage, and it is checked against nsys
below rather than trusted.

1000 queries, alpha=100, nprobe=128, mean of three. Raw: `v42.log`, and
`v41.log` for the attempt that did not work.

| | full | no compaction | neither | compaction | lookup | gather + rest |
|---|---|---|---|---|---|---|
| vogue-768 | 45.68 ms | 25.59 ms | 15.06 ms | **20.09 ms (44%)** | 10.52 ms (23%) | 15.06 ms (33%) |
| bge-m3 | 119.72 ms | 109.53 ms | 47.45 ms | 10.19 ms (9%) | **62.08 ms (52%)** | 47.45 ms (40%) |
| stella | 85.40 ms | 77.00 ms | 41.29 ms | 8.39 ms (10%) | **35.72 ms (42%)** | 41.29 ms (48%) |

Reconstructing the scan kernel's own time from these parts gives 41.64 ms on
vogue and 79.34 ms on stella, against nsys's 42.49 and 79.50 -- **2.0% and
0.2%**. Two methods that share no machinery agree, so the split is real.

## The lookup is the largest single component at scale

> **At BLOCK=256 only.** These runs did not set `JHQ_BLOCK` and so took its
> default of 256; the fronts run at 1024, where the lookup is 14-15% rather
> than 42-52% and is the *best*-scaling part of the kernel (-79 to -84% from
> 256 to 1024). The divergence described below is real -- its cost is latency,
> and warps hide latency. `results/block_sweep/` has the split at both sizes.

`g_lut[m * 256 + cm]` with `cm` a data-dependent byte. Across a warp the 32
threads read 32 unrelated 4-byte positions inside one 1 KB row, so a single
instruction becomes up to 32 separate 32-byte transactions, and it repeats M
times per candidate.

**This is divergence, not working set.** The warp walks `m` in lockstep, so only
one 1 KB row is live at a time and it fits anywhere. That reframes the two
failures in `results/v40_scan_lut/`: putting the table in shared memory does fix
the divergence, because shared memory is banked, but pays one block per SM for
it; the L1 carveout does not address divergence at all and only costs occupancy.
Neither was aimed at what is actually expensive.

## vogue is a different kernel with the same code

Compaction is 44% there and 9-10% on the large sets. Same `ck`, same `nprobe`,
so the difference is in how often `*s_cnt` crosses `ck` -- vogue scans 12.5% of
its base per query against 0.8-1.6% for the others, and its distance
distribution puts far more candidates under the running threshold. Any work on
the compaction would help vogue and be nearly invisible on bge-m3 and stella,
and the reverse for the lookup.

## The one combination never tried

`__half` in **global** memory. It halves the row to 512 B, so about 16
transactions where float takes 32, and it costs no occupancy at all. v39_lut16
never tested this: with a `__half` table `ex_lut` becomes true automatically and
it went to shared, which is the configuration that lost 65-89%. Forcing the
global path with a half table separates the two effects that have so far only
been measured together.
