# Re-measured at BLOCK=1024: three verdicts move, one of them reverses

Every standalone diagnostic in this project ran at `JHQ_BLOCK`'s default of
256. Every published frontier row ran at 1024. `results/parameter_coverage/BLOCK_SPLIT.md`
recorded the split and asked for one sweep to say whether the two groups could
be merged. They cannot: **256 -> 1024 is 39-46% off the latency**, and it does
not move the parts proportionally.

`scripts/run_at_block1024.sh`, raw in `results/pending/b1024.log`. 1000
queries, alpha=100, nprobe=128, k=10, two repeats interleaved. Repeat spread is
at most 1.62 ms and usually under 0.5, so nothing below is inside the noise.
Derived table in `derived.txt`.

## 1. The `__half` table in shared memory reverses: -79/-85% becomes +8%

| | vogue-768, M=96 | stella, M=128 |
|---|---|---|
| | **BLOCK=256 / BLOCK=1024** | **BLOCK=256 / BLOCK=1024** |
| fp32 table, global (v39, the head) | 46.64 / **27.29 ms** | 84.86 / **51.39 ms** |
| `__half` table, shared (`v39_lut16`) | 86.12 **+85%** / 25.03 **-8.3%** | 151.52 **+79%** / 47.33 **-7.9%** |
| fp32, global, L1 carveout (`v40_exp`) | 96.72 **+107%** / 27.17 -0.4% | 172.44 **+103%** / 51.53 +0.3% |
| both (`v40_lut16`) | 86.31 +85% / 25.07 **-8.1%** | 151.76 +79% / 47.28 **-8.0%** |

Recall is identical in every cell (0.9846/0.9847 on vogue, 0.9917 on stella),
so `__half`'s eleven mantissa bits summed M times still cost nothing.

`results/v40_scan_lut/` concluded "the scan kernel wants occupancy, not
locality" and kept its directory as a measured negative. **The negative was an
artefact of the block size.** The mechanism the file itself predicted is what
happened: a block asking 82 KB of a 100 KB SM leaves one block resident
whatever its size, so threads per SM *is* the block size -- 256 (17% of 1536)
against 1024 (67%). At 17% there are not enough warps to pay for the banking;
at 67% there are, and the banked table wins 8%.

The carveout is a genuine negative at both sizes: +103/+107% at 256, and
nothing at 1024.

## 2. The lookup is not the bottleneck -- it is the part warps fix

Scan split by compiling parts out (`jhq_v42_scan_split/`), same protocol:

| | | compaction | lookup | gather + rest | total |
|---|---|---|---|---|---|
| vogue | 256 | 20.51 | 10.79 | 15.28 | 46.58 ms |
| | 1024 | 8.52 | 3.84 | 14.91 | 27.27 ms |
| | | **-58%** | **-64%** | -2% | -41% |
| bge-m3 | 256 | 10.15 | 62.52 | 47.36 | 120.03 ms |
| | 1024 | 15.06 | 9.77 | 39.50 | 64.33 ms |
| | | **+48%** | **-84%** | -17% | -46% |
| stella | 256 | 7.75 | 35.38 | 41.65 | 84.78 ms |
| | 1024 | 9.91 | 7.43 | 34.11 | 51.45 ms |
| | | **+28%** | **-79%** | -18% | -39% |

`results/v42_scan_split/` called the lookup "the largest single component at
scale" -- 52% on bge-m3, 42% on stella. At the block size the fronts run it is
**15% and 14%**, and it is the component that responds most to more warps
(-79 to -84%). The divergence is real; its cost is latency, and latency is
what occupancy hides. That also explains item 1 from the other side: shared
memory removes the divergence, and only at 1024 are there enough warps left
over to profit from it.

## 3. Compaction moves in opposite directions on different datasets

`cap` is 2048 at ck=1000 for every block size, so the same bitonic sort gets
four times the threads. On vogue that halves it (-58%). On bge-m3 and stella
it gets **more expensive in absolute time** (+48%, +28%).

The prediction written into `scripts/run_at_block1024.sh` -- "its share must
fall" -- was right on vogue and wrong on the two large sets. `v42`'s own
explanation says why the sign can differ: vogue scans 12.5% of its base per
query against 0.8-1.6%, so `*s_cnt` crosses `ck` constantly and the sort runs
often, while on the large sets the block mostly spins in the threshold path,
where four times the threads is four times the contention on `s_cnt` for the
same small number of survivors.

## 4. The refine share rises, as predicted

| | BLOCK=256 | BLOCK=1024 |
|---|---|---|
| vogue-768 | 3.08 ms = 6.6% | 2.45 ms = **9.0%** |
| bge-m3 | 4.10 ms = 3.4% | 3.15 ms = **4.9%** |
| stella | 3.59 ms = 4.2% | 3.25 ms = **6.3%** |

`residual_refine_fused_kernel` launches with a hard-coded 256 threads, so it
gains little while the scan gains 40%. Its absolute time does fall 11-23%,
which the hard-coded launch does not explain and the L2 pressure from the
faster scan probably does; either way the ceiling on refine-stage work is now
**9% on vogue**, not 6%.

## What is now the head

At BLOCK=1024 the fastest configuration measured is `__half` in shared, worth
8% on both datasets tested, at identical recall. Deciding whether that becomes
the one path (this repo does not keep both behind a flag) needs the crossover
located and the paper's alpha checked, because at alpha=8 `cap` follows the
block size and the shared budget moves with it -- `scripts/run_lut_policy.sh`.
