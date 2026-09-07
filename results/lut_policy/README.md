# The LUT policy, decided on both axes it depends on

`results/block_sweep/` reversed `v40`'s verdict at one block size and one
alpha. Two points locate a crossover between them and nothing more, and both
sat at alpha=100 where `cap` is 2048 for every block size, so the shared
budget did not move. `scripts/run_lut_policy.sh` fills the grid: 4 block sizes
x 4 LUT variants x 2 datasets x 2 alphas, two repeats. Raw in `lutpol.log`.

Latency for 1000 queries, mean of two, percentages against fp32-global at the
same block size:

## alpha=100 (ck=1000, cap=2048, scan_base 16 KB at every BLOCK)

| | | fp32 global | `__half` shared | fp32 + L1 carveout | both |
|---|---|---|---|---|---|
| **vogue** M=96 | 128 | 90.24 | 204.81 **+127%** | 226.99 +152% | 204.94 +127% |
| | 256 | 46.63 | 86.06 **+85%** | 96.88 +108% | 86.10 +85% |
| | 512 | 30.19 | 40.15 **+33%** | 45.03 +49% | 40.05 +33% |
| | **1024** | 27.21 | **25.01 -8%** | 27.19 -0% | **25.01 -8%** |
| **stella** M=128 | 128 | 150.56 | 324.43 +115% | 370.03 +146% | 324.54 +116% |
| | 256 | 84.68 | 151.42 +79% | 172.37 +104% | 151.55 +79% |
| | 512 | 52.33 | 77.50 +48% | 86.99 +66% | 77.51 +48% |
| | **1024** | 51.44 | **47.23 -8%** | 51.45 +0% | **47.20 -8%** |

## alpha=8 (the paper's setting; cap follows BLOCK, so scan_base is 2/4/8/16 KB)

| | | fp32 global | `__half` shared | fp32 + L1 carveout | both |
|---|---|---|---|---|---|
| **vogue** | 128 | 56.02 *(shared -- see below)* | 52.68 -6% | 56.38 +1% | 52.64 -6% |
| | 256 | 22.26 | 30.48 +37% | 42.22 +90% | 30.45 +37% |
| | **512** | **15.61** | 18.95 +21% | 24.52 +57% | 18.93 +21% |
| | 1024 | 17.95 | **15.77 -12%** | 17.96 +0% | 15.77 -12% |
| **stella** | 128 | 96.18 | 188.84 +96% | 140.50 +46% | 188.82 +96% |
| | 256 | 80.81 | 102.44 +27% | 124.19 +54% | 102.34 +27% |
| | 512 | 43.44 | 57.73 +33% | 67.62 +56% | 57.65 +33% |
| | **1024** | 41.17 | **36.72 -11%** | 41.25 +0% | **36.68 -11%** |

Recall is constant down every column (0.9845-0.9847 and 0.9917 at alpha=100;
0.9479-0.9481 and 0.9888 at alpha=8), so `__half`'s eleven mantissa bits cost
nothing at either alpha.

## What the grid settles

**One combination wins, and only at one block size.** `__half` in shared at
BLOCK=1024: -8% at alpha=100 and **-11 to -12% at the paper's alpha=8**, on
both datasets, at identical recall. At 128, 256 and 512 the same table loses
21-127%. The mechanism is the one `v40` named and then mis-attributed: a block
holding the table leaves one block resident per SM, so threads per SM is the
block size, and below 1024 there are not enough warps to pay for the banking.

**The L1 carveout is dead.** It is worse than the baseline in 15 of 16 cells
and equal in the sixteenth. `results/v40_scan_lut/` keeps its directory as the
record; nothing should carry the knob forward.

**One cell is mislabelled, and it is the informative one.** `ex_lut` is chosen
automatically by whether the table fits, so the "fp32 global" column is not
always global. At alpha=8, BLOCK=128, vogue: `ex_base` is 2,056 B and an fp32
M=96 table is 98,304, which totals 100,360 against 101,376 available -- so
that cell ran **fp32 in shared**, and the -6% beside it is half-against-fp32
with residency held constant, not shared-against-global. It is the only
measurement of an fp32 shared table in this project, and it says the dtype on
its own is worth about 6%.

That is also the flaw the grid cannot fix: **dtype and residency are one
switch**, so every other cell moves both at once. Separating them needs a
residency control independent of the table's type.

## The block size is not a free choice either

At alpha=8 on vogue, BLOCK=512 with an fp32 global table (15.61 ms) is the
fastest cell in the whole quadrant -- faster than 1024 with either table.
Everywhere else 1024 wins. A single hard-coded policy of BLOCK=1024 plus
`__half` in shared is best or within 1% of best in three of the four
dataset x alpha quadrants, and 1.0% off the best in the fourth.
