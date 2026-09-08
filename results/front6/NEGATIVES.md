# Five changes that were measured and did not pay

Written down because the reasoning behind each still looks right, and the next
person to have the same idea should find the measurement rather than the idea.
Every row is at identical recall to its baseline unless noted — these are
execution changes, not search changes.

Raw logs: `v48.log` (in `results/v48_early_exit/`), `v49b.log`, `v53.log`.

The fifth, v54's table-free primary distance, has its own page:
`V54_SIGN_IP.md`. It is the largest loss here, −30 to −52%, and the most
informative: the identity it rests on is confirmed to four decimals of recall,
and it still lost on the one dataset whose occupancy it tripled.

## 1. A bigger selection buffer, sorted less often (v53, `JHQ_CAP_MULT`)

`cap` is the power of two at or above `ck + BLOCK`, and v45 compacts when the
buffer cannot take another chunk. At ck=1000 and BLOCK=1024 that is cap=2048,
firing at cnt > 1024 and leaving cnt at 1000 — so twenty-five survivors later
it fires again. Doubling cap moves the trigger to 3072, which takes 2,072
survivors: eighty times fewer sorts, of twice the length.

```
dataset           np    x1 QPS       x2       x4
vogue            128   136,107    -1.0%   -16.9%
bge              128   102,603    -4.1%   -24.0%
stella           128    85,985    -3.1%   -24.3%
openai3-3072     128    41,113    +3.7%    +3.1%
vogue            512    75,445    -5.7%   -18.2%
bge              512    37,866    -3.4%   -22.9%
stella           512    28,279    -3.2%   -23.4%
openai3-3072    1024    11,975    +2.1%    +2.3%
```

Two things were wrong with the premise. The arithmetic counted *appends*, and
appends are rare once the threshold tightens — the compaction was already
infrequent, so a bigger buffer barely cut its frequency while paying for the
space every cycle. And the space is what matters: `scan_base` goes 16,392 →
32,776 B, which with the table drops the SM from 3 resident blocks to 2.

**openai3-3072 is the control.** M=384 makes a 49 KB table, which already
forces one block per SM, so there is no occupancy left to lose — and it is the
one dataset that gains. That is the mechanism confirming itself.

The useful consequence is a number to price shared memory with on this card:
**16 KB more per block costs 17–26%** wherever occupancy is not already
pinned at one.

## 2. Sorting the survivors into position order before refinement (v50, `JHQ_REGROUP`)

The top-ck arrives ordered by distance, so the positions handed to the
refinement are scattered — ck reads of `bpv` bytes at random offsets, 1000
scattered 1 KB reads a query at d=1024. Sorted on position they run forward.

```
dataset           np   base QPS    group
vogue            128    106,191    +2.7%
stella           128     68,565    -0.2%
vogue            512     36,353    -0.9%
stella           512     16,126    -0.3%
```

Zero, within noise. The reads are 1 KB each, so each one is already many full
transactions — ordering them changes which DRAM pages are open, not how many
bytes move, and at this size that is not the binding cost.

## 3. Fusing the refinement into the scan kernel (v50, `JHQ_FUSE_REFINE`)

Unfused, the scan writes ck positions and ck primary distances to global and
the next kernel reads them straight back: 16 KB a query at ck=1000, plus a
launch.

```
dataset           np   base QPS     fuse     both
vogue            128    106,191    +0.6%    -0.3%
stella           128     68,565   -14.2%   -13.3%
vogue            512     36,353    -3.6%    -3.8%
stella           512     16,126   -14.8%   -14.8%
```

The round trip it removes is real but small. What it adds is that the scan
kernel now also needs the query staged in shared memory and the register
pressure of both phases at once, and the fused kernel is resident for the
whole scan rather than only for the refinement. Same family as v53: the
saving is in traffic, the cost is in occupancy, and occupancy wins.

## 4. Early exit on a bound (v48, and v49's suffix minimum)

v48 exits a candidate when the partial sum already exceeds the threshold.
v49 tightens it with `s_suf[]`, a suffix minimum over the remaining subspaces
(`min_c T_m[c] = min Thi + min Tlo`), so the test is `a + s_suf[m] > thr`.

```
dataset      np   base QPS    every-8   every-16   every-32
vogue       128     67,723      -1.3%      -0.4%      -0.5%
bge         128     60,947      -2.8%      -1.3%      -1.0%
stella      128     54,760      -1.9%      -0.7%      -0.5%
vogue       512     29,515      +1.6%      +2.3%      +0.9%
bge         512     20,046      -0.8%      +0.8%      +0.7%
stella      512     14,581      +3.7%      +4.2%      +2.9%
```

Between −2.8% and +4.2%, i.e. nothing. The scan is warp-synchronous over a
tile, so an exit only saves work when *every* lane in the warp exits, and the
candidates in a warp are unrelated. The vote costs a `__syncwarp` and a ballot
per check; the bound saves the tail of a loop that is memory-bound anyway.

**One bug worth recording.** v49's first version dropped recall to 0.08. v48
could write `dist = a` on an exit because its bound was `a > thr` alone — a
partial sum that already exceeds the threshold is a valid lower bound, so the
candidate is correctly rejected. With `a + s_suf > thr` that premise is gone:
`a` alone may be well below `thr`, and writing it admits a pruned candidate
with an understated distance. The comment explaining why `dist = a` was safe
had been copied across with its reasoning intact and its premise removed. The
fix is one line:

```cpp
if (live) dist = pruned ? INF : a;
```

## What they have in common

Three of them trade a memory saving for shared memory or registers, and lose.
On this card, at these sizes, **occupancy is worth more than the traffic these
changes remove.** v54 is the mirror image and completes the picture: it spent
instructions to buy occupancy, tripled it on openai3-3072, and lost 30-50%.
Neither resource is the one to trade against; the scan is bound by the
instructions in its inner loop, and only v51 and v52 -- which removed
instructions and memory requests without spending anything -- moved it. The two changes that did pay in the same period — v51's
probe cursor (+2.5 to +148%) and v52's word layout (+30 to +48%) — cost
nothing in either.
