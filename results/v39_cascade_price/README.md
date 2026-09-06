# JHQ-GPU-Cascade, priced on v39

The fidelity contract prices this variant at "38%, inside the build's own
+/-2e-4" from the v22 build and a 25-iteration codebook. Re-measured, that is
three numbers rather than one, it hides a second approximation the contract
folds in with the first, and it has an optimum that recall alone cannot find.

nprobe=128, Br=8, `JHQ_BLOCK=1024`, `KEEP=8`. QPS from runs without the dump
(2-3 repeats, spread under 0.3%); the mismatch column from separate runs with
`JHQ_DUMP_TOPCK`, which allocates and writes a file per search call and so is
not comparable on throughput. `casc2.log`, `anom.log`, `anom2.log`, `ex.log`.

| | recall | vs exact | QPS | vs exact | top-ck wrong |
|---|---|---|---|---|---|
| **vogue-768**, exact | 0.9848 | — | 36,812 | — | 0 / 1000 |
| retention only, PM=M | 0.9848 | 0 | 59,669 | **+62%** | 2 |
| cascade M/2 | 0.9846 | -0.0002 | 77,138 | **+110%** | 8 |
| cascade M/4 | 0.9762 | -0.0086 | **88,881** | **+141%** | 102 |
| cascade M/8 | 0.9157 | -0.0691 | 87,940 | +139% | 334 |
| **bge-m3**, exact | 0.9587 | — | 15,590 | — | |
| retention only | 0.9587 | 0 | 18,061 | +16% | |
| cascade M/2 | 0.9584 | -0.0003 | **21,537** | **+38%** | |
| cascade M/4 | 0.9497 | -0.0090 | 19,851 | +27% | |
| cascade M/8 | 0.8839 | -0.0748 | 18,009 | +16% | |
| **stella**, exact | 0.9911 | — | 19,437 | — | 0 |
| retention only | 0.9911 | 0 | 21,620 | +11% | 3 |
| cascade M/2 | 0.9911 | 0 | **23,874** | **+23%** | 17 |
| cascade M/4 | 0.9908 | -0.0003 | 21,098 | +9% | 174 |
| cascade M/8 | 0.9787 | -0.0124 | 18,844 | **-3%** | 357 |

## Three things the single "38%" hid

**It is two approximations, not one.** Keeping only KEEP candidates per thread
is separate from truncating the primary distance, and it applies at every
prefix including PM=M. On its own it is exactly recall-neutral to four decimals
on all three datasets and buys +62%, +16% and +11%. The contract folds it into
the cascade's price; it is the free half.

**Throughput has an interior optimum, and past it the variant is dominated.**
Shorter prefixes do less arithmetic -- at stella PM=M/4 is 5.2M subspace
lookups against M/2's 9.4M -- and run *slower*. Pass one reads
`list_primary_t[m*N + pos]` with pos contiguous across a warp and coalesces;
pass two completes only the KEEP survivors, whose positions are scattered. As
the prefix shortens, cheap coalesced work is traded for expensive scattered
work. The optimum is M/4 on vogue and M/2 on bge-m3 and stella, which tracks
the size of a subspace slice -- 932 KB, 10.1 MB, 17.8 MB -- and so the reach of
the cache behind the scattered pass. **At stella M/8 the variant is both 1.2
recall points worse and 3% slower than the exact path**: strictly dominated,
and nothing about recall alone says so.

**Recall cannot see the selection error.** At stella M/4, **174 of the returned
top-1000 are not in the true top-1000 by complete primary distance**, and
Recall@10 moves by 3e-4. The residual refinement re-ranks what it is given, so
a sixth of the candidate set can be wrong and the first ten still come out
right. The mismatch count comes from `scripts/test_jhq_fidelity.py`, which
compares the returned ck against the exact top-ck of the same run's own
complete distances; its docstring already said why recall is not consulted.

## What is reportable

The retention alone: recall-neutral, +11% to +62%, and it belongs to the exact
path as much as to the cascade -- Algorithm 1 line 4 is a global selection, and
retention only equals it when no thread holds more than KEEP of the answer,
which the mismatch count says is nearly always true here (1 to 3 in 1000).

The cascade at the per-dataset optimum: +110%, +38%, +23% for 2e-4, 3e-4 and 0
of recall. It is partial-distance search, which is not new; what is measured
here is where its optimum sits on a GPU and that going past it loses on both
axes at once.

---

# JHQ_BITONIC_SELECT, re-measured

The flag is read inside `scan_ivf_coalesced_kernel` and nowhere else, so **the
faithful path never sees it**. Its recorded claim -- "holds recall to four
decimal places ... and lifts QPS 46% at nprobe=128, 77% at 32 and 90% at 8" --
is a property of the cascade variant.

The technique is in the faithful path regardless: `jhq_compact_topck` is a
bitonic sort of the whole candidate buffer, and `scan_ivf_exact_kernel` calls
it unconditionally. What that costs there is in `results/v42_scan_split/`: 44%
of vogue's search time, 9-10% on the two large sets. There is no A/B for it
without writing the ck-reduction alternative into the exact kernel.

On the variant it does govern, two repeats each, spread under 0.5%:

| dataset | nprobe | bitonic on | off | gain | recall |
|---|---|---|---|---|---|
| vogue-768 | 8 | 212,745 | 85,476 | **+149%** | identical |
| vogue-768 | 32 | 144,492 | 70,760 | **+104%** | identical |
| vogue-768 | 128 | 76,997 | 40,727 | **+89%** | identical |
| bge-m3 | 8 | 62,850 | 59,901 | +4.9% | identical |
| bge-m3 | 32 | 35,443 | 33,658 | +5.3% | identical |
| bge-m3 | 128 | 21,507 | 14,568 | **+48%** | identical |
| stella | 8 | 68,963 | 61,900 | +11.4% | identical |
| stella | 32 | 37,258 | 36,699 | +1.5% | identical |
| stella | 128 | 23,908 | 16,655 | **+44%** | identical |

Recall is **exactly equal to four decimals in all eighteen pairs**, which is a
stronger statement than the record's "holds to four decimal places".

The recorded 46/77/90% understates today's code: vogue gives 89/104/149%. But
the recorded *explanation* -- "the gain is largest where there is least to scan
because the cost it removes, ck block-wide reductions, does not scale with
nprobe" -- fits vogue and inverts on the two large sets, where nprobe=8 gains
5-11% and nprobe=128 gains 44-48%. A plausible reason is that the scan itself
dominates on 10-18M vectors and the selection does not, which is the direction
`results/v42_scan_split/` reports for the exact kernel. That is an explanation
and not a measurement; it is not tested here.
