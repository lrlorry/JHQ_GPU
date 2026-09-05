# Where the residual Lloyd should stop

v31 found the residual codebook's iteration count was the largest source of
variation measured -- larger than initialisation, and far larger than the ALL
vs 100K sampling question -- and that recall was still climbing at 800. This
is the follow-through: three attempts at a convergence test, what each one
showed, and the setting the experiments should freeze on.

Raw: `v32.log`, `v33.log`, `v34.log`, `v34cap.log`.

## Lloyd does not converge here, and two versions were spent finding out why

**v32** tested the fixed point exactly, on the bin boundaries. With the values
sorted the partition *is* its Kr-1 boundaries, so boundaries that hold mean the
update writes back the centroids it read. Every dataset hit the cap of 1000.

**v33** relaxed it to a relative tolerance on the centroids -- converged when
`max |dc| <= 1e-6 * max |c|`, below float32's own 1.2e-7 resolution. Every
dataset hit the cap of **20,000**, with `rel` stuck at 2.5e-5 (vogue), 1.7e-4
(bge-m3) and 1.2e-5 (stella): 100-1000x anything float32 jitter explains.

**v34** held an empty cell in place instead of reseeding it to a quantile of
the data, on the theory that the reseed -- a jump into dense data -- was the
perpetual motion. **That theory was wrong.** The instrumentation v34 added
reports `0 cells empty` on all three datasets at every cap, so the reseed had
never fired. Holding is still the standard Lloyd-Max convention and v34 keeps
it, but it fixed nothing.

What actually moves is sparse tail bins. A bin holding tens of values rather
than the average 3125 shifts its mean by `(v - mean)/cnt ~ 6e-5` when a single
value crosses its boundary, which moves the boundary, which can move the value
back: a genuine two-cycle in the data, not a rounding artefact. Lloyd's
monotone-descent argument does not survive it, and `rel` measures the worst
subspace of 128, so one such bin holds the whole chunk short of the tolerance.

**Exact convergence is not reachable, so the protocol is a fixed count chosen
where recall plateaus, and reported.**

## Where it plateaus (`v34cap.log`, 100K sample, seed 0)

| cap | vogue | train | bge-m3 | train | stella | train |
|---|---|---|---|---|---|---|
| 25 | 0.9770 | 112.5 ms | 0.9514 | 212.1 ms | 0.9852 | 277.1 ms |
| 100 | 0.9804 | 116.0 ms | 0.9543 | 213.1 ms | 0.9880 | 272.7 ms |
| 200 | 0.9817 | 106.8 ms | 0.9563 | 209.7 ms | 0.9890 | 270.6 ms |
| 500 | 0.9829 | 127.3 ms | 0.9570 | 222.1 ms | 0.9903 | 275.1 ms |
| 1000 | 0.9838 | 129.7 ms | 0.9575 | 223.7 ms | 0.9906 | 283.5 ms |
| **2000** | **0.9846** | **134.5 ms** | **0.9585** | **242.6 ms** | **0.9915** | **297.5 ms** |
| 5000 | 0.9843 | 180.4 ms | 0.9592 | 290.0 ms | 0.9921 | 346.0 ms |
| 10000 | 0.9839 | 248.4 ms | 0.9585 | 394.7 ms | 0.9917 | 417.4 ms |
| 20000 | 0.9841 | 409.0 ms | 0.9585 | 508.7 ms | 0.9920 | 570.6 ms |
| 40000 | 0.9845 | 652.3 ms | 0.9586 | 797.1 ms | 0.9916 | 867.3 ms |

Everything from 2000 up is one band: the spread above it is 7e-4, 7e-4 and 6e-4
against repeat-run noise of 8e-4, 3e-4 and 3e-4 (`v34.log`, same config twice).
Below 2000 the trend is monotone and outside that noise.

**2000 buys +7.6e-3, +7.1e-3 and +6.3e-3 over 25, for +22, +30 and +20 ms.**
Going to 40000 buys nothing and costs 5x the residual training.

Per-subspace early stopping would let the converged subspaces drop out and cut
that 22 ms further. It is not worth the mask: residual training is 135 ms of a
multi-second build.

## On the ALL path

stella with all 17.8M vectors and cap 20000 takes 181.0 s against 171.6 s at
cap 1000 (`v34.log`). The iterations are noise there -- the cost is the 67.8 GB
written and read back -- and recall is 0.9921 against 0.9920 for the 100K
sample at the same cap. Consistent with v30: **sampling is still not what
matters.** The iteration count is.

## v35 confirms the default

`v35_confirm.log` -- `jhq_v35_iters_2000` with no environment variable set at
all, so 2000 comes from the source:

| | Recall@10 | vs v34 at cap 2000 | QPS | train | add |
|---|---|---|---|---|---|
| vogue-768 | 0.9848 | 0.9846 | 21229 | 6440.6 ms* | 416.4 ms |
| bge-m3 | 0.9589 | 0.9585 | 8182 | 306.8 ms | 2168.3 ms |
| stella | 0.9914 | 0.9915 | 11504 | 325.5 ms | 5530.0 ms |

All three inside repeat-run noise. QPS is unchanged, as it must be -- the
codebook changed, not the search.

\* the first configuration in the script again pays ~6 s of one-time CUDA
context and cuBLAS setup; bge and stella run later in the same script and show
the real 307 and 326 ms.
