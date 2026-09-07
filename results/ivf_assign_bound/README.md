# The int8 assignment bound is 30x wider than the error it bounds

## What the bound is

The IVF assignment computes `argmin_c (||c||^2 - 2 y.c)` for every database
vector. The dot products run on int8 tensor cores at 552 TOPS against fp16's
235, and the rounding of the inputs to eight bits is bounded rather than
assumed. With `y = s_y*y_q + e` and `c = s*c_q + f`, Cauchy-Schwarz gives

```
|y.c - s_y*s*(y_q.c_q)|  <=  ||y||*||f|| + ||e||*||s*c_q||
```

A centroid whose int8 distance cannot reach the running best within twice that
is discarded; anything inside it is recomputed in fp32 from the fp32 inputs.
The argmin is then the fp32 argmin, row for row.

## It is computed, not tuned

`jhq_gpu_index.cu`, inside `assign_from_dots8_kernel`:

```cpp
const float A = 2.f * nf * (rs.y + rs.z);      // 2*max||f||*(||y|| + ||e||)
const float B = 2.f * rs.z * (1.f + 1e-5f);    // 2*||e||
const float REL = 4e-6f, ABS = 1e-7f;
bound = A + B*sqrtf(q) + (q + 2*fabsf(v))*REL + ABS;
```

| term | where it comes from | tunable |
|---|---|---|
| `||y||`, `||e||` | measured per row when that row is quantised | no |
| `s`, `max||f||` | measured once over all centroids at build | no |
| `||c||` | `cent_norms`, which the argmin needs anyway | no |
| `REL`, `ABS` | headroom for fp32's own summation error | constants, not performance knobs |

There is no hyperparameter. Every row carries its own bound.

## Why it is loose

Cauchy-Schwarz is tight only when `e` and `f` are parallel. They are rounding
errors, effectively uncorrelated with the data and with each other, and two
near-random vectors in d dimensions have an inner product about `sqrt(d)`
smaller than the product of their norms. At d=1024 that is a factor of 32, and
the source comment records the measured looseness as "some thirty times wider
than the error" -- the same number.

The consequence is measured: at nlist 8192 and above, **85-88% of rows have at
least one centroid inside the bound** and go to the fp32 recompute. The product
itself halves; the settlement gives most of that back.

| | fp16 product | int8 + settlement | |
|---|---|---|---|
| vogue, 0.93M/768, nlist 1024 | 224 ms | 224 ms | 0% |
| bge-m3, 10.1M/1024, nlist 8192 | 1679 ms | 1574 ms | 6% |
| stella, 17.8M/1024, nlist 16384 | 4271 ms | 3705 ms | **13%** |

Rows whose assignment differs from the fp16 path are ties: 0 of 14838 on
vogue's last batch, 6 of 31748 on bge-m3, 3 of 16359 on stella, recall the same
to three decimals.

## The experiment to run

Replace Cauchy-Schwarz with a probabilistic bound. If `e` and `f` are treated
as independent, `|e.f|` concentrates around `||e||*||f||/sqrt(d)` and a bound at
a few standard deviations is roughly 30x tighter. The recompute fraction should
fall from 85-88% to single digits, and the 13% should approach the ~50% the
halved product alone would give.

**What it costs is the reason it was not done.** The assignment stops being
provably identical to fp32 and becomes identical with high probability. Every
comparison in this project -- equation 4's price at -0.027, the residual
iteration count's +6-8e-3, the cascade's ladder -- assumes the index itself did
not move between arms. A floating assignment makes those numbers unattributable.

So the experiment is worth running as a **measurement of what exactness costs**,
not as a change to the default:

1. add a `JHQ_ASSIGN_BOUND=prob` path with the tighter bound
2. measure the recompute fraction and `add()` time on all three datasets
3. count rows whose assignment differs from the exact path, and the recall delta
4. report both, and keep the provable bound as the default

That last point matters: the useful result is the size of the trade, not a
faster default. If the probabilistic bound moves 3 rows in 16359 and saves 35%
of add time, that is worth stating; adopting it silently is not.

## Report the assignment delta, not the recall delta

The recall impact of a looser bound is almost certainly unmeasurable, and the
existing evidence says so from the wrong direction: the fp16 path settles
nothing at all and still moves only 0-6 rows in 14838-31748, with "recall the
same to the third decimal". A probabilistic bound is strictly better than that.

But "the same to the third decimal" is the resolution of the report, not a
value. Build nondeterminism alone is +/-2e-4, so recall cannot separate these
configurations at all — the same limit this project has now hit four times
(91% of residual codes rewritten for 1.5e-3 of recall, 174 of 1000 top-ck wrong
for 3e-4, 96.8% of codes rewritten for 2e-4).

So the number that belongs in a write-up is **rows whose assignment differs,
out of rows assigned** — 0/14838, 6/31748, 3/16359 for the unsettled path. That
has resolution. A recall delta quoted at this magnitude would imply a certainty
the measurement does not have.
