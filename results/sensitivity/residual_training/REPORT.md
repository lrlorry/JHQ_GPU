# Residual-codebook training size: is 100K enough?

Paper Eq. 5 collects R^m over **all** y in Y. The implementation trains on a
deterministic 100,000-vector sample. This measures what that costs.

Run against `paper-faithful-v1` semantics; no algorithm change.

## Experimental controls

`n_train` used to drive four things at once — sigma, the Eq. 4 primary
codebook, the IVF centroids and the residual codebook — so sweeping it would
have varied all four and credited the result to one. `train()` now takes a
separate set used **only** for the residual level, rotated by the Pi already
built and encoded against the codebook already built.

**Held constant, and verified rather than asserted.** The encode dump carries
the primary codebook and the IVF centroids; `analyze_res_train.py` compares
them bit for bit before reporting anything else. Across every comparison:

```
primary codebook : same
IVF centroids    : same
```

Also constant: dataset, Pi, B=8, M, Ds=8, alpha=100, k=10, nlist, nprobe,
batch=1024, numerical mode (fp32 table, TF32 off, exact top-alpha*k), hardware,
and the frozen faithful search path. **The only variable is
`JHQ_RES_TRAIN_N`.**

Not held constant, and worth naming: the 1-D k-means on residuals runs 25
iterations from a quantile initialisation, so a different sample size reaches a
different local optimum. That is inherent to varying the training set and is
part of what is being measured.

## Raw results

`sweep_raw.log`, `sweep_table.txt` — 72 search runs, 8 codebook dumps, 0 errors.

## Recall@10 sensitivity

Deltas against ALL, per (dataset, Br, nprobe):

| dataset | Br | nprobe | 10K | 50K | 100K | 250K | 500K | ALL |
|---|---|---|---|---|---|---|---|---|
| vogue | 8 | 32 | −0.0003 | −0.0002 | −0.0005 | −0.0015 | −0.0016 | 0.9206 |
| vogue | 8 | 128 | −0.0009 | +0.0001 | −0.0002 | −0.0020 | −0.0025 | 0.9772 |
| vogue | 8 | 256 | −0.0010 | −0.0002 | −0.0003 | −0.0017 | −0.0023 | 0.9839 |
| vogue | 4 | 32 | +0.0004 | −0.0006 | −0.0007 | −0.0015 | −0.0015 | 0.8905 |
| vogue | 4 | 128 | −0.0002 | −0.0025 | −0.0018 | −0.0030 | −0.0023 | 0.9395 |
| vogue | 4 | 256 | +0.0005 | −0.0023 | −0.0012 | −0.0024 | −0.0022 | 0.9445 |
| o3072 | 8 | 32 | +0.0002 | +0.0001 | 0.0000 | −0.0004 | −0.0004 | 0.9101 |
| o3072 | 8 | 128 | −0.0001 | −0.0003 | −0.0003 | −0.0001 | −0.0006 | 0.9692 |
| o3072 | 8 | 256 | +0.0003 | 0.0000 | −0.0003 | −0.0001 | −0.0006 | 0.9840 |
| o3072 | 4 | 32 | +0.0001 | −0.0003 | −0.0004 | −0.0006 | +0.0001 | 0.9014 |
| o3072 | 4 | 128 | −0.0017 | −0.0013 | −0.0007 | −0.0011 | +0.0001 | 0.9576 |
| o3072 | 4 | 256 | −0.0017 | −0.0013 | −0.0005 | −0.0012 | +0.0002 | 0.9710 |

**|100K − ALL| = 0.0018 at worst** (vogue Br=4 nprobe=128). Band: **0.001–0.005**.

Two things the table shows that the single number does not.

*The sign is systematic.* 100K sits at or below ALL in eleven of twelve
configurations. The magnitude is tiny but the direction is not random.

*The trend is not monotone.* 250K and 500K are frequently **worse** than 100K
and than 10K — vogue Br=8 nprobe=128 reads −0.0020 and −0.0025 at 250K/500K
against −0.0002 at 100K. More data does not monotonically help, which places
these differences in the k-means's own convergence rather than in sample size.

QPS is flat across training sizes at each (dataset, Br, nprobe) — within 1% —
as it must be, since training size does not enter query execution.

## Training cost

Whole train phase, mean over configurations:

| n_res | vogue-768 | openai3-3072 |
|---|---|---|
| 10K | 99 ms | 279 ms |
| 100K | 124 ms | 258 ms |
| 250K | 227 ms | 563 ms |
| 500K | 314 ms | 877 ms |
| ALL | 388 ms | 1511 ms |

ALL costs **+264 ms** on vogue and **+1253 ms** on openai3-3072 over 100K —
against add times of 187 ms and 622 ms respectively, so it roughly doubles to
triples the build, on an operation performed once.

## Codebook convergence — the part recall cannot show

100K against ALL, same fixed sample, same primary codes:

| dataset | Br | Kr | mean abs Δcentroid | max abs Δ | normalized | **residual codes that change** |
|---|---|---|---|---|---|---|
| vogue | 8 | 256 | 1.37e-04 | 3.35e-03 | 2.74e-02 | **34.56%** |
| vogue | 4 | 16 | 1.88e-04 | 2.94e-03 | 3.17e-02 | **2.12%** |
| openai3-3072 | 8 | 256 | 2.11e-05 | 2.16e-03 | 3.70e-02 | **8.99%** |
| openai3-3072 | 4 | 16 | 3.80e-05 | 5.09e-04 | 1.22e-02 | **0.59%** |

**The codebooks have not converged.** At Br=8 on vogue, a third of residual
code assignments change between 100K and ALL, and recall moves 0.0003. The
search-level similarity is therefore **not** explained by codebook convergence.

The explanation is the density of the codebook: Kr=256 codewords over a narrow
residual range makes adjacent codewords nearly interchangeable, so a
reassignment moves the reconstruction by far less than the quantisation step.
Br=4, with sixteen codewords spread over the same range, has cells wide enough
that assignments are stable — 2.1% and 0.59% — which is why its assignment
churn is an order of magnitude lower even though its centroid distances are
comparable.

That relationship is the reason to report assignment churn rather than centroid
distance alone: the two point opposite ways at Br=8.

## Classification

**BORDERLINE.**

Not SUFFICIENT: the decision rule requires differences that are "negligible and
not systematic" *and* codebooks that have "substantially converged". The
differences are negligible in magnitude but systematic in sign, and the
codebooks demonstrably have not converged at Br=8.

Not INSUFFICIENT: no configuration shows a material recall improvement from
more data. The largest gain from 100K to ALL is 0.0018, and 250K/500K are often
worse than 100K, so there is no consistent benefit to buy.

## Recommended protocol

**ALL vectors** for the six-dataset main benchmark.

The reasoning is cost, not accuracy. ALL is what Eq. 5 defines; it is
consistently equal or marginally better; it costs about 1.3 s of build time on
the largest of these two datasets; and adopting it removes a documented P1
deviation from the paper's own specification rather than carrying it into the
results with a footnote. Choosing 100K would save a second per build and oblige
the paper to defend a deviation whose effect is systematic in sign even if
small.

If build time later proves to matter at 17.8M — where ALL means seventeen times
the vectors measured here and the cost is untested — 100K is defensible on this
evidence and should be re-measured at that scale rather than extrapolated.
