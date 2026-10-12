# The residual codebook: what is frozen, and why

This is the settled protocol for equation 5's per-dimension 1-D quantizer. It
supersedes the residual-training paragraph in `jhq_v23_faithful/`
`PAPER_EXPERIMENT_CONFIG.md`, which is frozen and says 25 iterations.

| | setting | env | why |
|---|---|---|---|
| training set | 100,000 sampled vectors | `JHQ_RES_TRAIN_N` | all of Y moves recall 2-8e-4, inside build noise |
| estimator | exact 1-D Lloyd off a segmented sort | — | the histogram one is 227x cheaper and rewrites 97% of codes |
| initialisation | quantile, `sorted[(i+0.5)/Kr * n]` | `JHQ_RES_SEED=0` | beats random on vogue, a wash elsewhere |
| **iterations** | **2000** | `JHQ_RES_MAX_ITER` | the plateau; 25 was 7e-3 of recall short |
| empty cell | holds its centroid | — | Lloyd-Max convention; never observed to fire |

## The evidence, in order

`results/v30_disk_stage/` — training on all of Y, reached in one pass with the
residuals spilled to disk, moves Recall@10 by +2e-4, +8e-4 and -3e-4 on
vogue-768, bge-m3 and stella. Build nondeterminism alone is +/-2e-4. Sampling
100K costs 309 ms against 172 s and 67.8 GB of scratch disk.

`results/v31_init_spread/` — the calibration that finding needed. Holding the
primary codebook and IVF centroids bit-identical, five random initialisations
move recall by 1.5e-3 and rewrite **91% of the residual codes**; the iteration
count moves it by 6-9e-3. Sampling is the smallest effect of the three. It also
settles that **recall cannot validate a codebook**: 91% of codes can change for
1.5e-3 of recall.

`results/v34_lloyd_iterations/` — where to stop. Exact convergence is not
reachable: sparse tail bins two-cycle, so v32's exact fixed-point test and
v33's relative one both ran to their caps, and v34's empty-cell hypothesis was
disproved by its own instrumentation (`0 cells empty` everywhere). The plateau
is at 2000.

## What this does not change

The algorithm. The training set, the initialisation, the stopping rule and the
estimator are all things the paper leaves to the implementation; equation 5,
`O(n * K_r)`, and the codebook it defines are unchanged. Raising the iteration
count moves *toward* the paper, since a Lloyd-Max quantizer is defined as the
fixed point of the Lloyd conditions and 25 iterations was well short of it.
