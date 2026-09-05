# How far does exact Lloyd move on its own?

v30 found that training the residual codebook on all of Y rather than a 100K
sample moves Recall@10 by +2e-4, +8e-4 and -3e-4. Build nondeterminism alone
is +/-2e-4, so those numbers say "small" without saying smaller than what.

The reference they need is Lloyd's own arbitrariness. The paper specifies
1-D k-means and neither of the two things an implementation must decide: where
the Kr seeds start, and how many iterations run. `jhq_v31_init_spread/` exposes
both as `JHQ_RES_SEED` (0 = the quantile placement every earlier version used,
anything else draws Kr positions uniformly at random and sorts them) and
`JHQ_RES_ITER` (default 25). Defaults reproduce v30: seed 0 / iter 25 gives
0.9770-0.9771, 0.9516-0.9519, 0.9850-0.9854 against v30's 0.9771, 0.9516,
0.9852.

Everything else is held. `analyze_res_train.py` confirms the primary codebook
and the IVF centroids are bit-identical across all 30 runs, so every difference
below is the residual codebook and nothing else.

RTX 5090, 100K residual training sample, M/nlist as in the frozen config.
Raw: `v31b.log` (recall), `v31cb.log` (codebooks), `v31c.log` (iterations).

## Recall

| source of variation | vogue-768 | bge-m3 | stella |
|---|---|---|---|
| build noise (seed 0, 3 repeats) | 0.9769-0.9771 | 0.9516-0.9519 | 0.9850-0.9854 |
| | **2e-4** | **3e-4** | **4e-4** |
| **sampling: ALL vs 100K** (v30) | **+2e-4** | **+8e-4** | **-3e-4** |
| initialisation (5 random seeds) | 0.9747-0.9761 | 0.9513-0.9528 | 0.9849-0.9865 |
| | **1.4e-3** | **1.5e-3** | **1.6e-3** |
| iterations (5 -> 800) | 0.9741-0.9833 | 0.9502-0.9559 | 0.9843-0.9906 |
| | **9.2e-3** | **5.7e-3** | **6.3e-3** |

The sampling decision is the smallest effect measured. It sits at or below
build noise, 2-5x under initialisation and 7-20x under iteration count. **100K
is inside Lloyd's own arbitrariness**, which is the claim the protocol needed.

Random initialisation is worse than the quantile placement on vogue (all five
seeds below it) and a wash on bge and stella. Quantile stays the default.

## Codebooks, which recall cannot see

Same runs, `mean |dc|` over the M*Kr codewords and the share of residual codes
that change on a fixed 512-vector sample re-encoded from each codebook:

| variant | mean abs dc | codes moved (vogue / bge / stella) |
|---|---|---|
| build noise (seed 0 repeat) | ~1e-6 | 0.38% / 0.70% / 0.24% |
| iter 25 -> 5 | ~3.3e-4 | 11.0% / 11.8% / 11.3% |
| iter 25 -> 100 | ~6.8e-4 | 24.5% / 25.3% / 24.2% |
| random seed | ~1.4e-3 | **91.2% / 91.4% / 91.0%** |

A different initialisation rewrites **91% of the residual codes** and moves
recall by 1.5e-3. This is the same insensitivity the histogram estimator showed
(96.84% of codes changed for 2e-4 of recall): **recall cannot be used to
validate a codebook implementation.** The isolation check plus the code-change
rate is what does that.

## 25 iterations is not converged, and converging is free

`v31c.log`, seed 0, `train` is the whole residual-codebook phase:

| iters | vogue recall | train | bge recall | train | stella recall | train |
|---|---|---|---|---|---|---|
| 5   | 0.9741 | 117.1 ms | 0.9502 | 230.5 ms | 0.9843 | 301.0 ms |
| 10  | 0.9749 | 114.5 ms | 0.9507 | 198.3 ms | 0.9858 | 258.7 ms |
| 25  | 0.9770 | 125.9 ms | 0.9516 | 200.5 ms | 0.9854 | 267.7 ms |
| 50  | 0.9789 | 104.8 ms | 0.9541 | 204.3 ms | 0.9864 | 267.0 ms |
| 100 | 0.9803 | 112.7 ms | 0.9543 | 221.2 ms | 0.9879 | 266.0 ms |
| 200 | 0.9818 | 112.6 ms | 0.9560 | 206.5 ms | 0.9892 | 264.7 ms |
| 400 | 0.9819 | 115.5 ms | 0.9555 | 207.1 ms | 0.9901 | 285.7 ms |
| 800 | 0.9833 | 120.6 ms | 0.9559 | 213.7 ms | 0.9906 | 271.4 ms |

Train time has no trend across a 160x change in iteration count -- 105-126,
198-231, 259-301 ms -- because the sort dominates and the iterations do not.
The sort is O(n log n) once, 800k*20 = 16M operations; an iteration is
O(Kr log n), 256*20 = 5120, so even 800 of them are a quarter of the sort.

Recall climbs the whole way: 25 -> 200 is **+4.8e-3, +4.4e-3, +3.8e-3** at no
measurable cost. That is an order of magnitude more than anything else varied
here, and it is free.

It is also the more faithful setting. A Lloyd-Max quantizer is *defined* as the
fixed point of the Lloyd conditions, so stopping at 25 iterations is a
deviation from the paper that raising the count repairs. Cutting the sweep off
at 800 is arbitrary in the same way; a convergence test is what belongs here.
