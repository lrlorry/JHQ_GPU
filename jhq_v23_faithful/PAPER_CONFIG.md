# The paper-faithful JHQ-GPU configuration

The one build the paper calls **JHQ-GPU**. Everything the fidelity tests
verified is compiled into it, not selected by a default, and the environment
variables that could undo one of those properties are refused rather than
ignored. A result produced by any other target is a result for that target.

## A. Commit

```
tag     paper-faithful-v1
branch  fix/recall-eval-v15
```

Resolve it with `git rev-parse paper-faithful-v1`. The tag is the identifier
rather than a SHA written into the file: a SHA committed here can only ever be
the SHA of the parent commit, since writing it changes the commit it names.

The fidelity work this rests on is `da2d9d7` and its ancestors. This file names
its own commit rather than pointing at whatever is checked out.

## B. Target

```
library     jhq_v23_faithful_exact
executable  demo_jhq_v23_faithful_exact
sources     jhq_v23_faithful/
```

The cascade lives at `demo_jhq_v23_cascade`. It is an approximate variant and
its numbers are not JHQ-GPU numbers.

## C. Build

```sh
cmake . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --target demo_jhq_v23_faithful_exact -j 16
```

Compiled in by the target definition:
`JHQ_PAPER_FAITHFUL=1 JHQ_LUT32=1 JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=1
JHQ_K_LOCAL=4 JHQ_PREFIX_KEEP=4 JHQ_TILE_C=2 JHQ_TILE_STRIDED=1
JHQ_BITONIC_SELECT=1`

## D. Fields every result CSV must carry

`commit`, `dirty`, `gpu`, `driver`, `nvcc`, `cuvs` (for baseline rows), `utc`,
`host_cores`, and **every `JHQ_*` variable the child process saw** — the last
one because a row that cannot say what produced it cannot be re-run, which is
how the OpenAI3 rows at M=96 came to be reported without anyone noticing they
were not running Equation 4.

## E. Algorithm configuration

| Parameter | Value |
|---|---|
| B | 8 (primary bits per subspace; the code byte) |
| Br | 8 or 4 (residual bits per dimension) |
| M | must satisfy the admissibility condition in G |
| Ds | d / M |
| alpha | 100.0 |
| k | 10 |
| nlist | 1024 (vogue, openai3), 2048 (arxiv), 8192 (bge), 16384 (stella) |
| nprobe sweep | 8, 32, 64, 128, 256, 512 |
| batch size | 1024 |
| ivf_iters | 8 |
| kmeans_iters | 5 |

Per-dataset M, chosen as the smallest admissible value:

| dataset | d | M | Ds |
|---|---|---|---|
| vogue-768 | 768 | 96 | 8 |
| arxiv-768 | 768 | 96 | 8 |
| openai3-1536 | 1536 | 192 | 8 |
| openai3-3072 | 3072 | 384 | 8 |
| bge-m3 | 1024 | 128 | 8 |
| stella-trec24 | 1024 | 128 | 8 |

## F. Numerical configuration

| Property | Setting | How |
|---|---|---|
| primary lookup table | fp32 | `JHQ_LUT32=1` compiled in; `JHQ_LUT32` is refused at runtime |
| accumulation | fp32 | — |
| TF32 rotation | off | `JHQ_TF32` not read |
| primary distance | complete over all M subspaces | the exact scan takes no prefix arguments |
| top-alpha*k | exact global selection | `exact_topck()` returns true unconditionally |
| final top-k ties | by ascending database id | in both the per-thread scan and the block reduction |
| residual refinement | fused | permitted: differentially validated against the materialised path, candidate sets identical and composite distances within 1.192e-07 |

## G. Admissibility

```
Ds = d / M
d % M == 0
B % Ds == 0
```

At B = 8 the admissible Ds are **{1, 2, 4, 8}**, so M ∈ {d, d/2, d/4, d/8}
among the M that divide d. `check_eq4_admissible()` throws otherwise, naming
the admissible M for that d.

**d/8 is the smallest admissible M, and `M >= d/8` is necessary but not
sufficient.** At d = 768, M = 128 gives Ds = 6: larger than d/8 = 96, and
inadmissible because 8 % 6 != 0.

## H. Training protocol

| Item | Value |
|---|---|
| residual codebook training size | **100,000 vectors** |
| sampling | deterministic — a fixed stride over the base set, seeded by the parameter set |
| 1-D k-means iterations | 25 |
| primary codebook | analytical, Equation 4; no training data beyond sigma |
| sigma | `sqrt(sum_sq / (n_train * d))`, Lemma 2's E[||x||^2]/d |

Equation 5 describes R^m as collected over **all** y in Y. Training on 100,000
is a deviation, recorded in I, and a sensitivity experiment against a larger
sample is planned separately. It is not a correctness question and does not
reopen the correctness phase.

## I. Remaining P1

| Item | Why it is P1 |
|---|---|
| residual codebook trained on 100K, Eq. 5 says all of Y | training protocol; changes the codebook, not the algorithm |
| composite score omits `-||q_JL||^2` | constant per query, so ranking and top-k are identical; the returned values are the metric plus that constant and are not distances. Only matters if an experiment consumes absolute values — none currently does |
| fp16 primary table, TF32 rotation | available under other targets for ablation; not in this one |
| `B % Ds != 0` undefined by the paper | PAPER AMBIGUITY, resolved by refusing to run |

The Equation 4 scale question is **not** on this list: all six datasets measure
unit-L2-norm to within 3e-7, so `sigma*sqrt(2)` and `sqrt(2/d)` agree to
between 7.6e-10 and 2.8e-08 relative. Resolved for this protocol.

## Regression at the freeze

Run against this binary, with the existing checkers unextended.

```
build_rc = 0

guard, each variable launched against the frozen target:
  JHQ_PFX_DEN=4         -> refused
  JHQ_EXACT_TOPCK=0     -> refused
  JHQ_TF32=1            -> refused
  JHQ_LUT32=0           -> refused
  JHQ_PAPER_CODEBOOK=0  -> refused

admissibility, d=768 M=128 (Ds=6, above d/8 and inadmissible):
  refused, naming admissible M: 768, 384, 192, 96

primary code mismatch     = 0   (exact ties: 0)
residual code mismatch    = 0   (exact ties: 0)
top-alpha*k mismatch      = 0   at nprobe 32, 128, 256
final top-k mismatch      = 0   at nprobe 32, 128, 256
primary distance err      = 0.000e+00
composite distance err    = 0.000e+00
```

The guard was exercised rather than assumed: a guard that compiles without
firing is worth nothing.

## J. What must not be reported as JHQ-GPU

**Results from v22, v24, `demo_jhq_v23_cascade`, or any `demo_jhq_v22_s*b*`
target must not be used as faithful JHQ-GPU results.** They differ in ways that
were measured: the cascade prunes on a partial primary distance, and the
pre-freeze targets used the per-thread retention buffer, whose selected
candidate set differs from the exact one.

The existing `results/final/p0_*.csv` rows were produced before this freeze and
are not JHQ-GPU results in this sense.
