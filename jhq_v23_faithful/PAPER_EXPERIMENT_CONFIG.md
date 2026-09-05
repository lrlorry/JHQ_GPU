# The frozen paper experiment

`paper-faithful-v1` fixes the algorithm. This fixes the experiment run on it:
the training protocol, the parameters, and what a result file must record.

Two tags, two things:

```
paper-faithful-v1     validated search semantics
paper-experiment-v1   those semantics + the final experimental protocol
```

`paper-faithful-v1` is not moved or overwritten.

## Base and commit

```
search semantics    paper-faithful-v1  (e038ceafff1cf05278abc643fd41c8ff0016eb9c)
experiment tag      paper-experiment-v1
branch              fix/recall-eval-v15
executable          demo_jhq_v23_faithful_exact
```

Resolve the experiment commit with `git rev-parse paper-experiment-v1^{}`.

Everything between the two tags is training-set selection and instrumentation:
`search.cu` is **byte-identical** across them, and no search symbol appears in
the diff.

## Final residual-training protocol

**100,000 vectors**, deterministic stride over the base set.

Chosen because all of Y — which is what Eq. 5 defines — does not run on this
hardware at the scales that matter. Measured: it fails within three seconds
having allocated nothing, at both 10.1M and 17.8M, and 4M already fails. The
largest workable size is 1M on both.

The reason is that this port materialises the rotated set: `rotate_on_gpu`
allocates N·d·4 twice and the residual trainer needs it resident, which is
38.5 GB at bge-m3 and 67.8 GB at stella against a 31.4 GB card.

**This is a limitation of this port and this hardware, not of JHQ.** The
paper's own platform has 576 GB of host memory and room for both. Equation 5
needs O(M·K_r) of state plus a histogram, not O(N·d); a streaming residual
collector would remove the limit and is a recorded follow-up. Nothing here
licenses the claim that Eq. 5 is impractical at scale.

Evidence that 100K costs little: `results/sensitivity/residual_training/`.
Recall@10 differs from ALL by at most 0.0018 across two datasets, two Br values
and three nprobe; 250K and 500K are frequently worse than 100K. At the two
large datasets, 100K against 1M moves Recall@10 by 0.0001 and 0.0002.

Carried forward as **P1**: the deviation from Eq. 5's "all y in Y" is real,
documented, and bounded by that measurement.

## Parameters

| | |
|---|---|
| B | 8 |
| Br | 8 and 4 |
| M | smallest admissible per dataset (below) |
| Ds | d / M, always 8 here |
| alpha | 100.0 |
| k | 10 |
| nprobe sweep | 8, 32, 64, 128, 256, 512 |
| batch size | 1024 |
| ivf_iters | 8 |
| kmeans_iters | 5 |
| residual training | 100,000, deterministic stride |

### Admissibility

```
Ds = d / M      d % M == 0      B % Ds == 0
```

At B=8: Ds ∈ {1,2,4,8}, so M ∈ {d, d/2, d/4, d/8}. **d/8 is the smallest
admissible M and `M >= d/8` is not sufficient** — at d=768, M=128 gives Ds=6,
above d/8 and inadmissible. `check_eq4_admissible()` refuses and names the
admissible M.

### Per dataset

| dataset | N | d | M | Ds | nlist |
|---|---|---|---|---|---|
| vogue-768 | 932,328 | 768 | 96 | 8 | 1024 |
| arxiv-768 | 2,253,000 | 768 | 96 | 8 | 2048 |
| openai3-1536 | 999,000 | 1536 | 192 | 8 | 1024 |
| openai3-3072 | 999,000 | 3072 | 384 | 8 | 1024 |
| bge-m3 | 10,091,524 | 1024 | 128 | 8 | 8192 |
| stella-trec24 | 17,776,615 | 1024 | 128 | 8 | 16384 |

## Numerical mode

| | |
|---|---|
| primary lookup table | fp32, compiled in; `JHQ_LUT32` refused at runtime |
| accumulation | fp32 |
| TF32 | off; `JHQ_TF32` not read |
| primary distance | complete over all M subspaces |
| top-alpha*k | exact global selection |
| final top-k ties | ascending database id |
| residual refinement | fused; validated equal to the materialised path to 1.192e-07 |

## Required CSV provenance

`commit`, `dirty`, `gpu`, `driver`, `nvcc`, `cuvs` for baseline rows, `utc`,
`host_cores`, and **every `JHQ_*` variable the child process saw**.

The last is not optional. The pre-freeze OpenAI3 rows recorded four variables
and could not say which primary codebook was in force — they were taken at
M=96, where Eq. 4 cannot be built, and nothing in the file said so.

## What must not be reported as JHQ-GPU

Results from v22, v24, `demo_jhq_v23_cascade`, any `demo_jhq_v22_s*b*` target,
or the pre-freeze rows in `results/final/p0_*.csv`. They differ in ways that
were measured, not supposed.
