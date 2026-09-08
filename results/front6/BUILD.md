# Build cost, against the baselines

`results/front6/front6.log`, cold caches, so JHQ's `train` and `add` are real
builds -- the first row of each group only; the three after it read the cache
the first one wrote. Baselines from `results/pre_freeze_v22_s2b1/`, commit
`305e763`.

| dataset | JHQ published cfg | JHQ this work | CAGRA int8 | CAGRA fp32 | IVF-PQ |
|---|---|---|---|---|---|
| vogue-768 | 0.43 s | **0.58 s** | 4.93 s **8.5x** | 5.95 s 10.3x | 1.94 s 3.3x |
| arxiv-768 | 0.66 s | **0.99 s** | 7.95 s **8.0x** | 9.32 s 9.4x | 4.15 s 4.2x |
| openai3-1536 | 0.69 s | **1.25 s** | 6.65 s **5.3x** | 7.70 s 6.2x | 3.32 s 2.7x |
| openai3-3072 | 1.42 s | **1.66 s** | 12.19 s **7.3x** | 14.19 s 8.5x | 6.11 s 3.7x |
| bge-m3 | 2.08 s | **4.73 s** | 40.51 s **8.6x** | OOM | 28.96 s 6.1x |
| stella | 6.03 s | **11.12 s** | 73.30 s **6.6x** | OOM | OOM |

**JHQ builds 5.3x to 10.3x faster than CAGRA and 2.7x to 6.1x faster than
IVF-PQ, on every dataset.** Stella's int8 graph takes 73 s where JHQ takes 11.

## The retuning is not what makes it cheap

The larger index costs five seconds on the largest dataset: 6.03 s to 11.12 s
on 17.8M vectors, for 2.3x the query throughput at matched recall and VRAM from
20,547 to 20,837 MiB. k-means over `n_train x nlist x d` is a GEMM this card
finishes in seconds, and `add()` -- which encodes all N vectors and does not
change with nlist -- dominates the build in both rows.

A graph has to be searched into existence rather than computed, which is why it
does not amortise the same way. That gap is a property of the method, not of
this tuning: even the published configuration, at 6.03 s on stella, is 12x
faster to build than the int8 graph.

## Two things this table is not

- **cuVS reports its whole build in `train_ms`** and leaves `add_ms` empty, so
  its column is one number where JHQ's splits into training and encoding.
- **VRAM is not compared here.** The two sides measure it differently -- JHQ's
  figure comes from `cudaMemGetInfo` after the build and includes the context
  and the search workspace. `results/front6/BYTES.md` compares memory properly,
  by resident bytes per vector.
