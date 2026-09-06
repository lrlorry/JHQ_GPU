# What actually differs between the published fronts and the re-run

Source-level audit, no GPU. It settles which of the candidate explanations for
vogue's 2.3-3.1x throughput drop and 0.45-2.55 point recall drop are real
differences and which are not.

The old target's definitions come from the commit that introduced it
(`7298f5c`); `demo_jhq_v22_s2b1` is no longer in CMakeLists and its sibling
`jhq_v22_jq` records the shared settings: sources `jhq_v21_cascade/`, and

```
JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=2 JHQ_PREFIX_KEEP=8
JHQ_TILE_C=2 JHQ_TILE_STRIDED=1 JHQ_BITONIC_SELECT=1
```

with **no `JHQ_LUT32`**, so `lut_t` was `__half`.

## The same in both, contrary to what was suspected

**The primary codebook.** Both builds read `JHQ_PAPER_CODEBOOK` as
`return !(e && e[0] == '0')` -- **on unless explicitly disabled** -- and the
harness records every `JHQ_*` the child saw, with no such variable in the old
vogue rows. So both used the analytical Cartesian construction of equation 4.
The suspicion that the old rows ran a different primary quantiser is
**refuted for vogue**, not merely unproven. (It remains open for the openai3
rows, which at M=96 cannot run under it at all.)

**The residual codebook layout.** `JHQ_GLOBAL_RESIDUAL_CB` defaults to 0 in
both: one scalar codebook per subspace, not one global one replicated.

**The IVF.** Neither frontier run sets `JHQ_GPU_CODEBOOK`, so both build the
coarse quantiser on the host, with the same `ivf_iters` and `kmeans_iters`.

**The primary training set.** `n_train` is 100,000 in both, the demo's default.

## The six real differences

| | old `v22_s2b1` | new `v39` |
|---|---|---|
| primary distance | cascade, PM = M/2, KEEP = 8 | exact, PM = M |
| primary table | `__half` | fp32 |
| where that table lives | 40,960 + 49,152 = **90,112 B, in shared** | 16,392 + 98,304 = **114,696 B > 101,376, in global** |
| top-alpha*k selection | per-thread KEEP, then one sort | block-wide threshold compaction |
| residual training set | 100K (old `train()` takes no residual set at all) | **all of Y**, 932,328 |
| residual Lloyd | 25 iterations, host | 2000 iterations, device sorted |

## The cascade was not pruning at low nprobe

`KEEP = 8` and `BLOCK = 1024`, and the kernel keeps `float pd[KEEP]` per
thread, evicting only when `acc[t] < pd[KEEP-1]`. Candidates per thread:

| nprobe | candidates | per thread | evicts at KEEP=8? |
|---|---|---|---|
| **8** | 7,284 | **7.1** | **no** |
| 32 | 29,135 | 28.5 | yes |
| 256 | 233,082 | 227.6 | yes |

At nprobe=8 every candidate fits in a thread's slots, so the old run completed
the full primary distance for all of them. **Its 2.33x throughput advantage
there cannot be the cascade**, because at that point there was no cascade.

## What this leaves

Throughput: three plausible contributors remain -- the extra completed
distances at higher nprobe (a factor between 1 and 2, not 2, since
`W_new/W_old = 2*N_C/(N_C+S)`), the table moving from shared to global, and
the block-wide compaction, which `results/v42_scan_split/` already measures at
**44% of vogue's search time** against 9-10% on the large sets.

Recall: the gap narrows from 2.55 points at nprobe=8 to 0.45 at nprobe=256,
which is the shape of something that matters most when coverage is thin. Note
that exact selection is exact *by approximate distance*: a true neighbour that
the primary code ranks 1001st is excluded by an exact top-1000 and may survive
a lossy one. Nothing here decides it; the staged coverage measurement does.
