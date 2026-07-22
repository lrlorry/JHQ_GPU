# HBlock v37_prr: Exact-Seed Threshold Diagnostic — Results

Diagnostic spec: `HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md`
Raw output: `results/hblock_v37_prr_seed_diag_20260722_110423.{csv,txt}`
Dataset: Vogue-768 (932,328 × 768, float), nq=1000, k=10, VECTOR_LEVEL epsilon,
one index built once (no rebuild between ef/seed_per_block configurations).

> **Superseded run.** An earlier run (`..._20260722_103525`) reported
> NO-GO with `candidate_set_top10_agreement` of only 33–57%. That run was
> invalid: the CPU-side diagnostic stats indexed the leaf-block-sorted pair
> arrays (`d_pair_leaf_b`/`d_pair_qid_b`, reordered by
> `gpu_build_and_sort_pairs_v29`) using offsets computed for the pre-sort,
> query-major layout — it was reading essentially arbitrary other queries'
> candidates. Fixed in `hblock_v37_prr/jhq_gpu_index.cu` by routing every
> access through the `d_prr_perm_` qid-major permutation (the same one the
> production `tau_U`/`tau_seed2` kernels already used correctly). This
> document reflects the corrected rerun, which reaches
> `candidate_set_top10_agreement = 100%` at every configuration.

## Decision: NO-GO

Deterministic PRR (candidate-set-exact, via seed-derived thresholds) does
not clear the spec's bar on the current Br=4 fine code for Vogue —
confirmed with valid data this time (100% candidate-set agreement
throughout). The margin is narrower than the retracted run suggested, and
the failure mode is more specific than "hopeless": roughly half of queries
already need less exact work than the fixed top-16 baseline; a persistent
tail of unprunable blocks drags the mean above the no-go line.

## 1. Exact-work ratio: best case still ~2.56×, never below baseline

| ef  | spb | exact_ratio_mean | exact_ratio_p50 | exact_ratio_p90 | queries_better_than_baseline |
|-----|-----|-------------------|-----------------|-------------------|-------------------------------|
| 128 | 1   | 2.930              | 1.408           | 7.709             | 43.2% |
| 128 | 2   | 2.803              | 1.177           | 7.707             | 46.6% |
| 128 | 4   | 2.817              | 1.176           | 7.706             | 46.0% |
| 128 | 8   | 2.939              | 1.316           | 7.706             | 41.8% |
| 256 | 1   | 2.635              | 0.956           | 7.616             | 50.9% |
| 256 | 2   | **2.563**          | **0.815**       | 7.607             | **53.8%** |
| 256 | 4   | 2.604              | 0.854           | 7.606             | 53.7% |
| 256 | 8   | 2.745              | 1.033           | 7.607             | 48.6% |

Spec's no-go criterion is `mean total_exact_required >= baseline_exact`
(ratio ≥ 1). The best cell in the full sweep (ef=256, spb=2: ratio 2.563×)
is still ~5.1× worse than strong-go (≤0.5×) and ~2.6× worse than weak-go
(<1×). Every one of the 16 (ef, spb) cells has `exact_ratio_mean` above 2.5.

**spb=2 is the practical sweet spot, not spb=8.** Increasing seeds per
block from 2 to 8 makes the mean ratio *worse* (2.563→2.745 at ef=256):
each extra seed adds directly to the exact-work numerator (seeds themselves
get exact-reranked), and beyond ~2 seeds the marginal pruning gain from a
tighter `tau_seed2` no longer pays for that added cost.

## 2. The median query already benefits — a tail of blocks does not

At ef=256, spb=2: `exact_ratio_p50 = 0.815` — the median query needs *less*
exact work than the fixed baseline, and 53.8% of queries do better than
baseline overall (`queries_better_than_baseline_pct`). But
`exact_ratio_p90 = 7.607` — a right tail of queries costs ~7.6× baseline,
which is what pulls the mean above 1.

The cause is visible in `p90_survivors_block = 128` (the *entire* block) at
**every single (ef, spb) configuration tested**, unmoved by seed count.
This is not diffuse looseness across all blocks; a persistent subset of
blocks (likely dense clusters where all member vectors sit at similar
distance from the query, so PQ/exact distances barely separate them) simply
cannot be pruned by any seed-derived threshold, however many seeds are
spent. Median-block behavior is much better: `p50_survivors_block` drops to
**10** (below the baseline's 16) at ef=256, spb∈{2,4,8} — for a typical
block, the interval mechanism works as intended.

`tau_seed/tau_U ≈ 0.72–0.78` throughout (seeds tighten the production
threshold by ~22–28%), consistent with the retracted run's number — this
part of the earlier analysis was unaffected by the indexing bug, since
`tau_U`/`tau_seed2` are computed entirely on the GPU via the (correct)
`d_prr_perm_` permutation.

## 3. Candidate-set correctness: 100% (bug fixed)

`candidate_set_top10_agreement = 100.0` at all 16 (ef, spb) cells,
confirming `tau_seed2`'s safety proof holds in practice once the CPU
aggregation bug was fixed. `queries_with_insufficient_seeds = 0` throughout
— every query had ≥k valid seeds at every ef tested.

Controls on the same index, no rebuild:
```
FIXED_PER_BLOCK  ef=128  recall@10=0.9903   ef=256  recall@10=0.9967
CORRECTED_FIXED  ef=128  recall@10=0.9906   ef=256  recall@10=0.9969
CERTIFIED_PRR    ef=128  recall@10=0.9891   ef=256  recall@10=0.9965
```

## Interpretation (per spec's required distinctions)

1. **The current `tau_U` policy is too loose** — confirmed; seeds tighten
   it by ~22–28%, still not enough on its own.
2. **Exact seeds do not provide a sufficiently tighter safe upper
   threshold** to clear weak-go at any tested config — confirmed, though
   the median query is much closer to viable than the mean suggests.
3. **The L2 lower bound remains too loose for a specific tail of blocks**
   even after `tau_seed2` improves — confirmed; `p90_survivors_block=128`
   at every config identifies this as a persistent minority of "unprunable"
   blocks, not uniform looseness.
4. Candidate-set correctness (the actual survivor-count gate) is satisfied
   (100% agreement) — the failure is purely on exact-work volume, not
   safety.

## Root cause

Structural, consistent with the per-vector reconstruction-error measurement
from the three-mode experiment (`results/hblock_v37_prr_vogue.csv`): 4-bit
scalar PQ (`Br=4`, `Kr=16`) reconstruction error (mean ε≈0.31, p50≈0.26) is
comparable in magnitude to the residual norm itself (‖r3‖≈0.65 over
d=768). For most blocks this still leaves enough separation for a
seed-tightened threshold to prune productively (median block: 10
survivors, below the 16-candidate baseline). But for a consistent ~10% tail
of blocks the bound stays uninformative regardless of seed count — and
because a single un-prunable block costs a full 128-candidate exact pass,
this tail dominates the per-query mean.

## Recommended next action

Stop deterministic PRR work on Br=4 in its current uniform form — mean
exact-work never clears even weak-go. Do not attempt further seed-count
tuning; §1 shows that knob is already past its optimum (spb=2). Two
directions are out of scope for this diagnostic and are not started here:

- **Query-level triage**: since the median query already beats baseline
  and the failure is concentrated in a tail of blocks (not diffuse), a
  policy that runs PRR only when the visited block set looks "easy" (e.g.
  cheap early signal correlated with `p90_survivors_block=128` blocks) and
  falls back to fixed top-16 otherwise might clear weak-go on average —
  untested, would need its own diagnostic.
- **Br=8** (`Kr=256`) as a separate, explicitly scoped experiment — doubles
  fine-code storage and LUT-related work; expected to shrink reconstruction
  error enough to also prune the current tail. Not started here per the
  spec's explicit exclusion.
- Return to the original v37 goal (SPACEV-100M, int8, full pipeline
  validation) — PRR's real payoff is at 1B scale (pruning a block saves an
  H2D transfer, not just an exact-distance computation), so this line of
  work is better revisited once the 1B batch-loading design
  (`HBLOCK_1B_DESIGN.md`) is further along.
