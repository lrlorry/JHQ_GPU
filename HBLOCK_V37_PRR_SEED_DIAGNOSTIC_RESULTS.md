# HBlock v37_prr: Exact-Seed Threshold Diagnostic — Results

Diagnostic spec: `HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md`
Raw output: `results/hblock_v37_prr_seed_diag_20260722_103525.{csv,txt}`
Dataset: Vogue-768 (932,328 × 768, float), nq=1000, k=10, VECTOR_LEVEL epsilon,
one index built once (no rebuild between ef/seed_per_block configurations).

## Decision: NO-GO

Deterministic PRR (candidate-set-exact, via tightened seed-derived thresholds)
does not work on the current Br=4 fine code for Vogue. Confirmed on two
independent grounds; either alone is sufficient to stop.

## 1. Exact-work ratio fails even weak-go, at every configuration tested

| ef  | spb | total_exact/query | baseline/query | ratio |
|-----|-----|-------------------|-----------------|-------|
| 32  | 8   | 2071.8             | 511.7           | 4.05× |
| 64  | 8   | 3861.3             | 1023.4          | 3.77× |
| 128 | 8   | 7174.0             | 2046.5          | 3.51× |
| 256 | 8   | 13473.2            | 4092.1          | 3.29× |

Spec's no-go criterion is `mean total_exact_required >= baseline_exact`
(ratio ≥ 1). Every (ef, seed_per_block) cell in the full sweep — ef ∈
{32,64,128,256}, seed_per_block ∈ {1,2,4,8} — lands between 3.18× and 4.23×.
The best cell (ef=256, spb=8: ratio 3.29×) is still ~6.6× worse than the
strong-go bar (≤0.5×) and ~3.3× worse than weak-go (<1×).

## 2. Seeds barely move survivor counts — the bottleneck is the L2 lower bound, not the threshold

| ef  | spb=1 surv/blk | spb=8 surv/blk | tau_seed/tau_U |
|-----|----------------|----------------|----------------|
| 32  | 67.3           | 62.2           | 0.73–0.78      |
| 256 | 51.8           | 49.0           | 0.72–0.75      |

Going from 1 seed/block to 8 seeds/block tightens the threshold
(`tau_seed2`) by 25–28% relative to the production `tau_U`, but survivors
per block drop by only 5–8%. p90 survivors/block is **128 — the entire
block** — at every single configuration. Adding more seeds cannot fix this:
the L2 lower bound itself (`max(0, sqrt(pq_dist) - eps)^2`) is too loose,
because per-vector reconstruction error under Br=4 (mean ε ≈ 0.31, p50 ≈
0.26) is comparable in magnitude to the residual norm itself (σ_r3 ≈
0.0236/dim → ‖r3‖ ≈ 0.65 over d=768). No threshold refinement narrows an
interval whose width is set by the code's own quantization error. This
matches interpretation note #3 in the spec.

## 3. Candidate-set correctness anomaly (separate issue, doesn't change the verdict)

Phase-4 validation agreement was 33–57%, far below the required 100%.
Mathematically `tau_seed2` (k-th smallest *exact* distance among ≥k seeds)
is a provably valid upper bound on the true candidate-set k-th distance —
agreement should be 100% modulo ties. This gap indicates a bug in the
diagnostic-only code (most likely seed/candidate ID mapping or a mismatch
between the interval computation and the phase-4 exact-scan reference set),
not a flaw in the PRR math. It does not affect the shipped `CERTIFIED_PRR`
search mode, which uses a different mechanism (direct L2-sorted top-16, no
seed-derived threshold) and was reconfirmed correct by the same run's
control block:

```
[control] CERTIFIED_PRR  ef=256  recall@10=0.9965  qps=54516
```

— consistent with the FIXED_PER_BLOCK (0.9967) and CORRECTED_FIXED (0.9969)
controls run on the identical index, no rebuild. Since criterion #1 alone
already forces NO-GO, this bug was not chased further; it is noted here for
anyone revisiting seed-based thresholds later.

## Interpretation (per spec's required distinctions)

1. **The current `tau_U` policy is too loose** — confirmed; seeds tighten it
   by ~25%, but that is not enough.
2. **Exact seeds do not provide a sufficiently tighter safe upper
   threshold** on Br=4 Vogue codes — confirmed at every tested config.
3. **The L2 lower bound remains too loose even after `tau_seed2`
   improves** — confirmed; this is the actual bottleneck (see §2).
4. This diagnostic did not reach the survivor-count gate, so no two-wave
   GPU search was implemented or benchmarked.

## Root cause

Structural, not a tuning problem: 4-bit scalar PQ (`Br=4`, `Kr=16`)
reconstructs each dimension with error comparable to the residual signal
itself. Any interval built from that reconstruction error — however the
upper threshold is chosen — stays wide enough to keep the majority of a
128-vector block "possibly in top-k." This is consistent with the earlier
v37_prr three-mode result (`results/hblock_v37_prr_vogue.csv`): the
production `CERTIFIED_PRR` mode already showed 59–90 survivors/block under
the same eps regime.

## Recommended next action

Stop deterministic PRR work on Br=4. Do not attempt further seed-policy or
threshold tuning on this code — per spec, that would be moving the
goalposts. Two options going forward, both out of scope for this
diagnostic:

- **Br=8** (`Kr=256`) as a separate, explicitly scoped experiment — doubles
  fine-code storage and LUT-related work; expected to shrink reconstruction
  error substantially, may bring survivor counts down enough to revisit.
  Not started here per the spec's explicit exclusion.
- Return to the original v37 goal (SPACEV-100M, int8, full pipeline
  validation) — PRR's real payoff is at 1B scale (pruning a block saves an
  H2D transfer, not just an exact-distance computation), so this line of
  work is better revisited once the 1B batch-loading design
  (`HBLOCK_1B_DESIGN.md`) is further along.
