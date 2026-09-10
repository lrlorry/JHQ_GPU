# ADC 2026: mother skeleton v3 for Sections 3, 4, and 6

This is the new drafting framework, not a replacement for any earlier skeleton or review. Repository branch: `fix/recall-eval-v15`. Evidence audit: 2026-09-10. Paths below are relative to the repository root; `data/` in this document means `paper_adc2026/data/`. Version identifiers appear only in internal evidence mapping, never in manuscript prose. Existing reviews are advice; frozen results and implementation semantics take precedence. `[TBD]` means the claim or required validation is not ready for publication.


## Superseded measurements — read before drafting

Three numbers that appear in earlier skeletons and reviews are older than the
evidence. They were true of the runs they came from and are not true of the
paper's current data. `README.md` is the evidence map and is current.

| older claim | what is measured now | source |
|---|---|---|
| "sample size around `S=32`" / "S=32 is the knee" | **The knee is per workload.** 2000 resampled draws scored on held-out queries: at S=32, 76% inside 1e-3 on openai3-3072 and **27% on vogue-768**, where the mean held-out loss is 0.0033 and the 95th percentile 0.0110. openai3-3072 reaches 98% at S=64; vogue-768 still loses 11% of draws at S=128. Give S against a stated risk, not as a constant. | `data/alpha_resample.json`, `figures/alpha_resample.py` |
| "factorised LUT: roughly +6% to +14.5%" | **−4% at M=96 to +53% at M=384.** All 28 original cells were M=96 and M=128. The 256-entry table is 96 KiB at M=96, which shared memory holds, and 384 KiB at M=384, which it does not: the factorisation matters exactly where the full table stops fitting. One M=96 cell is negative. | `data/lut_groups.log`, `figures/fig_lutgroups.py` |
| any statement that the coarse quantiser is undertrained | **It is not, in the reported runs.** `_pa.sh` passes `JHQ_N_TRAIN = 39 × nlist` on every dataset. The undertraining finding is historical. | `SKELETON_REVIEW_v2.md` |

Two results the skeleton could not have known to ask for, both now measured:

- **Why halves and not quarters.** G=2 beats G=1, G=4 and G=8 in all eight
  configurations, and G=4 and G=8 have *smaller* tables yet lose in proportion
  to their loads a candidate a subspace. The scan is issue-bound — the same
  conclusion as the table-free distance and the packed load, from a third
  direction. (§6.4.2, `fig_lutgroups`.)
- **The two payoffs multiply.** The factorisation is worth about the same with
  and without the packed layout, so "pays twice" holds as an independent
  product rather than an overlap — which the framing had been assuming without
  evidence. (`fig_lutgroups(b)`.)

## Paper contract and contribution hierarchy

Keep the teacher-required order: **1 Introduction; 2 Preliminaries; 3 Method Part 1; 4 Method Part 2; 5 Related Work; 6 Experiments; 7 Conclusion.** Detailed literature discussion stays in Section 5.

Paper story:

> We exploit JHQ's Cartesian primary representation to redesign its primary-distance path for GPU execution, and use output-stability calibration to decide how much residual refinement a workload actually needs.

Section 3 asks what JHQ-specific structure is worth exploiting on GPU. Section 4 asks how much refinement to use and when calibration pays back. Section 6 asks whether matched, reproducible experiments support those answers. This is representation, execution, policy, and evidence—not a chronology of CUDA changes.

| Component | Scientific status | Required treatment |
|---|---|---|
| Cartesian-factorized primary LUT | Strongest JHQ-specific contribution | Derive the exact identity and measure its execution benefit. |
| Packed primary-code loading | Representation-enabled optimization | Explain admissibility and instruction reduction; secondary to the algebraic contribution. |
| Subspace-major physical layout | Necessary, standard GPU engineering | Explain coalescing without claiming the transpose as novelty. |
| Batched GEMM / orthogonal transform | Enabling infrastructure | Preserve mathematical semantics. |
| Query-sized launch, cursor traversal removal, bug fixes | Implementation corrections | Separate from research contributions and from algebraic ablations. |
| Output-stability alpha selection | Policy/algorithm contribution | Validate selection and economics separately. |

Novelty and effect size are different axes: packing may yield a larger measured speedup than factorization. Report both honestly rather than changing their novelty labels to follow performance. Do not assert that no other quantizer can factorize; the exact factorization follows from this Cartesian representation and is not available to the compared baselines in their current representations. Baseline-specific representation citations remain `[TBD]` for the literature pass.

## 3 GPU-Native JHQ

### 3.1 Design Overview

Introduce the two-level representation and the GPU query path in `fig_pipeline`. A primary scan scores routed candidates, retains primary survivors, and refines only those survivors using residual information before final top-k selection. The representation determines which distance computations can be reorganized exactly; the refinement budget determines how much approximate filtering occurs.

Use consistent notation: database vector x, transformed vector y = Qx, transformed query q' = Qq, Q in R^(d x d), M subspaces, primary code c_m, requested output size k. The JL/orthogonal transform is square, not dimensionality reduction. For row batches, Y = X Q^T is the same convention.

### 3.2 GPU-Oriented Representation and Layout

Explain batched transformation, primary and residual encoding, IVF organization, and physical layout in a connected account. The residual is **r = y - y_hat_primary**, never the IVF-centroid residual. IVF organizes candidate routing; it does not redefine the residual target. Query ADC uses tables built from the full query, not a single quantized database-style query code.

Describe the logical candidate-major primary matrix [N,M] and the subspace-major scan layout [M,N], followed by the packed physical variant. Keep logical layout and packed storage distinct. Batched GEMM and the transpose are implementation infrastructure, not independent research claims.

Training dependencies in `fig_pipeline` must match code: transformation and primary construction precede IVF centroid training and residual codebook training; the residual codebook uses a sample and need not wait for all database assignments. Avoid hard-coded equation numbers or a schematic that implies a false dependency.

### 3.3 Structure-Aware Query Processing

#### 3.3.1 Cartesian-Factorized Primary Distance Table

For each subspace, T_m[c] = ||q'_m - C_m[c]||^2. In the current Cartesian 8-bit representation, split the coordinates and code into independent high and low halves. Squared Euclidean distance adds over disjoint coordinates, giving exactly:

    T_m[c] = T_m_hi[c >> 4] + T_m_lo[c mod 16].

Thus 256 entries become 16 + 16 = 32 entries per subspace. These counts are algebraic consequences, not measured speedups. Explain the chain from Cartesian structure to exact factorization to smaller query-specific tables and less construction work, and then evaluate GPU throughput. Extra lookups and arithmetic mean table-size reduction is not a proportional throughput promise.

`fig_lut` belongs here. The implementation checks Cartesian separability against the centroids; a generic refined codebook cannot be substituted silently. Algebraic exactness does not imply bitwise-identical floating-point evaluation or deterministic tie order.

#### 3.3.2 Coalesced Primary Scan and Packed Access

Explain contiguous candidate accesses at a fixed subspace, primary distance accumulation, and survivor selection. The current one-byte primary code permits packing four subspaces into one 32-bit word. Present packing as enabled by this representation and as reducing load/address/loop instructions. Do not imply packing necessarily reduces transferred bytes when the byte layout is already coalesced. Do not repeat the incorrect 128-byte-line / 75%-waste explanation.

`fig_layout` separates the coalescing effect of layout from the instruction effect of packing. Query-sized launch and redundant cursor removal belong in implementation notes and a separately labeled correction ablation.

#### 3.3.3 Selective Residual Refinement

Explain residual decoding/scoring of primary survivors and final top-k selection. Distinguish exact evaluation of a chosen approximate representation from exact nearest-neighbor search over original vectors: routing and primary filtering can still exclude true neighbors. Motivate the hierarchy using the primary-only ablation in Section 6.4, without implying that exhaustive routing fixes inadequate primary resolution.

### 3.4 The Remaining Policy Question

The execution path accepts a survivor budget, but does not determine its useful size. End explicitly with **ck = alpha * k** and the question:

> How many primary survivors should actually be refined?

## 4 Adaptive Hierarchical Refinement

### 4.1 Motivation and Workload Dependence

Original JHQ leaves alpha externally specified. Too little refinement can lose neighbors; too much can return the same top-k while spending unnecessary residual work.

Use this bounded empirical statement: **The sufficient refinement budget ranges from 4 to at least 200 at nprobe=128; the upper endpoint is a lower bound because the measured curve is still improving at the largest tested alpha.** Evidence: `data/alpha6.log`, Br=8 rows, not `alpha_ds.log`, which does not test 200. OpenAI-3072 recall is 0.9416 at both alpha=4 and 200; arxiv improves from 0.9760 at 100 to 0.9771 at 200. This is an operational, grid-limited saturation observation, not a precise ratio or a proof of a plateau beyond the grid. No “25x” saturation claim.

### 4.2 Output-Stability Criterion

Let Q_s be S sampled queries and R_max(q) the top-k ID set at a generous alpha_max. Define R_alpha(q) analogously and define **epsilon as a non-negative integer count of disagreeing slots over the whole sample**:

    D(alpha) = sum over q in Q_s of [k - |R_alpha(q) intersect R_max(q)|]
    alpha* = min {alpha in tested grid : D(alpha) <= epsilon}.

This is set overlap per query: returned neighbors in a different order agree. It is neither a fractional tolerance nor epsilon per query nor positional rank disagreement. It assumes valid distinct top-k IDs. Current defaults are S=32 and epsilon=1; these are empirical settings, not a correctness guarantee.

The rule uses no external ground-truth labels and no learned predictor. Ground truth is used only afterward to evaluate the policy. Agreement with alpha_max cannot certify exact-neighbor recall, recover IVF routing misses, or detect improvements beyond alpha_max.

### 4.3 Selection Procedure and Positioning

Algorithm box: choose a sample and descending alpha grid; compute the alpha_max reference; evaluate smaller budgets against D; select an accepted budget; apply it to subsequent workload queries. Report the actual search schedule, sample selection, and stopping conditions alongside `fig_rule`.

Internal implementation evidence: `examples/demo_jhq_alpha_fast.cu` defaults to grid {100,64,32,16,8,4,2}, selects evenly strided queries from the current batch, and defaults to bisection. Its optional linear mode descends and stops at first rejection. The conceptual minimum above is the target; claiming the implementation always finds the global minimum requires the accepted-prefix/monotonicity assumption. Tie behavior and budget-dependent selection require validation, not an unqualified guarantee. Use: **“The stopping behavior is conservative relative to the sampled agreement criterion.”** Do not say it never picks alpha too small.

Concise positioning paragraph for drafting: sampled parameter tuning and adaptive ANN search control already exist, including FAISS ParameterSpace and adaptive early-termination approaches. Our distinction is label-free, predictor-free top-k output stability applied directly to JHQ's externally specified alpha, with validation against the exhaustive sweep it is intended to replace. Do not claim sampling itself is novel or that full-grid validation is already complete. Primary literature references and precise comparisons: `[TBD]`, to be finalized in Section 5.

### 4.4 Calibration Cost and Reuse

Define T_cal as calibration time, T_fixed and T_rule as time per equally sized subsequent batch with fixed and selected alpha, and B as the number of subsequent batches:

    T_adaptive(B) = T_cal + B*T_rule
    T_fixed_total(B) = B*T_fixed
    B* = T_cal / (T_fixed - T_rule), when T_rule < T_fixed.

For integer batches, use the smallest integer meeting the desired nonnegative/strict savings condition. If T_rule >= T_fixed, there is no finite payback. A fractional B* describes amortization, not a fractional measured workload. The comparison is conditional on the observed quality difference being acceptable; a raw gain is not automatically matched-recall gain.

Reuse assumes a stable query distribution, routing configuration, k, and index. Drift and recalibration frequency are unmeasured limitations. Recalibrating every H batches would add T_cal/H per batch under this model; no fixed percentage overhead follows merely from sampling a small fraction of queries. Current calibration timing starts after sample copying and excludes setup outside that timer; distinguish logged calibration time from a future all-inclusive deployment overhead measurement.

## 6 Experiments

### 6.1 Experimental Setup and Protocol

Keep the main text concise but include each protocol principle below, with full grids and provenance in an artifact table/appendix. Missing entries stay `[TBD]` rather than borrowing assumptions from another cohort.

- Hardware and data: the frozen project uses one RTX 5090, nominally 32 GB. Report device-available memory readings as measurements, not specifications. Cover vogue-768, arxiv-768, bge-m3, stella-trec24, openai3-1536, and openai3-3072. Exact N, preprocessing, query counts, metric and GT provenance must be reconciled with benchmark inputs `[TBD]`; dataset display names may round N.
- Recall@10, k=10 throughout the current evaluation. Alpha and k are coupled; generalization across k is unmeasured. Share query IDs, GT, distance convention and timing boundary across systems. Host queries in to host results out is the stated common boundary; record synchronization, warmups and timed repetitions per executable. The rule benchmark uses a warmup and three timed full searches (`examples/demo_jhq_alpha_fast.cu`); this does not establish every baseline's protocol.
- **Recall-QPS curves report steady-state query throughput; calibration cost is reported separately and incorporated through break-even analysis.** `AF_RESULT` times calibration and full-query throughput separately. Do not include diagnostic alpha-sweep QPS in frontiers; its diagnostic readbacks change timing.
- Mandatory CPU JHQ: strongest defensible reproducible setup, source revision, thread count, affinity/pinning, training budget, matched recall and timing. Attempt the original authors' artifact; if the available baseline is `JHQ_repro`, identify it as a reimplementation rather than the authors' executable. Frozen artifact notes document build incompatibilities and unmatched residual training. The historical 32-thread versus 208-thread advantage is a protocol warning, not a finalized GPU speedup. CPU speedup: `[TBD]` until rerun with matched training and a defensible pinned configuration. Explicitly disclose absence if unresolved.
- Main GPU baselines: IVF-RaBitQ, CAGRA and IVF-PQ. Optional references: IVF-Flat, brute force, primary-only/JQ-like hierarchy ablation. Record source/library revisions, build and search grids, nlist, nprobe, PQ code size, RaBitQ mode, CAGRA graph/build/search parameters and precision. State memory/code-budget matching or explicitly acknowledge differing budgets. Final baseline grid/provenance table: `[TBD]`.
- Current JHQ FIX/RULE grid in `data/paper_fronts.log`: nprobe={8,32,128,256,512,1024}; (M,nlist) are vogue (96,4096), arxiv (96,8192), BGE (128,32768), stella (128,32768), OpenAI-1536 (192,8192), OpenAI-3072 (384,4096). These are observed settings, not proof of exhaustive tuning. Retain QUANT4 evidence and its frozen source rather than silently choosing a slower RaBitQ mode.
- Frontier selection follows the nondominated envelope in `figures/style.py`. QPS at a stated recall is log-linearly interpolated between adjacent retained measured points; label resulting ratios interpolated, and never extrapolate beyond either method's measured range. Publish exact source points and parameter configurations. A sweep endpoint is not an architectural recall ceiling.
- Disclose coarse-training history: earlier configurations included roughly six training points per centroid; later work used much denser training, around 39. `results/front6/README.md` and `NTRAIN.md` distinguish training regimes. Freeze the exact training recipe/cache provenance for every plotted cohort `[TBD]`; never combine pre-fix and post-fix JHQ points as one final frontier.
- Disclose Vogue's imbalanced coarse list as a characteristic of the observed trained index, without assigning an unproven cause or dismissing poor performance. README reports 135,489/932,328 vectors, about 14.5%, but its cited `data/export_ivf.log` is absent from the current frozen directory. Freeze raw histogram evidence `[TBD]` before publishing that measured percentage.
- Repeatability: `results/front6/README.md` reports cold-cache Vogue recalls 0.9852/0.9842/0.9849 with the same candidate set. Treat changes below roughly 1e-3 as within observed build-to-build variation unless repeated evidence resolves them. This is an observed range, not a confidence interval or a universal noise threshold. Raw repeated-run evidence and systematic timing uncertainty remain `[TBD]` to freeze.

### 6.2 End-to-End Recall-QPS — RQ1

Use `fig_frontier` to compare operating regions on the common recall interval. Evidence sources are `data/paper_fronts.log`, `data/paper_rabitq.log`, `data/bench_quant.log`, and **`report_adc2026/v47/fronts.json`**. The last source supplies older CAGRA/IVF-PQ sweeps, not current JHQ points. Verify protocol compatibility before treating the combined figure as a finalized comparison.

Draft around dataset-specific regions: JHQ can be strong at moderate/high recall, particularly on the higher-dimensional OpenAI workloads; RaBitQ can catch or pass it in high-recall tails. Quantify only interpolated matched-recall comparisons supported by overlapping curves. No universal winner or causation from dimension alone.

CPU JHQ belongs in a compact matched-recall table with source and protocol, not an optional footnote; current entries are `[TBD]`. Keep CPU throughput separate from the GPU frontier axes. Allocation failures must name the measured configuration and library path; `data/rabitq_bigsets.log` and `data/paper_rabitq.log` support limited failure disclosures, not a claim that the algorithm can never index those datasets. CAGRA precision variants must be identified individually.

### 6.3 Adaptive Refinement Evaluation — RQ2

#### 6.3.1 Variation in Sufficient Alpha

Use `fig_alpha` and the Br=8 `alpha6.log` sweep at nprobe=128. Show the grid and right-censored arxiv endpoint; use “4 to at least 200.” The current plotted quantity is **ivf_recall - recall**, so “ranking loss” is appropriate with that explicit definition: routing loss has been removed. If changing to 1-Recall@10, label it recall loss/error instead. Do not use diagnostic QPS as performance evidence.

#### 6.3.2 Calibration versus Exhaustive Sweep

Use `fig_calibration` with references keyed by **(dataset,nprobe)**. Current panel (a) compares four sample-rule configurations at nprobe=128; nprobe=512 has no corresponding exhaustive sweep and must not reuse another nprobe's answer. The script derives swept alpha using a 3e-4 tolerance to the largest-grid ranking loss. That operational threshold is smaller than the observed inter-build range; do not describe it as an established noise floor. Re-evaluate sensitivity to the plateau definition `[TBD]`.

`data/alpha_sample.log` supports sample-size sensitivity, including OpenAI-3072 at S=8 selecting alpha=2 and losing 0.0058 recall. However, this older implementation uses a fractional tolerance; it is not a complete sensitivity study of the final integer-slot rule. `data/alpha_fast.log` supports epsilon={0,1,2} slot experiments; distinguish linear and bisection rows rather than letting a parser overwrite repeated configurations. For Vogue at nprobe=512, the two-slot bisection row selects 32 and loses 0.0035 recall. Do not generalize one dataset's tolerance choice to all workloads.

Required final validation `[TBD]`: rerun final-policy sample-size and slot sensitivity, compare selected alpha and full-batch recall against the exhaustive grid at each tested (dataset,nprobe), document alpha_max censoring, repeat samples/builds, and check agreement monotonicity and tie behavior. Separate selection accuracy from speed and from reference quality.

#### 6.3.3 Steady-State Gain and Break-Even

Make economics a headline result, using all RULE rows and their paired within-row `qps_max`, not unrelated FIX timings. Include selected alpha, steady-state gain, recall delta, calibration ms, and batches_to_repay. The table below was recomputed directly from frozen `data/paper_fronts.log`:

| Quantity | Verified value and interpretation |
|---|---|
| RULE configurations | 36 |
| Logged calibration time | 2.0–47.5 ms |
| Finite logged payback range | **0.5–178.6 batches**, correcting the review's 0.6 minimum |
| Median finite payback | 2.7 batches over 35 finite rows; exclude the -1 sentinel |
| Logged finite rows requiring more than two batches | 18 of 36 total configurations; the additional non-repaying row never amortizes |
| No finite payback | arxiv, nprobe=128: gain=0.996, sentinel=-1.0 |
| Largest raw gain | 2.653 at OpenAI-3072, nprobe=32; logged payback=0.5 |
| Longest finite payback | arxiv, nprobe=256: gain=1.003, payback=178.6 |
| Largest logged recall loss | BGE, nprobe=1024: 0.0048 |

BGE losses also exceed 0.003 at nprobe=128 (0.0036), 256 (0.0041), and 512 (0.0045). Therefore do not repeat the README's “all except one <=0.003” or label all raw gains equal-recall. Sub-1e-3 deltas such as -0.0003 are within observed build-to-build variation; larger losses remain visible.

Finalize **`fig_economics` [TBD: review and integration]**: x=steady-state QPS gain, y=logged finite batches_to_repay on a log scale, color/marker identifying dataset and optionally nprobe. Add gain=1 and payback=1 reference lines. Represent the non-repaying configuration separately with an explicit annotation; never plot -1 on the log axis or silently drop it. All 36 configurations must be accounted for. An untracked economics script and rendered outputs appeared during this audit through concurrent workspace work; they are not part of this skeleton commit. Its current “one real cost” annotation and claim that only one recall delta lies outside the observed variation band conflict with the log and require correction. Caption specifies the 35-row finite median, rounding, quality deltas and stable-workload assumption.

Interpret jointly: substantial gains tend to repay quickly; negligible gains produce long or no repayment. A finite 178.6 is long, not mathematically “never,” and tiny timing differences require repetition. The stale “eight batches / 8000 queries” note is not the current result. Economics measures conditional workload reuse, not observed generalization across future drifting batches.

### 6.4 Representation and Execution Ablation — RQ3

#### 6.4.1 Value of the Residual Hierarchy

Use `fig_hierarchy`, `data/hierarchy_ablation.log` and the full JHQ frozen frontier. Compare primary-only/JQ-like versus full JHQ under matched routing/training. Report attained recall across the sweep and work added by refinement. Call maxima “highest measured recall,” not architectural ceilings. This ablation tests the hierarchy, not a fully tuned independent JQ system.

#### 6.4.2 Cartesian-Factorized LUT

Isolate full versus factorized primary tables with identical codes, candidates and policy. State exact representation equivalence, then measured construction/search effects. Historical evidence is under `results/v47_split_lut/`; freezing a matched final-implementation ablation and uncertainty is `[TBD]`. Do not substitute the algebraic entry ratio for a speedup.

#### 6.4.3 Representation-Enabled Optimizations and Engineering

Separate packed access, physical layout, and cursor/launch corrections by status. `fig_ablation` should not visually present them as equally novel. Historical packing and correction evidence is in `results/front6/NEGATIVES.md`, `results/front6/v52.log`, `results/front6/v51.log`, and `data/v57_launch.log`; final matched ablation matrix is `[TBD]`. Do not add percentage gains from different cohorts to infer a combined speedup. Report actual effect sizes side by side even if a simpler optimization has the larger effect.

### 6.5 Mechanistic Analysis and Negative Results — RQ4

Use `fig_negatives`; discuss larger selection buffers, regrouping, scan/refinement fusion, early exit, and table-free distance. Evidence: `data/v54.log`, `data/qdup.log`, `data/qdup_stella.log`, with internal experiment descriptions in `results/front6/NEGATIVES.md`, `V54_SIGN_IP.md`, and `QDUP.md`. Freeze other raw negative-result cohorts before quoting numerical ranges `[TBD]`.

The table-free experiment shows that fewer table bytes need not mean faster execution. Instruction count, occupancy, cache, shared-memory residency and memory traffic all matter. Separate measured runtime/resource counts from causal explanations requiring profiling. One-card observations do not establish a cross-architecture law.

Duplicate-query experiments are a **controlled reuse proxy**, not an upper bound on another scheduler. If L2 reuse is inferred, say the pattern is “consistent with L2 already capturing much of the reuse.” Direct attribution and selection-stage profiling remain `[TBD]`. Prefer shortening setup before removing this evidence; if necessary fold negative results into the ablation discussion without converting the experiment organization to version history.

### 6.6 Batch Sensitivity — RQ5

Use `fig_batch` and `data/batch_sweep.log`, explicitly labeled **batch sensitivity at fixed nprobe=128**. The logged sweep covers OpenAI-3072 and Vogue with batch labels from 32 to 1024. Show both recalls alongside throughput ratios: at the largest logged batch, OpenAI-3072 recalls are 0.9414/0.9436 and Vogue recalls are 0.9645/0.9586 for JHQ/RaBitQ. These are not matched-recall winner comparisons.

Reconcile logged batch=1024 with actual query-file row counts and benchmark batching before describing the workload as 1024 independent queries `[TBD]`. Do not manufacture an independent 10k workload by duplicating approximately 1000 real queries. A matched-recall batch sweep, independent larger query set and causal attribution of batch effects remain `[TBD]`. Do not extrapolate curves to another GPU or batch size.

### 6.7 Build Time and Memory — RQ6

Use `fig_cost` and `fig_memory`. `data/vram.log` measures RaBitQ resident GPU memory on four datasets; `data/paper_fronts.log` measures JHQ. State measurement point (index plus allocated search workspace), units and nprobe/workspace configuration. For example, the nprobe=8 JHQ / measured RaBitQ values are Vogue 1405.2/1012.0 MiB and OpenAI-3072 4047.2/3324.0 MiB. Measured JHQ residency varies with nprobe; the current figure loader retains the last FIX row, so do not mix its bars with first-row prose values.

Model primary/residual codes, IDs/corrections, centroids and query/candidate workspace separately from measured totals. Add **other / allocator / runtime workspace = measured minus modeled**. This residual segment reconciles the accounting but does not independently measure its causes; a negative gap signals a model error. Report competitor components only where measured. RaBitQ has the smaller observed resident footprint on the four measured datasets. Say “JHQ occupies a different memory-accuracy-throughput regime,” not universal memory superiority.

For build time, separate training from encoding, cold training from cached loads, and input loading from index construction. `fig_cost` selects the first FIX row for build cost and converts logged milliseconds to seconds. Cold-cache provenance, units and equivalent competitor timing boundaries need final verification `[TBD]`. Do not treat later cached training rows or unmatched CPU training as full build comparisons.

### 6.8 Experimental Takeaways

Conclude only what the finalized evidence supports: Cartesian structure enables exact reorganization of primary-distance evaluation; refinement is valuable but its useful budget depends on workload; output-stability selection must be judged by both quality and payback; competitive throughput occupies dataset/recall/batch regions with explicit memory and build tradeoffs. CPU acceleration claims remain `[TBD]` until the mandatory baseline lands.

## Figure and artifact completion map

This task creates only the mother framework. Existing scripts and data are preserved; the following are drafting/experiment deliverables, not claims that they have been completed.

| Figure | Placement | Remaining check or action |
|---|---|---|
| fig_pipeline | 3.1 | Consistent Q, actual training dependencies, no equation numbers. |
| fig_layout | 3.2 / 3.3.2 | Separate coalescing from packed instruction reduction; no cache-line waste myth. |
| fig_lut | 3.3.1 | Exact current representation; replace hard-coded equation references. |
| fig_rule | 4.3 | Align set-slot criterion, actual sample selection and bisection/linear mode. |
| fig_frontier | 6.2 | Audit cohort provenance, tuning and hard-coded RaBitQ points in style.py against bench_quant.log. No ceiling shading. |
| fig_alpha | 6.3.1 | Current ranking-loss axis is valid; stale “25x” and equal-recall/gain captions require correction. |
| fig_calibration | 6.3.2 | Preserve dataset+nprobe keys; distinguish old fractional-tolerance evidence, repeated modes and plateau threshold sensitivity. |
| fig_economics (new working-tree draft) | 6.3.3 | Review/integrate scatter, correct recall-loss overclaim, and distinguish non-repayment from finite y values. |
| fig_hierarchy | 6.4.1 | Highest observed recall, matched setup, no universal ceiling. |
| fig_ablation | 6.4 | Separate novelty tiers and freeze final matched ablations. |
| fig_negatives | 6.5 | Controlled reuse proxy; measured facts versus inferred mechanism. |
| fig_batch | 6.6 | Fixed nprobe, both recalls; reconcile actual query counts. |
| fig_cost / fig_memory | 6.7 | Cold-build provenance, measurement configuration, model reconciliation and measured competitor values. |

Use text at least approximately 7 pt at final print size, embedded readable fonts and distinguishable markers. Caption and script comments must agree with the plotted quantity. Earlier reviews saying all figure fixes are complete do not supersede current script inspection.

## Evidence readiness and remaining experimental TODOs

| Claim | Current readiness | Required before final prose |
|---|---|---|
| Exact Cartesian table factorization | Semantics/algebra complete; implementation precondition checked in `jhq_v57_launch_nq/jhq_gpu_index.cu` | Final isolated timing ablation is separate. |
| Alpha variation and censored endpoint | Frozen diagnostic recall evidence complete within tested grid (`alpha6.log`) | State nprobe and plateau convention; no diagnostic QPS. |
| Calibration economics summary | Frozen row-level arithmetic complete (`paper_fronts.log`) | Finalize economics figure; qualify timing scope, quality and stationarity. |
| Final policy universally recovers sufficient alpha | **Blocked [TBD]** | Full matched sweep, final slot-rule sensitivity, sample/build repetition and censoring checks. |
| GPU operating-region comparisons | Frozen curves exist; protocol completion pending | Full baseline grids, training/cache lineage, interpolation provenance and matched-recall tables. |
| Faster than original CPU JHQ | **Blocked [TBD]** | Reproducible strongest CPU setup, explicit source, matched training/recall and pinned timings. |
| Residual hierarchy value | Frozen ablation exists | Audit matching and avoid calling sweep maxima ceilings. |
| Memory tradeoff | Measured totals available for JHQ and four RaBitQ datasets | Align workspace configuration and modeled/observed accounting. |
| Fixed-nprobe batch sensitivity | Frozen measurements available | Verify batch construction; matched-recall batch conclusions remain blocked. |
| Generalization across k, drifting workloads, GPUs | **Unmeasured [TBD]** | Run relevant studies or explicitly retain limitations. |
| Vogue imbalance and repeatability numbers | Documented in notes, frozen raw provenance incomplete | Freeze histogram and repeated-run evidence before numerical publication. |

Priority order: finalize mandatory CPU and baseline protocol; validate the final alpha policy and quality exceptions; finalize economics figure; freeze matched representation/execution ablations and build/measurement provenance; resolve batch counts, repeatability and diagnostic evidence. Larger-k, larger independent batches, drift and a second GPU may remain explicit scope limitations if not measured. No experiment data or implementation should be changed to make a narrative claim true.
