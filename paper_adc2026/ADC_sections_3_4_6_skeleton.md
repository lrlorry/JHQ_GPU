# ADC 2026 Paper Skeleton — Sections 3, 4, and 6

> Purpose: this is a **paper-structure skeleton**, not final prose.  
> It is designed so Claude can inspect the current `JHQ_GPU` project, verify each claim against code/logs, and then turn the skeleton into LaTeX.
>
> Core rule: **do not write the paper as a version history (`v47/v51/v52/...`)**.  
> Write it as a research story:
>
> **Section 3 = GPU-native JHQ execution**  
> **Section 4 = adaptive refinement-budget selection**  
> **Section 6 = experimental evidence answering explicit research questions**

---

# 3. GPU-Native JHQ

**Suggested length:** ~1400–1700 words

## 3.1 Overview and Design Goals

### Goal
Explain what must change when moving JHQ from the CPU implementation to a GPU-native execution pipeline, while preserving JHQ's quantization semantics.

### Main message
Original JHQ provides the hierarchical quantization formulation, but its execution is not designed around GPU memory access, query-level parallelism, shared-memory pressure, or large-batch search.

### Suggested overview figure

```text
                    INDEX CONSTRUCTION
                           |
                    JL Transformation
                           |
                 Primary Quantization
                           |
                 Residual Quantization
                           |
                    IVF Assignment
                           |
                  GPU-Oriented Layout
                           |
                           v
QUERY -> JL -> IVF Probe -> Primary Scan -> top-alpha*k
                                      |
                               Residual Refine
                                      |
                                   top-k
```

### What this subsection should establish

- JHQ's mathematical representation is preserved.
- GPU redesign focuses on:
  - batched transforms,
  - physical layout,
  - compact/factorized distance evaluation,
  - coalesced candidate scanning,
  - exact candidate selection,
  - selective residual refinement.
- Separate **JHQ-specific structural optimizations** from ordinary CUDA engineering.

### Do not overclaim
Do not present every implementation fix as a research contribution.

---

## 3.2 GPU-Native Index Construction

### 3.2.1 Batched JL Transformation

Original operation:

\[
y = Qx
\]

GPU form:

\[
Y = XQ^T
\]

Explain:

- same orthogonal transform / same mathematical semantics;
- vectors are transformed in batches;
- GPU implementation turns repeated matrix-vector operations into matrix-matrix multiplication;
- transformed vectors are used for both primary and residual quantization.

### Key implementation points to verify in code

- cuBLAS GEMM path;
- staged H2D transfers;
- pinned host buffers / asynchronous streams;
- faithful mode vs optional TF32 mode.

### Paper emphasis
This is enabling infrastructure, **not the central novelty**.

---

### 3.2.2 Primary and Residual Quantization

Explain only what is needed for the GPU design.

Primary representation:

\[
y \rightarrow c_P \rightarrow \hat y_P
\]

Residual:

\[
r = y - \hat y_P
\]

Residual quantization:

\[
r \rightarrow c_R
\]

Important distinction:

- the JHQ residual is the residual of the **primary JQ approximation**;
- it is **not** the IVF centroid residual.

### Mention

- analytical Cartesian primary codebook;
- scalar residual codebook;
- hierarchy preserved from original JHQ.

### Do not turn this into a long tutorial
Detailed JHQ theory belongs in Section 2.

---

### 3.2.3 IVF Assignment and GPU-Oriented Physical Layout

Explain:

1. train IVF coarse centroids;
2. assign each database vector to one list;
3. sort/group vectors by list ID;
4. store list contents contiguously.

Primary-code layout transformation:

\[
[N,M] \rightarrow [M,N]
\]

Meaning:

> candidate codes are physically reorganized so threads scanning the same subspace read adjacent bytes.

Suggested micro-figure:

```text
Row-major candidate layout

x0: c00 c01 c02 ...
x1: c10 c11 c12 ...
x2: c20 c21 c22 ...

            |
            v

Subspace-major layout

m0: c00 c10 c20 ...
m1: c01 c11 c21 ...
m2: c02 c12 c22 ...
```

### Claim
This changes physical organization, not JHQ's distance function.

---

## 3.3 GPU-Native Query Processing

This is the main technical subsection.

Suggested flow:

```text
query
  |
JL transform
  |
IVF probe selection
  |
factorized primary distance table
  |
coalesced primary scan
  |
exact top-alpha*k
  |
residual refinement
  |
final top-k
```

---

### 3.3.1 Query Transform and IVF Routing

For each query:

\[
q' = Qq
\]

Then choose the nearest `nprobe` IVF lists.

Explain briefly:

- query-centroid distances are evaluated on GPU;
- only candidates from selected lists enter JHQ primary scanning.

Keep this short.

---

### 3.3.2 Cartesian-Factorized Primary Distance Table

This should be one of the strongest technical points in Section 3.

Standard ADC form:

\[
T_m[c] = \|q_m - C_m[c]\|^2
\]

For the analytical Cartesian primary codebook, exploit code structure:

\[
T_m[c]
=
T_m^{hi}[c \gg 4]
+
T_m^{lo}[c \& 15]
\]

For the current 8-bit primary code:

```text
Full table:
256 entries / subspace

Factorized table:
16 + 16 = 32 entries / subspace
```

Therefore:

\[
256 \rightarrow 32
\]

### What to explain

- factorization is exact for the current Cartesian structure;
- it reduces the per-query primary LUT working set;
- smaller working set reduces shared-memory pressure and LUT construction work;
- the arithmetic meaning of the JHQ primary distance is unchanged.

### Current evidence to verify

- `results/v47_split_lut/README.md`
- expected reported reduction:
  - `M=96`: 96 KiB -> 12 KiB per query
  - `M=384`: 384 KiB -> 48 KiB per query
- measured QPS improvement: approximately +6% to +14.5% across tested cells.

### Suggested figure

```text
8-bit code c = [high 4 bits][low 4 bits]

                 /--> LUT_hi[high]
candidate code --|
                 \--> LUT_lo[low]

distance contribution = LUT_hi + LUT_lo
```

### Research framing
This is not merely "use less shared memory".

Frame it as:

> exploiting the Cartesian algebraic structure of JHQ to redesign GPU distance evaluation.

---

### 3.3.3 Coalesced Primary Scan and Exact Candidate Selection

Primary distance:

\[
D_P(q,x)
=
\sum_{m=1}^{M} T_m[c_m(x)]
\]

GPU execution:

```text
selected IVF lists
       |
contiguous candidate positions
       |
coalesced primary-code reads
       |
factorized LUT lookup
       |
complete primary distance
       |
exact top-ck selection
```

where

\[
ck = \alpha k
\]

### Implementation ideas that may be described here

- monotonic per-thread probe cursor;
- packed primary-code loading (e.g. four subspaces via one 32-bit word);
- exact threshold-compaction candidate selection;
- launch sized to actual number of queries.

### Important writing rule
Do **not** write:

> v51 does X, v52 does Y, v57 does Z.

Write:

> we remove redundant probe-boundary traversal;  
> we coarsen code loads;  
> we size the launch to the active query batch.

Version numbers only belong in internal evidence mapping.

### Exactness
State clearly which parts preserve the exact JHQ primary score / exact top-`ck` semantics.

---

### 3.3.4 Residual Refinement

Only the primary survivors are refined:

\[
ck = \alpha k
\]

Residual-aware distance is evaluated for these candidates and the final top-`k` is selected.

Relevant expansion:

\[
\|q-(\hat y+\hat r)\|^2
=
\|q-\hat y\|^2
+
\|\hat r\|^2
-
2q^T\hat r
+
2\hat y^T\hat r
\]

Explain:

- primary term comes from the primary scan;
- residual contribution is computed only for survivors;
- per-vector correction can be precomputed;
- residual codebook is small and reused.

### Main message
The hierarchy turns residual processing into a **selective refinement stage**, not a full-database computation.

### Avoid
Do not claim residual refinement dominates total runtime unless current profiling supports it.

---

## 3.4 Summary of Section 3

End the section with a compact synthesis:

```text
JHQ structure             GPU realization
------------------------------------------------
Orthogonal transform   -> batched GEMM
Cartesian primary code -> factorized LUT
Subspace codes         -> coalesced / packed loads
Hierarchical residual  -> selective refinement
IVF lists              -> contiguous GPU layout
```

### Main takeaway
Section 3 should read as **JHQ-aware GPU co-design**, not a CUDA optimization diary.

---

# 4. Adaptive Hierarchical Refinement

**Suggested length:** ~750–1000 words

## 4.1 Motivation: Why a Fixed Alpha Is Wasteful or Risky

Original JHQ uses:

\[
ck = \alpha k
\]

but leaves `alpha` externally specified.

Explain the two failure modes:

```text
alpha too small
    ->
not enough candidates refined
    ->
ranking loss

alpha too large
    ->
same final answer
    ->
unnecessary residual work
```

### Empirical observation to motivate the method
The refinement saturation point varies strongly across datasets / search configurations.

Current project evidence reports a range of roughly **25x**.

Use careful wording:

> "sufficient / saturation refinement budget"

Prefer this over:

> "globally optimal alpha"

unless an explicit optimization objective is defined.

### Evidence files

- `data/alpha_ds.log`
- `data/bge_alpha_scansplit.log`
- `data/paper_fronts.log`

---

## 4.2 Calibration Principle

Goal:

> select the smallest refinement budget that reproduces the stable top-`k` result sufficiently closely.

Use a small sample of queries:

\[
Q_s,\quad |Q_s| = S
\]

Run a generous reference budget:

\[
\alpha_{\max}
\]

to obtain:

\[
R_{\max}(q)
\]

For candidate alpha values, obtain:

\[
R_{\alpha}(q)
\]

Then measure output disagreement.

Generic rule:

\[
\alpha^*
=
\min
\left\{
\alpha:
\operatorname{Diff}(R_\alpha,R_{\max})
\le \epsilon
\right\}
\]

### Important property

- no external ground-truth labels are required;
- the rule asks whether reducing refinement changes JHQ's own returned result;
- it avoids an exhaustive full-query alpha sweep.

Do **not** call it "no pre-scan" if calibration requires running sample queries.

---

## 4.3 Adaptive-Alpha Selection Algorithm

Suggested paper algorithm:

```text
Algorithm: Adaptive Refinement Budget Selection

Input:
    sample queries Qs
    candidate alpha values A
    generous alpha_max
    tolerance epsilon

1. Search Qs with alpha_max and store reference top-k.
2. Test smaller alpha values.
3. Compare returned top-k with the reference.
4. Select the smallest alpha whose disagreement is within epsilon.
5. Use this alpha for the query workload.
```

### Current configuration to verify experimentally

- sample size around `S=32`;
- one-slot tolerance;
- stopping rule / ordering of alpha candidates.

### Discuss

- calibration overhead;
- one-sided conservative behavior;
- failure mode when the sample is not representative;
- this is an empirical policy, not a formal worst-case guarantee.

---

## 4.4 Integration with JHQ-GPU

Show:

```text
small calibration sample
         |
      alpha*
         |
         v
query -> IVF -> primary scan -> top-alpha*k -> residual refinement -> top-k
```

### Important point
Adaptive alpha changes only the refinement budget.

It does not change:

- JL transform;
- primary codebook;
- residual codebook;
- IVF routing;
- distance definition.

Therefore it is modular and can be enabled independently of most Section 3 execution optimizations.

---

## 4.5 Expected Cost / Benefit

Calibration cost should be compared with the savings it produces over subsequent batches.

Current evidence to verify:

- `data/v57_launch.log`
- reported calibration: roughly 6–29 ms in current experiments;
- some workloads repay the calibration cost within about one batch.

Keep detailed numbers for Section 6; Section 4 should explain the mechanism.

---

# 6. Experiments

**Suggested length:** ~1800–2200 words

The experiments should be organized by **research questions**, not by implementation versions.

---

## 6.1 Experimental Setup

### 6.1.1 Hardware

Current project record:

- NVIDIA RTX 5090
- 32,607 MiB GPU memory
- 170 SMs
- 1792 GB/s memory bandwidth
- host machine with 208 CPU cores and 754 GB RAM

Verify final values before submission.

### 6.1.2 Datasets

Current six datasets:

| Dataset | N | d |
|---|---:|---:|
| vogue-768 | 1.0M | 768 |
| arxiv-768 | 2.25M | 768 |
| bge-m3 | 10.1M | 1024 |
| stella-trec24 | 17.8M | 1024 |
| openai3-1536 | 1.0M | 1536 |
| openai3-3072 | 1.0M | 3072 |

### 6.1.3 Query / Metric Protocol

Current protocol:

- 999 or 1000 queries, depending on dataset file;
- one search call for the complete query file;
- timed region: host queries in -> host results out;
- same timing definition for JHQ, cuVS, and IVF-RaBitQ;
- Recall@10 against true top-10.

### Important TODO
If batch size or timing protocol changes, re-run all methods under the same protocol.

### 6.1.4 Baselines

Main baselines:

- original CPU JHQ
- JQ-GPU / hierarchy ablation
- cuVS IVF-PQ
- cuVS CAGRA
- cuVS IVF-RaBitQ
- optional IVF-Flat / brute-force reference

### Fairness checklist

- same GPU for all GPU systems;
- same query set;
- same `k`;
- same metric;
- same host/device timing boundary;
- compare at matched Recall@10, not only one parameter point.

### Current open item
CPU JHQ protocol still needs final validation:
- `max_iter`
- `max_train_n`
- thread count / pinning
- official-repository parity

---

# RQ1. How Does JHQ-GPU Compare with Modern GPU ANN Methods?

## 6.2 End-to-End Recall-QPS

### Main figure
Six-panel Recall-QPS frontier:

```text
Vogue        Arxiv        BGE
Stella       OAI-1536     OAI-3072
```

Curves:

- JHQ-GPU
- IVF-RaBitQ
- CAGRA
- IVF-PQ

### Current evidence

- `paper_adc2026/figures/fronts.json`
- `data/paper_fronts.log`
- current RaBitQ:
  - `data/bench_quant.log`
  - `data/openai3072_v57_front.log`

### Current RaBitQ state
At the time this skeleton was created, the repository records the direct IVF-RaBitQ comparison as complete on **openai3-3072 only**; the other five are marked missing/running.

Do not generalize the OpenAI3-3072 conclusion to all datasets until the remaining five are measured.

### Current OpenAI3-3072 pattern

Approximate reported JHQ / RaBitQ ratio at matched recall:

| Recall@10 | JHQ / IVF-RaBitQ |
|---:|---:|
| 0.95 | 1.36x |
| 0.96 | 1.22x |
| 0.97 | 1.06x |
| 0.98 | 0.93x |
| 0.99 | 0.72x |

Crossover:

\[
Recall \approx 0.975
\]

### Good research framing

Avoid:

> JHQ always beats RaBitQ.

Prefer:

> the two systems occupy different operating regions; JHQ is stronger in some recall regimes while RaBitQ is stronger at very high recall.

Only generalize after all six datasets support it.

---

# RQ2. Does Adaptive Alpha Select a Good Refinement Budget?

## 6.3 Adaptive Refinement Evaluation

This subsection should directly validate Section 4.

### 6.3.1 How much does the saturation alpha vary?

Plot:

\[
\alpha \rightarrow Recall / QPS
\]

for representative datasets.

Goal:

> show that one global fixed alpha is either wasteful or insufficient.

Evidence:

- `data/alpha_ds.log`
- `data/bge_alpha_scansplit.log`

Current reported observation:

\[
\text{saturation budget span} \approx 25\times
\]

---

### 6.3.2 Does calibration recover the swept saturation point?

Compare:

```text
exhaustive sweep result
vs
adaptive calibration result
```

Evidence:

- `data/alpha_sample.log`
- `data/alpha_fast.log`

Current project record:

- recovers swept saturation alpha in 7/8 tested configurations;
- `S=32` appears to be the knee;
- one-slot tolerance appears to be the knee.

Explain failures, not just successes.

---

### 6.3.3 End-to-end benefit

Report:

- equal-recall QPS gain;
- recall delta;
- selected alpha;
- calibration overhead.

Evidence:

- `data/paper_fronts.log`
- `data/v57_launch.log`

Current reported range:

\[
1.0\times \text{ to } 2.65\times
\]

with small recall change in most tested configurations.

### Important wording
Do not say "2.65x faster than JHQ algorithm" without qualification.

Say:

> relative to the fixed-alpha configuration, adaptive refinement removes unnecessary residual work.

---

# RQ3. Which GPU Design Choices Actually Matter?

## 6.4 GPU Design Ablation

Do not organize by version number.

Suggested table:

| Design | Mechanism | Evidence |
|---|---|---|
| Factorized Cartesian LUT | reduce per-query LUT working set | `results/v47_split_lut/README.md` |
| Packed primary-code loading | fewer / wider memory operations | `results/front6/v52.log` |
| Monotonic probe traversal | remove repeated list-boundary scans | `results/front6/v51.log` |
| Query-sized launch | avoid zero-padded query work | `data/v57_launch.log` |
| Adaptive alpha | reduce residual refinement work | `data/paper_fronts.log` |

### Current measured evidence to verify

- factorized LUT: roughly +6% to +14.5%;
- packed 32-bit code loads: roughly +30% to +48%;
- probe cursor: roughly +2.5% to +148%, configuration-dependent;
- query-sized launch: roughly +1% to +9%;
- adaptive alpha: up to 2.65x relative to fixed alpha.

### Warning
Large gains caused by removing an implementation pathology should not be presented as deep algorithmic novelty.

---

# RQ4. Why Do Plausible GPU Optimizations Fail?

## 6.5 Performance Analysis and Negative Results

This can be one of the most interesting parts of the paper if kept concise.

Suggested table:

| Design | Result |
|---|---:|
| larger selection buffer | often negative |
| residual regrouping | about neutral |
| scan/refinement fusion | negative on Stella |
| exact early termination | small mixed effect |
| table-free primary distance | -30% to -52% |

Evidence:

- `results/front6/NEGATIVES.md`
- `results/front6/V54_SIGN_IP.md`
- `data/v54.log`

### Focus case: table-free primary distance

At binary Cartesian levels:

\[
D_P
=
\|q'\|^2 + da^2 - 2a\langle q', s_y \rangle
\]

This identity reduces LUT storage, but current RTX 5090 experiments show large slowdowns.

Interpretation:

> less memory traffic does not automatically imply higher GPU throughput when the alternative increases instruction cost and the architecture already provides high memory bandwidth / cache capacity.

Do not claim architecture-independent conclusions from one GPU.

---

### Cross-query reuse / cluster-centric scheduling study

Evidence:

- `results/front6/QDUP.md`
- `data/qdup.log`
- `data/qdup_stella.log`

Artificially increase query-list overlap by duplicating queries.

Current reported gain ceiling:

\[
+4\% \text{ to } +26\%
\]

Important observed pattern:

- much of the gain appears after the first reuse increase;
- additional duplication produces diminishing returns;
- current analysis attributes this to L2 already serving a large fraction of repeated primary-code accesses.

### Paper use
This is useful to explain why a larger cross-query regrouping rewrite was not automatically adopted.

Be careful:

- this is an empirical upper-bound-style experiment, not a proof;
- do not claim the RaBitQ scheduler is ineffective in general.

---

# RQ5. What Is the Cost in Build Time and Memory?

## 6.6 Build Time, Memory, and Scalability

Suggested table:

| Method | Build Time | GPU Memory | Recall Regime | QPS |
|---|---:|---:|---:|---:|
| JHQ-GPU | ... | ... | ... | ... |
| IVF-RaBitQ | ... | ... | ... | ... |
| CAGRA | ... | ... | ... | ... |
| IVF-PQ | ... | ... | ... | ... |

### Important existing observation

For large high-dimensional datasets:

- CAGRA fp32 may not fit in 32 GB GPU memory;
- current repo notes:
  - Stella raw fp32 storage is far beyond card capacity;
  - BGE-M3 fp32 is also beyond card capacity.

Frame carefully:

> JHQ occupies a different memory-accuracy-throughput regime.

Do not claim JHQ is universally more memory-efficient than RaBitQ unless directly measured.

---

# 6.7 Experimental Takeaways

Finish Section 6 with 3–4 compact conclusions, for example:

1. JHQ-GPU is competitive with modern GPU ANN systems, with operating-region tradeoffs against IVF-RaBitQ and CAGRA.
2. A fixed refinement budget is poorly matched across datasets; lightweight calibration removes substantial unnecessary residual work.
3. The largest GPU gains come from structure-aware layout / distance evaluation, not from every seemingly natural kernel fusion.
4. Some intuitive optimizations fail because cache behavior, shared-memory residency, and instruction cost dominate simple byte-count reasoning.

Only keep conclusions directly supported by final submitted experiments.

---

# Recommended Figures and Tables

## Figures

### Figure 1 — JHQ-GPU pipeline
Section 3.1.

### Figure 2 — Cartesian factorized LUT
Section 3.3.2.

### Figure 3 — Adaptive-alpha calibration
Section 4.

### Figure 4 — Six-panel Recall-QPS frontier
Section 6.2.

### Figure 5 — Alpha saturation / adaptive selection
Section 6.3.

### Figure 6 — Optional ablation or negative-result chart
Section 6.4 or 6.5.

---

## Tables

### Table 1 — Datasets and hardware
Section 6.1.

### Table 2 — GPU design ablation
Section 6.4.

### Table 3 — Build time / GPU memory
Section 6.6.

---

# What Claude Should Verify Before Turning This into LaTeX

1. Read the latest `paper_adc2026/README.md`.
2. Check all frozen `paper_adc2026/data/` logs before copying any number.
3. Verify the current branch / commit used for final experiments.
4. Do not use stale `v42/v47/...` results if a newer frozen paper result supersedes them.
5. Confirm which optimizations are:
   - exact / faithful,
   - approximate,
   - bug fix,
   - performance optimization,
   - algorithmic contribution.
6. Confirm the final IVF-RaBitQ results on all six datasets.
7. Confirm CPU-JHQ protocol before writing CPU speedup claims.
8. Do not claim:
   - architecture independence from one GPU,
   - universal superiority over RaBitQ,
   - universal memory superiority over RaBitQ,
   - a theoretical guarantee for adaptive alpha.
9. Keep repository version numbers out of the paper body.
10. Every important numeric claim should map to one frozen evidence file.

---

# Suggested Final Section Titles

```text
3 GPU-Native JHQ
  3.1 Overview
  3.2 GPU-Native Index Construction
  3.3 GPU-Native Query Processing
      3.3.1 Query Transformation and IVF Routing
      3.3.2 Cartesian-Factorized Primary Distance Table
      3.3.3 Coalesced Primary Scan and Candidate Selection
      3.3.4 Residual Refinement
  3.4 Summary

4 Adaptive Hierarchical Refinement
  4.1 Motivation
  4.2 Calibration Principle
  4.3 Adaptive Refinement-Budget Selection
  4.4 Integration and Overhead

6 Experiments
  6.1 Experimental Setup
  6.2 End-to-End Recall-QPS
  6.3 Adaptive Refinement Evaluation
  6.4 GPU Design Ablation
  6.5 Performance Analysis and Negative Results
  6.6 Build Time and Memory
```

---

# One-Sentence Paper Story

> **JHQ-GPU exploits the structural properties of JHQ to redesign its GPU execution path, and introduces an adaptive refinement-budget policy that avoids unnecessary residual work across datasets with widely different refinement requirements.**

This sentence should be checked against the final evidence and then used to keep Sections 3, 4, and 6 aligned.
