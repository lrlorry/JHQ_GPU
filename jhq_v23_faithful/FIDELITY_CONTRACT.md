# JHQ fidelity contract

What the paper (Han et al., PVLDB 2026, p1530) defines, what this directory
executes, and how the two are compared. The paper is the specification. Where
the paper does not define something, that is recorded as an ambiguity rather
than resolved silently in favour of whatever the code happens to do.

Severities are as agreed: **P0** breaks the claim that this is a faithful GPU
realisation of JHQ; **P1** can change results without changing the algorithm;
**P2** is systems-only. Recall is not used as a correctness test anywhere in
this document — it is an aggregate over queries and two different result sets
can share one.

Citations are to the paper's own numbering: §3.1–3.2 (JL transform, JQ), §4.1–4.3
(JHQ architecture, codebooks, query processing), Eq. 3–9, Lemma 1/2/5,
Theorem 6, Algorithm 1.

---

## 1. JL / orthogonal transformation

**Paper.** Definition 1 and §3.1: Π ∈ ℝ^{d×d} is obtained by sampling
G_ij ~ N(0,1) and taking Q from the QR decomposition G = QR; y = Πx. Lemma 1
states inner product, cosine and Euclidean distance are preserved, so distance
*order* is preserved.

**Implementation.** `cpu/jl_transform.cpp` builds Π by LAPACK QR of a Gaussian
matrix; the rotation is applied as a cuBLAS SGEMM. `JHQ_TF32` is off by default.

**Status: MATCH** (fp32), **P1** when `JHQ_TF32=1` — TF32 keeps 10 mantissa bits
and can reorder candidates near a tie. Measured cost is recorded in
`jhq_gpu_index.cu`: 0.9444 → 0.9410 at d=768.

**Tested by.** Differential test A: rotated query against a CPU fp64 Πx.

---

## 2. Primary codebook construction

**Paper.** §3.2, Eq. 4:

    c^m_ji = √(2/d) · erf⁻¹(2 q_i − 1),   q_i = (i − 0.5)/K^{1/Ds},  i = 1…K^{1/Ds}

with C^m the Cartesian product of the per-dimension codeword sets, |C^m| = K = 2^B.

**Implementation.** `cpu/codebook.cpp`, `LloydMaxCodebook`:

    K1D = 2^{B/Ds};  q_i = (i + 0.5)/K1D  (0-indexed);
    c1d[i] = σ · √2 · erfinv(2 q_i − 1)

**Two differences, and they are not the same kind.**

*Quantile indexing* — the paper's 1-indexed (i−0.5)/K and the code's 0-indexed
(i+0.5)/K enumerate the same set. **MATCH.**

*Scale* — the paper writes √(2/d); the code writes σ·√2 with σ² = E[‖x‖²]/d,
the per-dimension variance §3.2 defines. `train()` computes exactly that, as
`sqrt(sum_sq / (n_train·d))`. The two forms coincide when E[‖x‖²] = 1, so the
question reduces to whether the vectors reaching JHQ are unit-norm.

**Measured, not assumed** (`scripts/check_norms.py`). The demo hands what
`load_fvecs_mmap` returns straight to `train()` and `add()` with no
transformation, so the files are what JHQ consumes. Reading them, over a
200,000-vector deterministic stride sample of each base set and every query:

| dataset | d | min ‖x‖ | mean ‖x‖ | max ‖x‖ | std |
|---|---|---|---|---|---|
| vogue-768 | 768 | 0.99999973 | 1.00000000 | 1.00000028 | 6.2e-08 |
| arxiv-768 | 768 | 0.99999989 | 1.00000003 | 1.00000013 | 3.2e-08 |
| openai3-1536 | 1536 | 0.99999992 | 1.00000003 | 1.00000011 | 2.9e-08 |
| openai3-3072 | 3072 | 0.99999992 | 1.00000003 | 1.00000011 | 3.0e-08 |
| bge-m3 | 1024 | 0.99999992 | 1.00000003 | 1.00000011 | 3.0e-08 |
| stella-trec24 | 1024 | 0.99999993 | 1.00000003 | 1.00000014 | 3.1e-08 |

Queries match their base sets to the same tolerance. Every vector is unit-norm
to within 3e-7, which is float32 rounding on a sum of d squares.

The two scales therefore agree to fp32 epsilon:

| dataset | σ | σ·√2 | √(2/d) | abs diff | rel diff |
|---|---|---|---|---|---|
| vogue-768 | 0.03608439 | 0.05103104 | 0.05103104 | 3.9e-11 | 7.6e-10 |
| arxiv-768 | 0.03608439 | 0.05103104 | 0.05103104 | 1.3e-09 | 2.5e-08 |
| openai3-1536 | 0.02551552 | 0.03608439 | 0.03608439 | 9.2e-10 | 2.6e-08 |
| openai3-3072 | 0.01804220 | 0.02551552 | 0.02551552 | 6.7e-10 | 2.6e-08 |
| bge-m3 | 0.03125000 | 0.04419418 | 0.04419417 | 1.2e-09 | 2.7e-08 |
| stella-trec24 | 0.03125000 | 0.04419418 | 0.04419417 | 1.2e-09 | 2.8e-08 |

**Status: RESOLVED FOR OUR EXPERIMENTS.** On this benchmark protocol the
implementation computes the paper's constant. The general form is kept because
it is the one Lemma 2 states and it stays correct if a non-normalised set is
ever indexed; on non-unit-norm data the two would genuinely differ, and that
is a property of the input, not a defect of either.

---

## 3. B not divisible by Ds

**Paper.** Eq. 4 needs K^{1/Ds} = 2^{B/Ds} codewords per dimension and asserts
|C^m| = K. For that to hold, B/Ds must be a whole number. **The paper never
defines the case B % Ds ≠ 0**, and its own experiments do not appear to require
it.

**Implementation.** `LloydMaxCodebook` throws:
`"B must be divisible by Ds = d/M"`, and `cartesian_admissible(B, Ds)` returns
`Ds > 0 && B % Ds == 0`.

**The exact condition is `Ds = d/M` and `Ds | B`, not `M >= d/8`.** With B = 8
that means `Ds ∈ {1, 2, 4, 8}` and so `M ∈ {d, d/2, d/4, d/8}` among the M that
divide d. `M >= d/8` is necessary but not sufficient: at d = 768, M = 128 gives
Ds = 6, which is larger than d/8 = 96 and still inadmissible because 8 % 6 ≠ 0.
The smallest valid M is d/8; the valid ones above it are not every integer but
the three that make Ds a power of two dividing 8.

The executable has always tested the right thing. Earlier fidelity reports
stated the restriction as `M >= d/8`, which is the weaker necessary condition,
and that wording was wrong.

**Status: PAPER AMBIGUITY, resolved conservatively.** Refusing to run is the
choice that does not invent an interpretation, and it is the right one. A
comment elsewhere in the tree claims "the uneven bit split makes it
constructible at every Ds" — that **disagrees with the executable**, which
throws, and is stale. The restriction has to be stated as a parameter
constraint of the faithful mode, not hidden.

**Consequence already measured.** The OpenAI3 rows in `results/final/p0_*` were
taken at M=96, where d/Ds is 16 and 32 and this construction cannot be built at
all. Those rows therefore did not use Eq. 4.

---

## 4. Primary encoding

**Paper.** §3.2: ŷ^{(m)} = argmin_{c ∈ C^m} ‖y^{(m)} − c‖².

**Implementation.** Because C^m is a Cartesian product, the argmin factorises
into Ds independent 1-D nearest-codeword searches; `nearest_1d` binary-searches
the midpoints of the sorted codeword list. This is an identity, not an
approximation: the squared distance is a sum of per-dimension terms and each is
minimised independently.

**Status: VERIFIED.** `scripts/test_jhq_encode.py` recomputes the code by
enumerating all K codewords of the subspace in float64 -- it does not take the
separable shortcut, which is what makes it a check of that shortcut rather than
a restatement of it.

**Result.** 3,440,640 primary codes over 30 configurations (5 seeds x M in
{96,192,384} x Br in {8,4}, 512 vectors each): **0 mismatches, 0 ties.**

---

## 5. Residual definition

**Paper.** §4.2: r^{(m)} = y^{(m)} − ŷ^{(m)}, the error of the primary level.

**Implementation.** `encode.cu:residual_encode_kernel`: `resid = y[j] − yhat_j`
with `yhat_j` read from the primary centroid. **MATCH.**

---

## 6. Residual codebook training

**Paper.** Eq. 5: R^m = { r^{(m)}_j : 1 ≤ j ≤ Ds, **∀y ∈ 𝒴** } — every
one-dimensional residual value in subspace m, pooled across that subspace's Ds
dimensions, over the whole dataset. One-dimensional k-means on R^m gives
K_r = 2^{B_r} codewords, one shared codebook per subspace, O(M·K_r) space.

**Implementation.** One codebook per subspace, values pooled across the
subspace's dimensions — **MATCH** on structure. Trained on `n_train = 100,000`
sampled vectors, not all of 𝒴 — **P1**, a training-protocol approximation. At
17.8M vectors that is a 0.6% sample.

**Tested by.** Not a differential test; a training-protocol note. A strict mode
would set n_train = N.

---

## 7. Residual encoding

**Paper.** §4.2: each residual dimension is quantised to its nearest scalar
codeword in C_R^m.

**Implementation.** `nearest_sorted_dev`, binary search on midpoints of the
sorted codebook — exact nearest for a sorted 1-D codebook. **MATCH.**

---

## 8. IVF usage / candidate generation

**Paper.** §4.4: "we only use IVF to partition the dataset into clusters and
index the vectors with the cluster ID, to roughly prune the candidates and
narrow down the search space… the quantization quality of our algorithm is not
affected by the IVF." Quantisation is of y itself, not of an IVF residual.

**Implementation.** Coarse assignment by exact fp32 ranking of
‖c‖² − 2⟨q,c⟩; encoding quantises y, not y − centroid. **MATCH.**

The int8 tensor-core path for *building* the assignment is exact by
construction: rows whose Cauchy–Schwarz bound leaves the argmin ambiguous are
recomputed in fp32, so the assignment equals the fp32 one row for row. **MATCH.**

---

## 9. Full primary-distance evaluation

**Paper.** Algorithm 1, lines 2–3: **for all y ∈ 𝒴**, calculate the primary
distance d(q_JL, ŷ) — Eq. 6, a sum over all M subspaces.

**Implementation.** `scan_ivf_coalesced_kernel` evaluates subspaces
[0, PM) first, keeps KEEP candidates per thread by that partial sum, and
completes [PM, M) only for those. With PM < M a candidate can be discarded
before its full primary distance exists.

**Status: P0 when PM < M.** Isolated as the named variant `JHQ-GPU-Cascade`;
the faithful path sets PM = M. Measured cost of the variant on vogue:
0.9747 → 0.9647 at PM = M/4, and inside the build's own ±2e-4 at PM = M/2,
buying 38%.

**Tested by.** Differential test D: primary distance per candidate against the
CPU reference.

---

## 10. Top-αk selection

**Paper.** Algorithm 1, line 4: select the top-αk nearest 𝒵 from 𝒴 **by primary
distance**, |𝒵| = αk. This is a global selection over all evaluated candidates.

**Implementation (before this audit).** Each thread keeps its own best K_LOCAL
and the block merges those. A thread holding more than K_LOCAL of the true
top-ck drops the excess before the merge sees it.

**Status: P0.** Measured, by diffing returned ids with the index pinned and
BLOCK fixed at 512: depth 4 against depth 8 on vogue at nprobe=128 differs in
11 of 10,000 id positions, 4 queries in 1000, 1 of them a genuinely different
set. Depth is capped by registers rather than tuned — pd/pp/ld/lp are 4·K_LOCAL
registers against the 64 a thread gets at BLOCK=1024, so 8 does not launch there
at all.

**Fix.** A threshold-compaction selection that is exact regardless of
distribution: the block keeps a shared buffer of capacity 2·ck; a candidate is
appended when its distance is below the current threshold; when the buffer
fills it is sorted, the best ck are kept, and the threshold becomes the ck-th
value. A candidate is discarded only when its distance is ≥ the ck-th best seen
so far, which is ≥ the final ck-th best, so it cannot belong to the final
top-ck. Exactness does not depend on how candidates are distributed over
threads.

**Status: VERIFIED.** Over 18 search configurations -- both residual paths,
nprobe in {8,32,128,256}, M in {96,192}, plus arXiv-768 at two nprobe --
**top-alpha*k mismatch = 0** in every one, with the primary distances the
selector returned matching a recomputation to **0.000e+00**. The control
`JHQ_EXACT_TOPCK=0` fails the same test (3 missing/3 extra at nprobe=32, 1/1 at
128), which is what makes the passes mean something.

---

## 11. Residual refinement

**Paper.** Algorithm 1, lines 5–7 and Eq. 8: for z ∈ 𝒵 only, compute
d(q_JL, r̂) = Σ_m Σ_j T_R^{mj}[q'_{mj}], then the composite distance.

**Implementation.** Residual distance computed only for the ck selected
candidates. Two paths, both computing the same value: the materialised tables
(§4.3's own O(d·K_r) construction) and a fused kernel that recomputes each entry
from q_JL and C_R^m, which Lemma 5 permits since one codebook is shared across a
subspace's dimensions. **MATCH.**

**Status: VERIFIED.** Fused against materialised, on one pinned index, matched
by candidate rather than by slot: candidate sets **identical**, max
|fused - materialised| composite **1.192e-07** absolute and 6.9e-08 relative --
fp32 rounding on a quantity computed two different ways -- and final top-k
identical at both nprobe tested.

---

## 12. Composite JHQ distance

**Paper.** Theorem 6:

    d(q_JL, ŷ + r̂) = d(q_JL, ŷ) + d(q_JL, r̂) − ‖q_JL‖² + 2⟨ŷ, r̂⟩

**Implementation.** `comp_dists = topck_primary + d_res + list_corr[pos]`, with
`list_corr[vid] = 2⟨ŷ, r̂⟩` precomputed at encode time. The −‖q_JL‖² term is
omitted: it is constant per query and cannot change the order within a query.

**Status: MATCH for ranking.** The returned *distances* differ from the paper's
by +‖q_JL‖² and are not metric values; anything reported as a distance must say
so. **P1 for reporting, not for semantics.**

**Tested by.** Differential test F: composite distances against the reference
computed with the ‖q_JL‖² term, checking the difference is constant per query.

---

## 13. Final top-k

**Paper.** Algorithm 1, line 8: select top-k from 𝒵 by composite distance.

**Implementation.** `batched_topk_final_kernel`, k rounds of a block-wide
minimum over the ck composite distances. Exact for k ≪ ck.

**Status: VERIFIED, after a fix.** Ties used to go to whichever candidate the
reduction reached first, which varies with block size and scan order: on vogue
at nprobe=32 the tenth and eleventh composite distances are bit-identical at
1.7044189 and the two residual paths returned different tenth neighbours for
that tie. Ties now go to the lower database id in both the per-thread scan and
the block reduction, the rule the reference uses.

**Result.** Over 18 configurations, the returned top-k is the exact top-k of
the composite distances, max |reference - returned| **0.000e+00**.

---

## 14. Numerical modes that can change ordering

| Mode | Same operation? | Can reorder? | Status |
|---|---|---|---|
| fp32 rotation | yes | — | strict |
| TF32 rotation | yes | yes, 10-bit mantissa | P1, off by default |
| fp32 primary table | yes | — | strict; also 11% faster than half |
| fp16 primary table | Eq. 6 summed after rounding each term to 11 bits | yes | P1 |
| fused residual | same value, no materialisation | no (bit-identical with `__fmul_rn`) | strict |
| materialised residual | the paper's own O(d·K_r) construction | no | strict |

The strict validation configuration is: fp32 rotation, fp32 primary table, no
TF32, PM = M, exact top-αk, and either residual path.

---

## What the differential test asserts

Run on one index — the same Π, codebooks, IVF centroids and encoded database
for both sides, so nothing is attributable to retraining.

All discrete checks below returned zero mismatch on the runs recorded above.

| Check | Criterion |
|---|---|
| A rotated query | max abs err ≤ 1e-4 relative |
| B primary codes | exact equality |
| C residual codes | exact equality |
| D primary distance per candidate | max abs err ≤ 1e-3 relative |
| E top-αk id set | **mismatch = 0** |
| F composite distances | difference from reference constant per query |
| G final top-k ids | mismatch = 0 except documented ties |

E is the acceptance criterion that the faithful claim rests on.
