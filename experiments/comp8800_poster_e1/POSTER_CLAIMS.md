# Poster claims supported by E1

Three claims, each with its exact numbers, what it compares against, where and at what recall, the
measurement protocol, the raw files behind it and the qualifiers it needs. Every number comes from
`E1_matched.csv`, `E1_points.csv` and `E1_TABLES.md`, which `analyze.py` regenerates from `RAW_RESULTS/`.

**Common protocol for all three.**
- **Hardware:** one NVIDIA RTX 5090.
- **Workload:** batch of 1,000 queries per call, k = 10.
- **Timing:** host to host. The clock starts with FP32 queries in host memory and stops when the top-10 ids and distances are back in host memory; query upload, any query preprocessing, search, download and synchronisation are all inside it.
- **Throughput:** median over 5 rounds of 24 calls, after warm-up.
- **Accuracy:** Recall@10 against the same ground truth for every method.
- **Comparison:** each method is compared on its own Pareto frontier over its own parameter grid.
- **Ratios:** "interpolated" means log-QPS interpolated between two measured frontier points, never extrapolated. "Measured" means the best measured point at or above the recall.
- **Datasets:**

  | dataset | vectors | dimensions |
  |---|---:|---:|
  | Vogue-768 | 0.93 M | 768 |
  | arXiv-768 | 2.25 M | 768 |
  | OpenAI3-1536 | 1.0 M | 1,536 |
  | OpenAI3-3072 | 1.0 M | 3,072 |

---

## Claim 1: against graph search (NVIDIA cuVS CAGRA, fp32)

> **On a single RTX 5090, JHQ-GPU answered 1.1–2.4× as many queries per second as NVIDIA cuVS CAGRA (fp32) at the same Recall@10, anywhere between 0.90 and 0.985, on all four embedding datasets tested, while using about one-third of CAGRA's GPU memory.**

Short version for the poster: *Up to 2.4× faster than NVIDIA's CAGRA graph index at the same accuracy, with about a third of the GPU memory.*

**Throughput ratio, JHQ-GPU / CAGRA fp32, Recall@10 0.90–0.985:**

| dataset | interpolated | best measured points |
|---|---:|---:|
| Vogue-768 | 1.27–1.51 | 1.33–1.71 |
| arXiv-768 | 1.10–1.41 | 1.10–1.42 |
| OpenAI3-1536 | 1.75–2.44 | 1.29–2.33 |
| OpenAI3-3072 | 1.58–1.95 | 1.38–2.20 |

**GPU memory:**

| dataset | JHQ-GPU* | CAGRA fp32 index | ratio |
|---|---:|---:|---:|
| Vogue-768 | ≈ 0.9–1.0 GiB | 2.9 GiB | 0.31–0.35 |
| arXiv-768 | ≈ 2.0–2.2 GiB | 6.8–7.0 GiB | 0.29–0.32 |
| OpenAI3-1536 | ≈ 1.8–2.3 GiB | 6.0 GiB | 0.30–0.39 |
| OpenAI3-3072 | ≈ 3.5–3.8 GiB | 11.7 GiB | 0.30–0.33 |

\* JHQ-GPU's figure is the process total minus a 507 MiB CUDA context, and includes its search workspace.

**Source:**
- `E1_matched.csv` (other = "CAGRA fp32");
- `RAW_RESULTS/jhq_*.json`, `RAW_RESULTS/cuvs_*_cagra.jsonl`;
- `figures/fig_recall_qps_*.pdf`, `figures/fig_speedup_*.pdf`.

**Qualifiers:**
- CAGRA fp32 reaches a higher maximum recall (0.992–0.9995) than JHQ-GPU (0.985–0.991).
- At Recall@10 0.99 on arXiv-768, CAGRA is 1.4–2.1× faster.
- One GPU and one batch size only.
- CAGRA was built with cuVS defaults (graph degree 32 or 64, intermediate degree 128).

---

## Claim 2: against the strongest compressed index (cuVS IVF-RaBitQ)

> **JHQ-GPU matched or exceeded the throughput of IVF-RaBitQ up to Recall@10 0.97 on all four datasets: 1.0–1.6× on the two 768-dimensional sets and 1.1–1.9× on the 1,536- and 3,072-dimensional sets. Above Recall@10 0.98, IVF-RaBitQ was faster.**

Short version for the poster: *As fast as or faster than IVF-RaBitQ up to 97% recall, and the margin is larger on higher-dimensional embeddings; IVF-RaBitQ is faster above 98%.*

**Throughput ratio, JHQ-GPU / IVF-RaBitQ, Recall@10 0.90–0.97:**

| dataset | interpolated | best measured points |
|---|---:|---:|
| Vogue-768 | 1.06–1.26 | 1.05–1.37 |
| arXiv-768 | 1.00–1.56 | 1.02–1.59 |
| OpenAI3-1536 | 1.13–1.93 | 0.96–1.80 |
| OpenAI3-3072 | 1.16–1.86 | 1.04–1.74 |

The measured 0.96 on OpenAI3-1536 is a single grid cell at Recall@10 0.96: JHQ-GPU has no measured point between 0.955 and 0.972 there.

**At Recall@10 ≥ 0.98, IVF-RaBitQ was faster** (ratios JHQ-GPU / IVF-RaBitQ, interpolated):

| dataset | Recall@10 | ratio | IVF-RaBitQ faster by |
|---|---|---:|---|
| Vogue-768 | 0.98–0.99 | 0.89–0.98 | 2–12% |
| OpenAI3-3072 | 0.98–0.985 | 0.89–0.95 | 5–12% |
| arXiv-768 | 0.98 | 0.86 | 1.2× |
| arXiv-768 | 0.985 | 0.77 | 1.3× |
| arXiv-768 | 0.99 | 0.44 | 2.3× |
| OpenAI3-1536 | 0.985 | 0.96 | 4% |

On OpenAI3-1536 at 0.98 the interpolated ratio (1.10) and the measured one (0.79) disagree, so no direction is claimed there.

**Storage budget:** the IVF-RaBitQ points that set these ratios used 9 bits per dimension, the setting whose code size matches JHQ-GPU's (1 + 8 bits per dimension), so this is a like-for-like storage comparison. The one exception is OpenAI3-3072 at Recall@10 0.985 and 0.99, where an 8-bit point is 1.7% faster than the best 9-bit one. IVF-RaBitQ's index is 1.0–3.5 GiB, against ≈ 0.9–3.8 GiB for JHQ-GPU.

**Robustness:** IVF-RaBitQ shows rare 30–60 ms search calls. Recomputing throughput as 1000 / median call latency, which ignores those calls, leaves the direction unchanged:

| dataset | ratio at Recall@10 0.94–0.97 |
|---|---:|
| Vogue-768 | 1.06–1.27 |
| arXiv-768 | 1.00–1.06 |
| OpenAI3-1536 | 1.12–1.48 |
| OpenAI3-3072 | 1.10–1.40 |

**Source:**
- `E1_matched.csv` (other = "IVF-RaBitQ" and "IVF-RaBitQ 9-bit", identical);
- `E1_TABLES.md` (Sensitivity sections);
- `RAW_RESULTS/rabitq_*.jsonl`, `RAW_RESULTS/jhq_*.json`.

**Qualifiers:**
- IVF-RaBitQ reaches a higher maximum recall (0.994–0.996).
- Its returned ids vary slightly from call to call (recall ±0.001).
- It was given a 2 GiB workspace pool and pinned query uploads, both in its favour.

---

## Claim 3: where JHQ-GPU leads

> **At Recall@10 = 0.97, JHQ-GPU had the highest throughput of the six GPU methods tested on every dataset (tied with IVF-RaBitQ on arXiv-768). CAGRA int8, the fastest method at Recall@10 0.90–0.92 on all four datasets, cannot reach 0.97 on any of them: its maximum is 0.945–0.968.**

Short version for the poster: *At 97% recall, JHQ-GPU was the fastest of six GPU methods on all four datasets.*

**JHQ-GPU throughput at Recall@10 0.97 (interpolated) and its margin:**

| dataset | JHQ-GPU QPS | over the next fastest | over CAGRA fp32 |
|---|---:|---|---:|
| Vogue-768 | 231 K | IVF-RaBitQ, 1.06× (measured 1.05×) | 1.42× |
| arXiv-768 | 259 K | IVF-RaBitQ, 1.00× (measured 1.02×) | 1.32× |
| OpenAI3-1536 | 127 K | IVF-RaBitQ, 1.13× (measured 1.11×) | 1.75× |
| OpenAI3-3072 | 56 K | IVF-RaBitQ, 1.16× (measured 1.35×) | 1.75× |

The other methods at Recall@10 0.97:
- **IVF-PQ:** 41–117× slower; it cannot reach 0.97 on OpenAI3-3072.
- **Exact GPU brute force:** 25–71× slower.
- **CAGRA int8:** maximum recall 0.9453, 0.9620, 0.9667 and 0.9683 on the four datasets.

**Source:**
- `E1_matched.csv` at recall 0.97;
- `E1_TABLES.md` (frontier ceilings, matched-recall tables);
- `figures/fig_recall_qps_*.pdf`.

**Qualifiers:**
- The margins over IVF-RaBitQ at 0.97 are small: 0–16% interpolated. On Vogue-768 the 5–6% lead is about the size of IVF-RaBitQ's own round-to-round spread (median 2.9–5.3%). Say "fastest or tied fastest".
- At Recall@10 0.90–0.92, CAGRA int8 is faster than JHQ-GPU by 1.2–2.1× on every dataset, and it stays faster up to 0.95 on arXiv-768 and 0.94 on the OpenAI3 sets. On Vogue-768 at 0.94, JHQ-GPU is already faster (1.26×).

---

## What the poster should not say

- **"Faster than every baseline":** not true at low recall (CAGRA int8 is faster at 0.90–0.92 everywhere) or above Recall@10 0.98 (IVF-RaBitQ).
- **The single cells where the interpolated and measured ratios disagree in direction,** e.g. OpenAI3-1536 against IVF-RaBitQ at 0.96 and 0.98.
- **A dimensionality mechanism.** The lead is larger at 1,536 and 3,072 dimensions in this experiment, but JHQ's primary code also grows with d here (M = d/8), so E1 does not show why.
- **Historical numbers.** September's CAGRA/RaBitQ results and the ADC draft's numbers predate v61–v69 and were measured on another protocol. Do not mix them with these.

## Suggested figures

- **Main panel:** `figures/fig_recall_qps_arxiv.pdf` (768-d, the largest dataset, and where IVF-RaBitQ wins at the top) beside `figures/fig_recall_qps_openai3072.pdf` (3,072-d, where JHQ-GPU leads most of the range). Showing one figure where JHQ-GPU loses part of the range keeps the comparison honest.
- **Supporting:** the matching `figures/fig_speedup_*.pdf` ("above 1 = JHQ-GPU faster") for a non-expert audience.

All figures are vector PDF/SVG with large fonts.
