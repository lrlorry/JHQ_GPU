# Compressed CAGRA: what is and is not reachable

Checked on the machine the results were produced on, not inferred.

| | |
|---|---|
| cuVS | 26.08.01 (Python API) |
| GPU / driver | RTX 5090 / 595.71.05 |

## PQ-compressed dataset (`vpq_dataset`) — **not available**

`cagra.IndexParams` has no compression parameter. Its full signature in this
version is

```
IndexParams(metric='sqeuclidean', *, intermediate_graph_degree=128,
            graph_degree=64, build_algo='ivf_pq',
            nn_descent_niter=20, ivf_pq_build_params=None,
            ivf_pq_search_params=None, ace_params=None, refinement_rate=1.0)
```

`ivf_pq_build_params` configures the IVF-PQ used to *build the knn graph*; it
does not compress the stored vectors.

The standalone quantiser `cuvs.preprocessing.quantize.pq` does produce PQ
codes, and `cagra.update_dataset` accepts a `Dataset`, so the two were tried
together. The index rejects the codes:

```
cuvsCagraUpdateDataset: dtype mismatch between index and dataset
  RAFT failure at c/src/neighbors/cagra.cpp line=1648
```

The graph is built over float32 and `update_dataset` requires the same dtype,
so uint8 PQ codes cannot be swapped in. C++ cuVS has a `vpq_dataset` type; it
is not exposed through this Python API at this version.

## int8 dataset — **available, and used**

`cuvs.preprocessing.quantize.scalar` produces an int8 dataset that CAGRA
builds and searches directly. This is the compressed CAGRA baseline in the
results:

| | bytes per vector |
|---|---|
| CAGRA fp32, degree 64 | 4d + 4·64 = 3328 at d=768 |
| CAGRA int8, degree 64 | d + 4·64 = 1024 at d=768 |
| JHQ M=96 Br=4 | 96 + 384 + 4 = 484 |
| JHQ M=96 Br=8 | 96 + 768 + 4 = 868 |

int8 brings CAGRA to roughly the same budget as JHQ at Br=8 for the first
time, which is the comparison worth reporting. Both are measured with
`cudaMemGetInfo` around the build, not derived from the table above.
