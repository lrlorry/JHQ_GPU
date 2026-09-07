# v44-v46: the gap to int8 CAGRA on stella goes from 2.69x to 1.29x

Three changes, none of them to the algorithm. Raw: `v46.log`,
`../v45_compact/v45.log`, `../v44_probe/v44.log`. BLOCK=1024, k=10, alpha=100,
1000 queries, three repeats, `JHQ_DIAG=1` (whose cost is under 0.3% -- v43 with
and without it came back 19,446 against 19,402).

| | what it was | what it is |
|---|---|---|
| **v44** | `select_probes` held all nlist scores in shared and ran nprobe argmin passes; 4*nlist B capped nlist at 16384 | streams the centroids against a running threshold; cap*8 + 8 B, independent of nlist |
| **v45** | both selection buffers sorted as soon as the count passed the target -- a full cap-element bitonic sort per chunk | sorts when the buffer cannot absorb another chunk |
| **v46** | `cublasSdot(n * d_, ...)` overflowed int above 2.1M training rows; sigma came out 0 and equation 4 put every centroid at the origin | chunked at 2^28, accumulated in double |

## v45 is +11 to +38% for two lines

QPS against v43 at the same recall, same configuration (`../v45_compact/`):

| | nprobe=8 | 32 | 128 | 256 |
|---|---|---|---|---|
| vogue-768 | +1.3% | **+22.3%** | **+37.6%** | +28.9% |
| bge-m3 | +3.8% | **+23.8%** | **+21.1%** | +13.8% |
| stella | +2.9% | **+22.4%** | **+17.2%** | +10.8% |

v44 alone was *slower* -- -18% at nprobe=8 on stella -- and that was this bug,
not the streaming: at nprobe=8 with BLOCK=1024 the buffer has 2048 slots and
the old test fired on the ninth survivor, so every chunk paid a 2048-element
sort to absorb two elements. `results/block_sweep/` had already measured the
scan's compaction at 19-31% and rising with the block size.

Recall is bit-identical on bge-m3 and stella and moves by 1e-4 on vogue
(0.9849 -> 0.9848 at nprobe=128). `jhq_compact_topck` compares distance alone,
so among exactly equal distances the order is arbitrary and a different fill
pattern resolves a boundary tie differently. One neighbour in ten thousand.

## v46: the threshold is exactly 2^31, and it was read as an nlist ceiling

nlist=16384, nprobe=128, stella (d=1024), varying only `JHQ_N_TRAIN`:

| rows | n*d | v45 | v46 |
|---|---|---|---|
| 1,800,000 | 1.84e9 | 0.9913 | 0.9913 |
| 2,000,000 | 2.05e9 | 0.9923 | 0.9923 |
| **2,097,152** | **exactly 2^31** | **0.4772** | **0.9915** |
| 2,200,000 | 2.25e9 | 0.4792 | 0.9923 |
| 2,600,000 | 2.66e9 | 0.4826 | 0.9914 |

`ivf_recall` is 0.9959-0.9964 in every one of those rows. Routing never
noticed, because routing uses the IVF centroids and only the primary level
depends on sigma. The failure looked like "large nlist does not work" for two
runs, because nlist and the training set had always been raised together.

The first attempt at v46 reported 0.477 as well: `cache_path()` keys the
trained-state cache on the data and the parameters, not on the code, so it
loaded v45's broken codebook. That run is kept in `v46_cached.log`.

## What the coarse quantizer was costing

100,000 training points over 16,384 centroids is 6.1 per centroid, against the
39 FAISS treats as a floor. Every front this project has published used it.
Matched-recall QPS, stella, nlist=16384, everything else fixed:

| recall | nt=100K | 320K | 640K | 1.3M | 2.6M |
|---|---|---|---|---|---|
| 0.975 | 49,561 | 68,810 | 74,721 | -- | -- |
| 0.985 | 30,592 | 43,500 | 46,373 | **50,595** | 48,440 |
| 0.990 | 24,035 | 34,587 | 36,532 | **38,375** | 37,028 |
| 0.993 | 15,044 | **29,415** | 28,028 | 28,868 | 23,760 |

**+50 to +96%, and it saturates by 640K-1.3M.** The mechanism is in the
`cand` column: at nprobe=128 the scan reads 226,410 candidates with 100K
training points and 146,535 with 1.3M -- **35% fewer**, at higher recall.
Better centroids make the lists more even, and probing 128 of a more even
partition touches less. It is not a subtler quantizer; it is a less lopsided
one.

bge-m3 behaves the same way at matched recall (+44% at 0.95, +32% at 0.97)
while its recall at fixed nprobe falls slightly, which is the same trade seen
from the other side.

## nlist, now that it can be raised

Each with a training set at ~39 points per centroid:

| recall | nlist=16384 | 32768 | 65536 |
|---|---|---|---|
| 0.975 | 74,721 | **81,462** | 78,137 |
| 0.985 | 46,373 | 59,093 | **65,393** |
| 0.990 | 36,532 | **50,329** | 48,124 |
| 0.993 | 28,028 | **33,573** | -- |

## Against int8 CAGRA

int8 CAGRA on stella tops out at R=0.9780 and 95,575 QPS; fp32 CAGRA cannot
build the index at all (67.81 GiB of a 31.36 GiB card). At that recall:

| configuration | JHQ QPS | gap |
|---|---|---|
| as published (nlist=16384, nt=100K, v39) | 35,566 | 2.69x |
| + v45 | 42,883 | 2.23x |
| + a trained coarse quantizer (nt=640K) | 64,757 | **1.48x** |
| + nlist=32768 (nt=1.3M) | 73,983 | **1.29x** |
| nlist=65536 (nt=2.6M) | 74,073 | 1.29x |

**2.69x to 1.29x, with the algorithm untouched.** Above R=0.9780 nothing else
in the comparison runs on this dataset.

None of it came from the scan's arithmetic. Two of the three were bugs -- a
sort scheduled on the wrong predicate and an integer overflow -- and the third
was a training set that had never been sized.
