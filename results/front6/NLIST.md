# nlist has an optimum, it is not the one used, and it moves with recall

`scripts/run_nlist6.sh`, raw in `nlist6.log`. Four nlist per dataset, each with
a training set at ~39 points per centroid, nprobe 8-1024 capped at nlist,
Br=8, v47. 120 rows, none failed.

## The optimum

| dataset | used in the report | best mid-front | best at the top | what it was worth |
|---|---|---|---|---|
| vogue-768 | 4096 | 4096 | **2048** | +7% at the top |
| arxiv-768 | 8192 | 8192 | **16384** | **+38%** at the top |
| bge-m3 | 16384 | **32768** | **32768** | +5 to 9% throughout |
| stella | 32768 | 32768 | **65536** | **+139%** at the top |
| openai3-1536 | 4096 | **8192** | 8192 | +4 to 7% |
| openai3-3072 | 4096 | 8192 | 4096 | +8% at the low end |

**It moves with recall on four of six.** A single nlist per dataset is
leaving throughput on the table at one end or the other, and the crossings are
not small -- stella's top point is 12,819 QPS at nlist=65536 against 5,355 at
32768.

## There is a ceiling, and vogue is past it

vogue at nlist=8192 collapses to 84,896 QPS where 4096 gives 186,239 -- **2.2x
slower at identical recall**. 1M vectors over 8192 lists is 122 vectors a list,
and at that size the per-probe overhead (the probe cursor, the boundary search,
a coarse scan over 8192 centroids to pick 8) costs more than the scan it is
directing. More lists is not monotonically better; the useful range ends when
lists stop being long enough to amortise opening them.

## bge-m3's optimum was not bracketed

32768 wins at every recall level measured and is the largest value tried. The
optimum on that dataset is at or above it and is still unmeasured.

## The sigma fix, verified in a real configuration

stella at nlist=65536 trains on 39 x 65536 = 2,555,904 rows, which is past the
2,097,152 where `n * d_` wraps `int` at d=1024. It returns 0.9952 at 12,819
QPS. Before `results/v46_sigma/`, that configuration returned 0.48.
