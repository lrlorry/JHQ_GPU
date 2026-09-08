# The top of the front cannot be compared by interpolating two single builds

At high recall the front is nearly vertical, and recall varies between builds:
`results/front6/` measured a 1.0e-3 build-to-build floor on vogue. In a
vertical region 1e-3 of recall is worth 2-3x of apparent QPS, so interpolating
one front against another there reads noise as a result. It produced two
spurious readings on the way: a "0.32x regression" on vogue and "0.19x" on
arXiv, both from comparing an endpoint against an interpolation.

## Three builds per configuration (`topband.log`)

The spread is in recall, not throughput:

| | nlist / nprobe | recall spread over 3 builds | QPS spread |
|---|---|---|---|
| vogue-768 | 1024 / 512 | 4e-4 | 1.4% |
| vogue-768 | 1024 / 1024 | 7e-4 | 0.5% |
| arxiv-768 | 2048 / 512 | **1.0e-3** | 0.7% |
| stella | 16384 / 512 | 2e-4 | 0.7% |
| openai3-1536 | 1024 / 512 | 4e-4 | 0.7% |

## The protocol that does work: matched configuration, matched nprobe

No interpolation, median of three builds against the published single build at
the same nlist and nprobe:

| dataset | nlist / np | v39 recall | v39 QPS | v47 median recall | v47 QPS | speedup |
|---|---|---|---|---|---|---|
| vogue-768 | 1024 / 512 | 0.9950 | 11,536 | 0.9944 | 14,566 | **1.26x** |
| vogue-768 | 1024 / 1024 | 0.9954 | 3,776 | 0.9948 | 4,272 | **1.13x** |
| arxiv-768 | 2048 / 512 | 0.9918 | 7,465 | 0.9902 | 8,849 | **1.19x** |
| arxiv-768 | 2048 / 1024 | 0.9920 | 2,838 | 0.9899 | 3,130 | **1.10x** |
| openai3-1536 | 1024 / 512 | 0.9959 | 6,273 | 0.9951 | 7,343 | **1.17x** |
| openai3-1536 | 1024 / 1024 | 0.9966 | 2,593 | 0.9959 | 2,881 | **1.11x** |
| stella | 16384 / 512 | 0.9958 | 4,688 | 0.9954 | 5,240 | **1.12x** |

**1.10x to 1.26x, with no interpolation anywhere.** That is the code speedup
alone at the top -- the configuration change is not in it, since nlist and
n_train are held at the published values.

## The recall deficit is equation 4, and it was already recorded

v47 sits 4e-4 to 1.6e-3 below the published rows in every one of those cells,
consistently and beyond the build spread. That is not a regression from any
change made here. `results/frontier_v39/WHY_RECALL_MOVED.md` located it
already: the published fronts were built at `7298f5c`, before
`build_analytical_cartesian` existed, so their primary level is a
Lloyd-refined product quantiser rather than equation 4's Cartesian product.
The primary level alone is 0.027 better there; the 2000-iteration residual
codebook lifts the equation-4 build further and closes it to about 0.004 in
the finished system, which is the size seen here.

**So the comparison is not code-against-code.** It is faithful-to-the-paper
against a stronger non-paper codebook, and the faithful one is 1.10x to 1.26x
faster for 4e-4 to 1.6e-3 of recall.
