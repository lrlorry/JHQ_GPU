# The training set matters below about 20 points per centroid, and costs above it

`results/v46_sigma/` measured +50 to +96% from raising `JHQ_N_TRAIN` on
stella and attributed it to six datasets. Separated at fixed nlist on the other
five (`scripts/run_final3.sh`, raw in `final3.log`), it does not generalise --
and on vogue it reverses.

Matched-recall QPS, `JHQ_N_TRAIN` 640,000 against 100,000, nlist held fixed:

| dataset | N | nlist | points/centroid at 100K | low recall | mid | high |
|---|---|---|---|---|---|---|
| **vogue-768** | 1.0M | 4,096 | 24.4 | **-47%** | **-39%** | -8% |
| openai3-1536 | 999K | 4,096 | 24.4 | +9% | +9% | +14% |
| openai3-3072 | 999K | 4,096 | 24.4 | +4% | +9% | +14% |
| arxiv-768 | 2.25M | 8,192 | 12.2 | +19% | +19% | +53% |
| bge-m3 | 10.1M | 16,384 | 6.1 | **+42%** | **+60%** | +58% |
| stella | 17.8M | 32,768 | 3.1 | +50% | +96% | +75% |

**The gain is monotone in points per centroid, and crosses zero around 20.**

## The mechanism is in the candidate counts, and it runs both ways

| | candidates scanned, 100K -> 640K |
|---|---|
| stella, 6.1 pts/centroid | 226,410 -> 146,535 &nbsp; **-35%** |
| vogue, 24.4 pts/centroid | 46,541 -> 90,578 &nbsp; **+95%** |

Opposite directions, at matched recall, which is the whole effect.

Below roughly ten points per centroid k-means is degenerate: most centroids
never receive enough data to move far from where initialisation put them, the
partition is wildly uneven, and a query that opens `nprobe` lists opens some
enormous ones. More training repairs that, the lists even out, and the same
nprobe touches 35% fewer vectors at higher recall. That is the stella result
and it holds on bge-m3.

Above about twenty, the quantizer is already reasonable and more training only
makes it **follow the density more sharply** -- more, smaller centroids where
the data is dense, fewer where it is sparse. Queries land in dense regions by
construction, so they probe exactly the lists that grew. Candidates scanned
rise 95% and the throughput goes with them.

So it is not "a better-trained quantizer is faster". It is that an
under-trained one is pathologically uneven, and the fix is worth a lot until
the pathology is gone.

## The claim that has to be narrowed

`results/v46_sigma/` and the first version of the report say the training set
is worth +50 to +96% and imply it generally. It is worth that on the two
largest datasets here, where 100,000 rows is 0.6% and 1.0% of the base. On the
three around 1M vectors, where 100,000 is 10% of the base, it is worth +4 to
+14% -- and on vogue it is negative.

## bge-m3's nlist optimum stays unbracketed, for a memory reason

nlist=65536 and 131072 both fail in `add()` at
`jhq_gpu_index.cu:2490`, the `n * bpv_` residual-code buffer: 10.09M rows at
`bpv_ = (1024*8+7)/8 = 1024` bytes is 10.3 GB, and the extra centroid copies at
65536 lists push the total past the 32 GB card. So 32768 remains the largest
value measurable on bge-m3 at Br=8, and it is still the best one tried.
