# IVF-RaBitQ on this box returns garbage, and it is the library, not the call

`examples/bench_ivf_rabitq.cu` links `cuvs::neighbors::ivf_rabitq` directly out
of the installed `libcuvs.so`. It builds, it runs, it times -- and every
configuration returns `Recall@10 = 0.0000`.

## What was ruled out

| | how |
|---|---|
| the reader | host brute force over all 1M vogue vectors puts the true nearest neighbour of query 0 at **id 8775, distance 0.629839** -- exactly `gt[0][0]` |
| the ground truth | same line: it agrees with an exhaustive scan |
| the metric | all four of `L2Expanded`, `L2SqrtExpanded`, `CosineExpanded`, `InnerProduct` return 0.0000, at **identical QPS**, so the parameter is not reaching the kernel |
| streaming construction | a host matrix logs "Using streaming construction: dataset size (2.67 GB) exceeds comfortable GPU memory limit" even on a 32 GiB card; uploading first removes the message and cuts the build from 13.4 s to 1.49 s, and recall stays 0.0000 |

## What it looks like instead

- Returned ids are in range and distinct but **65% of them are 0** (3,500 of
  10,000 nonzero on vogue).
- They **change between runs of the same configuration** -- 586992, then
  893503, then 481927 for query 0.
- The distances beside them are **smaller than the true minimum**: cuVS
  reports 0.1787 for the first neighbour where an exhaustive scan of the whole
  base finds nothing closer than 0.629839, and the id it actually returned
  measures 1.092698.

A distance below the true minimum cannot come from the data that was handed
in. Together with the run-to-run variation and the zeros, this reads as
uninitialised memory.

## The sm_120 guess was wrong

`cuobjdump --list-elf libcuvs.so` lists **sm_75 sm_80 sm_86 sm_90 sm_100
sm_120**, and this card is 12.0. CAGRA, IVF-PQ, IVF-Flat and IVF-SQ all go
through the same `jit_lto/` machinery and all of them returned sensible recall
in the same session's re-run. The architecture is not the problem.

## What else was ruled out afterwards

| | how |
|---|---|
| the row count | the file is 2,867,840,928 bytes at 3,076 a row = **932,328 exactly**, and JHQ's own loader prints `base=932328x768`. Both readers agree. |
| an unbuilt index | there is no `extend()` in this API -- only build, search, serialize, deserialize -- and `idx.size()` returns **932,328**, every row |
| the search mode | LUT16 and QUANT4 both return near-random results |
| `bits_per_dim` | 3 (their default) and 9 both do |
| the allocator | cuVS's own harness runs behind a pool and an explicit raft workspace. Setting one up is not possible against this wheel without a further migration: RMM 26.8 has dropped `rmm/mr/device/pool_memory_resource.hpp` for the CCCL-style `cuda/memory_pool` API. |

Recall across all of it: **0.0000 to 0.0006**, where chance alone on
932,328 rows is about 0.0001.

## What is left, and what it would take

Everything testable from outside the library now checks out, and the search
still returns ids that are essentially random with distances **below the true
minimum** -- 0.1787 where an exhaustive host scan of the whole base finds
nothing under 0.629839. A distance smaller than the minimum cannot be computed
from the data that was handed in.

The one structural clue left is that `ivf_rabitq` has **no Python binding in
this release** while every other algorithm here does. The C++ symbols and the
header shipped; the feature did not. That is consistent with a path that is
present but not yet functional in 26.08.01, which is also what the paper
implies -- it names its own fork as the artifact rather than a cuVS version.

Everything else about the comparison is ready: the timed region matches
`demo_jhq_v36.cu` (host queries in, host results out), the byte budgets line up
(`bits_per_dim` counts the 1-bit code, so JHQ Br=4 is 5 and Br=8 is 9), and
OpenAI-3072-1M is the one dataset this project and their paper share.

**What is missing is a libcuvs with sm_120 kernels for ivf_rabitq**, which
means building `github.com/Stardust-SJF/cuvs_rabitq` (branch
`cuvs_ivf_rabitq`) from source. The build flags this file needs are already
worked out in `scripts/run_rabitq.sh`:

- every wheel that ships headers, `rapids_logger` included -- raft's
  `logger_macros.hpp` includes it and it lives in its own package
- the CCCL the RAPIDS wheels vendor under `<pkg>/include/rapids` (3.4.3),
  ahead of the toolkit's, since RMM refuses below 3.3 and CUDA 13.0 here has
  3.0.1
- `g++-12`, because GCC 11's `<atomic>` rejects a type CCCL hands
  `std::atomic`
- `librmm.so` as well as `libcuvs.so`

## One thing the comparison cannot have

Their default batch is **10,000 queries**. Every query file here holds
**1,000** -- openai3-3072's is 12,292,000 bytes at 12,292 bytes a row.
Replicating the set tenfold would let the tables and caches be reused across
duplicates and inflate the throughput, so the comparison has to run at 1,000
on both sides. By their own Figure 4 that is the batch where IVF-RaBitQ's
advantage over CAGRA is smallest, so it is not a setting that flatters JHQ.

## The self-query test, which ends it

Every check above went through a real dataset on disk, and linking the fork's
own `libivf_rabitq.a` ahead of the wheel's `libcuvs.so` changes nothing --
`compile_rc=0`, recall still 0.0000.

`examples/rabitq_selftest.cu` removes the dataset from the question. It
generates its own Gaussian data, uses **rows of the index as the queries**,
and probes **every list**. The correct answer is the row's own id at distance
zero, and it cannot be missed by routing because there is no routing left.

Both link paths, four configurations each (`selftest.log`):

| N | d | nlist | bits | nprobe | index size | self in top-1 | self in top-10 |
|---|---|---|---|---|---|---|---|
| 50,000 | 128 | 256 | 8 | 256 (all) | 50,000 | 0 / 20 | 0 / 20 |
| 50,000 | 128 | 64 | 8 | 64 (all) | 50,000 | 0 / 20 | 0 / 20 |
| 200,000 | 768 | 512 | 8 | 512 (all) | 200,000 | 0 / 20 | 0 / 20 |
| 50,000 | 128 | 256 | 4 | 256 (all) | 50,000 | 0 / 20 | 0 / 20 |

The wheel build and the static build agree to the row. One cell reads 1 / 20,
which is what chance gives.

`build()` reports the right row count every time. Query 0 asks for id 0, whose
distance to itself is 0, and gets id 21,397 at 117.11 back instead.

**So: `cuvs::neighbors::ivf_rabitq` on this card cannot retrieve a vector that
is in its own index, with every list probed, on data it was handed directly.**
Nothing about the call site, the dataset, the reader, the metric, the search
mode, `bits_per_dim`, the architecture or the link path is left to blame.

## What this costs, and what to say instead

A same-machine, same-batch comparison against IVF-RaBitQ is not available
here. The comparison that is available runs through the baseline both papers
share: each method's speedup over CAGRA fp32 measured on its own machine.
PVLDB 19(11) reports 0.8x-8.2x at Recall 0.95 across eight datasets, 3.3x on
average; this work measures 1.19x on openai3-3072, the one dataset in common.
That normalisation assumes CAGRA and the compressed method scale alike across
an L40S and a 5090, which is not verified.
