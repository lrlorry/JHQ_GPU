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

## The likely cause, and what it would take

`libcuvs/include/cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp`
says these kernels are assembled at run time through JIT-LTO. The wheel is
built for the architectures RAPIDS ships; this card is **sm_120**, and a
missing fragment for it would fail exactly this quietly -- the call returns,
the timing is real, the output is whatever was in the buffer.

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
