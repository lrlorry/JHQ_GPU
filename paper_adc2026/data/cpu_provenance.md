# CPU baseline provenance — what it is, and what it is not

Read this before quoting any GPU-over-CPU number.

## It is not a reproduction of the published JHQ figures

`p1530-han.pdf` §5 measures on an **AMD EPYC 9654 @ 3.7 GHz (96 cores / 192
threads), 576 GB DDR5**, and states: *"construction was parallelized over 32
threads, and **search performance was measured in a single-thread setting**."*

Four protocol differences, and they do not all point the same way:

| | published JHQ | our CPU baseline | direction |
|---|---|---|---|
| search threads | **1** | 16 and 32, best taken | **generous to CPU** — conservative for us |
| alpha | **{2.0, 4.0, 8.0}** (tuned) | 100 (first run) | **generous to us** — see below |
| nprobe | ≤ 128 | ≤ 1024 | wider, not unfair |
| primary code length | 128 bit at openai3-3072 / R=0.90 | M=384, i.e. 3072 bit | different index |
| host | AMD EPYC 9654 | Intel Xeon Platinum 8470Q | different machine |

So the paper must not say it reproduces the published numbers. It builds a CPU
baseline from the authors' artifact **at the GPU's parameters**, which is a
different and defensible thing, and says so.

## The one number that does line up

Published: **154 QPS** on openai3-3072 at 90% recall, single-threaded.
Ours: **1199 QPS** at the same dataset and recall, best of 16/32 threads.

**7.8x**, which is what a memory-bound scan does across 16-32 threads — and it
agrees with our own thread sweep, where 32 threads barely beat 16 because
bandwidth saturates near 16. Two independent observations of the same
saturation.

## The alpha mismatch, which inflates our speedup

The first CPU run used **alpha=100** to match the GPU's FIX arm. That is
internally consistent, and it is not the strongest CPU configuration: at
alpha=100 the CPU refines 1000 survivors a query where the published
configuration refines 40, so it does about 25x more residual work than the
authors would have run.

**The 37x-60x figures computed from that run are therefore inflated and must
not be quoted.** Both alpha and nprobe trade recall for throughput, so the
fair CPU baseline is the **Pareto envelope over both**, which is what
`cpu4.sh` measures (alpha in {2,4,8,16} added to the nprobe sweep already at
100). Expect the CPU frontier to move up and the speedup to land lower.

## What to write in 6.1

1. The baseline is the authors' artifact, built from the user's tree, not a
   reimplementation — and not upstream, which does not build (8 modified
   files, `bench_vogue768.cpp` untracked).
2. Thread count and pinning, with the sweep as justification: 208 threads
   segfaults, 64 is clearly worse, 32 and 16 are inside the noise of a single
   timing.
3. **The published protocol measures search single-threaded; ours does not.**
   Multiple threads is the more generous CPU configuration, so a
   GPU-over-CPU claim made against it is conservative. Say this rather than
   leave a reader to find the difference.
4. The CPU frontier is an envelope over alpha and nprobe, not a fixed alpha.
5. Two defects were fixed in the artifact to run it at all: an int overflow in
   `read_fvecs` at d=3072, and a `recall_at_k` that scanned the whole
   ground-truth row. See `../../results/cpu_baseline/README.md`.
