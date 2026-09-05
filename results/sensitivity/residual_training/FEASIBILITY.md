# Can the residual codebook train on all of Y at 10M and 17.8M?

Measured, not predicted. B=8, Br=8, M=128, Ds=8, on the frozen faithful target.

## Result

| dataset | n_res | ran | wall | peak host | peak VRAM | train | Recall@10 |
|---|---|---|---|---|---|---|---|
| bge-m3 | 100,000 | yes | 39 s | 40.4 GB | 11,963 MiB | 213.5 ms | 0.9517 |
| bge-m3 | 1,000,000 | yes | 64 s | 43.2 GB | 17,318 MiB | 656.5 ms | 0.9518 |
| bge-m3 | 4,000,000 | **no** | 21 s | 45.8 GB | 16,556 MiB | — | — |
| bge-m3 | **ALL (10,091,524)** | **no** | 3 s | — | 2 MiB | — | — |
| stella | 100,000 | yes | 48 s | 69.7 GB | 20,661 MiB | 272.1 ms | 0.9853 |
| stella | 1,000,000 | yes | 150 s | 73.2 GB | 22,896 MiB | 892.7 ms | 0.9855 |
| stella | 4,000,000 | **no** | 26 s | 69.1 GB | 16,556 MiB | — | — |
| stella | **ALL (17,776,615)** | **no** | 3 s | — | 2 MiB | — | — |

ALL fails within three seconds having allocated nothing, which is the shape of
an allocation refused outright rather than a run that degrades. The largest
size that completes is 1M on both; 4M already fails.

## Why, and whose limit it is

`rotate_on_gpu` allocates `d_x` and `d_y` at N·d·4 each and the residual
trainer needs `d_y` resident, so the rotated set lives in VRAM:

| | rotated set | + residual buffer | our VRAM 31.4 GB | our cgroup 96.6 GB | the paper's 576 GB host |
|---|---|---|---|---|---|
| bge-m3 | 38.5 GB | 77.0 GB | does not fit | fits | fits |
| stella | 67.8 GB | 135.6 GB | does not fit | does not fit | fits |

**The two datasets fail for different reasons and the distinction matters.**
bge-m3's 77 GB fits this machine's host memory; it fails only because this port
puts the rotated set on the device. stella's 135.6 GB exceeds the host budget
as well, so it would fail on a CPU path here too.

**The paper's own platform — 96 cores, 576 GB — has room for both.** So JHQ-CPU
as published can train on all of Y at these scales. What cannot is this GPU
port on a 32 GB card.

## O(N·d) is this implementation's choice, not equation 5's requirement

Equation 5 collects the one-dimensional residual values of each subspace and
runs 1-D k-means on them. The state that needs is O(M·K_r) plus a histogram —
not O(N·d). A batch's residuals can be accumulated and the batch discarded;
`launch_residual_codebook_hist` already replaces the sort with a histogram,
though it still wants `d_y` whole.

So "ALL is impractical" is a statement about this port, and writing it as a
statement about the algorithm would be wrong. A streaming residual collector
would remove the limit. That is a redesign, deliberately not attempted here,
and it is recorded as a known and reachable follow-up rather than as a defect
of JHQ.

## Consequence for the protocol

ALL is unavailable on this hardware for the two datasets that most need it, so
a single protocol across all six datasets cannot be ALL. It is **100K**, with
the sensitivity study as the evidence that this costs at most 0.0018 Recall@10
and that 250K/500K buy nothing.

The sanity points here agree: 100K against 1M moves Recall@10 by 0.0001 on
bge-m3 (0.9517 → 0.9518) and 0.0002 on stella (0.9853 → 0.9855), while the
build cost rises from 213 ms to 657 ms and from 272 ms to 893 ms.
