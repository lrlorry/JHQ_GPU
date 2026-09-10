# `batch_matched_CONTAMINATED.log` — do not use the RaBitQ column

Item 7, the matched-recall batch sweep, ran while the CPU baseline was using
20+ cores and repeatedly reading a 12 GB file. **The JHQ column survived; the
RaBitQ column did not.**

## How it fails

QPS must be non-decreasing in batch and non-increasing in nprobe. RaBitQ
breaks both, on both datasets:

| | | |
|---|---|---|
| vogue-768, np=128 | batch 512 -> 1024 | 140,297 -> **27,802** |
| vogue-768, np=512 | batch 512 -> 1024 | 76,008 -> **18,341** |
| vogue-768, batch 1024 | np=128 -> 256 | 27,802 -> **45,750** (more probes, faster) |
| openai3-3072, batch 1024 | np=32 -> 64 | 32,083 -> **66,379** (more probes, faster) |
| openai3-3072, batch 32 | np=256 -> 512 | 4,723 -> **6,453** (more probes, faster) |

JHQ is monotone in both directions at every point on both datasets.

## What it produced, and why it must not be quoted

Interpolating those points gives **4.38x at vogue-768, batch 1024, R=0.95**,
against 1.05x at batch 512. That number is an artefact of a collapsed RaBitQ
denominator, not a result. It was caught before it reached a figure.

## The likely mechanism, and why it hit one system and not the other

`bench_rq_batch` round-trips the index through `JHQ_RQ_IDX` on disk --
serialize then deserialize, because cuVS leaves `build()` in a layout search
does not read. JHQ touches no file. So host disk contention has a path into
the RaBitQ measurement and none into JHQ's, which is consistent with one
column degrading and the other not. Not verified; the re-run does not depend
on it being right.

## What to do

Re-run on a quiet host: nothing else on the GPU **and** nothing heavy on the
host, because the timed region is host-queries-in to host-results-out. The
same caution applies to `k_and_stages.log`, whose QPS column was collected
under the same conditions and is also non-monotonic; its recall and its stage
*shares* are unaffected and are the columns that file is for.
