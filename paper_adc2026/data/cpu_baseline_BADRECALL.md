# `cpu_baseline_BADRECALL.log` — the QPS is usable, the recall is not

The first complete CPU baseline. Twelve invocations, all rc=0, vogue-768 and
openai3-3072 at the GPU's parameters. **Its recall column measures a different
metric from the GPU side and no matched-recall comparison against it means
anything.**

## The defect

`bench_vogue768.cpp`'s `recall_at_k` scanned the whole ground-truth row:

```c
for (int p = 0; p < k; p++)
    for (int g = 0; g < gt_k; g++)          // gt_k is the file's width
        if (pred[i*k+p] == gt[i*gt_k+g]) { hits++; break; }
```

`gt_k` is however wide the file happens to be — **100 on vogue-768, 20 on
openai3-3072** — so a returned neighbour counted as correct whenever it landed
in the true top-100. That is Recall@10 against the true top-100, which is a
much easier metric than Recall@10.

The GPU path has measured this correctly since `common/recall.cuh`, whose
header says so explicitly: "standard set-intersection against the true top-k,
not against the whole ground-truth row". The CPU bench still had the old loop.

## How it surfaced

The two recall curves could not both be right:

| | vogue-768 |
|---|---|
| CPU, this log | 1.0000 from nprobe=16 |
| GPU, `paper_fronts.log` | 0.9939 at nprobe=1024 |

A quantised index does not reach perfect Recall@10 at nprobe=16 on a 932K-vector
set. Comparing the two curves is what caught it.

## Which direction it errs

**Against us.** It lets the CPU reach high recall at low nprobe, so any
GPU-over-CPU speedup computed at matched recall from this log is understated.
That does not make it safe to keep: a reviewer laying the two recall curves
side by side sees they disagree, and then doubts the rest.

## What survives

**The QPS column.** Search work does not depend on how recall is scored
afterwards, so the throughputs and the thread sweep conclusion stand:

* 208 threads segfaults (rc=139)
* 64 threads is clearly worse than both 32 and 16
* 32 and 16 are inside the noise of a single timing

Fixed by intersecting against `min(k, gt_k)`. Re-run is `cpu3.sh`, which also
waits for the GPU sweep to finish before touching the CPUs -- see
`batch_matched_CONTAMINATED.md` for why that matters.

## The second defect found in this file

`read_fvecs` sized its buffer as `*n * dim` with both operands `int`, which
overflows at d=3072 (999000 x 3072 = 3,069,072,000). See
`../../results/cpu_baseline/README.md`. Both defects were invisible until the
binary met data it had not been run against.
