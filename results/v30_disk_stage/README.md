# v30 disk-stage: residual codebook trained on ALL of Y

`jhq_v30_disk_stage/` computes residuals in **one pass** over the base
(rotate -> primary encode -> residual, all on device), spills them
subspace-major to `/root/autodl-tmp/jhq_resid_spill.bin`, then reads back
chunks of `JHQ_RES_CHUNK` (default 8) subspaces and runs the **exact
sorted-Lloyd** estimator on the device.

RTX 5090 (32,111 MiB), `JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1`.
Raw log: `v30_run.log`.

| dataset | N | n_res | res-train | add | Recall@10 | QPS |
|---|---|---|---|---|---|---|
| vogue-768  | 932,328    | 100,000    | 5985.1 ms* | 244.4 ms   | 0.9771 | 21550 |
| vogue-768  | 932,328    | **ALL**    | 416.9 ms   | 247.5 ms   | 0.9773 | 21099 |
| bge-m3     | 10,091,524 | 100,000    | 201.9 ms   | 1772.3 ms  | 0.9516 | 8201  |
| bge-m3     | 10,091,524 | **ALL**    | 84214.0 ms | 5797.6 ms  | 0.9524 | 8149  |
| stella     | 17,776,615 | 100,000    | 308.5 ms   | 10879.0 ms | 0.9852 | 11408 |
| stella     | 17,776,615 | **ALL**    | 171768.7 ms| 14692.0 ms | 0.9849 | 11395 |

\* first configuration in the script; ~5.8 s of it is one-time CUDA context
and cuBLAS handle creation, not residual training. Compare the other two
100K rows (202 / 309 ms), which run later in the same process-per-config
script and pay only their own cost.

## ALL buys nothing measurable

Recall deltas ALL - 100K: vogue +2e-4, bge +8e-4, stella -3e-4. Index build
is nondeterministic at +/-2e-4 (float `atomicAdd` in the IVF assignment), so
only bge's +8e-4 is even marginally outside build noise, and it is well
inside the spread of exact Lloyd under different initialisations.

Cost of ALL on stella: **172 s** and **67.8 GB** of scratch disk, against
**309 ms** and nothing for the 100K sample.

## Resource profile

Spill sizes: vogue 2.7 GB, bge 38.5 GB, stella 67.8 GB (`N*d*4`), against
265 GB free on `/root/autodl-tmp`. Root has only 7.4 GB, hence `JHQ_RES_SPILL`.

Predicted device peak during the estimator is
`C * N * Ds * 16 bytes` (values + sort copy + double prefix sum) =
0.89 / 9.62 / 16.95 GB for vogue / bge / stella at C=8. This prediction is
**not** confirmed by the `VRAM used` column, which is measured after `add`
and reports the resident index, not the training peak; all six runs
completed without OOM, which bounds the peak below 32,111 MiB but no
tighter. Halving `JHQ_RES_CHUNK` halves the peak and doubles the number of
chunks, and chunks re-read the spill file rather than the base, so the
trade is linear.

`cub::DeviceSegmentedRadixSort` takes 32-bit offsets: `C*N*Ds` is 1.14e9 at
C=8 on stella, inside `2^31`; C=16 would overflow and the code rejects it.
