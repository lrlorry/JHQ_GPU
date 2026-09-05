# Moving the residuals off disk, and a premise that was wrong

Same box, same run, RTX 5090 / 96.6 GB cgroup / 264 GB scratch. Raw:
`v37_run.log`.

## The disk, measured rather than inferred

`dd` with `oflag=direct` / `iflag=direct` on `/root/autodl-tmp`:
**write 242 MB/s, read 1.3 GB/s.**

v36's residual pass was reported here as "203 GiB at 1.27 GB/s, disk-bound".
That attribution was wrong twice over. The 1.27 GB/s was the sum of four host
stages in series, not a device rate; and the spill's buffered write and read
never touched the disk at that rate anyway, because 96 GB of page cache
absorbed them. The direct-I/O numbers above are what the disk actually does.

## What v37 changed, isolated

stella, all 17.8M vectors, M=128, `train` is the whole training phase:

| | train | recall |
|---|---|---|
| v36: host transpose, spill | 161.4 s | 0.9914 |
| v37: GPU transpose, spill | 145.2 s | 0.9916 |
| v37: GPU transpose, host memory | **110.8 s** | 0.9914 |

The transpose is worth 16.2 s, dropping the spill another 34.4 s, together
**31%**. bge-m3 goes 74.4 s -> 53.0 s on the same two changes.

**Predicting 172 -> 40 s was too optimistic**, and the reason is instructive.
The spill was costed as real disk traffic; buffered, it was page cache. And
holding 67.3 GiB of residuals *evicts the base's page cache*, so the base read
turns into a genuine 1.3 GB/s disk read -- about 56 s of the remaining 111.
The two changes partially cancel: the memory that stops the spill is the memory
that was caching the input.

## The layout check passed

v37 stores subspace-major from the kernel, which a wrong index would corrupt
silently. Running the same binary with the residuals in memory and forced
through the spill file gives 0.9584 / 0.9584 on bge-m3 (identical), 0.9914 /
0.9916 on stella and 0.9851 / 0.9846 on vogue -- inside the +/-2e-4 to 8e-4 the
IVF assignment's float atomicAdd produces between repeats of one config.

## M=64 could not have OOMed, because M=64 does not exist

v36 derived the chunk count from a device budget, and the reason given was that
stella at M=64 would want 42.4 GiB of a 31.4 GiB card. **It would not, because
equation 4 rejects M=64 before any of that runs:**

```
equation 4 needs Ds = d/M to divide B. Here d=1024 M=64 gives Ds=16,
and B=8 % 16 != 0. Admissible M: 1024, 512, 256, 128.
```

At B=8, `Ds | B` forces Ds <= 8, so `M >= d/8` always, and the chunk is at most
`8*N*8*20` bytes. That exceeds a 31.4 GiB card only above N ~ 26M; stella is
17.8M. So the derived C is prophylactic, not a fix -- it bounds the peak at
~16 GiB independently of N and M, which is worth having, but nothing that ran
here was ever going to fail. At M=256 it uses *more* than the fixed 8 did
(15.8 vs 10.6 GiB).

This also means the paper's own sweep is not fully reachable at B=8: it lists
M in {64, 128, 256} for 1024-d datasets, and 64 is inadmissible under its
equation 4.

## All of Y buys nothing, now measured at the converged iteration count

Same box, 2000 Lloyd iterations both ways:

| | ALL | 100K | train, ALL | train, 100K |
|---|---|---|---|---|
| vogue-768 | 0.9851 | 0.9853 | ~0.44 s | 0.147 s |
| bge-m3 | 0.9584 | 0.9587 | 53.0 s | 0.244 s |
| stella | 0.9914 | 0.9914 | 110.8 s | 0.305 s |

**Two of the three are marginally lower with all of Y**, and all three
differences are inside build noise. On stella that is 363x the training time
for nothing measurable. v30 found this at 25 iterations; it survives at 2000,
which is the setting where the codebook is actually converged.

The 100K column also reproduces v35 (0.9848 / 0.9589 / 0.9914), so nothing in
v36 or v37 moved the sampled path.
