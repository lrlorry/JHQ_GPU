# The scan kernel wants occupancy, not locality

> **Superseded. This file's verdict was an artefact of `JHQ_BLOCK`'s default
> of 256, which is not the block size the fronts run.** Re-measured at 1024,
> the `__half` table in shared memory is **8% FASTER**, not 65-89% slower, at
> identical recall. See `results/block_sweep/`. The L1 carveout row below
> survives as a negative at both block sizes. Everything below is kept because
> the mechanism it names is the one that turned out to be operating -- read it
> with "at BLOCK=256" attached to every number.

`scan_ivf_exact_kernel` is 91% of search (`results/v38_refine_share/`), and it
reads its per-query table out of global memory: with a float `lut_t` the table
is `M*256*4` = 131,072 B against this card's 101,376 of opt-in shared, so
`ex_lut` is false on every dataset. A `__half` table is 65,536 and fits.

Two changes were made on that basis and both were wrong. Latency for 1000
queries, mean of three interleaved repeats:

| | vogue-768 | bge-m3 | stella |
|---|---|---|---|
| **v39: float table, global** | **45.29 ms** | **119.71 ms** | **85.52 ms** |
| `__half` table, shared | 85.59 ms **+89%** | 197.74 ms **+65%** | 151.66 ms **+77%** |
| float, global, L1 carveout | 96.38 ms **+113%** | 225.24 ms **+88%** | 172.76 ms **+102%** |
| `__half`, shared, carveout | 85.65 ms +89% | 197.61 ms +65% | 151.65 ms +77% |

Recall is unchanged across all four -- 0.9845-0.9851, 0.9583-0.9591,
0.9908-0.9916 -- so `__half`'s eleven mantissa bits, summed M=128 times, cost
nothing measurable. The precision worry was unfounded. The cost is entirely
occupancy.

## Both changes cut blocks per SM

The SM has 100 KB of unified L1/shared. A block already needs `ex_base` =
16,392 B for the candidate scratch:

- **`__half` table in shared** takes the block to 81,928 B, so **one block per
  SM**. The scan is latency-bound on the strided `list_primary_t` reads and has
  nothing left to hide them with.
- **`PreferredSharedMemoryCarveout = 0`** asks for the smallest shared
  partition. The block still needs its 16,392 B, so the driver sizes the
  partition to fit one or two blocks where the default fitted several. Same
  loss, arrived at from the opposite direction.

The fourth row is the control: with the table already in shared there is
nothing for the carveout to protect, and it does nothing, which is what the
code intended.

## This is the opposite of the refine kernel

The same carveout won 2.2% on `residual_refine_fused_kernel`
(`results/v38_refine_share/`). That kernel is 7% of search, its shared use is
small, and it is bound by repeated reads of a 128 KB codebook -- locality is
what it lacked. The scan kernel is the throughput-critical one, streams
`list_primary_t` once, and is bound by having enough warps in flight. **The
same knob helps one and halves the other**, and nothing about the first
measurement predicted the second.

## What this settles

An earlier claim that "fp32 LUT is 10% faster" was retracted, because the run
behind it had `d_lut_r` sized for halves and filled with floats and so measured
an overrun. Made properly, fp32 is 65-89% faster -- but for the occupancy
reason above, not for precision or arithmetic. v39 stays the head; v40 keeps
its directory as the record of a measured negative.

## Where the scan time actually goes is still open

91% of search in one kernel, ~29 cycles a table lookup, and the two obvious
cache hypotheses are now both disproved. What remains untested is whether it is
bound by the strided `list_primary_t` gather, by the threshold compaction that
shares the same block, or by instruction issue. `ncu` is installed on the box
and would answer it directly; nothing here should be guessed at again.

---

## The shared-memory result was measured at one BLOCK, and it is the worst one

`v39_lut16` put a `__half` table in shared memory and came out 65-89% slower.
That run set `JHQ_GPU_CODEBOOK`, `JHQ_ENCODE_GROUPED_OFF` and
`JHQ_Y_TRANSPOSED` and **not `JHQ_BLOCK`**, so it ran at the default of 256.
The frontier runs at 1024.

When shared memory is what limits residency, a larger block is strictly better:
the SM has 100 KB, a block asking 82 KB leaves room for exactly one block
either way, so the threads per SM are the block size.

| BLOCK | blocks/SM at 82 KB | threads/SM | of 1536 |
|---|---|---|---|
| **256** | 1 | 256 | **17%** |
| 512 | 1 | 512 | 33% |
| **1024** | 1 | 1024 | **67%** |

So the configuration that was measured is the one where a shared-memory table
costs the most occupancy, by a factor of four against the one the fronts
actually use. **The conclusion "the scan wants occupancy, and shared memory
takes it away" is supported only at BLOCK=256.**

### Where the table fits at all

`scan_base = cap*8 + 8` with `cap` the power of two at or above `ck + BLOCK`,
and the table has to fit in what is left of 101,376 B:

| alpha | ck | BLOCK | scan_base | left for LUT | max M, fp32 | max M, half |
|---|---|---|---|---|---|---|
| 100 | 1000 | 256 / 512 / 1024 | 16.0 K | 83.0 K | **82** | 165 |
| 8 | 80 | **256** | **4.0 K** | **95.0 K** | **94** | 189 |
| 8 | 80 | 512 | 8.0 K | 91.0 K | 90 | 181 |
| 8 | 80 | 1024 | 16.0 K | 83.0 K | 82 | 165 |

The measured configurations are M=96 (vogue) and M=128 (bge-m3, stella), so an
fp32 table never fits. At the paper's `alpha=8` with BLOCK=256, vogue's M=96
**misses by two subspaces** — 96 against a ceiling of 94.

`ck = alpha*k` is therefore a second lever on this question, and alpha has never
been run at the paper's range either (`results/parameter_coverage/`).

### What is untested

| | measured | result |
|---|---|---|
| half table, shared, **BLOCK=256** | yes | 65-89% slower |
| half table, shared, **BLOCK=1024** | **no** | — |
| fp32 table, shared (needs M ≤ 82, or a smaller ck) | **no** | — |

Re-running `demo_jhq_v39_lut16` with `JHQ_BLOCK=1024` closes the first gap and
takes about ten minutes. Until then the shared-memory verdict should be quoted
with its block size attached.
