# The primary LUT, factorised: +6 to +14% and the half-table question closes

Equation 4 builds each subspace's K codewords as the Cartesian product of L
per-dimension levels, so a code byte is Ds base-L digits and

    T_m[c] = sum_j (q_j - level[digit_j(c)])^2

splits along any partition of the Ds dimensions. Splitting it in half:

    T_m[c] = Thi_m[c >> 4] + Tlo_m[c & 15]

exact whenever K = 256 and Ds is even -- then the high nibble carries the first
Ds/2 digits and the low nibble the rest. 32 entries a subspace instead of 256.

`jhq_v47_split_lut/`, `scripts/run_v47.sh`, raw in `v47.log`. BLOCK=1024,
1000 queries, three repeats.

## Three things get smaller at once

| | 256-entry table | factorised |
|---|---|---|
| table, M=96 / M=384 | 96 / 384 KiB fp32 | **12 / 48 KiB** |
| build work | `B*M*256` entries of Ds terms | `B*M*32` of Ds/2 -- **16x less** |
| shared-memory banks | addresses `m*256 + c`, 8 per bank -> up to 8-way conflict | `m*32 + h`, `h` in 0..15, one address per bank -> **none** |
| per candidate per subspace | 1 load, 1 add | 2 loads, 2 adds |

The last row is the cost, and it is paid on the cheap side: the loads are
conflict-free shared rather than an 8-way-conflicting shared or a global read.

## Measured, 28 cells, none negative

| | nprobe=8 | 32 | 128 | 256 |
|---|---|---|---|---|
| **alpha=100** | | | | |
| vogue-768 | +6.7% | +6.5% | +7.8% | +7.6% |
| bge-m3 | +6.8% | +8.3% | +8.9% | +8.8% |
| stella (nlist=16384) | +6.3% | +7.7% | +7.8% | +7.0% |
| stella (nlist=32768) | +5.9% | +6.7% | +7.0% | +6.4% |
| **alpha=8 (the paper's)** | | | | |
| vogue-768 | **+13.6%** | **+12.7%** | **+11.0%** | +9.6% |
| bge-m3 | **+14.5%** | **+10.9%** | +9.5% | +8.7% |
| stella | **+12.4%** | **+11.1%** | +8.5% | +7.8% |

It helps most where the scan is shortest, which is the LUT build being 16x
cheaper: at alpha=8 there are fewer candidates to amortise the build over, so
its share is larger and cutting it shows more.

Recall agrees to 5e-4 or better in every cell and is as often above as below
(0.9849 -> 0.9847 on vogue at nprobe=128; 0.9902 -> 0.9904 on stella at
nlist=32768). The identity is exact in real arithmetic; the two halves are
summed separately, so the last bits differ.

## The precondition is checked, not assumed

`verify_split_lut()` runs at train time and compares the centroids themselves:
for every subspace and code, the first Ds/2 coordinates must equal those of
code `(c>>4)<<4` and the last Ds/2 those of code `c & 15`. A Lloyd-refined
product quantiser (`JHQ_PAPER_CODEBOOK=0`) does not have the property -- its
centroids move independently -- and would give wrong distances silently. The
check throws instead, and names v46 as the version to use for that codebook.

## This closes the `__half` question by removing it

`results/lut_policy/` spent a 16-cell grid on whether a `__half` table in
shared beats an fp32 table in global, and could not separate the dtype from
the residency because `ex_lut` picks residency by whether the table fits. The
factorised table is 12-48 KiB: **it fits in fp32 at every M this project
runs**, so residency is no longer a choice and there is nothing left to buy by
giving up mantissa. `JHQ_LUT32` and the second table type are gone from v47.

The open question that motivated a further experiment -- "half in global, to
separate precision from placement" -- is therefore moot for the head. It would
still answer what the half was worth on the 256-entry table, but the 256-entry
table is not what the scan reads any more.
