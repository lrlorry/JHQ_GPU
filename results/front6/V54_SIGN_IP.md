# v54: the primary distance without a table, measured

At L=2 the two levels of equation 3 are `+/- sigma*sqrt(2)*erfinv(0.5)` —
symmetric, because equation 3 puts them at the quantiles of a zero-mean
Gaussian — and equation 4 makes the codeword a Cartesian product of them. So

```
D_P(q,y) = ||q'||^2 + d*a^2 - 2a<q', s_y>,   s_y in {-1,+1}^d
```

is an identity. The point of building it was shared memory: the factorised
table needs `M*32` floats resident a query, the permuted query needs `M*8`.
At M=384 that is 49,152 B against 12,288, and `NEGATIVES.md` prices 16 KB a
block at 17–26% wherever occupancy is not already pinned at one.

Both builds share a cache directory per dataset, so the trained state is
bit-identical and any recall difference is the scan's. Raw in `v54.log`.

| dataset | M | nprobe | LUT recall | sign | LUT QPS | sign QPS | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| vogue | 96 | 128 | 0.9644 | 0.9644 | 137,689 | 84,914 | −38.3% |
| bge | 128 | 128 | 0.9184 | 0.9184 | 102,294 | 68,291 | −33.2% |
| stella | 128 | 128 | 0.9890 | 0.9890 | 86,126 | 56,354 | −34.6% |
| openai3-3072 | 384 | 128 | 0.9418 | 0.9417 | 40,969 | 28,486 | −30.5% |
| vogue | 96 | 512 | 0.9902 | 0.9902 | 73,931 | 37,334 | −49.5% |
| bge | 128 | 512 | 0.9667 | 0.9667 | 37,533 | 23,389 | −37.7% |
| stella | 128 | 512 | 0.9946 | 0.9946 | 28,333 | 17,310 | −38.9% |
| openai3-3072 | 384 | 512 | 0.9820 | 0.9819 | 19,959 | 10,866 | −45.6% |
| vogue | 96 | 1024 | 0.9933 | 0.9933 | 57,560 | 27,570 | −52.1% |
| bge | 128 | 1024 | 0.9820 | 0.9820 | 25,844 | 14,197 | −45.1% |
| stella | 128 | 1024 | 0.9954 | 0.9954 | 17,208 | 9,833 | −42.9% |
| openai3-3072 | 384 | 1024 | 0.9932 | 0.9931 | 11,962 | 6,004 | −49.8% |

## The identity holds

**Recall is equal to four decimals in nine of the twelve pairs**, and the
three that move by 1e-4 are all openai3-3072 — d=3072, the longest summation
and so the one where reordering it shows. That is what an identity computed a
different way looks like, and `verify_sign_levels()` passing on all four
datasets says the `+/-a` structure it rests on is real and not an assumption
about how the encoder happens to be written.

## And it costs 30 to 52%

**On the dataset it was aimed at.** openai3-3072's table is what pins that
dataset at one block an SM: `16,392 + 49,152 = 65,544 B` of the 101,376 this
card allows. The sign form needs `16,392 + 12,288 = 28,680`, which fits three.
**Occupancy tripled and it still lost 30 to 50%.**

**The loss grows with nprobe** — −30 to −38% at 128, −43 to −52% at 1024 —
so it is the inner loop's instruction count, not a fixed cost. More
candidates, more of it.

## Why, in hindsight

**The factorised table *is* the optimal precomputation of this inner product.**
`Thi_m[h]` and `Tlo_m[h]` are partial sums over four dimensions each; the sign
form is the same arithmetic with that precomputation removed. It can only save
space and only lose instructions, and at these sizes space is not what binds.

The estimate that said otherwise was wrong in a specific way. "The scan runs
at about 7% of this card's arithmetic throughput, so there is headroom" is a
statement about FLOP/s. The inner loop is not FLOP-bound: it is shared-memory
loads and predicated adds, so what binds is issue slots and LSU throughput —
and 32 shared reads a word against the table's 8 is 4x on exactly the
resource that was already the limit.

## What it also settles

The cluster-centric proposal's remaining case was that dropping per-query
residency from 4d floats to d frees the shared memory a multi-query block
needs. That relief is now measured on its own: **on the one dataset where
shared memory demonstrably binds, taking occupancy from one block to three
bought −30% rather than a gain.** Together with `QDUP.md` — the reuse it would
capture is worth +4 to +26%, and 87% of that is already served by L2 — both
halves of the argument now have a number, and both are negative.
