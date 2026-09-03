# The GPU operating point

Measured 2026-09-04 on one RTX 5090, one job at a time. Every row here comes
from the binary at the commit that carries this directory; the numbers that
preceded it were invalid, and the reason is worth stating plainly.

A trained-state cache hit restored the codebooks but not the per-dimension
levels the separable encoder reads, so the first run of a configuration was
correct and every later run that hit the cache wrote garbage codes and returned
recall near 1e-4. Every sweep that set `JHQ_INDEX_CACHE` was measuring that.
The fix rebuilds the levels on load; `asg23_sweep.log` opens with the gate that
now guards it -- the same configuration run twice, which must agree.

## What is here

| file | what it answers |
|---|---|
| `fronts_gpu.csv` | recall against QPS on six datasets, sweeping nlist and nprobe at alpha=32 |
| `alpha_mechanism.csv` | where alpha's knee sits, and what moves it: queries in flight, and block width |
| `codebook_across_ds.csv` | equation 4's analytical codebook against a trained product quantiser, as the subspace widens |
| `residual_sample_size.csv` | how much of the dataset the residual codebook has to be estimated from |
| `raw/` | the logs the four tables were read from |

## Three things they show

**Alpha's knee is not the paper's.** The paper sweeps alpha in {2, 4, 8} and
settles on 4, which is the right answer for a CPU: refinement there is a serial
loop over alpha*k candidates, so its cost is linear in alpha. On the GPU it is
a parallel gather, nearly free until the parallelism runs out, and the knee
moves to 16-32. On Vogue, going from the paper's 4 to 32 buys 6.7 points of
recall for 3.7% of the throughput. Past the knee the cost keeps climbing while
recall does not: 100 -- which earlier runs of ours used -- gives away 13% of
the throughput on Vogue and 5% on Stella for nothing.

The knee is the smaller of two quantities, only one of which is a property of
the card:

    alpha* = min(alpha at which recall saturates, alpha the parallelism affords)

The first is a property of the quantiser and the data, identical on any
machine. The second is what a bigger or smaller card moves, and
`alpha_mechanism.csv` moves it two ways without a second GPU -- by saturating
the card with queries, and by narrowing the block, which caps the retained
candidates at 4*BLOCK and so puts a hard ceiling on the reachable ck.

**Equation 4 holds up where it was not supposed to reach.** The paper's
construction gives every dimension B/Ds bits, a whole number only when Ds
divides B, which confines it to M >= d/B -- 384 subspaces at 3072 dimensions.
Its own experiments sweep M down to 32 there, and Table 3's headline JHQ figure
needs a primary code of about 129 bits. Splitting the bits unevenly (r = B mod
Ds dimensions carry one more) keeps the product at exactly 2^B codewords and
reaches every M, and the construction stays analytical and O(MK). Measured
against a product quantiser trained by Lloyd at the same code length, it is
about a point behind at Ds = 8 to 24, ahead by 3.9 points at Ds = 32 on Stella,
and still running at Ds = 48 and 96 where the trained quantiser fails.

**The residual codebook does not have to see the dataset.** Equation 5 collects
residuals over all of Y and the paper gives the stage O(n*Kr), which is where
its extra minutes over JQ go. But the codebook is estimating the quantiles of a
one-dimensional distribution -- after the JL transformation each dimension is
near-Gaussian and the residual is the mixture the primary codebook cuts out of
it -- and that distribution does not depend on how many vectors are in the
database. A 100k sample already draws S*Ds = 800k scalars per subspace to place
at most 256 levels. Ten times the sample moves recall by 0.0006 on Stella and
costs 5.8x the training time. Encoding still visits every vector; only the
estimation is sampled.
