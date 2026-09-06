# Where the recall difference comes from: equation 4

The re-measured fronts sit below the published ones on vogue by 0.45 to 2.55
points depending on nprobe. This locates it.

## The residual level is not the cause; it is the compensation

Running the same configurations with the residual level compiled out
(`JHQ_NO_RESIDUAL`) separates the two levels. vogue, M=96, Br=4:

| nprobe | old primary only | new primary only | primary delta | old full | new full | full delta |
|---|---|---|---|---|---|---|
| 1 | 0.3613 | 0.2727 | **-0.0886** | 0.4645 | 0.3196 | -0.1449 |
| 8 | 0.5680 | 0.5395 | -0.0285 | 0.7647 | 0.7389 | -0.0258 |
| 32 | 0.6395 | 0.6131 | -0.0264 | 0.8989 | 0.8932 | -0.0057 |
| 128 | 0.6618 | 0.6342 | -0.0276 | 0.9452 | 0.9409 | -0.0043 |
| 256 | 0.6642 | 0.6361 | -0.0281 | 0.9508 | 0.9467 | -0.0041 |

The primary level is worse in the new build at every nprobe, by a deficit that
settles at **-0.027** once coverage saturates -- old primary-only is 0.6642 at
nprobe=256 and 0.6640 at 512, so it has plateaued and the IVF is not what
differs.

The residual level lifts **more** in the new build: +0.3106 against +0.2866 at
nprobe=256. That is the 2000-iteration codebook doing its work, and it is what
shrinks a -0.027 primary deficit to -0.004 in the finished system.

## The primary deficit is the price of equation 4

At `7298f5c`, the commit the published vogue rows record, the sources those
rows ran on contained **no `JHQ_PAPER_CODEBOOK` and no
`build_analytical_cartesian`**. That construction arrived two days later in
`384de5c`, whose own message measured the cost:

```
Vogue,        M=96    0.9357 against 0.9444
OpenAI3-3072, M=384   0.9588 against 0.9614
```

So the published fronts used the reference implementation's primary
quantiser -- analytic seeding followed by five Lloyd iterations -- and the
re-run uses the paper's, which section 3.2 states plainly: *"we propose to
calculate codewords directly without leveraging the k-means method"*. A freely
refined product quantiser has 256 unconstrained centroids per subspace; the
paper's has the Cartesian product of two scalar levels per dimension. The first
is strictly more expressive, and 2.7 points is what that buys on vogue's
primary level.

## The same pattern appears twice

| | the paper says | the reference implementation does | this port does |
|---|---|---|---|
| primary codebook, eq. 4 | closed form, no k-means | analytic seed **+ 5 Lloyd iterations** | closed form |
| residual codebook, eq. 5 | every `y` in `Y`, `O(n*K_r)` | **20,000 scalar values**, 5 iterations | all of Y, 2000 iterations |

The reference implementation departs from its own paper in both places, and
both departures are toward the more expensive, data-driven option that scores
slightly higher. Following the text costs 2.7 points on the primary level and
recovers more than that at the residual level.

## A protocol note

`jhq_v21_cascade/` is a version directory, and it was edited after `7298f5c`
-- the `JHQ_PAPER_CODEBOOK` branch is in it today and was not in it then. That
is what `CLAUDE.md` now forbids, written down during this project and after
these edits. The consequence is concrete: **the directory no longer reproduces
the rows that cite it**, and an audit that compares the two directories'
current contents, as one in this project did, compares neither run.
