# The JHQ side of the frontier, re-measured on the frozen path

`demo_jhq_v39_exp`, prefix 1/1, 3 repeats, `JHQ_BLOCK=1024`, index cache on.
63 rows, every one `ok` -- the harness rejects a configuration whose QPS spread
across repeats exceeds what an idle card produces, and none did. 2 h 57 min.

The cuVS baselines are untouched by anything that changed and are not re-run;
they come from `results/pre_freeze_v22_s2b1/`.

`OLD_VS_NEW.md` audits what differs between the two builds.

## Ownership

"Exclusive" counts JHQ points above the best recall any baseline reaches on
that dataset -- the region where no other method produces a point at all.

| dataset | best baseline R | old JHQ max R | old exclusive | new JHQ max R | new exclusive |
|---|---|---|---|---|---|
| vogue-768 | 0.9986 | 0.9826 | 0 | 0.9957 | 0 |
| arxiv-768 | 1.0000 | 0.9928 | 0 | 0.9920 | 0 |
| **bge-m3** | **0.9425** | 0.9881 | **5** | **0.9932** | **5** |
| **stella-trec24** | **0.9780** | 0.9936 | **5** | **0.9960** | **5** |
| openai3-1536 | 0.9971 | 0.9941 | 0 | 0.9651\* | 0 |
| openai3-3072 | 0.9970 | 0.9921 | 0 | 0.9719\* | 0 |

**The claim the paper rests on is stronger, not weaker.** On the two largest
sets JHQ's reachable recall rises (0.9881 -> 0.9932, 0.9936 -> 0.9960) and it
still holds every point above the baselines' ceilings -- 0.9425 for bge-m3,
where fp32 CAGRA needs 39.70 GiB of a 31.4 GiB card and IVF-PQ tops out, and
0.9780 for stella, where only int8 CAGRA runs at all.

The two 768-d sets held nothing before and hold nothing now.

\* openai3 is not comparable: see below.

## The openai3 rows are a grid that collapsed, not a regression

The old openai3 fronts were built on **M=96**, which equation 4 rejects at both
dimensions -- Ds = 16 at d=1536 and Ds = 32 at d=3072, and B=8 divides neither.
Copying the old grid onto the frozen path therefore left only `M=192 Br=4` and
`M=384 Br=4`: four residual bits, and the ceiling that implies. `Br=8` at the
admissible M was never in the old grid and so was never run. It is running now.

## Throughput is dataset-dependent, and one earlier reading of it was wrong

New / old QPS at matched nprobe, Br=4:

| nprobe | vogue-768 | arxiv-768 | bge-m3 |
|---|---|---|---|
| 1 | x2.25 | x0.63 | -- |
| 8 | x1.41 | x0.66 | **x2.20** |
| 32 | x0.86 | x0.54 | **x1.81** |
| 128 | **x0.68** | x0.53 | **x1.74** |
| 256 | x0.72 | x0.60 | **x1.65** |

bge-m3 is uniformly faster, arxiv uniformly slower, vogue faster below
nprobe=16 and slower above. An earlier note in this project put vogue at
"2.3-3.1x slower"; that was read off the contaminated run in which two jobs
shared the card, the one the harness flagged at 97.8% QPS spread. The worst
case on vogue is x0.68.

Recall deltas are equally dataset-dependent: vogue -0.1445 at nprobe=1 falling
to -0.0045 by nprobe=128, arxiv +0.0037 rising to -0.0204, bge-m3 +0.0110
falling to -0.0028.

## What isolates the recall difference

At vogue nprobe=1 there are 910 candidates and `ck = alpha*k = 1000`, so every
candidate is refined and the top-alpha\*k selection does nothing at all. The
-0.1445 there therefore cannot come from exact-versus-cascade selection; only
the quantiser or the probe choice can produce it. arxiv at nprobe=1 has 1100
candidates and moves the other way, +0.0037, under the same isolation. That
pair is the cheapest available lead and is where the diagnosis starts.
