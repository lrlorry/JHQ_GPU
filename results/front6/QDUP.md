# Is the cross-query reuse already served by L2?

Grouping search tasks by IVF list — one list's primary codes loaded once and
scored against every query that probes it — is a large rewrite. Its premise is
that those reads are repeated *and that the repeats cost DRAM traffic*. At
nprobe=128 with nlist=4096 a list is probed by about 31 queries of a
1000-query batch, so they are certainly repeated. But this card has 96 MB of
L2, and if the repeats are already served there the rewrite buys nothing.

`JHQ_QUERY_DUP=D` (examples/demo_jhq_qdup.cu) settles it without touching a
kernel: keep nq/D queries and repeat each D times, adjacent. Query count,
block count, LUT builds, candidate count and selection work are all unchanged.
The only thing that moves is how much the probe lists overlap, and it moves by
exactly D — the D copies of a query probe an *identical* list, which is more
reuse than any real clustering could produce.

Binary is `demo_jhq_qdup`, linking `jhq_v53_x1`: v51's probe cursor and v52's
word layout, the current default. Raw in `qdup.log` and `qdup_stella.log`.

## QPS against D=1

```
dataset           np      D=1 QPS      D=2      D=4      D=8
vogue            128     138,525    +3.1%    +1.8%    +6.6%
bge              128     102,581    +5.9%    +5.7%    +6.1%
stella           128      86,082    +9.5%   +10.5%   +11.6%
openai3-3072     128      40,818    +6.3%    +7.2%   +11.4%

vogue           1024      57,580    +0.8%    +1.6%    +4.2%
bge             1024      25,635   +17.1%   +17.7%   +19.5%
stella          1024      17,163   +22.6%   +24.6%   +25.9%
openai3-3072    1024      12,004    +5.9%    +9.2%   +11.4%
```

Recall drifts with D because the query set shrinks — 125 distinct queries at
D=8 — and is printed only to confirm the run is the same search. It is not
comparable across D.

## What it says

**The gain saturates at D=2.** stella at nprobe=1024 takes +22.6% from the
first doubling and only +3.3 more points from the next 4x; bge takes +17.1%
then +2.4. That is the signature of L2 already holding the working set: once
two copies are in flight the line is resident, and six more cost nothing extra
to serve.

**The ceiling on the whole idea is +4 to +26%, not the 2x the traffic
arithmetic suggested.** That arithmetic — 28.9 GB of gather a batch on stella,
cut 4x by sharing — counted L2 traffic as if it were DRAM traffic. It is not:
the two are separated by 96 MB of cache and a hit rate high enough that
removing 87% of the repeats buys nothing beyond removing the first half.

**Where it is worth most is where the reads are worst.** bge and stella at
nprobe=1024 on nlist=32768 give each query ~31,000 candidates spread over
1024 lists — the shortest, most scattered per-list runs in the grid, and the
two largest gains. vogue at nprobe=1024 on nlist=4096 gives 250,000
candidates over 1024 lists, runs four times longer, and gains +4.2%.

## Against the cost

`NEGATIVES.md` prices shared memory on this card: **16 KB more per block costs
17–26%** wherever occupancy is not already pinned at one block.

A Q-query block needs Q selection buffers — `cap * 8` is 16,392 B each — so
Q=4 adds 49 KB and Q=2 adds 16 KB. Tiling the LUT over subspaces (Q=4,
M_tile=16 is 8 KB) fixes the *table* residency but not the *selection*
residency, which is the larger of the two.

So the honest arithmetic for the cheapest useful version, Q=2 with an M-tiled
table, on stella at nprobe=1024:

```
  reuse gain at D=2                          +22.6%
  one more 16 KB selection buffer            -10 to -17%
                                             ------------
                                             +5 to +12%
```

A large restructuring for something under half of what v52's word layout
returned for a repack loop. The alternative that does not pay this tax is the
sign inner product: at L=2 the primary levels are +/-a, so
`D_P = ||q'||^2 + d*a^2 - 2a<q', s_y>` and the per-query shared residency
falls from the table's 4d floats to the query's d — 49 KB to 12 KB on
openai3-3072, the one dataset v53 showed is bound by exactly this.
