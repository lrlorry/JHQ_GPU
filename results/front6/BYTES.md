# Br=8 is the matched-bytes point, not the expensive one

`bpv_ = (d * Br + 7) / 8` -- the residual stores a code per **dimension**, not
per subspace. So JHQ's resident footprint is

    M (primary) + (d*Br+7)/8 (residual) + 8 (corr + id)

and at Br=8 that lands within 1-8% of int8 CAGRA's `d + 4*graph_degree` on
every dataset here. `results/front6/br4.log` measures Br=4 as the second point
on the curve; `Br=2` is rejected by the index (`Br must be 4 or 8`).

| | JHQ Br=8 | CAGRA int8 | ceiling, JHQ | ceiling, int8 | | JHQ Br=4 |
|---|---|---|---|---|---|---|
| vogue-768 | 872 B | 896 B | 0.9946 | 0.9459 | **+4.9 pts** | 488 B, 0.9479 |
| arxiv-768 | 872 B | 896 B | 0.9900 | 0.9620 | **+2.8** | 488 B, 0.9371 |
| bge-m3 | 1160 B | 1152 B | 0.9930 | 0.9376 | **+5.5** | 648 B, 0.9562 |
| stella | 1160 B | 1152 B | 0.9959 | 0.9780 | **+1.8** | 648 B, 0.9623 |
| openai3-1536 | 1736 B | 1664 B | 0.9958 | 0.9722 | **+2.4** | 968 B, 0.9688 |
| openai3-3072 | 3464 B | 3200 B | 0.9969 | 0.9733 | **+2.4** | 1928 B, 0.9761 |

This reproduces the published report's 1.8-5.6 point claim, on re-measured
JHQ fronts.

**Br=4 is not a free improvement on it.** At 0.56x the memory the ceiling
falls below int8 CAGRA on four of six -- vogue 0.9479 against 0.9459 and
openai3-3072 0.9761 against 0.9733 are the two that hold. So the advantage is
at matched bytes; it is not that JHQ dominates at every size.

## The reading that was wrong on the way

Seeing `1024 B/vec` of residual at Br=8 and reading it as "JHQ has no memory
advantage here" inverts the comparison. Having the same footprint as the
baseline is the condition under which a recall comparison means anything; it
is the operating point the claim needs, not one to move away from. The Br=4
sweep was still worth taking -- it is what shows the trade is not free -- but
it does not replace Br=8.

## The audit that came out clean

The sigma overflow (`results/v46_sigma/`) triggers above `2^31/d` training
rows: 699,050 at d=3072, 1,398,101 at d=1536, 2,097,152 at d=1024, 2,796,202
at d=768. Every recorded `env` in `results/**/*.csv` was checked and none sets
`JHQ_N_TRAIN`; `examples/demo_jhq_v36.cu:128` defaults it to
`min(nb, 100000)`. **No published result is affected.** The bug became
reachable only when this work raised the training set.
