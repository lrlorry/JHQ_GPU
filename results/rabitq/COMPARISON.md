# JHQ against IVF-RaBitQ on the same card

`cuvs::neighbors::ivf_rabitq` is the search half of *GPU-Native ANNS with
IVF-RaBitQ* (PVLDB 19(11), 2026, NTU + NVIDIA), shipped in cuVS. This
compares it against JHQ on the one dataset both papers use, with everything
else held equal: one RTX 5090, a 999-query batch, and the timed region of
`examples/bench_ivf_rabitq.cu` matched to `examples/demo_jhq_v36.cu` — queries
up, search, results back.

Raw: `bench_lut.log`, `bench_quant.log`, and
`../front6/openai3072_hirecall.log` for JHQ's high-recall end.

## openai3-3072 (999,000 × 3072)

| Recall@10 | JHQ | IVF-RaBitQ | CAGRA fp32 | JHQ / RaBitQ | JHQ / fp32 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 68,033 | 53,763 | 56,941 | **1.27×** | 1.19× |
| 0.96 | 54,901 | 48,456 | 45,646 | **1.13×** | 1.20× |
| 0.97 | 41,151 | 42,062 | 34,149 | 0.98× | 1.21× |
| 0.98 | 28,964 | 33,942 | 20,350 | 0.85× | 1.42× |
| 0.99 | 16,735 | 25,489 | — | 0.66× | — |
| 0.994 | 11,368 | 22,316 | — | 0.51× | — |

**The crossover is at Recall ≈ 0.97.** Below it JHQ leads by up to 1.27×;
above it IVF-RaBitQ pulls away, to about 2× at 0.994.

Measured points, no interpolation:

```
JHQ         0.9413/81,988  0.9646/49,744  0.9815/27,478  0.9926/14,380  0.9965/7,472
IVF-RaBitQ  0.8606/76,165  0.9055/67,883  0.9414/58,790  0.9666/45,244
            0.9839/31,218  0.9943/22,095
```

## The step that made it work

`build()` does not leave the index in the layout search reads. cuVS's own test
round-trips it through `serialize`/`deserialize` with the comment *"Serialize
and deserialize to reorganize data for efficient search"*; nothing in the
public header says so. Without that step the search returns ids that are
essentially random beside distances **below the true minimum**, which is what
`README.md` spent a session ruling out one cause at a time.

`examples/rabitq_selftest.cu` isolates it. Generated data, queries that ARE
rows of the index, every list probed, so the answer is the row's own id at
distance zero:

| | self in top-1 | self in top-10 |
|---|---|---|
| searched as built | 0 / 20 | 0 / 20 |
| round-tripped | **20 / 20** | **20 / 20** |

at N=200,000 d=768 and again at N=50,000 d=128.

## Which mode, and why it matters

`search_mode` is `{LUT16=0, LUT32=1, QUANT4=2, QUANT8=3}`. QUANT4 is its best
here and the paper says why: it calls the LUT kernel bounded by shared-memory
capacity at high dimensionality while the bitwise formulation stays scalable,
"reflected by the gap between the two methods on the OpenAI-3072-1M dataset".
Measured at nprobe=1024: QUANT4 22,095, QUANT8 20,134, LUT16 12,055, LUT32
lower still. Reporting the LUT number would have overstated JHQ by 1.4–1.8×.

`nlist` was also checked: 1024 is far behind 4096 (0.9891 at 16,031 against
0.9839 at 31,218), so 4096 — the value JHQ uses on this dataset — is the fair
choice for both.

## What is not equal here

Both systems run on this card at this batch, so the comparison is direct. It
is not the paper's setting: it reports an L40S at batch 10⁴, where its own
advantage over CAGRA is 0.8×–8.2× at Recall 0.95 across eight datasets, 3.3×
on average. Against CAGRA fp32 measured here, IVF-RaBitQ is 0.77× at 0.95 and
1.67× at 0.98 — well below its published average, which is what a 2× wider
memory bus and a tenth of the batch would do to a bandwidth-bound method.
**So the JHQ/RaBitQ column is sound and the RaBitQ/CAGRA column is not
comparable to the paper's.**
