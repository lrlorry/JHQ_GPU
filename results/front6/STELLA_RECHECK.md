# The published stella front, re-run on today's default

`report_adc2026/v47/` was produced by `demo_jhq_v47_diag` at `JHQ_BLOCK=1024`,
before v51's probe cursor and v52's word layout. This re-runs the same
nprobe sweep on `demo_jhq_v53_x1` — v51 and v52 both on, `JHQ_BLOCK=512` —
to check that the search is unchanged and to say by how much the published
numbers are stale. One dataset, because v51 and v52 were each already
measured off-versus-on at identical recall (`v51.log`, `v52.log`).

stella-trec24, M=128, B=8, Br=8, alpha=100, k=10, nlist=32768, nt=1277952,
1000-query batch. Raw in `stella_front_v53.log`; the published rows are the
nlist=32768 block of `front6.log`.

| nprobe | published recall | now | published QPS | now | x |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.9026 | 0.9028 | 139,814 | 174,353 | 1.25 |
| 32 | 0.9685 | 0.9689 | 100,611 | 144,106 | 1.43 |
| 128 | 0.9890 | 0.9893 | 53,442 | 86,436 | 1.62 |
| 256 | 0.9932 | 0.9935 | 31,293 | 55,471 | 1.77 |
| 512 | — | 0.9949 | — | 28,195 | — |
| 1024 | — | 0.9957 | — | 17,141 | — |

**Recall agrees to 2-3e-4 at every point** — inside the 1e-3 floor that three
cold-cache runs of one binary already established, and expected here because
the two runs used different cache directories and so different codebook last
bits. The search is unchanged.

**The speedup rises with nprobe, 1.25x to 1.77x.** That is the shape v51's
probe cursor should have: it removed a per-candidate re-walk of about nprobe/2
list boundaries, so its value grows with nprobe while the rest of the scan
does not.

Two changes are folded together here and not separated: BLOCK went 1024 to
512 as well, which `results/block_sweep/` had already shown wins 20 of 24
cells. Both are what the default should be, so the row is the right number to
report — it is just not attributable to v51 and v52 alone.

At R=0.9957 this is 17,141 QPS against the ~1,800 the PVLDB paper reports for
its own implementation on this dataset.
