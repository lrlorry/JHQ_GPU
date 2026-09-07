# Where recall is lost, and how much the scan actually reads

Two numbers this project has been estimating rather than measuring
(`jhq_v43_diag/`, `scripts/run_diag.sh`, raw in `diag.log`). 1000 queries,
BLOCK=1024, k=10, three repeats. `JHQ_DIAG=1`, so the QPS these runs print is
not a result -- the readbacks are on the search stream.

## 1. Almost all of the missing recall is routing, not ranking

`ivf_recall` is the share of the true top-10 whose IVF list the query opened at
all. What is missing from it was lost before the scan ran. What survives it and
still does not come back was lost by the primary filter or the refinement.

| | nprobe | recall | ivf_recall | lost to routing | lost to ranking |
|---|---|---|---|---|---|
| vogue-768 | 8 | 0.7538 | 0.7555 | **0.2445** | 0.0017 |
| | 32 | 0.9264 | 0.9295 | **0.0705** | 0.0031 |
| | 128 | 0.9849 | 0.9899 | 0.0101 | 0.0050 |
| | 256 | 0.9922 | 0.9973 | 0.0027 | 0.0051 |
| bge-m3 | 8 | 0.7320 | 0.7333 | **0.2667** | 0.0013 |
| | 32 | 0.8798 | 0.8826 | **0.1174** | 0.0028 |
| | 128 | 0.9589 | 0.9620 | 0.0380 | 0.0031 |
| | 256 | 0.9785 | 0.9819 | 0.0181 | 0.0034 |
| stella | 8 | 0.9010 | 0.9031 | 0.0969 | 0.0021 |
| | 32 | 0.9710 | 0.9745 | 0.0255 | 0.0035 |
| | 128 | 0.9910 | 0.9951 | 0.0049 | 0.0041 |
| | 256 | 0.9941 | 0.9982 | 0.0018 | 0.0041 |

**The ranking loss is 0.0013-0.0051 everywhere.** Across three datasets and a
32x range of nprobe, everything downstream of coarse routing -- the equation-4
primary code, the ADC scan, the top-ck selection, the residual refinement, the
final sort -- together loses at most half a point of the neighbours routing
hands it. At nprobe=8 it is under 0.2%, and 99.3% of vogue's shortfall is that
the right list was never opened.

Two consequences:

- **No amount of work on the scan can raise recall.** The compaction, the
  lookup, the LUT precision, an early-exit bound: all of them are speed items
  with a recall budget of 0.5% at most, and most of that budget is alpha's
  (below). Recall belongs to the coarse quantizer.
- **The quantization is not the weak part.** A 1 bit/dim primary code plus a
  Br-bit residual reproduces the exact ranking of what it is given to within
  0.5%. That is a result about JHQ, and it is invisible in the recall number
  because routing dominates it.

## 2. alpha is a ranking knob, and it saturates

nprobe=128, so routing is fixed and only `ck = alpha*k` moves:

| | alpha=8 | 20 | 50 | 100 |
|---|---|---|---|---|
| vogue recall | 0.9483 | 0.9767 | 0.9835 | 0.9848 |
| vogue lost to ranking | **0.0416** | 0.0132 | 0.0064 | 0.0051 |
| stella recall | 0.9881 | 0.9909 | 0.9910 | 0.9910 |
| stella lost to ranking | 0.0070 | 0.0042 | 0.0041 | 0.0041 |

Routing loss is identical in every column (0.0101 vogue, 0.0049 stella), which
is the control: alpha cannot touch it.

The paper's alpha=8 costs vogue **4.2 points of ranking loss**; alpha=50
recovers all but 0.6 of it and alpha=100 adds 0.13 more. stella is done by
alpha=20. So alpha=100 is not over-provisioned on vogue and is roughly 2x
over-provisioned on stella -- the first evidence in this project for a
per-dataset alpha rather than one global value.

## 3. The scan reads up to 2.2x what every table here has claimed

`cand` is what `select_probes_kernel` summed for its own loop bound. The
estimate is the `N*nprobe/nlist` that `results/` has been quoting:

| | nprobe | measured | N*nprobe/nlist | ratio |
|---|---|---|---|---|
| vogue-768 | 8 / 32 / 128 / 256 | 9,876 / 37,611 / 135,024 / 246,658 | 7,284 / 29,135 / 116,541 / 233,082 | **1.36 / 1.29 / 1.16 / 1.06** |
| bge-m3 | 8 / 32 / 128 / 256 | 21,870 / 77,240 / 278,379 / 533,823 | 9,855 / 39,420 / 157,680 / 315,360 | **2.22 / 1.96 / 1.77 / 1.69** |
| stella | 8 / 32 / 128 / 256 | 16,824 / 60,873 / 226,410 / 440,070 | 8,680 / 34,720 / 138,880 / 277,760 | **1.94 / 1.75 / 1.63 / 1.58** |

The estimate assumes equal list lengths. IVF lists are not equal, and the
lists a query probes are not a random sample of them -- a query lands near a
dense region and the dense region's lists are the long ones, so probing is
biased towards exactly the lists that cost most.

The bias shrinks as nprobe grows (bge 2.22 -> 1.69) because opening more lists
averages more of the distribution, and it is smallest on vogue, whose 1024
lists over 1M vectors are the most even.

**Every efficiency comparison in this project that used the estimate was
flattering to JHQ**, worst by 2.2x on bge-m3, which is also the dataset where
the gap to a graph index is widest. Nothing in `results/` should quote a
candidate count that is not this measured one.
