## Isolated scoring (median us per launch; B0/impl > 1 = faster than B0)

### vogue-768, b = 8 (6081662 candidates a pass, 16 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 23.5 | 1.000 | 8.53e-07 | 3.54e-07 | 0 | 0 |
| H1_r8 | 176.1 | 0.134 | 6.24e-07 | 2.47e-07 | 1 | 1 |
| H1z_r8 | 115.6 | 0.203 | 6.24e-07 | 1.17e-07 | 1 | 1 |
| H1_r4 | 391.9 | 0.060 | 6.83e-07 | 3.05e-07 | 0 | 0 |
| H2_r8 | 128.6 | 0.183 | 6.20e-07 | 2.46e-07 | 0 | 0 |
| H2_r6 | 51.2 | 0.459 | 6.71e-07 | 2.67e-07 | 0 | 0 |
| H2_r4 | 64.4 | 0.365 | 6.20e-07 | 3.01e-07 | 1 | 1 |
| B0 at the scan's grid | 81.0 | | | | | |
| query preparation: B0 table / Mailman C0 | 4.2 / 4.5 | | | | | |

scan stage (v70 nsys) 225 us of 372 us latency; B0's scoring at the scan's grid is 36% of the scan and f = 22% of the latency. Best Mailman here: H2_r6 at s = 0.459 -> modelled end-to-end x0.796

### vogue-768, b = 1000 (47999848 candidates a pass, 1 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 2188.0 | 1.000 | 9.15e-07 | 3.47e-07 | 1 | 0 |
| H1_r8 | 21256.5 | 0.103 | 7.89e-07 | 2.72e-07 | 4 | 3 |
| H1z_r8 | 11735.3 | 0.186 | 7.89e-07 | 1.17e-07 | 3 | 4 |
| H1_r4 | 47615.3 | 0.046 | 8.97e-07 | 3.30e-07 | 4 | 5 |
| H2_r8 | 13705.4 | 0.160 | 8.03e-07 | 2.69e-07 | 2 | 3 |
| H2_r6 | 6032.9 | 0.363 | 7.69e-07 | 2.94e-07 | 1 | 2 |
| H2_r4 | 5116.1 | 0.428 | 8.37e-07 | 3.32e-07 | 3 | 4 |
| B0 at the scan's grid | 2180.6 | | | | | |
| query preparation: B0 table / Mailman C0 | 9.5 / 4.8 | | | | | |

scan stage (v70 nsys) 3004 us of 3943 us latency; B0's scoring at the scan's grid is 73% of the scan and f = 55% of the latency. Best Mailman here: H2_r4 at s = 0.428 -> modelled end-to-end x0.575

### arxiv-768, b = 8 (6056619 candidates a pass, 16 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 22.6 | 1.000 | 5.16e-07 | 2.31e-07 | 0 | 0 |
| H1_r8 | 148.4 | 0.152 | 9.86e-07 | 4.89e-07 | 0 | 0 |
| H1z_r8 | 83.0 | 0.272 | 9.86e-07 | 1.60e-07 | 0 | 0 |
| H1_r4 | 296.8 | 0.076 | 1.12e-06 | 5.75e-07 | 1 | 1 |
| H2_r8 | 125.0 | 0.181 | 1.15e-06 | 4.84e-07 | 0 | 0 |
| H2_r6 | 55.5 | 0.407 | 1.04e-06 | 5.09e-07 | 1 | 1 |
| H2_r4 | 38.8 | 0.581 | 1.16e-06 | 5.58e-07 | 1 | 1 |
| B0 at the scan's grid | 49.1 | | | | | |
| query preparation: B0 table / Mailman C0 | 4.4 / 4.5 | | | | | |

scan stage (v70 nsys) 233 us of 433 us latency; B0's scoring at the scan's grid is 21% of the scan and f = 11% of the latency. Best Mailman here: H2_r4 at s = 0.581 -> modelled end-to-end x0.924

### arxiv-768, b = 1000 (47143349 candidates a pass, 1 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 2033.0 | 1.000 | 5.31e-07 | 2.35e-07 | 3 | 0 |
| H1_r8 | 17278.5 | 0.118 | 1.24e-06 | 5.03e-07 | 2 | 1 |
| H1z_r8 | 6873.3 | 0.296 | 1.15e-06 | 1.59e-07 | 3 | 0 |
| H1_r4 | 37126.3 | 0.055 | 1.17e-06 | 5.83e-07 | 6 | 5 |
| H2_r8 | 19978.7 | 0.102 | 1.15e-06 | 5.03e-07 | 3 | 4 |
| H2_r6 | 7542.4 | 0.270 | 1.15e-06 | 5.34e-07 | 6 | 5 |
| H2_r4 | 4233.5 | 0.480 | 1.23e-06 | 5.82e-07 | 3 | 6 |
| B0 at the scan's grid | 2029.2 | | | | | |
| query preparation: B0 table / Mailman C0 | 10.1 / 4.9 | | | | | |

scan stage (v70 nsys) 2933 us of 4191 us latency; B0's scoring at the scan's grid is 69% of the scan and f = 48% of the latency. Best Mailman here: H2_r4 at s = 0.480 -> modelled end-to-end x0.656

### openai3-3072, b = 8 (2355077 candidates a pass, 16 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 27.7 | 1.000 | 1.95e-06 | 9.64e-07 | 0 | 0 |
| H1_r8 | 173.0 | 0.160 | 6.81e-07 | 1.39e-07 | 0 | 0 |
| H1z_r8 | 161.6 | 0.172 | 6.81e-07 | 1.17e-07 | 0 | 0 |
| H1_r4 | 316.4 | 0.088 | 6.69e-07 | 2.13e-07 | 0 | 0 |
| H2_r8 | 346.7 | 0.080 | 6.81e-07 | 1.39e-07 | 0 | 0 |
| H2_r6 | 178.9 | 0.155 | 7.42e-07 | 1.62e-07 | 0 | 0 |
| H2_r4 | 148.2 | 0.187 | 6.82e-07 | 2.16e-07 | 0 | 0 |
| B0 at the scan's grid | 65.4 | | | | | |
| query preparation: B0 table / Mailman C0 | 4.2 / 5.2 | | | | | |

scan stage (v70 nsys) 177 us of 335 us latency; B0's scoring at the scan's grid is 37% of the scan and f = 20% of the latency. Best Mailman here: H2_r4 at s = 0.187 -> modelled end-to-end x0.541

### openai3-3072, b = 1000 (18316479 candidates a pass, 1 batches)
| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3703.4 | 1.000 | 2.18e-06 | 9.62e-07 | 0 | 0 |
| H1_r8 | 14117.4 | 0.262 | 9.29e-07 | 1.40e-07 | 0 | 0 |
| H1z_r8 | 14129.8 | 0.262 | 9.29e-07 | 1.17e-07 | 0 | 0 |
| H1_r4 | 30799.5 | 0.120 | 1.25e-06 | 2.14e-07 | 0 | 0 |
| H2_r8 | 34275.6 | 0.108 | 9.89e-07 | 1.40e-07 | 0 | 0 |
| H2_r6 | 13434.6 | 0.276 | 8.73e-07 | 1.63e-07 | 0 | 0 |
| H2_r4 | 7130.6 | 0.519 | 1.07e-06 | 2.16e-07 | 0 | 0 |
| B0 at the scan's grid | 3701.3 | | | | | |
| query preparation: B0 table / Mailman C0 | 26.7 / 6.3 | | | | | |

scan stage (v70 nsys) 4290 us of 6322 us latency; B0's scoring at the scan's grid is 86% of the scan and f = 59% of the latency. Best Mailman here: H2_r4 at s = 0.519 -> modelled end-to-end x0.649

## Where the Mailman kernels spend their cycles (clock64 per warp, MB_PROFILE=1)
| dataset | b | impl | zero | accumulate | recover | write |
|---|---:|---|---:|---:|---:|---:|
| vogue-768 | 8 | H1_r8 | 0.4% | 95.0% | 3.2% | 1.4% |
| vogue-768 | 8 | H1_r4 | 0.1% | 97.3% | 1.3% | 1.3% |
| vogue-768 | 8 | H2_r8 | 1.4% | 94.3% | 3.7% | 0.6% |
| vogue-768 | 8 | H2_r4 | 0.2% | 98.7% | 0.4% | 0.8% |
| vogue-768 | 8 | H2_r6 | 0.3% | 98.1% | 1.0% | 0.5% |
| vogue-768 | 8 | H1z_r8 | 0.5% | 89.6% | 6.4% | 3.6% |
| vogue-768 | 1000 | H1_r8 | 0.3% | 94.7% | 3.3% | 1.7% |
| vogue-768 | 1000 | H1_r4 | 0.1% | 97.3% | 1.5% | 1.1% |
| vogue-768 | 1000 | H2_r8 | 2.1% | 90.4% | 5.9% | 1.7% |
| vogue-768 | 1000 | H2_r4 | 0.3% | 97.9% | 0.9% | 0.9% |
| vogue-768 | 1000 | H2_r6 | 0.2% | 97.5% | 1.5% | 0.8% |
| vogue-768 | 1000 | H1z_r8 | 0.5% | 88.6% | 7.5% | 3.4% |
| arxiv-768 | 8 | H1_r8 | 0.3% | 95.9% | 2.7% | 1.0% |
| arxiv-768 | 8 | H1_r4 | 0.1% | 97.6% | 1.1% | 1.2% |
| arxiv-768 | 8 | H2_r8 | 1.3% | 94.5% | 3.5% | 0.7% |
| arxiv-768 | 8 | H2_r4 | 0.2% | 98.6% | 0.3% | 0.9% |
| arxiv-768 | 8 | H2_r6 | 0.3% | 98.2% | 1.0% | 0.5% |
| arxiv-768 | 8 | H1z_r8 | 0.5% | 88.8% | 6.8% | 3.8% |
| arxiv-768 | 1000 | H1_r8 | 0.2% | 96.3% | 2.3% | 1.2% |
| arxiv-768 | 1000 | H1_r4 | 0.1% | 97.7% | 1.3% | 0.9% |
| arxiv-768 | 1000 | H2_r8 | 1.4% | 93.5% | 4.0% | 1.1% |
| arxiv-768 | 1000 | H2_r4 | 0.2% | 98.3% | 0.6% | 0.9% |
| arxiv-768 | 1000 | H2_r6 | 0.1% | 98.2% | 1.1% | 0.6% |
| arxiv-768 | 1000 | H1z_r8 | 0.4% | 89.3% | 6.9% | 3.4% |
| openai3-3072 | 8 | H1_r8 | 0.2% | 97.6% | 1.5% | 0.7% |
| openai3-3072 | 8 | H1_r4 | 0.0% | 99.0% | 0.5% | 0.5% |
| openai3-3072 | 8 | H2_r8 | 0.3% | 98.5% | 0.9% | 0.2% |
| openai3-3072 | 8 | H2_r4 | 0.0% | 99.7% | 0.1% | 0.2% |
| openai3-3072 | 8 | H2_r6 | 0.1% | 99.6% | 0.2% | 0.1% |
| openai3-3072 | 8 | H1z_r8 | 0.2% | 97.3% | 1.8% | 0.8% |
| openai3-3072 | 1000 | H1_r8 | 0.1% | 97.7% | 1.2% | 1.0% |
| openai3-3072 | 1000 | H1_r4 | 0.0% | 98.8% | 0.6% | 0.6% |
| openai3-3072 | 1000 | H2_r8 | 0.3% | 98.5% | 0.9% | 0.2% |
| openai3-3072 | 1000 | H2_r4 | 0.0% | 99.7% | 0.1% | 0.2% |
| openai3-3072 | 1000 | H2_r6 | 0.0% | 99.6% | 0.2% | 0.1% |
| openai3-3072 | 1000 | H1z_r8 | 0.1% | 97.6% | 1.4% | 1.0% |

## Structure of the codes (top 512 lists by scan reads)

### vogue-768: 512 lists, 60.7% of the scan's reads; |p1 - 1/2| per (list, dim) mean 0.284, share > 0.25 58%
- Mailman r4: pattern entropy 3.31/4 bits, all-zero 27.2% all-one 25.4%; a warp over 32 dims: 10.9 distinct patterns, largest multiplicity 10.1 (random bits 4.8)
- Mailman r8: pattern entropy 6.09/8 bits, all-zero 18.6% all-one 17.0%; a warp over 32 dims: 20.5 distinct patterns, largest multiplicity 7.1 (random bits 1.9)
- dictionary: best bytes n64_g8 -> 0.976 of raw (13% of blocks use it); best table work n64_g8 -> 0.959

### arxiv-768: 512 lists, 28.8% of the scan's reads; |p1 - 1/2| per (list, dim) mean 0.361, share > 0.25 75%
- Mailman r4: pattern entropy 2.90/4 bits, all-zero 29.4% all-one 35.5%; a warp over 32 dims: 9.6 distinct patterns, largest multiplicity 12.4 (random bits 4.8)
- Mailman r8: pattern entropy 5.01/8 bits, all-zero 23.6% all-one 28.9%; a warp over 32 dims: 15.9 distinct patterns, largest multiplicity 10.3 (random bits 1.9)
- dictionary: best bytes n128_g8 -> 0.864 of raw (39% of blocks use it); best table work n256_g16 -> 0.731

### openai3-3072: 512 lists, 30.2% of the scan's reads; |p1 - 1/2| per (list, dim) mean 0.195, share > 0.25 34%
- Mailman r4: pattern entropy 3.84/4 bits, all-zero 15.3% all-one 14.4%; a warp over 32 dims: 13.1 distinct patterns, largest multiplicity 6.3 (random bits 4.8)
- Mailman r8: pattern entropy 7.44/8 bits, all-zero 6.4% all-one 5.7%; a warp over 32 dims: 27.3 distinct patterns, largest multiplicity 3.1 (random bits 1.9)
- dictionary: best bytes n64_g8 -> 1.000 of raw (0% of blocks use it); best table work n64_g8 -> 1.000

## Shared FP32 atomicAdd in SASS
```
192 ATOMS.CAST.SPIN
     24 RED.E.ADD.64.STRONG.GPU
sm_120     224 ATOMS.CAST.SPIN
```
