
## Vogue-768 (0.93M x 768)

| method | points | recall range (frontier) | best QPS | frontier points | device memory, MiB |
|---|---:|---|---:|---:|---|
| JHQ-GPU v69 | 27 | 0.7714-0.9907 | 702,867 | 19 | process 1,425-1,537 (incl. context) |
| CAGRA fp32 | 64 | 0.8564-0.9985 | 475,836 | 25 | index 2,958-2,980 |
| CAGRA int8 | 64 | 0.8304-0.9453 | 1,182,799 | 24 | index 910-930 |
| IVF-RaBitQ | 132 | 0.7532-0.9948 | 507,519 | 23 | index 572-998 |
| IVF-RaBitQ 9-bit | 44 | 0.7532-0.9948 | 507,519 | 20 | index 998 |
| IVF-PQ | 54 | 0.6885-0.9805 | 292,289 | 21 | index 416-1,052 |
| GPU brute force | 1 | 0.9993-0.9993 | 8,865 | 1 | index 2,746 |

### Frontier points (protocol A)


**JHQ-GPU v69**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7714 | 702,867 (676,255-706,169) | - | 1.415 / 1.447 | `jhq-v69` {"nprobe": 16, "alpha": 16, "ck": 160, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.7743 | 642,311 (639,968-645,622) | - | 1.549 / 1.577 | `jhq-v69` {"nprobe": 16, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.8565 | 573,862 (545,118-577,683) | - | 1.729 / 1.780 | `jhq-v69` {"nprobe": 32, "alpha": 16, "ck": 160, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.8610 | 533,957 (533,042-535,474) | - | 1.872 / 1.893 | `jhq-v69` {"nprobe": 32, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.8619 | 408,870 (407,725-410,638) | - | 2.443 / 2.470 | `jhq-v69` {"nprobe": 32, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9240 | 407,813 (403,851-408,181) | - | 2.449 / 2.494 | `jhq-v69` {"nprobe": 64, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9257 | 330,948 (327,669-332,848) | - | 3.017 / 3.060 | `jhq-v69` {"nprobe": 64, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9494 | 329,529 (326,760-332,355) | - | 3.029 / 3.100 | `jhq-v69` {"nprobe": 96, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9622 | 284,210 (273,569-287,201) | - | 3.510 / 3.574 | `jhq-v69` {"nprobe": 128, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9644 | 244,218 (242,319-245,484) | - | 4.090 / 4.162 | `jhq-v69` {"nprobe": 128, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9739 | 221,928 (216,085-222,298) | - | 4.503 / 4.588 | `jhq-v69` {"nprobe": 192, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9767 | 196,581 (193,775-197,175) | - | 5.093 / 5.154 | `jhq-v69` {"nprobe": 192, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9796 | 185,563 (184,240-186,721) | - | 5.375 / 5.458 | `jhq-v69` {"nprobe": 256, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9827 | 168,370 (167,237-169,132) | - | 5.940 / 6.013 | `jhq-v69` {"nprobe": 256, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9853 | 140,941 (139,113-141,147) | - | 7.101 / 7.237 | `jhq-v69` {"nprobe": 384, "alpha": 32, "ck": 320, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9884 | 128,803 (125,964-129,904) | - | 7.711 / 7.835 | `jhq-v69` {"nprobe": 384, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9887 | 110,708 (109,720-112,042) | - | 8.917 / 9.175 | `jhq-v69` {"nprobe": 384, "alpha": 128, "ck": 1280, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9904 | 108,380 (107,149-108,959) | - | 9.160 / 9.371 | `jhq-v69` {"nprobe": 512, "alpha": 64, "ck": 640, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |
| 0.9907 | 95,186 (95,050-95,746) | - | 10.448 / 10.602 | `jhq-v69` {"nprobe": 512, "alpha": 128, "ck": 1280, "nlist": 4096, "M": 96, "Br": 8, "batch": 1000} |

**CAGRA fp32**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8564 | 475,836 (473,355-479,188) | 539,840 | 2.087 / 2.108 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 32, "search_width": 1} |
| 0.8618 | 445,134 (438,372-446,199) | 497,078 | 2.245 / 2.260 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 32, "search_width": 2} |
| 0.8740 | 403,432 (403,260-404,047) | 445,795 | 2.474 / 2.489 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 32, "search_width": 4} |
| 0.8926 | 331,782 (325,591-332,306) | 359,153 | 3.013 / 3.029 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 32, "search_width": 8} |
| 0.9370 | 268,660 (265,680-268,902) | 286,923 | 3.719 / 3.740 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 64, "search_width": 1} |
| 0.9402 | 247,889 (247,759-248,006) | 263,120 | 4.032 / 4.051 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 64, "search_width": 4} |
| 0.9477 | 222,173 (221,995-222,404) | 234,477 | 4.497 / 4.519 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 64, "search_width": 8} |
| 0.9650 | 186,406 (186,301-186,619) | 194,983 | 5.364 / 5.378 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 96, "search_width": 2} |
| 0.9655 | 181,780 (180,666-181,964) | 189,829 | 5.499 / 5.524 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 96, "search_width": 1} |
| 0.9688 | 167,010 (166,935-167,114) | 173,818 | 5.983 / 6.015 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 96, "search_width": 8} |
| 0.9761 | 144,365 (144,275-144,510) | 149,377 | 6.925 / 6.950 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 128, "search_width": 1} |
| 0.9763 | 144,103 (143,986-144,201) | 149,063 | 6.935 / 6.963 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 128, "search_width": 2} |
| 0.9773 | 139,390 (138,370-139,457) | 144,142 | 7.172 / 7.194 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 128, "search_width": 4} |
| 0.9788 | 133,441 (132,861-133,495) | 137,830 | 7.493 / 7.522 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 128, "search_width": 8} |
| 0.9874 | 98,469 (98,441-98,499) | 101,059 | 10.152 / 10.197 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 192, "search_width": 4} |
| 0.9875 | 96,298 (96,262-96,389) | 98,760 | 10.380 / 10.429 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 192, "search_width": 2} |
| 0.9878 | 94,325 (94,229-94,351) | 96,672 | 10.605 / 10.636 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 192, "search_width": 8} |
| 0.9901 | 76,111 (76,069-76,184) | 77,702 | 13.137 / 13.172 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 256, "search_width": 2} |
| 0.9903 | 75,932 (75,883-75,939) | 77,423 | 13.170 / 13.204 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 256, "search_width": 4} |
| 0.9904 | 73,598 (73,586-73,632) | 75,063 | 13.585 / 13.632 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 256, "search_width": 8} |
| 0.9958 | 52,516 (52,453-52,573) | 53,213 | 19.049 / 19.080 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 384, "search_width": 8} |
| 0.9969 | 40,253 (40,209-40,322) | 40,674 | 24.787 / 25.270 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.48, "index_mib": 2980.0, "itopk_size": 512, "search_width": 4} |
| 0.9978 | 26,611 (26,604-26,617) | 26,841 | 37.580 / 37.674 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.092, "index_mib": 2958.0, "itopk_size": 384, "search_width": 4} |
| 0.9980 | 25,678 (25,641-25,714) | 25,939 | 38.914 / 39.040 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.092, "index_mib": 2958.0, "itopk_size": 384, "search_width": 2} |
| 0.9985 | 20,244 (20,233-20,259) | 20,384 | 49.377 / 49.569 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 3.092, "index_mib": 2958.0, "itopk_size": 512, "search_width": 8} |

**CAGRA int8**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8304 | 1,182,799 (1,156,870-1,198,440) | 1,716,578 | 0.834 / 0.874 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 32, "search_width": 1} |
| 0.8406 | 1,169,148 (1,151,106-1,172,093) | 1,668,261 | 0.852 / 0.868 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 32, "search_width": 2} |
| 0.8487 | 1,074,280 (1,068,858-1,082,447) | 1,482,223 | 0.927 / 0.952 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 32, "search_width": 4} |
| 0.8592 | 858,316 (856,055-860,734) | 1,104,557 | 1.163 / 1.183 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 32, "search_width": 8} |
| 0.8996 | 807,338 (803,045-808,733) | 1,022,484 | 1.238 / 1.245 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 32, "search_width": 1} |
| 0.9009 | 775,289 (772,299-775,886) | 967,898 | 1.288 / 1.301 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 32, "search_width": 2} |
| 0.9059 | 753,486 (750,647-754,884) | 932,170 | 1.323 / 1.352 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 64, "search_width": 4} |
| 0.9087 | 650,107 (648,227-651,783) | 778,586 | 1.536 / 1.551 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 64, "search_width": 8} |
| 0.9221 | 568,601 (536,284-570,920) | 666,380 | 1.754 / 1.783 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 96, "search_width": 4} |
| 0.9254 | 512,986 (512,565-515,298) | 589,375 | 1.941 / 1.973 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 96, "search_width": 8} |
| 0.9311 | 478,898 (477,248-479,057) | 542,893 | 2.087 / 2.099 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 64, "search_width": 2} |
| 0.9317 | 432,365 (425,394-433,598) | 484,888 | 2.308 / 2.321 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 128, "search_width": 8} |
| 0.9364 | 337,594 (337,123-337,982) | 365,302 | 2.961 / 2.973 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 96, "search_width": 2} |
| 0.9366 | 332,407 (330,856-333,194) | 361,185 | 3.004 / 3.046 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 192, "search_width": 4} |
| 0.9375 | 317,993 (316,523-320,555) | 346,676 | 3.126 / 3.212 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 192, "search_width": 8} |
| 0.9395 | 261,170 (260,110-261,654) | 277,236 | 3.823 / 3.837 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 128, "search_width": 4} |
| 0.9398 | 252,492 (251,415-253,055) | 268,119 | 3.955 / 4.000 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 128, "search_width": 2} |
| 0.9432 | 184,977 (184,761-185,517) | 192,582 | 5.406 / 5.420 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 192, "search_width": 4} |
| 0.9434 | 183,285 (182,970-184,245) | 191,164 | 5.457 / 5.470 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 192, "search_width": 2} |
| 0.9439 | 138,319 (138,018-138,647) | 142,646 | 7.229 / 7.246 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 256, "search_width": 4} |
| 0.9440 | 135,416 (135,398-135,522) | 139,728 | 7.384 / 7.403 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 256, "search_width": 2} |
| 0.9445 | 120,688 (120,142-121,123) | 124,981 | 8.255 / 8.518 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 3.154, "index_mib": 930.0, "itopk_size": 512, "search_width": 8} |
| 0.9447 | 87,912 (87,596-87,951) | 89,778 | 11.360 / 11.660 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 384, "search_width": 4} |
| 0.9453 | 72,696 (72,436-72,932) | 73,983 | 13.764 / 13.841 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.594, "kind": "cagra-int8", "build_s": 2.654, "index_mib": 910.0, "itopk_size": 512, "search_width": 8} |

**IVF-RaBitQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7532 | 507,519 (414,280-510,178) | 583,887 | 1.962 / 1.991 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.7534 | 493,110 (479,100-494,694) | 575,846 | 2.028 / 2.040 | `ivf-rabitq-b9-QUANT8` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 16, "mode": 3, "mode_name": "QUANT8"} |
| 0.8485 | 399,139 (393,262-403,674) | 456,168 | 2.418 / 2.833 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 32, "mode": 1, "mode_name": "LUT32"} |
| 0.9152 | 337,725 (331,038-341,822) | 374,481 | 2.915 / 3.210 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 64, "mode": 1, "mode_name": "LUT32"} |
| 0.9153 | 330,496 (307,949-333,897) | 369,764 | 2.934 / 3.596 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9434 | 297,852 (294,348-298,636) | 325,683 | 3.351 / 3.385 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 96, "mode": 1, "mode_name": "LUT32"} |
| 0.9576 | 262,547 (252,201-264,344) | 278,063 | 3.769 / 3.942 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9579 | 236,654 (221,106-270,595) | 292,090 | 4.294 / 4.885 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 128, "mode": 1, "mode_name": "LUT32"} |
| 0.9740 | 212,106 (210,346-212,347) | 224,174 | 4.709 / 4.737 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 192, "mode": 1, "mode_name": "LUT32"} |
| 0.9813 | 185,432 (183,838-187,070) | 195,954 | 5.354 / 5.435 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 256, "mode": 1, "mode_name": "LUT32"} |
| 0.9818 | 178,544 (175,717-181,775) | 189,628 | 5.539 / 5.839 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9884 | 137,793 (90,743-138,744) | 145,046 | 7.149 / 7.529 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 384, "mode": 1, "mode_name": "LUT32"} |
| 0.9886 | 134,584 (129,106-136,572) | 138,812 | 7.347 / 7.754 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9905 | 117,633 (116,927-119,543) | 122,364 | 8.358 / 8.502 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 512, "mode": 1, "mode_name": "LUT32"} |
| 0.9908 | 110,312 (78,185-115,215) | 116,158 | 8.662 / 10.018 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9917 | 102,235 (99,658-103,844) | 103,806 | 9.746 / 10.036 | `ivf-rabitq-b8-LUT32` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 912.0, "build_ms": 1398.6, "serialize_ms": 805.2, "deserialize_ms": 992.3, "n_probes": 512, "mode": 1, "mode_name": "LUT32"} |
| 0.9927 | 90,818 (89,797-91,829) | 92,450 | 10.863 / 11.368 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 768, "mode": 1, "mode_name": "LUT32"} |
| 0.9932 | 87,785 (87,222-88,421) | 90,350 | 11.259 / 12.304 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9935 | 82,378 (78,582-83,574) | 85,072 | 11.963 / 12.806 | `ivf-rabitq-b8-QUANT4` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 912.0, "build_ms": 1398.6, "serialize_ms": 805.2, "deserialize_ms": 992.3, "n_probes": 768, "mode": 2, "mode_name": "QUANT4"} |
| 0.9937 | 79,783 (78,098-80,331) | 81,479 | 12.353 / 13.645 | `ivf-rabitq-b8-LUT32` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 912.0, "build_ms": 1398.6, "serialize_ms": 805.2, "deserialize_ms": 992.3, "n_probes": 768, "mode": 1, "mode_name": "LUT32"} |
| 0.9939 | 74,373 (73,707-74,698) | 75,885 | 13.392 / 13.879 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 1024, "mode": 1, "mode_name": "LUT32"} |
| 0.9943 | 73,184 (72,919-73,259) | 74,761 | 13.655 / 13.705 | `ivf-rabitq-b8-QUANT4` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 912.0, "build_ms": 1398.6, "serialize_ms": 805.2, "deserialize_ms": 992.3, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |
| 0.9948 | 70,099 (52,719-70,868) | 71,825 | 14.133 / 15.374 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |

**IVF-RaBitQ 9-bit**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7532 | 507,519 (414,280-510,178) | 583,887 | 1.962 / 1.991 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.7534 | 493,110 (479,100-494,694) | 575,846 | 2.028 / 2.040 | `ivf-rabitq-b9-QUANT8` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 16, "mode": 3, "mode_name": "QUANT8"} |
| 0.8485 | 399,139 (393,262-403,674) | 456,168 | 2.418 / 2.833 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 32, "mode": 1, "mode_name": "LUT32"} |
| 0.9152 | 337,725 (331,038-341,822) | 374,481 | 2.915 / 3.210 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 64, "mode": 1, "mode_name": "LUT32"} |
| 0.9153 | 330,496 (307,949-333,897) | 369,764 | 2.934 / 3.596 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9434 | 297,852 (294,348-298,636) | 325,683 | 3.351 / 3.385 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 96, "mode": 1, "mode_name": "LUT32"} |
| 0.9576 | 262,547 (252,201-264,344) | 278,063 | 3.769 / 3.942 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9579 | 236,654 (221,106-270,595) | 292,090 | 4.294 / 4.885 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 128, "mode": 1, "mode_name": "LUT32"} |
| 0.9740 | 212,106 (210,346-212,347) | 224,174 | 4.709 / 4.737 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 192, "mode": 1, "mode_name": "LUT32"} |
| 0.9813 | 185,432 (183,838-187,070) | 195,954 | 5.354 / 5.435 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 256, "mode": 1, "mode_name": "LUT32"} |
| 0.9818 | 178,544 (175,717-181,775) | 189,628 | 5.539 / 5.839 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9884 | 137,793 (90,743-138,744) | 145,046 | 7.149 / 7.529 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 384, "mode": 1, "mode_name": "LUT32"} |
| 0.9886 | 134,584 (129,106-136,572) | 138,812 | 7.347 / 7.754 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9905 | 117,633 (116,927-119,543) | 122,364 | 8.358 / 8.502 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 512, "mode": 1, "mode_name": "LUT32"} |
| 0.9908 | 110,312 (78,185-115,215) | 116,158 | 8.662 / 10.018 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9909 | 98,162 (97,614-99,170) | 99,976 | 10.075 / 10.317 | `ivf-rabitq-b9-QUANT8` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 512, "mode": 3, "mode_name": "QUANT8"} |
| 0.9927 | 90,818 (89,797-91,829) | 92,450 | 10.863 / 11.368 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 768, "mode": 1, "mode_name": "LUT32"} |
| 0.9932 | 87,785 (87,222-88,421) | 90,350 | 11.259 / 12.304 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9939 | 74,373 (73,707-74,698) | 75,885 | 13.392 / 13.879 | `ivf-rabitq-b9-LUT32` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 1024, "mode": 1, "mode_name": "LUT32"} |
| 0.9948 | 70,099 (52,719-70,868) | 71,825 | 14.133 / 15.374 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 998.0, "build_ms": 1390.8, "serialize_ms": 874.5, "deserialize_ms": 1108.1, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |

**IVF-PQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.6885 | 292,289 (287,428-293,120) | 311,624 | 3.414 / 3.442 | `ivfpq-pq192-f16` {"n_lists": 4096, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 6.854, "index_mib": 416.0, "n_probes": 16, "lut": "f16", "refine_ratio": 1} |
| 0.7275 | 162,098 (161,718-165,334) | 165,751 | 6.170 / 6.183 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.7652 | 160,677 (157,095-164,344) | 164,022 | 6.220 / 6.263 | `ivfpq-pq192-f32` {"n_lists": 4096, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 6.854, "index_mib": 416.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.8162 | 85,932 (85,888-86,591) | 86,807 | 11.636 / 11.661 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.8281 | 57,835 (57,826-57,974) | 58,547 | 17.280 / 17.323 | `ivfpq-pq192-f16` {"n_lists": 4096, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 6.854, "index_mib": 416.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.8291 | 57,693 (57,606-57,905) | 58,334 | 17.330 / 17.373 | `ivfpq-pq192-f32` {"n_lists": 4096, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 6.854, "index_mib": 416.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.8298 | 51,552 (51,493-51,569) | 52,201 | 19.392 / 19.433 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1} |
| 0.8380 | 44,210 (44,187-44,302) | 44,573 | 22.622 / 22.682 | `ivfpq-pq192-f16` {"n_lists": 4096, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 6.854, "index_mib": 416.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.8792 | 44,206 (44,190-44,395) | 44,608 | 22.623 / 22.717 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 64, "lut": "f32", "refine_ratio": 1} |
| 0.9061 | 29,922 (29,907-29,959) | 30,045 | 33.421 / 33.470 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.9187 | 22,620 (22,616-22,629) | 22,722 | 44.173 / 44.381 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 128, "lut": "f32", "refine_ratio": 1} |
| 0.9271 | 17,741 (17,725-17,768) | 17,827 | 56.365 / 56.449 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.9328 | 15,205 (15,202-15,236) | 15,254 | 65.707 / 66.020 | `ivfpq-pq384-f32` {"n_lists": 4096, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 8.032, "index_mib": 522.0, "n_probes": 192, "lut": "f32", "refine_ratio": 1} |
| 0.9399 | 13,426 (13,411-13,430) | 13,470 | 74.478 / 74.594 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.9521 | 9,047 (9,045-9,050) | 9,068 | 110.516 / 110.694 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1} |
| 0.9588 | 6,808 (6,806-6,814) | 6,825 | 146.831 / 146.998 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1} |
| 0.9638 | 5,064 (5,057-5,067) | 5,074 | 197.424 / 197.874 | `ivfpq-pq768-f32` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 192, "lut": "f32", "refine_ratio": 1} |
| 0.9640 | 4,590 (4,574-4,595) | 4,597 | 217.849 / 218.175 | `ivfpq-pq768-f16` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1} |
| 0.9712 | 3,851 (3,849-3,851) | 3,854 | 259.676 / 259.717 | `ivfpq-pq768-f32` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 256, "lut": "f32", "refine_ratio": 1} |
| 0.9774 | 2,662 (2,660-2,663) | 2,664 | 375.617 / 375.882 | `ivfpq-pq768-f32` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1} |
| 0.9805 | 2,020 (2,020-2,021) | 2,022 | 494.948 / 495.131 | `ivfpq-pq768-f32` {"n_lists": 4096, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 11.883, "index_mib": 1052.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1} |

**GPU brute force**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.9993 | 8,865 (8,832-8,892) | 8,926 | 112.848 / 113.720 | `bruteforce-fp32` {"metric": "sqeuclidean", "dtype": "fp32", "kind": "bruteforce", "build_s": 2.488, "index_mib": 2746.0} |

### Matched recall: JHQ-GPU v69 throughput / other method (protocol A)

measured = best measured point at recall >= r (no interpolation); interp = log-linear between bracketing frontier points; '-' = the method does not cover r.

| r | JHQ QPS (measured / interp) | CAGRA fp32: QPS, ratio | CAGRA int8: QPS, ratio | IVF-RaBitQ: QPS, ratio | IVF-RaBitQ 9-bit: QPS, ratio | IVF-PQ: QPS, ratio | GPU brute force: QPS, ratio |
|---:|---|---|---|---|---|---|---|
| 0.9 | 407,813 / 408,221 | 268,660 / 320,315, x1.52 / x1.27 | 775,289 / 807,338, x0.53 / x0.51 | 337,725 / 350,832, x1.21 / x1.16 | 337,725 / 350,832, x1.21 / x1.16 | 29,922 / 32,691, x13.63 / x12.49 | 8,865, x46.00 (meas.) |
| 0.92 | 407,813 / 407,881 | 268,660 / 291,269, x1.52 / x1.40 | 568,601 / 580,664, x0.72 / x0.70 | 297,852 / 324,797, x1.37 / x1.26 | 297,852 / 324,797, x1.37 / x1.26 | 17,741 / 21,785, x22.99 / x18.72 | 8,865, x46.00 (meas.) |
| 0.94 | 329,529 / 330,091 | 247,889 / 247,889, x1.33 / x1.33 | 184,977 / 261,170, x1.78 / x1.26 | 297,852 / 301,624, x1.11 / x1.09 | 297,852 / 301,624, x1.11 / x1.09 | 9,047 / 13,426, x36.42 / x24.59 | 8,865, x37.17 (meas.) |
| 0.95 | 284,210 / 327,252 | 186,406 / 217,048, x1.52 / x1.51 | - | 262,547 / 280,888, x1.08 / x1.17 | 262,547 / 280,888, x1.08 / x1.17 | 9,047 / 9,683, x31.41 / x33.79 | 8,865, x32.06 (meas.) |
| 0.96 | 284,210 / 291,530 | 186,406 / 196,107, x1.52 / x1.49 | - | 212,106 / 233,298, x1.34 / x1.25 | 212,106 / 233,298, x1.34 / x1.25 | 5,064 / 6,342, x56.12 / x45.97 | 8,865, x32.06 (meas.) |
| 0.97 | 221,928 / 230,822 | 144,365 / 163,058, x1.54 / x1.42 | - | 212,106 / 217,956, x1.05 / x1.06 | 212,106 / 217,956, x1.05 / x1.06 | 3,851 / 3,965, x57.63 / x58.21 | 8,865, x25.04 (meas.) |
| 0.98 | 168,370 / 185,563 | 98,469 / 127,900, x1.71 / x1.45 | - | 185,432 / 189,923, x0.91 / x0.98 | 185,432 / 189,923, x0.91 / x0.98 | 2,020 / 2,112, x83.33 / x87.85 | 8,865, x18.99 (meas.) |
| 0.985 | 140,941 / 140,941 | 98,469 / 107,185, x1.43 / x1.31 | - | 137,793 / 157,468, x1.02 / x0.90 | 137,793 / 157,468, x1.02 / x0.90 | - | 8,865, x15.90 (meas.) |
| 0.99 | 108,380 / 108,380 | 76,111 / 76,111, x1.42 / x1.42 | - | 117,633 / 121,875, x0.92 / x0.89 | 117,633 / 121,875, x0.92 / x0.89 | - | 8,865, x12.23 (meas.) |
| 0.995 | - | 52,516 (JHQ -) | - | - | - | - | 8,865 (JHQ -) |

### Sensitivity: matched recall on QPS = 1000 / median call latency (interp ratio)

| r | CAGRA fp32 | CAGRA int8 | IVF-RaBitQ | IVF-RaBitQ 9-bit |
|---:|---:|---:|---:|---:|
| 0.9 | x1.28 | x0.51 | x1.14 | x1.14 |
| 0.92 | x1.40 | x0.70 | x1.22 | x1.22 |
| 0.94 | x1.33 | x1.26 | x1.09 | x1.09 |
| 0.95 | x1.51 | - | x1.16 | x1.16 |
| 0.96 | x1.49 | - | x1.27 | x1.27 |
| 0.97 | x1.42 | - | x1.06 | x1.06 |
| 0.98 | x1.45 | - | x0.97 | x0.97 |
| 0.985 | x1.31 | - | x0.88 | x0.88 |
| 0.99 | x1.43 | - | x0.88 | x0.88 |
| 0.995 | - | - | - | - |

### Matched throughput: best Recall@10 at QPS >= q (measured points, protocol A)

| method | >= 100K | >= 150K | >= 200K | >= 250K | >= 300K | >= 400K |
|---|---:|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 0.9904 | 0.9827 | 0.9739 | 0.9622 | 0.9494 | 0.9240 |
| CAGRA fp32 | 0.9788 | 0.9688 | 0.9477 | 0.9370 | 0.8926 | 0.8740 |
| CAGRA int8 | 0.9445 | 0.9434 | 0.9398 | 0.9398 | 0.9375 | 0.9317 |
| IVF-RaBitQ | 0.9917 | 0.9818 | 0.9740 | 0.9576 | 0.9153 | 0.7534 |
| IVF-RaBitQ 9-bit | 0.9908 | 0.9818 | 0.9740 | 0.9576 | 0.9153 | 0.7534 |
| IVF-PQ | 0.7652 | 0.7652 | 0.6885 | 0.6885 | - | - |
| GPU brute force | - | - | - | - | - | - |

### Variation across the rounds of a point: (max - min) / median of the per-round QPS

| method | points | median spread | 90th pct | max | points over 5% |
|---|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 27 | 1.34% | 3.06% | 5.67% | 1 |
| CAGRA fp32 | 64 | 0.17% | 0.67% | 2.02% | 0 |
| CAGRA int8 | 64 | 0.55% | 1.73% | 6.09% | 1 |
| IVF-RaBitQ | 132 | 3.80% | 33.12% | 60.24% | 53 |
| IVF-PQ | 54 | 0.21% | 0.74% | 4.51% | 0 |
| GPU brute force | 1 | 0.67% | 0.67% | 0.67% | 0 |

## arXiv-768 (2.25M x 768)

| method | points | recall range (frontier) | best QPS | frontier points | device memory, MiB |
|---|---:|---|---:|---:|---|
| JHQ-GPU v69 | 29 | 0.8635-0.9903 | 692,683 | 13 | process 2,567-2,753 (incl. context) |
| CAGRA fp32 | 64 | 0.8981-0.9995 | 457,413 | 36 | index 7,008-7,148 |
| CAGRA int8 | 64 | 0.8742-0.9620 | 1,207,945 | 22 | index 2,056-2,198 |
| IVF-RaBitQ | 132 | 0.8696-0.9964 | 433,668 | 13 | index 1,094-2,126 |
| IVF-RaBitQ 9-bit | 44 | 0.8696-0.9964 | 433,668 | 13 | index 2,126 |
| IVF-PQ | 54 | 0.7775-0.9844 | 287,392 | 20 | index 810-2,604 |
| GPU brute force | 1 | 1.0000-1.0000 | 3,667 | 1 | index 6,630 |

### Frontier points (protocol A)


**JHQ-GPU v69**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8635 | 692,683 (689,375-695,548) | - | 1.442 / 1.468 | `jhq-v69` {"nprobe": 16, "alpha": 25, "ck": 250, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9157 | 563,087 (550,339-566,156) | - | 1.771 / 1.801 | `jhq-v69` {"nprobe": 32, "alpha": 25, "ck": 250, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9255 | 382,225 (375,785-386,672) | - | 2.606 / 2.634 | `jhq-v69` {"nprobe": 32, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9581 | 312,792 (311,847-314,838) | - | 3.194 / 3.219 | `jhq-v69` {"nprobe": 64, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9700 | 258,567 (256,966-259,643) | - | 3.870 / 3.914 | `jhq-v69` {"nprobe": 96, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9766 | 225,588 (224,105-226,549) | - | 4.430 / 4.501 | `jhq-v69` {"nprobe": 128, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9831 | 174,751 (173,821-175,611) | - | 5.721 / 5.788 | `jhq-v69` {"nprobe": 192, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9856 | 145,818 (145,754-147,520) | - | 6.836 / 6.933 | `jhq-v69` {"nprobe": 256, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9865 | 120,580 (120,485-120,686) | - | 8.282 / 8.358 | `jhq-v69` {"nprobe": 256, "alpha": 200, "ck": 2000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9879 | 108,057 (106,942-108,142) | - | 9.226 / 9.344 | `jhq-v69` {"nprobe": 384, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9892 | 91,686 (91,584-91,913) | - | 10.904 / 10.984 | `jhq-v69` {"nprobe": 384, "alpha": 200, "ck": 2000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9899 | 74,676 (74,556-74,884) | - | 13.377 / 13.481 | `jhq-v69` {"nprobe": 512, "alpha": 200, "ck": 2000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |
| 0.9903 | 47,737 (47,675-47,788) | - | 20.933 / 21.158 | `jhq-v69` {"nprobe": 1024, "alpha": 100, "ck": 1000, "nlist": 8192, "M": 96, "Br": 8, "batch": 1000} |

**CAGRA fp32**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8981 | 457,413 (451,755-460,030) | 518,073 | 2.174 / 2.200 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 32, "search_width": 1} |
| 0.9027 | 425,962 (420,737-426,215) | 474,197 | 2.348 / 2.360 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 32, "search_width": 2} |
| 0.9154 | 378,765 (377,582-379,433) | 416,911 | 2.639 / 2.657 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 32, "search_width": 4} |
| 0.9239 | 303,825 (296,064-304,247) | 326,069 | 3.288 / 3.337 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 32, "search_width": 8} |
| 0.9568 | 262,475 (260,013-263,462) | 279,805 | 3.802 / 3.829 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 64, "search_width": 1} |
| 0.9571 | 252,051 (246,780-252,648) | 267,525 | 3.961 / 3.975 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 64, "search_width": 2} |
| 0.9619 | 234,521 (230,271-238,823) | 250,570 | 4.197 / 4.437 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 64, "search_width": 4} |
| 0.9635 | 210,774 (206,626-211,137) | 221,665 | 4.741 / 4.760 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 32, "search_width": 4} |
| 0.9649 | 208,830 (204,839-209,406) | 220,544 | 4.779 / 4.804 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 64, "search_width": 8} |
| 0.9764 | 181,770 (179,641-184,134) | 192,185 | 5.439 / 5.582 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 96, "search_width": 1} |
| 0.9766 | 179,737 (175,992-181,028) | 188,681 | 5.518 / 5.746 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 96, "search_width": 2} |
| 0.9774 | 169,595 (164,078-172,246) | 179,817 | 5.792 / 5.910 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 96, "search_width": 4} |
| 0.9789 | 158,716 (155,770-158,955) | 164,769 | 6.294 / 6.329 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 96, "search_width": 8} |
| 0.9840 | 142,604 (140,942-143,675) | 148,493 | 6.954 / 6.992 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 128, "search_width": 1} |
| 0.9846 | 138,202 (137,102-141,390) | 145,781 | 7.081 / 7.124 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 128, "search_width": 2} |
| 0.9849 | 135,818 (133,942-135,954) | 140,183 | 7.350 / 7.399 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 128, "search_width": 4} |
| 0.9851 | 130,955 (129,488-132,036) | 135,951 | 7.568 / 7.611 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 64, "search_width": 2} |
| 0.9857 | 127,184 (126,456-127,692) | 131,366 | 7.832 / 7.914 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 128, "search_width": 8} |
| 0.9859 | 118,090 (117,194-118,391) | 121,688 | 8.453 / 8.505 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 64, "search_width": 8} |
| 0.9905 | 98,505 (98,302-98,864) | 101,017 | 10.123 / 10.164 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 192, "search_width": 1} |
| 0.9906 | 96,710 (95,816-96,797) | 98,895 | 10.336 / 10.373 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 192, "search_width": 4} |
| 0.9917 | 94,903 (94,740-95,081) | 97,165 | 10.535 / 10.583 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 96, "search_width": 1} |
| 0.9924 | 86,946 (86,362-87,049) | 88,742 | 11.488 / 11.547 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 96, "search_width": 8} |
| 0.9931 | 76,491 (76,402-76,544) | 77,886 | 13.074 / 13.104 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 256, "search_width": 1} |
| 0.9934 | 75,933 (75,777-75,994) | 77,337 | 13.163 / 13.205 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 256, "search_width": 2} |
| 0.9944 | 72,943 (72,930-73,009) | 74,277 | 13.696 / 13.749 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 128, "search_width": 1} |
| 0.9945 | 71,668 (71,498-71,755) | 72,963 | 13.936 / 14.004 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 128, "search_width": 2} |
| 0.9949 | 68,696 (68,560-68,736) | 69,764 | 14.557 / 14.583 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 128, "search_width": 8} |
| 0.9966 | 52,046 (51,984-52,072) | 52,779 | 19.215 / 19.254 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 384, "search_width": 1} |
| 0.9967 | 51,895 (51,371-51,903) | 52,606 | 19.271 / 19.324 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 384, "search_width": 2} |
| 0.9969 | 50,086 (50,063-50,111) | 50,681 | 19.966 / 20.012 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 192, "search_width": 2} |
| 0.9970 | 49,107 (49,099-49,125) | 49,728 | 20.369 / 20.407 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 192, "search_width": 4} |
| 0.9971 | 48,187 (48,164-48,232) | 48,731 | 20.749 / 20.799 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 192, "search_width": 8} |
| 0.9978 | 40,270 (40,223-40,286) | 40,727 | 24.832 / 24.888 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.185, "index_mib": 7008.0, "itopk_size": 512, "search_width": 2} |
| 0.9990 | 26,260 (26,251-26,271) | 26,490 | 38.077 / 38.226 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 384, "search_width": 4} |
| 0.9995 | 20,163 (20,155-20,189) | 20,319 | 49.575 / 49.739 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 512, "search_width": 1} |

**CAGRA int8**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8742 | 1,207,945 (1,073,919-1,210,779) | 1,766,322 | 0.826 / 0.838 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 32, "search_width": 1} |
| 0.8840 | 1,134,875 (1,127,118-1,137,204) | 1,606,521 | 0.881 / 0.890 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 32, "search_width": 2} |
| 0.8949 | 981,858 (805,449-991,194) | 1,335,872 | 1.009 / 1.028 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 32, "search_width": 4} |
| 0.9307 | 788,159 (782,607-791,153) | 974,093 | 1.267 / 1.280 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 32, "search_width": 1} |
| 0.9331 | 773,712 (732,127-778,539) | 970,513 | 1.286 / 1.302 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 64, "search_width": 1} |
| 0.9351 | 757,738 (754,926-760,468) | 948,930 | 1.315 / 1.332 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 64, "search_width": 2} |
| 0.9390 | 621,028 (619,034-621,879) | 731,799 | 1.607 / 1.621 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 32, "search_width": 4} |
| 0.9478 | 574,516 (570,624-574,764) | 673,075 | 1.738 / 1.759 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 96, "search_width": 2} |
| 0.9504 | 480,594 (464,152-482,070) | 551,024 | 2.074 / 2.099 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 96, "search_width": 8} |
| 0.9536 | 469,099 (454,253-470,721) | 527,415 | 2.125 / 2.136 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 64, "search_width": 1} |
| 0.9546 | 412,308 (390,477-413,096) | 459,019 | 2.421 / 2.438 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 64, "search_width": 4} |
| 0.9585 | 337,020 (333,506-337,667) | 364,927 | 2.966 / 2.982 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 1} |
| 0.9586 | 327,610 (307,815-328,882) | 354,507 | 3.046 / 3.078 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 2} |
| 0.9591 | 309,073 (306,483-309,474) | 333,696 | 3.232 / 3.251 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 4} |
| 0.9593 | 261,149 (252,339-261,669) | 276,521 | 3.828 / 3.846 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 128, "search_width": 1} |
| 0.9594 | 257,978 (249,801-258,385) | 274,174 | 3.870 / 3.895 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 128, "search_width": 2} |
| 0.9600 | 257,867 (250,963-258,688) | 275,637 | 3.866 / 3.902 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 256, "search_width": 2} |
| 0.9601 | 245,765 (240,795-247,470) | 262,296 | 4.045 / 4.093 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 256, "search_width": 8} |
| 0.9614 | 180,311 (179,009-181,808) | 189,618 | 5.505 / 5.598 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 192, "search_width": 2} |
| 0.9616 | 140,413 (140,345-140,511) | 145,057 | 7.121 / 7.145 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 256, "search_width": 2} |
| 0.9617 | 138,942 (137,893-138,968) | 143,139 | 7.196 / 7.212 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 256, "search_width": 1} |
| 0.9620 | 75,750 (74,978-75,801) | 77,018 | 13.190 / 13.282 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 512, "search_width": 8} |

**IVF-RaBitQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8696 | 433,668 (422,181-439,572) | 482,932 | 2.271 / 2.365 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.9217 | 353,508 (320,341-353,639) | 389,271 | 2.828 / 2.843 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.9572 | 294,962 (292,054-295,380) | 319,439 | 3.389 / 3.409 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9719 | 254,707 (253,853-254,857) | 271,481 | 3.925 / 3.939 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 96, "mode": 0, "mode_name": "LUT16"} |
| 0.9798 | 230,576 (226,618-231,648) | 245,696 | 4.323 / 4.354 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9872 | 186,860 (185,450-187,155) | 195,918 | 5.348 / 5.374 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9910 | 161,574 (157,041-161,935) | 167,246 | 6.127 / 6.424 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 256, "mode": 1, "mode_name": "LUT32"} |
| 0.9912 | 157,864 (90,947-164,102) | 170,202 | 6.094 / 6.892 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9936 | 124,043 (122,006-124,652) | 127,426 | 8.013 / 8.271 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 384, "mode": 1, "mode_name": "LUT32"} |
| 0.9939 | 119,134 (110,910-125,277) | 129,916 | 7.969 / 8.238 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9950 | 105,527 (102,797-105,728) | 107,896 | 9.458 / 9.511 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9959 | 79,209 (78,970-79,677) | 80,681 | 12.565 / 12.613 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9964 | 63,365 (49,180-63,585) | 63,914 | 15.676 / 16.111 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |

**IVF-RaBitQ 9-bit**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8696 | 433,668 (422,181-439,572) | 482,932 | 2.271 / 2.365 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.9217 | 353,508 (320,341-353,639) | 389,271 | 2.828 / 2.843 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.9572 | 294,962 (292,054-295,380) | 319,439 | 3.389 / 3.409 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9719 | 254,707 (253,853-254,857) | 271,481 | 3.925 / 3.939 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 96, "mode": 0, "mode_name": "LUT16"} |
| 0.9798 | 230,576 (226,618-231,648) | 245,696 | 4.323 / 4.354 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9872 | 186,860 (185,450-187,155) | 195,918 | 5.348 / 5.374 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9910 | 161,574 (157,041-161,935) | 167,246 | 6.127 / 6.424 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 256, "mode": 1, "mode_name": "LUT32"} |
| 0.9912 | 157,864 (90,947-164,102) | 170,202 | 6.094 / 6.892 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9936 | 124,043 (122,006-124,652) | 127,426 | 8.013 / 8.271 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 384, "mode": 1, "mode_name": "LUT32"} |
| 0.9939 | 119,134 (110,910-125,277) | 129,916 | 7.969 / 8.238 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9950 | 105,527 (102,797-105,728) | 107,896 | 9.458 / 9.511 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9959 | 79,209 (78,970-79,677) | 80,681 | 12.565 / 12.613 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9964 | 63,365 (49,180-63,585) | 63,914 | 15.676 / 16.111 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 2126.0, "build_ms": 3549.9, "serialize_ms": 2007.5, "deserialize_ms": 2672.8, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |

**IVF-PQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7775 | 287,392 (280,821-289,612) | 301,591 | 3.465 / 3.499 | `ivfpq-pq192-f32` {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.8118 | 155,862 (154,277-159,378) | 159,918 | 6.409 / 6.453 | `ivfpq-pq192-f32` {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.8432 | 148,698 (148,402-151,075) | 152,242 | 6.724 / 6.739 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.8554 | 84,608 (84,498-84,736) | 86,491 | 11.808 / 11.959 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 16, "lut": "f16", "refine_ratio": 1} |
| 0.8923 | 83,724 (83,709-84,956) | 84,598 | 11.941 / 11.967 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.9038 | 50,684 (50,643-50,947) | 51,230 | 19.727 / 19.764 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1} |
| 0.9217 | 43,726 (43,645-43,789) | 43,983 | 22.868 / 23.051 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 64, "lut": "f32", "refine_ratio": 1} |
| 0.9348 | 29,658 (29,618-29,713) | 29,767 | 33.713 / 33.804 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.9367 | 26,362 (26,323-26,388) | 26,530 | 37.936 / 38.020 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1} |
| 0.9403 | 22,460 (22,439-22,535) | 22,547 | 44.495 / 44.715 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 128, "lut": "f32", "refine_ratio": 1} |
| 0.9506 | 17,578 (17,570-17,601) | 17,676 | 56.849 / 57.017 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.9568 | 13,272 (13,258-13,294) | 13,330 | 75.341 / 75.468 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.9609 | 9,756 (9,748-9,760) | 9,796 | 102.504 / 102.586 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.9649 | 8,909 (8,908-8,916) | 8,944 | 112.237 / 112.440 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1} |
| 0.9677 | 7,718 (7,714-7,719) | 7,745 | 129.595 / 129.715 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 128, "lut": "f32", "refine_ratio": 1} |
| 0.9680 | 6,746 (6,723-6,772) | 6,763 | 148.374 / 148.643 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1} |
| 0.9759 | 5,185 (5,184-5,191) | 5,196 | 192.824 / 192.959 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 192, "lut": "f32", "refine_ratio": 1} |
| 0.9798 | 3,954 (3,952-3,955) | 3,963 | 252.870 / 253.141 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 256, "lut": "f32", "refine_ratio": 1} |
| 0.9829 | 2,658 (2,656-2,659) | 2,661 | 376.325 / 376.401 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1} |
| 0.9844 | 2,009 (2,007-2,009) | 2,010 | 497.823 / 497.931 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1} |

**GPU brute force**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 1.0000 | 3,667 (3,660-3,672) | 3,674 | 272.643 / 272.789 | `bruteforce-fp32` {"metric": "sqeuclidean", "dtype": "fp32", "kind": "bruteforce", "build_s": 14.81, "index_mib": 6630.0} |

### Matched recall: JHQ-GPU v69 throughput / other method (protocol A)

measured = best measured point at recall >= r (no interpolation); interp = log-linear between bracketing frontier points; '-' = the method does not cover r.

| r | JHQ QPS (measured / interp) | CAGRA fp32: QPS, ratio | CAGRA int8: QPS, ratio | IVF-RaBitQ: QPS, ratio | IVF-RaBitQ 9-bit: QPS, ratio | IVF-PQ: QPS, ratio | GPU brute force: QPS, ratio |
|---:|---|---|---|---|---|---|---|
| 0.9 | 563,087 / 599,283 | 425,962 / 444,151, x1.32 / x1.35 | 788,159 / 951,597, x0.71 / x0.63 | 353,508 / 384,917, x1.59 / x1.56 | 353,508 / 384,917, x1.59 / x1.56 | 50,684 / 59,827, x11.11 / x10.02 | 3,667, x153.55 (meas.) |
| 0.92 | 382,225 / 475,061 | 303,825 / 336,166, x1.26 / x1.41 | 788,159 / 841,662, x0.48 / x0.56 | 353,508 / 355,873, x1.08 / x1.33 | 353,508 / 355,873, x1.08 / x1.33 | 43,726 / 44,343, x8.74 / x10.71 | 3,667, x104.23 (meas.) |
| 0.94 | 312,792 / 349,619 | 262,475 / 282,834, x1.19 / x1.24 | 574,516 / 615,558, x0.54 / x0.57 | 294,962 / 322,006, x1.06 / x1.09 | 294,962 / 322,006, x1.06 / x1.09 | 22,460 / 22,460, x13.93 / x15.57 | 3,667, x85.30 (meas.) |
| 0.95 | 312,792 / 328,767 | 262,475 / 270,533, x1.19 / x1.22 | 480,594 / 480,594, x0.65 / x0.68 | 294,962 / 305,994, x1.06 / x1.07 | 294,962 / 305,994, x1.06 / x1.07 | 17,578 / 17,831, x17.79 / x18.44 | 3,667, x85.30 (meas.) |
| 0.96 | 258,567 / 303,427 | 234,521 / 241,309, x1.10 / x1.26 | 257,867 / 257,867, x1.00 / x1.18 | 254,707 / 286,832, x1.02 / x1.06 | 254,707 / 286,832, x1.02 / x1.06 | 9,756 / 10,438, x26.50 / x29.07 | 3,667, x70.51 (meas.) |
| 0.97 | 258,567 / 258,567 | 181,770 / 196,365, x1.42 / x1.32 | - | 254,707 / 259,584, x1.02 / x1.00 | 254,707 / 259,584, x1.02 / x1.00 | 5,185 / 6,311, x49.87 / x40.97 | 3,667, x70.51 (meas.) |
| 0.98 | 174,751 / 197,383 | 142,604 / 155,093, x1.23 / x1.27 | - | 186,860 / 230,576, x0.94 / x0.86 | 186,860 / 230,576, x0.94 / x0.86 | 2,658 / 3,954, x65.75 / x49.92 | 3,667, x47.65 (meas.) |
| 0.985 | 145,818 / 152,292 | 130,955 / 138,202, x1.11 / x1.10 | - | 186,860 / 198,911, x0.78 / x0.77 | 186,860 / 198,911, x0.78 / x0.77 | - | 3,667, x39.76 (meas.) |
| 0.99 | 47,737 / 74,676 | 98,505 / 100,466, x0.48 / x0.74 | - | 161,574 / 167,876, x0.30 / x0.44 | 161,574 / 167,876, x0.30 / x0.44 | - | 3,667, x13.02 (meas.) |
| 0.995 | - | 52,046 (JHQ -) | - | 105,527 (JHQ -) | 105,527 (JHQ -) | - | 3,667 (JHQ -) |

### Sensitivity: matched recall on QPS = 1000 / median call latency (interp ratio)

| r | CAGRA fp32 | CAGRA int8 | IVF-RaBitQ | IVF-RaBitQ 9-bit |
|---:|---:|---:|---:|---:|
| 0.9 | x1.35 | x0.63 | x1.55 | x1.55 |
| 0.92 | x1.42 | x0.56 | x1.34 | x1.34 |
| 0.94 | x1.24 | x0.57 | x1.09 | x1.09 |
| 0.95 | x1.22 | x0.68 | x1.08 | x1.08 |
| 0.96 | x1.25 | x1.17 | x1.06 | x1.06 |
| 0.97 | x1.31 | - | x1.00 | x1.00 |
| 0.98 | x1.12 | - | x0.76 | x0.76 |
| 0.985 | x1.08 | - | x0.77 | x0.77 |
| 0.99 | x0.74 | - | x0.44 | x0.44 |
| 0.995 | - | - | - | - |

### Matched throughput: best Recall@10 at QPS >= q (measured points, protocol A)

| method | >= 100K | >= 150K | >= 200K | >= 250K | >= 300K | >= 400K |
|---|---:|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 0.9879 | 0.9831 | 0.9766 | 0.9700 | 0.9581 | 0.9157 |
| CAGRA fp32 | 0.9859 | 0.9789 | 0.9649 | 0.9571 | 0.9239 | 0.9027 |
| CAGRA int8 | 0.9617 | 0.9614 | 0.9601 | 0.9600 | 0.9591 | 0.9546 |
| IVF-RaBitQ | 0.9950 | 0.9912 | 0.9798 | 0.9719 | 0.9217 | 0.8696 |
| IVF-RaBitQ 9-bit | 0.9950 | 0.9912 | 0.9798 | 0.9719 | 0.9217 | 0.8696 |
| IVF-PQ | 0.8432 | 0.8118 | 0.7775 | 0.7775 | - | - |
| GPU brute force | - | - | - | - | - | - |

### Variation across the rounds of a point: (max - min) / median of the per-round QPS

| method | points | median spread | 90th pct | max | points over 5% |
|---|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 29 | 1.08% | 2.81% | 6.06% | 1 |
| CAGRA fp32 | 64 | 0.70% | 2.33% | 4.82% | 0 |
| CAGRA int8 | 64 | 1.37% | 4.26% | 18.92% | 7 |
| IVF-RaBitQ | 132 | 3.37% | 25.79% | 59.61% | 48 |
| IVF-PQ | 54 | 0.26% | 1.01% | 3.27% | 0 |
| GPU brute force | 1 | 0.34% | 0.34% | 0.34% | 0 |

## OpenAI3-1536 (1.0M x 1536)

| method | points | recall range (frontier) | best QPS | frontier points | device memory, MiB |
|---|---:|---|---:|---:|---|
| JHQ-GPU v69 | 37 | 0.8456-0.9862 | 568,840 | 11 | process 2,345-2,889 (incl. context) |
| CAGRA fp32 | 64 | 0.8761-0.9924 | 242,603 | 27 | index 6,096-6,110 |
| CAGRA int8 | 64 | 0.8610-0.9667 | 703,417 | 22 | index 1,706-1,718 |
| IVF-RaBitQ | 264 | 0.7918-0.9943 | 318,845 | 26 | index 980-1,918 |
| IVF-RaBitQ 9-bit | 88 | 0.7918-0.9943 | 318,845 | 24 | index 1,894-1,918 |
| IVF-PQ | 54 | 0.7467-0.9715 | 145,147 | 19 | index 780-2,254 |
| GPU brute force | 1 | 1.0000-1.0000 | 4,140 | 1 | index 5,876 |

### Frontier points (protocol A)


**JHQ-GPU v69**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8456 | 568,840 (567,485-569,064) | - | 1.756 / 1.781 | `jhq-v69` {"nprobe": 32, "alpha": 4, "ck": 40, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.8468 | 547,448 (538,690-547,761) | - | 1.827 / 1.838 | `jhq-v69` {"nprobe": 32, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.8934 | 451,017 (437,398-451,838) | - | 2.215 / 2.235 | `jhq-v69` {"nprobe": 64, "alpha": 4, "ck": 40, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.8948 | 437,534 (418,481-438,312) | - | 2.283 / 2.309 | `jhq-v69` {"nprobe": 64, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9294 | 320,494 (304,323-322,130) | - | 3.105 / 3.148 | `jhq-v69` {"nprobe": 128, "alpha": 4, "ck": 40, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9312 | 312,303 (305,446-314,826) | - | 3.190 / 3.230 | `jhq-v69` {"nprobe": 128, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9523 | 208,724 (208,400-209,836) | - | 4.790 / 4.834 | `jhq-v69` {"nprobe": 256, "alpha": 4, "ck": 40, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9545 | 205,492 (202,427-205,699) | - | 4.862 / 4.910 | `jhq-v69` {"nprobe": 256, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9719 | 119,423 (118,598-119,489) | - | 8.369 / 8.437 | `jhq-v69` {"nprobe": 512, "alpha": 4, "ck": 40, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9746 | 117,691 (117,631-118,501) | - | 8.483 / 8.562 | `jhq-v69` {"nprobe": 512, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |
| 0.9862 | 62,794 (62,637-63,002) | - | 15.925 / 16.016 | `jhq-v69` {"nprobe": 1024, "alpha": 8, "ck": 80, "nlist": 8192, "M": 192, "Br": 8, "batch": 1000} |

**CAGRA fp32**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8761 | 242,603 (235,953-245,430) | 276,066 | 4.077 / 4.090 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 32, "search_width": 1} |
| 0.8905 | 228,273 (224,890-228,472) | 255,038 | 4.379 / 4.394 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 32, "search_width": 2} |
| 0.8996 | 206,242 (204,742-206,260) | 227,763 | 4.850 / 4.865 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 32, "search_width": 4} |
| 0.9126 | 171,967 (170,666-172,113) | 186,771 | 5.816 / 5.839 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 32, "search_width": 8} |
| 0.9415 | 137,697 (135,486-138,130) | 147,231 | 7.237 / 7.320 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 64, "search_width": 1} |
| 0.9422 | 126,720 (126,119-127,167) | 134,540 | 7.867 / 7.896 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 64, "search_width": 4} |
| 0.9452 | 116,335 (114,557-116,384) | 123,031 | 8.593 / 8.618 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 64, "search_width": 8} |
| 0.9584 | 96,421 (96,406-96,436) | 100,929 | 10.370 / 10.391 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 96, "search_width": 1} |
| 0.9593 | 96,067 (96,033-96,109) | 100,621 | 10.405 / 10.436 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 96, "search_width": 2} |
| 0.9603 | 92,278 (92,240-92,301) | 96,332 | 10.835 / 10.867 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 96, "search_width": 4} |
| 0.9613 | 87,130 (87,118-87,242) | 90,831 | 11.474 / 11.503 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 96, "search_width": 8} |
| 0.9686 | 75,579 (74,858-75,748) | 78,516 | 13.197 / 13.254 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 128, "search_width": 1} |
| 0.9688 | 74,077 (73,515-74,557) | 77,301 | 13.403 / 13.460 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 128, "search_width": 2} |
| 0.9698 | 72,520 (71,339-72,625) | 75,103 | 13.779 / 13.817 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 128, "search_width": 4} |
| 0.9779 | 52,062 (51,849-52,078) | 53,348 | 19.204 / 19.241 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 192, "search_width": 1} |
| 0.9816 | 40,722 (40,574-40,758) | 41,504 | 24.543 / 24.619 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 256, "search_width": 1} |
| 0.9825 | 40,290 (40,022-40,359) | 41,248 | 24.807 / 24.864 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 256, "search_width": 2} |
| 0.9830 | 40,087 (39,941-40,098) | 40,814 | 24.942 / 24.989 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 256, "search_width": 4} |
| 0.9852 | 27,418 (27,338-27,434) | 27,786 | 36.469 / 36.552 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 384, "search_width": 2} |
| 0.9862 | 27,206 (27,156-27,209) | 27,568 | 36.749 / 36.829 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 384, "search_width": 4} |
| 0.9867 | 27,186 (27,095-27,194) | 27,549 | 36.772 / 36.932 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 384, "search_width": 8} |
| 0.9888 | 21,301 (21,204-21,382) | 21,583 | 46.834 / 47.183 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 512, "search_width": 1} |
| 0.9889 | 21,299 (21,218-21,327) | 21,490 | 46.923 / 47.137 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 512, "search_width": 2} |
| 0.9899 | 21,174 (21,160-21,193) | 21,383 | 47.222 / 47.383 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 512, "search_width": 4} |
| 0.9902 | 20,901 (20,848-20,924) | 21,111 | 47.833 / 48.016 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.738, "index_mib": 6110.0, "itopk_size": 512, "search_width": 8} |
| 0.9914 | 10,704 (10,673-10,712) | 10,788 | 93.355 / 93.785 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.701, "index_mib": 6096.0, "itopk_size": 512, "search_width": 1} |
| 0.9924 | 10,620 (10,611-10,633) | 10,694 | 94.156 / 94.413 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 5.701, "index_mib": 6096.0, "itopk_size": 512, "search_width": 8} |

**CAGRA int8**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8610 | 703,417 (638,403-703,568) | 1,046,476 | 1.420 / 1.433 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 32, "search_width": 1} |
| 0.8719 | 662,522 (613,029-663,464) | 959,003 | 1.508 / 1.518 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 32, "search_width": 2} |
| 0.8824 | 607,311 (556,243-608,663) | 843,628 | 1.646 / 1.661 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 32, "search_width": 4} |
| 0.8959 | 510,896 (489,024-511,835) | 670,647 | 1.956 / 1.966 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 32, "search_width": 8} |
| 0.9205 | 451,161 (428,164-452,045) | 571,135 | 2.212 / 2.228 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 64, "search_width": 1} |
| 0.9223 | 436,319 (412,538-437,025) | 548,232 | 2.289 / 2.302 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 64, "search_width": 2} |
| 0.9238 | 417,071 (408,008-417,309) | 517,173 | 2.397 / 2.409 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 64, "search_width": 4} |
| 0.9246 | 375,678 (369,334-375,824) | 454,828 | 2.660 / 2.677 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 64, "search_width": 8} |
| 0.9353 | 332,390 (321,258-333,143) | 393,284 | 3.003 / 3.021 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 96, "search_width": 2} |
| 0.9371 | 332,057 (331,969-332,232) | 393,279 | 3.011 / 3.028 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 96, "search_width": 1} |
| 0.9380 | 294,016 (288,692-294,298) | 340,497 | 3.400 / 3.415 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 96, "search_width": 8} |
| 0.9450 | 269,285 (256,423-269,706) | 307,155 | 3.709 / 3.726 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 128, "search_width": 1} |
| 0.9451 | 248,059 (247,102-254,601) | 288,110 | 3.926 / 4.207 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 128, "search_width": 4} |
| 0.9463 | 241,272 (234,866-241,426) | 270,310 | 4.141 / 4.171 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 128, "search_width": 8} |
| 0.9546 | 190,162 (186,045-190,254) | 208,052 | 5.255 / 5.285 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 192, "search_width": 1} |
| 0.9589 | 150,541 (148,306-150,782) | 161,509 | 6.627 / 6.678 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 256, "search_width": 2} |
| 0.9617 | 99,897 (98,412-100,209) | 105,121 | 9.964 / 10.007 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 384, "search_width": 2} |
| 0.9623 | 99,247 (97,803-99,787) | 104,449 | 10.026 / 10.068 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 384, "search_width": 4} |
| 0.9627 | 98,307 (97,807-98,404) | 103,028 | 10.171 / 10.269 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 384, "search_width": 8} |
| 0.9651 | 75,917 (74,520-76,081) | 78,969 | 13.183 / 13.283 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 512, "search_width": 4} |
| 0.9654 | 75,445 (74,686-76,082) | 78,958 | 13.139 / 13.419 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.354, "index_mib": 1718.0, "itopk_size": 512, "search_width": 8} |
| 0.9667 | 42,894 (42,811-43,019) | 43,916 | 23.269 / 23.410 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 0.584, "kind": "cagra-int8", "build_s": 4.36, "index_mib": 1706.0, "itopk_size": 512, "search_width": 8} |

**IVF-RaBitQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7918 | 318,845 (121,237-324,722) | 382,900 | 3.064 / 3.261 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.8232 | 306,207 (298,272-309,128) | 357,395 | 3.218 / 3.394 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.8458 | 267,416 (260,343-272,963) | 314,620 | 3.642 / 3.801 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.8459 | 263,636 (262,353-263,689) | 297,299 | 3.793 / 3.811 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 32, "mode": 1, "mode_name": "LUT32"} |
| 0.8723 | 252,662 (247,325-253,620) | 286,610 | 3.916 / 4.149 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.8932 | 225,032 (217,224-228,042) | 254,614 | 4.380 / 4.537 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.8933 | 218,657 (215,360-218,917) | 241,388 | 4.569 / 4.605 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 64, "mode": 1, "mode_name": "LUT32"} |
| 0.9198 | 209,688 (202,871-211,337) | 230,856 | 4.719 / 4.966 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9199 | 189,490 (182,689-189,602) | 206,483 | 5.278 / 5.295 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 96, "mode": 1, "mode_name": "LUT32"} |
| 0.9389 | 177,588 (175,402-178,571) | 194,112 | 5.537 / 6.000 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 96, "mode": 0, "mode_name": "LUT16"} |
| 0.9503 | 160,531 (158,948-160,682) | 173,075 | 6.223 / 6.253 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9516 | 143,390 (142,959-143,686) | 153,422 | 6.952 / 7.066 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9517 | 137,972 (137,295-138,088) | 145,171 | 7.245 / 7.266 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 192, "mode": 1, "mode_name": "LUT32"} |
| 0.9615 | 124,892 (124,066-125,338) | 131,976 | 7.979 / 8.117 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9643 | 123,976 (116,330-127,645) | 134,210 | 7.839 / 9.717 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9724 | 107,539 (107,202-107,897) | 110,988 | 9.268 / 9.475 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9730 | 83,731 (81,324-83,960) | 85,861 | 11.922 / 12.008 | `ivf-rabitq-b8-LUT16` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 1712.0, "build_ms": 2141.5, "serialize_ms": 1657.0, "deserialize_ms": 2088.1, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9781 | 80,572 (78,656-81,210) | 83,938 | 12.332 / 12.492 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9817 | 79,128 (78,547-79,678) | 82,316 | 12.477 / 13.302 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9871 | 64,483 (63,914-64,711) | 66,019 | 15.425 / 15.671 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9872 | 58,366 (48,052-58,402) | 60,010 | 17.084 / 17.186 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 768, "mode": 1, "mode_name": "LUT32"} |
| 0.9913 | 47,972 (39,087-48,649) | 49,935 | 20.472 / 24.151 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |
| 0.9914 | 47,042 (39,063-47,407) | 48,342 | 21.195 / 21.272 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 1024, "mode": 1, "mode_name": "LUT32"} |
| 0.9920 | 46,701 (39,196-46,834) | 47,578 | 21.330 / 21.541 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9941 | 39,794 (35,461-39,888) | 41,217 | 25.022 / 25.681 | `ivf-rabitq-b8-QUANT4` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 1712.0, "build_ms": 2141.5, "serialize_ms": 1657.0, "deserialize_ms": 2088.1, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |
| 0.9943 | 38,369 (35,366-38,464) | 39,187 | 26.038 / 26.174 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |

**IVF-RaBitQ 9-bit**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7918 | 318,845 (121,237-324,722) | 382,900 | 3.064 / 3.261 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.8232 | 306,207 (298,272-309,128) | 357,395 | 3.218 / 3.394 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 16, "mode": 0, "mode_name": "LUT16"} |
| 0.8458 | 267,416 (260,343-272,963) | 314,620 | 3.642 / 3.801 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.8459 | 263,636 (262,353-263,689) | 297,299 | 3.793 / 3.811 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 32, "mode": 1, "mode_name": "LUT32"} |
| 0.8723 | 252,662 (247,325-253,620) | 286,610 | 3.916 / 4.149 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 32, "mode": 0, "mode_name": "LUT16"} |
| 0.8932 | 225,032 (217,224-228,042) | 254,614 | 4.380 / 4.537 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.8933 | 218,657 (215,360-218,917) | 241,388 | 4.569 / 4.605 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 64, "mode": 1, "mode_name": "LUT32"} |
| 0.9198 | 209,688 (202,871-211,337) | 230,856 | 4.719 / 4.966 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 64, "mode": 0, "mode_name": "LUT16"} |
| 0.9199 | 189,490 (182,689-189,602) | 206,483 | 5.278 / 5.295 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 96, "mode": 1, "mode_name": "LUT32"} |
| 0.9389 | 177,588 (175,402-178,571) | 194,112 | 5.537 / 6.000 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 96, "mode": 0, "mode_name": "LUT16"} |
| 0.9503 | 160,531 (158,948-160,682) | 173,075 | 6.223 / 6.253 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 128, "mode": 0, "mode_name": "LUT16"} |
| 0.9516 | 143,390 (142,959-143,686) | 153,422 | 6.952 / 7.066 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9517 | 137,972 (137,295-138,088) | 145,171 | 7.245 / 7.266 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 192, "mode": 1, "mode_name": "LUT32"} |
| 0.9615 | 124,892 (124,066-125,338) | 131,976 | 7.979 / 8.117 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9643 | 123,976 (116,330-127,645) | 134,210 | 7.839 / 9.717 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 192, "mode": 0, "mode_name": "LUT16"} |
| 0.9724 | 107,539 (107,202-107,897) | 110,988 | 9.268 / 9.475 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 256, "mode": 0, "mode_name": "LUT16"} |
| 0.9781 | 80,572 (78,656-81,210) | 83,938 | 12.332 / 12.492 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9817 | 79,128 (78,547-79,678) | 82,316 | 12.477 / 13.302 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 384, "mode": 0, "mode_name": "LUT16"} |
| 0.9871 | 64,483 (63,914-64,711) | 66,019 | 15.425 / 15.671 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 512, "mode": 0, "mode_name": "LUT16"} |
| 0.9872 | 58,366 (48,052-58,402) | 60,010 | 17.084 / 17.186 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 768, "mode": 1, "mode_name": "LUT32"} |
| 0.9913 | 47,972 (39,087-48,649) | 49,935 | 20.472 / 24.151 | `ivf-rabitq-b9-LUT16` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 1024, "mode": 0, "mode_name": "LUT16"} |
| 0.9914 | 47,042 (39,063-47,407) | 48,342 | 21.195 / 21.272 | `ivf-rabitq-b9-LUT32` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 1918.0, "build_ms": 3203.5, "serialize_ms": 1865.4, "deserialize_ms": 2623.1, "n_probes": 1024, "mode": 1, "mode_name": "LUT32"} |
| 0.9920 | 46,701 (39,196-46,834) | 47,578 | 21.330 / 21.541 | `ivf-rabitq-b9-LUT16` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 768, "mode": 0, "mode_name": "LUT16"} |
| 0.9943 | 38,369 (35,366-38,464) | 39,187 | 26.038 / 26.174 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 1894.0, "build_ms": 2101.2, "serialize_ms": 1894.5, "deserialize_ms": 2441.4, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |

**IVF-PQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7467 | 145,147 (143,935-147,088) | 152,012 | 6.891 / 6.906 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 16, "lut": "f16", "refine_ratio": 1} |
| 0.7469 | 138,852 (137,547-143,716) | 142,506 | 7.146 / 7.245 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.7912 | 80,058 (79,807-80,159) | 82,485 | 12.481 / 12.544 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1} |
| 0.7913 | 75,858 (74,373-77,146) | 77,061 | 13.118 / 13.366 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.8243 | 43,161 (43,061-43,271) | 44,032 | 23.103 / 23.233 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1} |
| 0.8279 | 41,333 (41,314-41,346) | 42,150 | 24.191 / 24.233 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1} |
| 0.8405 | 29,837 (29,792-29,870) | 30,224 | 33.475 / 33.684 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.8412 | 28,346 (28,304-28,390) | 28,593 | 35.287 / 35.601 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.8514 | 22,732 (22,681-22,756) | 23,000 | 43.992 / 44.118 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.8706 | 21,566 (21,510-21,588) | 21,781 | 46.362 / 46.514 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1} |
| 0.8911 | 14,680 (14,626-14,687) | 14,799 | 68.091 / 68.287 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.9035 | 11,081 (11,065-11,087) | 11,145 | 90.227 / 90.556 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.9194 | 7,443 (7,434-7,445) | 7,474 | 134.325 / 134.469 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1} |
| 0.9297 | 5,598 (5,594-5,599) | 5,614 | 178.677 / 178.701 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1} |
| 0.9399 | 3,770 (3,768-3,771) | 3,776 | 265.315 / 265.455 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1} |
| 0.9462 | 2,852 (2,851-2,853) | 2,856 | 350.594 / 350.997 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1} |
| 0.9530 | 2,018 (2,013-2,019) | 2,024 | 496.331 / 496.623 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 256, "lut": "f32", "refine_ratio": 1} |
| 0.9655 | 1,362 (1,357-1,364) | 1,364 | 734.926 / 736.593 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1} |
| 0.9715 | 1,025 (1,024-1,025) | 1,026 | 975.620 / 976.268 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1} |

**GPU brute force**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 1.0000 | 4,140 (4,135-4,149) | 4,152 | 241.227 / 242.387 | `bruteforce-fp32` {"metric": "sqeuclidean", "dtype": "fp32", "kind": "bruteforce", "build_s": 7.11, "index_mib": 5876.0} |

### Matched recall: JHQ-GPU v69 throughput / other method (protocol A)

measured = best measured point at recall >= r (no interpolation); interp = log-linear between bracketing frontier points; '-' = the method does not cover r.

| r | JHQ QPS (measured / interp) | CAGRA fp32: QPS, ratio | CAGRA int8: QPS, ratio | IVF-RaBitQ: QPS, ratio | IVF-RaBitQ 9-bit: QPS, ratio | IVF-PQ: QPS, ratio | GPU brute force: QPS, ratio |
|---:|---|---|---|---|---|---|---|
| 0.9 | 320,494 / 417,536 | 171,967 / 206,242, x1.86 / x2.02 | 451,161 / 500,417, x0.71 / x0.83 | 209,688 / 216,354, x1.53 / x1.93 | 209,688 / 216,354, x1.53 / x1.93 | 11,081 / 11,997, x28.92 / x34.80 | 4,140, x77.42 (meas.) |
| 0.92 | 320,494 / 348,777 | 137,697 / 162,454, x2.33 / x2.15 | 451,161 / 451,161, x0.71 / x0.77 | 177,588 / 209,688, x1.80 / x1.66 | 177,588 / 209,688, x1.80 / x1.66 | 5,598 / 7,321, x57.26 / x47.64 | 4,140, x77.42 (meas.) |
| 0.94 | 208,724 / 263,991 | 137,697 / 139,295, x1.52 / x1.90 | 269,285 / 286,727, x0.78 / x0.92 | 160,531 / 175,866, x1.30 / x1.50 | 160,531 / 175,866, x1.30 / x1.50 | 2,852 / 3,770, x73.19 / x70.02 | 4,140, x50.42 (meas.) |
| 0.95 | 208,724 / 218,097 | 96,421 / 108,658, x2.16 / x2.01 | 190,162 / 216,980, x1.10 / x1.01 | 160,531 / 160,531, x1.30 / x1.36 | 160,531 / 160,531, x1.30 / x1.36 | 2,018 / 2,350, x103.45 / x92.79 | 4,140, x50.42 (meas.) |
| 0.96 | 119,423 / 173,097 | 92,278 / 92,278, x1.29 / x1.88 | 99,897 / 128,140, x1.20 / x1.35 | 124,892 / 126,810, x0.96 / x1.37 | 124,892 / 126,810, x0.96 / x1.37 | 1,362 / 1,619, x87.70 / x106.92 | 4,140, x28.85 (meas.) |
| 0.97 | 119,423 / 126,715 | 52,062 / 72,520, x2.29 / x1.75 | - | 107,539 / 112,168, x1.11 / x1.13 | 107,539 / 112,168, x1.11 / x1.13 | 1,025 / 1,100, x116.54 / x115.17 | 4,140, x28.85 (meas.) |
| 0.98 | 62,794 / 87,849 | 40,722 / 45,286, x1.54 / x1.94 | - | 79,128 / 79,806, x0.79 / x1.10 | 79,128 / 79,806, x0.79 / x1.10 | - | 4,140, x15.17 (meas.) |
| 0.985 | 62,794 / 67,010 | 27,418 / 27,418, x2.29 / x2.44 | - | 64,483 / 69,825, x0.97 / x0.96 | 64,483 / 69,825, x0.97 / x0.96 | - | 4,140, x15.17 (meas.) |
| 0.99 | - | 20,901 (JHQ -) | - | 47,972 (JHQ -) | 47,972 (JHQ -) | - | 4,140 (JHQ -) |
| 0.995 | - | - | - | - | - | - | 4,140 (JHQ -) |

### Sensitivity: matched recall on QPS = 1000 / median call latency (interp ratio)

| r | CAGRA fp32 | CAGRA int8 | IVF-RaBitQ | IVF-RaBitQ 9-bit |
|---:|---:|---:|---:|---:|
| 0.9 | x2.03 | x0.83 | x1.93 | x1.93 |
| 0.92 | x2.15 | x0.77 | x1.65 | x1.65 |
| 0.94 | x1.89 | x0.92 | x1.48 | x1.48 |
| 0.95 | x2.01 | x1.00 | x1.36 | x1.36 |
| 0.96 | x1.88 | x1.35 | x1.32 | x1.32 |
| 0.97 | x1.75 | - | x1.12 | x1.12 |
| 0.98 | x1.94 | - | x1.09 | x1.09 |
| 0.985 | x2.44 | - | x0.95 | x0.95 |
| 0.99 | - | - | - | - |
| 0.995 | - | - | - | - |

### Matched throughput: best Recall@10 at QPS >= q (measured points, protocol A)

| method | >= 100K | >= 150K | >= 200K | >= 250K | >= 300K | >= 400K |
|---|---:|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 0.9746 | 0.9545 | 0.9545 | 0.9312 | 0.9312 | 0.8948 |
| CAGRA fp32 | 0.9452 | 0.9126 | 0.8996 | - | - | - |
| CAGRA int8 | 0.9589 | 0.9589 | 0.9463 | 0.9450 | 0.9371 | 0.9238 |
| IVF-RaBitQ | 0.9724 | 0.9503 | 0.9198 | 0.8723 | 0.8232 | - |
| IVF-RaBitQ 9-bit | 0.9724 | 0.9503 | 0.9198 | 0.8723 | 0.8232 | - |
| IVF-PQ | 0.7469 | - | - | - | - | - |
| GPU brute force | - | - | - | - | - | - |

### Variation across the rounds of a point: (max - min) / median of the per-round QPS

| method | points | median spread | 90th pct | max | points over 5% |
|---|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 37 | 1.19% | 4.53% | 5.76% | 3 |
| CAGRA fp32 | 64 | 0.45% | 1.57% | 4.02% | 0 |
| CAGRA int8 | 64 | 1.21% | 4.16% | 9.26% | 5 |
| IVF-RaBitQ | 264 | 2.88% | 29.48% | 63.82% | 104 |
| IVF-PQ | 54 | 0.21% | 3.26% | 27.96% | 4 |
| GPU brute force | 1 | 0.33% | 0.33% | 0.33% | 0 |

## OpenAI3-3072 (1.0M x 3072)

| method | points | recall range (frontier) | best QPS | frontier points | device memory, MiB |
|---|---:|---|---:|---:|---|
| JHQ-GPU v69 | 39 | 0.8423-0.9853 | 266,419 | 11 | process 4,137-4,415 (incl. context) |
| CAGRA fp32 | 64 | 0.8803-0.9926 | 122,235 | 26 | index 11,950-11,964 |
| CAGRA int8 | 64 | 0.8599-0.9683 | 328,772 | 29 | index 3,170-3,182 |
| IVF-RaBitQ | 198 | 0.7765-0.9942 | 163,554 | 15 | index 1,760-3,638 |
| IVF-RaBitQ 9-bit | 66 | 0.7765-0.9938 | 163,554 | 14 | index 3,590-3,638 |
| IVF-PQ | 54 | 0.6644-0.9516 | 85,182 | 23 | index 900-2,364 |
| GPU brute force | 1 | 1.0000-1.0000 | 2,049 | 1 | index 11,746 |

### Frontier points (protocol A)


**JHQ-GPU v69**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8423 | 266,419 (266,247-266,484) | - | 3.753 / 3.763 | `jhq-v69` {"nprobe": 32, "alpha": 2, "ck": 20, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.8456 | 264,527 (258,477-264,733) | - | 3.778 / 3.791 | `jhq-v69` {"nprobe": 32, "alpha": 4, "ck": 40, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.8860 | 209,419 (199,133-212,620) | - | 4.702 / 4.714 | `jhq-v69` {"nprobe": 64, "alpha": 2, "ck": 20, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.8903 | 206,294 (202,020-211,473) | - | 4.729 / 4.745 | `jhq-v69` {"nprobe": 64, "alpha": 4, "ck": 40, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9267 | 150,110 (149,210-150,513) | - | 6.661 / 6.672 | `jhq-v69` {"nprobe": 128, "alpha": 4, "ck": 40, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9459 | 94,302 (93,502-94,551) | - | 10.571 / 10.682 | `jhq-v69` {"nprobe": 256, "alpha": 2, "ck": 20, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9523 | 94,143 (93,020-94,495) | - | 10.602 / 10.674 | `jhq-v69` {"nprobe": 256, "alpha": 4, "ck": 40, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9524 | 92,844 (92,482-92,983) | - | 10.739 / 10.886 | `jhq-v69` {"nprobe": 256, "alpha": 8, "ck": 80, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9716 | 53,392 (52,808-53,527) | - | 18.600 / 19.503 | `jhq-v69` {"nprobe": 512, "alpha": 4, "ck": 40, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9717 | 52,812 (52,402-52,884) | - | 18.791 / 19.476 | `jhq-v69` {"nprobe": 512, "alpha": 8, "ck": 80, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |
| 0.9853 | 28,441 (28,177-28,487) | - | 35.028 / 35.448 | `jhq-v69` {"nprobe": 1024, "alpha": 8, "ck": 80, "nlist": 8192, "M": 384, "Br": 8, "batch": 1000} |

**CAGRA fp32**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8803 | 122,235 (120,825-122,613) | 138,957 | 8.160 / 8.186 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 32, "search_width": 1} |
| 0.8935 | 113,290 (111,747-113,505) | 127,564 | 8.821 / 8.856 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 32, "search_width": 2} |
| 0.8995 | 101,576 (100,089-102,233) | 113,618 | 9.784 / 9.911 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 32, "search_width": 4} |
| 0.9134 | 86,258 (85,603-86,687) | 94,517 | 11.584 / 11.652 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 32, "search_width": 8} |
| 0.9411 | 68,155 (67,854-68,545) | 74,030 | 14.584 / 14.685 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 64, "search_width": 1} |
| 0.9433 | 65,758 (65,314-66,332) | 71,203 | 15.082 / 15.148 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 64, "search_width": 2} |
| 0.9478 | 57,870 (57,517-58,058) | 62,001 | 17.270 / 17.375 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 64, "search_width": 8} |
| 0.9568 | 47,544 (47,505-47,552) | 50,514 | 21.033 / 21.094 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 96, "search_width": 1} |
| 0.9573 | 47,237 (46,721-47,454) | 50,329 | 21.068 / 21.241 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 96, "search_width": 2} |
| 0.9597 | 43,124 (42,987-43,177) | 45,615 | 23.167 / 23.325 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 96, "search_width": 8} |
| 0.9672 | 37,306 (37,051-37,371) | 39,379 | 26.727 / 27.452 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 128, "search_width": 1} |
| 0.9686 | 34,264 (34,186-34,417) | 36,141 | 29.035 / 29.164 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 128, "search_width": 8} |
| 0.9746 | 25,574 (25,553-25,579) | 26,503 | 39.086 / 39.243 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 192, "search_width": 4} |
| 0.9756 | 25,408 (25,257-25,557) | 26,481 | 39.126 / 39.325 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 192, "search_width": 2} |
| 0.9764 | 24,571 (24,486-24,654) | 25,503 | 40.566 / 40.810 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 192, "search_width": 8} |
| 0.9792 | 20,218 (20,174-20,238) | 20,825 | 49.425 / 49.598 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 256, "search_width": 1} |
| 0.9798 | 19,802 (19,797-19,812) | 20,353 | 50.485 / 50.597 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 256, "search_width": 4} |
| 0.9803 | 19,226 (19,131-19,265) | 19,773 | 52.000 / 52.155 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 256, "search_width": 8} |
| 0.9864 | 13,616 (13,556-13,620) | 13,882 | 73.459 / 73.616 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 384, "search_width": 1} |
| 0.9874 | 13,552 (13,518-13,560) | 13,807 | 73.769 / 74.031 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 384, "search_width": 8} |
| 0.9889 | 10,706 (10,695-10,709) | 10,861 | 93.411 / 93.531 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 512, "search_width": 1} |
| 0.9899 | 10,646 (10,574-10,662) | 10,812 | 93.824 / 94.157 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 512, "search_width": 2} |
| 0.9900 | 10,594 (10,561-10,604) | 10,739 | 94.405 / 94.638 | `cagra-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.909, "index_mib": 11964.0, "itopk_size": 512, "search_width": 4} |
| 0.9914 | 5,403 (5,403-5,404) | 5,446 | 184.997 / 185.395 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.778, "index_mib": 11950.0, "itopk_size": 512, "search_width": 1} |
| 0.9925 | 5,386 (5,377-5,388) | 5,419 | 185.615 / 186.230 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.778, "index_mib": 11950.0, "itopk_size": 512, "search_width": 2} |
| 0.9926 | 5,228 (5,210-5,233) | 5,272 | 190.999 / 192.866 | `cagra-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 10.778, "index_mib": 11950.0, "itopk_size": 512, "search_width": 8} |

**CAGRA int8**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.8599 | 328,772 (309,251-337,985) | 528,373 | 2.875 / 3.706 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 32, "search_width": 1} |
| 0.8726 | 316,604 (290,397-327,821) | 480,427 | 3.048 / 3.745 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 32, "search_width": 2} |
| 0.8805 | 279,451 (269,943-291,803) | 417,466 | 3.371 / 4.099 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 32, "search_width": 4} |
| 0.8938 | 248,732 (240,897-253,053) | 352,029 | 3.842 / 4.566 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 32, "search_width": 8} |
| 0.9224 | 217,080 (210,007-218,929) | 285,288 | 4.510 / 5.138 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 64, "search_width": 1} |
| 0.9249 | 210,724 (208,944-213,359) | 272,710 | 4.612 / 5.474 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 64, "search_width": 2} |
| 0.9268 | 185,786 (182,377-189,032) | 235,271 | 5.204 / 5.958 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 64, "search_width": 8} |
| 0.9282 | 171,673 (168,360-173,735) | 207,530 | 5.771 / 5.851 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 32, "search_width": 8} |
| 0.9386 | 161,849 (157,830-163,933) | 196,167 | 6.072 / 6.686 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 96, "search_width": 1} |
| 0.9388 | 158,133 (156,226-159,988) | 193,037 | 6.173 / 6.823 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 96, "search_width": 2} |
| 0.9400 | 146,659 (140,297-148,228) | 174,302 | 6.739 / 7.226 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 96, "search_width": 8} |
| 0.9451 | 130,382 (127,363-131,208) | 152,458 | 7.620 / 7.917 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 128, "search_width": 1} |
| 0.9452 | 128,730 (125,956-129,402) | 149,573 | 7.706 / 8.182 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 128, "search_width": 2} |
| 0.9453 | 123,916 (123,084-125,871) | 145,471 | 7.875 / 8.414 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 128, "search_width": 4} |
| 0.9472 | 118,713 (117,094-120,642) | 139,138 | 8.245 / 8.672 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 128, "search_width": 8} |
| 0.9482 | 91,020 (89,422-91,293) | 100,842 | 10.971 / 11.072 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 96, "search_width": 1} |
| 0.9532 | 90,525 (89,801-90,755) | 101,933 | 10.911 / 11.839 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 192, "search_width": 4} |
| 0.9543 | 87,770 (87,049-88,338) | 98,128 | 11.356 / 11.703 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 192, "search_width": 8} |
| 0.9581 | 71,799 (71,264-72,158) | 78,343 | 13.887 / 14.127 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 256, "search_width": 4} |
| 0.9584 | 70,252 (69,727-70,309) | 76,377 | 14.240 / 14.363 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 256, "search_width": 8} |
| 0.9616 | 49,419 (48,861-49,522) | 52,827 | 20.213 / 20.384 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 384, "search_width": 8} |
| 0.9639 | 39,112 (39,049-39,174) | 41,300 | 25.544 / 25.653 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 512, "search_width": 2} |
| 0.9640 | 39,063 (39,042-39,092) | 41,260 | 25.605 / 25.696 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 512, "search_width": 4} |
| 0.9643 | 38,403 (38,262-38,528) | 40,647 | 25.939 / 26.260 | `cagra-int8-gd32` {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 7.456, "index_mib": 3182.0, "itopk_size": 512, "search_width": 8} |
| 0.9664 | 26,395 (26,366-26,411) | 27,390 | 37.860 / 38.003 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 384, "search_width": 4} |
| 0.9665 | 25,558 (25,502-25,566) | 26,493 | 39.112 / 39.214 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 384, "search_width": 8} |
| 0.9671 | 20,339 (20,215-20,365) | 20,950 | 49.138 / 49.330 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 512, "search_width": 1} |
| 0.9682 | 20,326 (20,299-20,360) | 20,963 | 49.155 / 49.437 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 512, "search_width": 2} |
| 0.9683 | 19,658 (19,648-19,673) | 20,207 | 50.836 / 50.976 | `cagra-int8-gd64` {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.289, "kind": "cagra-int8", "build_s": 8.155, "index_mib": 3170.0, "itopk_size": 512, "search_width": 8} |

**IVF-RaBitQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7765 | 163,554 (158,996-168,874) | 196,308 | 5.978 / 7.020 | `ivf-rabitq-b9-QUANT8` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 16, "mode": 3, "mode_name": "QUANT8"} |
| 0.8047 | 150,601 (145,471-151,225) | 172,627 | 6.628 / 6.670 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 16, "mode": 2, "mode_name": "QUANT4"} |
| 0.8257 | 131,979 (130,358-134,088) | 154,336 | 7.347 / 8.451 | `ivf-rabitq-b9-QUANT4` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 32, "mode": 2, "mode_name": "QUANT4"} |
| 0.8613 | 120,818 (118,307-123,731) | 137,276 | 8.073 / 8.218 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 32, "mode": 2, "mode_name": "QUANT4"} |
| 0.8782 | 110,431 (106,382-111,710) | 126,394 | 8.790 / 10.041 | `ivf-rabitq-b9-QUANT8` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 64, "mode": 3, "mode_name": "QUANT8"} |
| 0.9065 | 99,548 (98,286-100,632) | 109,328 | 9.939 / 10.561 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 64, "mode": 2, "mode_name": "QUANT4"} |
| 0.9278 | 86,240 (85,499-87,061) | 93,135 | 11.402 / 12.490 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 96, "mode": 2, "mode_name": "QUANT4"} |
| 0.9408 | 75,450 (75,179-76,198) | 82,158 | 12.981 / 13.329 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 128, "mode": 2, "mode_name": "QUANT4"} |
| 0.9567 | 60,526 (48,839-61,240) | 65,423 | 16.232 / 17.491 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 192, "mode": 2, "mode_name": "QUANT4"} |
| 0.9671 | 51,553 (41,323-52,254) | 55,089 | 19.118 / 20.795 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 256, "mode": 2, "mode_name": "QUANT4"} |
| 0.9788 | 39,655 (35,554-39,802) | 41,702 | 25.049 / 25.853 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 384, "mode": 2, "mode_name": "QUANT4"} |
| 0.9838 | 33,403 (29,533-33,444) | 34,776 | 29.916 / 30.065 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 512, "mode": 2, "mode_name": "QUANT4"} |
| 0.9907 | 26,073 (23,542-26,251) | 26,848 | 38.258 / 38.406 | `ivf-rabitq-b8-QUANT4` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 3224.0, "build_ms": 3470.2, "serialize_ms": 3275.9, "deserialize_ms": 4188.6, "n_probes": 768, "mode": 2, "mode_name": "QUANT4"} |
| 0.9911 | 25,626 (23,590-25,702) | 26,647 | 38.797 / 39.573 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 768, "mode": 2, "mode_name": "QUANT4"} |
| 0.9942 | 22,262 (20,863-22,283) | 22,756 | 44.795 / 46.178 | `ivf-rabitq-b8-QUANT4` {"n_lists": 4096, "bits_per_dim": 8, "index_mib": 3224.0, "build_ms": 3470.2, "serialize_ms": 3275.9, "deserialize_ms": 4188.6, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |

**IVF-RaBitQ 9-bit**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.7765 | 163,554 (158,996-168,874) | 196,308 | 5.978 / 7.020 | `ivf-rabitq-b9-QUANT8` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 16, "mode": 3, "mode_name": "QUANT8"} |
| 0.8047 | 150,601 (145,471-151,225) | 172,627 | 6.628 / 6.670 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 16, "mode": 2, "mode_name": "QUANT4"} |
| 0.8257 | 131,979 (130,358-134,088) | 154,336 | 7.347 / 8.451 | `ivf-rabitq-b9-QUANT4` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 32, "mode": 2, "mode_name": "QUANT4"} |
| 0.8613 | 120,818 (118,307-123,731) | 137,276 | 8.073 / 8.218 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 32, "mode": 2, "mode_name": "QUANT4"} |
| 0.8782 | 110,431 (106,382-111,710) | 126,394 | 8.790 / 10.041 | `ivf-rabitq-b9-QUANT8` {"n_lists": 8192, "bits_per_dim": 9, "index_mib": 3638.0, "build_ms": 5204.8, "serialize_ms": 4185.1, "deserialize_ms": 5195.2, "n_probes": 64, "mode": 3, "mode_name": "QUANT8"} |
| 0.9065 | 99,548 (98,286-100,632) | 109,328 | 9.939 / 10.561 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 64, "mode": 2, "mode_name": "QUANT4"} |
| 0.9278 | 86,240 (85,499-87,061) | 93,135 | 11.402 / 12.490 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 96, "mode": 2, "mode_name": "QUANT4"} |
| 0.9408 | 75,450 (75,179-76,198) | 82,158 | 12.981 / 13.329 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 128, "mode": 2, "mode_name": "QUANT4"} |
| 0.9567 | 60,526 (48,839-61,240) | 65,423 | 16.232 / 17.491 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 192, "mode": 2, "mode_name": "QUANT4"} |
| 0.9671 | 51,553 (41,323-52,254) | 55,089 | 19.118 / 20.795 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 256, "mode": 2, "mode_name": "QUANT4"} |
| 0.9788 | 39,655 (35,554-39,802) | 41,702 | 25.049 / 25.853 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 384, "mode": 2, "mode_name": "QUANT4"} |
| 0.9838 | 33,403 (29,533-33,444) | 34,776 | 29.916 / 30.065 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 512, "mode": 2, "mode_name": "QUANT4"} |
| 0.9911 | 25,626 (23,590-25,702) | 26,647 | 38.797 / 39.573 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 768, "mode": 2, "mode_name": "QUANT4"} |
| 0.9938 | 21,366 (19,884-21,406) | 22,038 | 46.691 / 47.234 | `ivf-rabitq-b9-QUANT4` {"n_lists": 4096, "bits_per_dim": 9, "index_mib": 3590.0, "build_ms": 3427.7, "serialize_ms": 3605.3, "deserialize_ms": 4624.6, "n_probes": 1024, "mode": 2, "mode_name": "QUANT4"} |

**IVF-PQ**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 0.6644 | 85,182 (85,152-85,260) | 90,560 | 11.746 / 11.807 | `ivfpq-pq384-f16` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 14.634, "index_mib": 900.0, "n_probes": 16, "lut": "f16", "refine_ratio": 1} |
| 0.6665 | 82,942 (82,124-84,002) | 89,233 | 11.942 / 12.078 | `ivfpq-pq384-f32` {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 14.634, "index_mib": 900.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.7352 | 69,295 (68,724-69,525) | 72,865 | 14.386 / 14.443 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 16, "lut": "f16", "refine_ratio": 1} |
| 0.7369 | 47,993 (47,488-48,259) | 50,881 | 20.763 / 20.853 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 16, "lut": "f32", "refine_ratio": 1} |
| 0.7812 | 39,190 (38,944-39,232) | 40,517 | 25.478 / 25.628 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1} |
| 0.7838 | 24,611 (24,512-24,617) | 25,402 | 40.645 / 40.698 | `ivfpq-pq768-f32` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 32, "lut": "f32", "refine_ratio": 1} |
| 0.8213 | 21,274 (21,197-21,285) | 21,728 | 46.998 / 47.121 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1} |
| 0.8379 | 14,676 (14,624-14,696) | 14,892 | 68.022 / 68.675 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.8484 | 11,199 (11,195-11,211) | 11,316 | 89.281 / 89.553 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.8603 | 7,609 (7,583-7,620) | 7,664 | 131.355 / 131.775 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1} |
| 0.8647 | 6,780 (6,774-6,786) | 6,842 | 147.502 / 147.666 | `ivfpq-pq1536-f16` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1} |
| 0.8679 | 5,770 (5,764-5,771) | 5,800 | 173.350 / 173.581 | `ivfpq-pq768-f16` {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.817, "index_mib": 1294.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1} |
| 0.8731 | 5,630 (5,590-5,633) | 5,678 | 177.540 / 177.758 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 64, "lut": "f32", "refine_ratio": 1} |
| 0.8844 | 4,684 (4,677-4,686) | 4,708 | 213.483 / 213.659 | `ivfpq-pq1536-f16` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1} |
| 0.8933 | 3,822 (3,822-3,823) | 3,843 | 261.594 / 261.660 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 96, "lut": "f32", "refine_ratio": 1} |
| 0.8965 | 3,572 (3,570-3,610) | 3,572 | 281.337 / 281.553 | `ivfpq-pq1536-f16` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1} |
| 0.9063 | 2,891 (2,888-2,896) | 2,907 | 345.330 / 346.918 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 128, "lut": "f32", "refine_ratio": 1} |
| 0.9125 | 2,451 (2,446-2,452) | 2,459 | 407.989 / 408.376 | `ivfpq-pq1536-f16` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1} |
| 0.9224 | 1,950 (1,950-1,950) | 1,955 | 512.809 / 512.875 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 192, "lut": "f32", "refine_ratio": 1} |
| 0.9314 | 1,473 (1,471-1,473) | 1,476 | 679.080 / 679.133 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 256, "lut": "f32", "refine_ratio": 1} |
| 0.9327 | 1,287 (1,287-1,288) | 1,290 | 776.602 / 777.493 | `ivfpq-pq1536-f16` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1} |
| 0.9438 | 988 (986-990) | 991 | 1010.659 / 1013.207 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1} |
| 0.9516 | 745 (745-746) | 747 | 1341.309 / 1343.665 | `ivfpq-pq1536-f32` {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 27.846, "index_mib": 2364.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1} |

**GPU brute force**

| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |
|---:|---:|---:|---|---|
| 1.0000 | 2,049 (2,044-2,053) | 2,049 | 488.055 / 488.743 | `bruteforce-fp32` {"metric": "sqeuclidean", "dtype": "fp32", "kind": "bruteforce", "build_s": 11.38, "index_mib": 11746.0} |

### Matched recall: JHQ-GPU v69 throughput / other method (protocol A)

measured = best measured point at recall >= r (no interpolation); interp = log-linear between bracketing frontier points; '-' = the method does not cover r.

| r | JHQ QPS (measured / interp) | CAGRA fp32: QPS, ratio | CAGRA int8: QPS, ratio | IVF-RaBitQ: QPS, ratio | IVF-RaBitQ 9-bit: QPS, ratio | IVF-PQ: QPS, ratio | GPU brute force: QPS, ratio |
|---:|---|---|---|---|---|---|---|
| 0.9 | 150,110 / 189,536 | 86,258 / 100,980, x1.74 / x1.88 | 217,080 / 241,500, x0.69 / x0.78 | 99,548 / 101,949, x1.51 / x1.86 | 99,548 / 101,949, x1.51 / x1.86 | 2,891 / 3,312, x51.93 / x57.23 | 2,049, x73.27 (meas.) |
| 0.92 | 150,110 / 159,157 | 68,155 / 81,550, x2.20 / x1.95 | 217,080 / 219,573, x0.69 / x0.72 | 86,240 / 90,893, x1.74 / x1.75 | 86,240 / 90,893, x1.74 / x1.75 | 1,950 / 2,061, x76.97 / x77.21 | 2,049, x73.27 (meas.) |
| 0.94 | 94,302 / 108,783 | 68,155 / 68,795, x1.38 / x1.58 | 146,659 / 146,659, x0.64 / x0.74 | 75,450 / 76,073, x1.25 / x1.43 | 75,450 / 76,073, x1.25 / x1.43 | 988 / 1,082, x95.40 / x100.54 | 2,049, x46.03 (meas.) |
| 0.95 | 94,143 / 94,200 | 47,544 / 55,156, x1.98 / x1.71 | 90,525 / 90,842, x1.04 / x1.04 | 60,526 / 66,416, x1.56 / x1.42 | 60,526 / 66,416, x1.56 / x1.42 | 745 / 790, x126.35 / x119.30 | 2,049, x45.95 (meas.) |
| 0.96 | 53,392 / 74,584 | 37,306 / 43,124, x1.43 / x1.73 | 49,419 / 58,922, x1.08 / x1.27 | 51,553 / 57,521, x1.04 / x1.30 | 51,553 / 57,521, x1.04 / x1.30 | - | 2,049, x26.06 (meas.) |
| 0.97 | 53,392 / 55,911 | 25,574 / 32,003, x2.09 / x1.75 | - | 39,655 / 48,307, x1.35 / x1.16 | 39,655 / 48,307, x1.35 / x1.16 | - | 2,049, x26.06 (meas.) |
| 0.98 | 28,441 / 36,199 | 19,226 / 19,802, x1.48 / x1.83 | - | 33,403 / 38,055, x0.85 / x0.95 | 33,403 / 38,055, x0.85 / x0.95 | - | 2,049, x13.88 (meas.) |
| 0.985 | 28,441 / 28,441 | 13,616 / 14,738, x2.09 / x1.93 | - | 26,073 / 31,994, x1.09 / x0.89 | 25,626 / 31,979, x1.11 / x0.89 | - | 2,049, x13.88 (meas.) |
| 0.99 | - | 10,594 (JHQ -) | - | 26,073 (JHQ -) | 25,626 (JHQ -) | - | 2,049 (JHQ -) |
| 0.995 | - | - | - | - | - | - | 2,049 (JHQ -) |

### Sensitivity: matched recall on QPS = 1000 / median call latency (interp ratio)

| r | CAGRA fp32 | CAGRA int8 | IVF-RaBitQ | IVF-RaBitQ 9-bit |
|---:|---:|---:|---:|---:|
| 0.9 | x1.87 | x0.76 | x1.84 | x1.84 |
| 0.92 | x1.87 | x0.68 | x1.66 | x1.66 |
| 0.94 | x1.58 | x0.73 | x1.40 | x1.40 |
| 0.95 | x1.71 | x0.89 | x1.39 | x1.39 |
| 0.96 | x1.50 | x1.10 | x1.11 | x1.11 |
| 0.97 | x1.67 | - | x1.10 | x1.10 |
| 0.98 | x1.84 | - | x0.95 | x0.95 |
| 0.985 | x1.94 | - | x0.89 | x0.89 |
| 0.99 | - | - | - | - |
| 0.995 | - | - | - | - |

### Matched throughput: best Recall@10 at QPS >= q (measured points, protocol A)

| method | >= 100K | >= 150K | >= 200K | >= 250K | >= 300K | >= 400K |
|---|---:|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 0.9267 | 0.9267 | 0.8903 | 0.8456 | - | - |
| CAGRA fp32 | 0.8995 | - | - | - | - | - |
| CAGRA int8 | 0.9472 | 0.9388 | 0.9249 | 0.8805 | 0.8726 | - |
| IVF-RaBitQ | 0.8782 | 0.8047 | - | - | - | - |
| IVF-RaBitQ 9-bit | 0.8782 | 0.8047 | - | - | - | - |
| IVF-PQ | - | - | - | - | - | - |
| GPU brute force | - | - | - | - | - | - |

### Variation across the rounds of a point: (max - min) / median of the per-round QPS

| method | points | median spread | 90th pct | max | points over 5% |
|---|---:|---:|---:|---:|---:|
| JHQ-GPU v69 | 39 | 1.35% | 4.58% | 7.34% | 3 |
| CAGRA fp32 | 64 | 0.59% | 1.26% | 5.11% | 1 |
| CAGRA int8 | 64 | 1.47% | 3.71% | 11.82% | 4 |
| IVF-RaBitQ | 198 | 5.28% | 19.61% | 90.05% | 103 |
| IVF-PQ | 54 | 0.40% | 1.15% | 6.28% | 1 |
| GPU brute force | 1 | 0.44% | 0.44% | 0.44% | 0 |

## Points whose results vary between calls (kept; 401 points)

- arxiv-768 CAGRA int8 R=0.9474 {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 96, "search_width": 4}: protocol B ids differ (0 rows as sets); B recall 0.9474
- arxiv-768 CAGRA int8 R=0.9504 {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 96, "search_width": 8}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9504
- arxiv-768 CAGRA int8 R=0.9531 {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 128, "search_width": 2}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9531
- arxiv-768 CAGRA int8 R=0.9616 {"graph_degree": 32, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 6.829, "index_mib": 2056.0, "itopk_size": 512, "search_width": 8}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9616; protocol B ids differ (0 rows as sets); B recall 0.9616
- arxiv-768 CAGRA int8 R=0.9330 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 32, "search_width": 2}: protocol B ids differ (0 rows as sets); B recall 0.9330
- arxiv-768 CAGRA int8 R=0.9390 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 32, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9390; protocol B ids differ (0 rows as sets); B recall 0.9390
- arxiv-768 CAGRA int8 R=0.9429 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 32, "search_width": 8}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9429; protocol B ids differ (0 rows as sets); B recall 0.9429
- arxiv-768 CAGRA int8 R=0.9546 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 64, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.9546; protocol B ids differ (0 rows as sets); B recall 0.9546
- arxiv-768 CAGRA int8 R=0.9546 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 64, "search_width": 8}: protocol B ids differ (0 rows as sets); B recall 0.9546
- arxiv-768 CAGRA int8 R=0.9586 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 2}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9586
- arxiv-768 CAGRA int8 R=0.9591 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.9591; protocol B ids differ (0 rows as sets); B recall 0.9591
- arxiv-768 CAGRA int8 R=0.9592 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 96, "search_width": 8}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9592; protocol B ids differ (1 rows as sets); B recall 0.9592
- arxiv-768 CAGRA int8 R=0.9594 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 128, "search_width": 2}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9594
- arxiv-768 CAGRA int8 R=0.9598 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 128, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9598; protocol B ids differ (0 rows as sets); B recall 0.9598
- arxiv-768 CAGRA int8 R=0.9597 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 128, "search_width": 8}: ids vary between calls: 2 rows differ as sets, 3 only in order; last-call recall 0.9597; protocol B ids differ (1 rows as sets); B recall 0.9597
- arxiv-768 CAGRA int8 R=0.9614 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 192, "search_width": 2}: protocol B ids differ (0 rows as sets); B recall 0.9614
- arxiv-768 CAGRA int8 R=0.9614 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 192, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9614; protocol B ids differ (0 rows as sets); B recall 0.9614
- arxiv-768 CAGRA int8 R=0.9614 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 192, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 3 only in order; last-call recall 0.9614; protocol B ids differ (1 rows as sets); B recall 0.9614
- arxiv-768 CAGRA int8 R=0.9617 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 256, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9617; protocol B ids differ (0 rows as sets); B recall 0.9617
- arxiv-768 CAGRA int8 R=0.9617 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 256, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 2 only in order; last-call recall 0.9617; protocol B ids differ (2 rows as sets); B recall 0.9617
- arxiv-768 CAGRA int8 R=0.9617 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 384, "search_width": 2}: protocol B ids differ (0 rows as sets); B recall 0.9617
- arxiv-768 CAGRA int8 R=0.9617 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 384, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.9617; protocol B ids differ (0 rows as sets); B recall 0.9617
- arxiv-768 CAGRA int8 R=0.9617 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 384, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 1 only in order; last-call recall 0.9617; protocol B ids differ (0 rows as sets); B recall 0.9617
- arxiv-768 CAGRA int8 R=0.9620 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 512, "search_width": 4}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9620; protocol B ids differ (0 rows as sets); B recall 0.9620
- arxiv-768 CAGRA int8 R=0.9620 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "quantile": 0.99, "quantise_dataset_s": 1.236, "kind": "cagra-int8", "build_s": 7.531, "index_mib": 2198.0, "itopk_size": 512, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 1 only in order; last-call recall 0.9620; protocol B ids differ (2 rows as sets); B recall 0.9620
- arxiv-768 CAGRA fp32 R=0.9924 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 96, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 0 only in order; last-call recall 0.9924
- arxiv-768 CAGRA fp32 R=0.9949 {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq", "kind": "cagra", "build_s": 9.972, "index_mib": 7148.0, "itopk_size": 128, "search_width": 8}: ids vary between calls: 1 rows differ as sets, 0 only in order; last-call recall 0.9949; protocol B ids differ (1 rows as sets); B recall 0.9949
- arxiv-768 IVF-PQ R=0.8498 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1}: ids vary between calls: 1 rows differ as sets, 0 only in order; last-call recall 0.8498; protocol B ids differ (1 rows as sets); B recall 0.8498
- arxiv-768 IVF-PQ R=0.8504 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1}: ids vary between calls: 1 rows differ as sets, 2 only in order; last-call recall 0.8504; protocol B ids differ (1 rows as sets); B recall 0.8504
- arxiv-768 IVF-PQ R=0.8115 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1}: protocol B ids differ (0 rows as sets); B recall 0.8115
- arxiv-768 IVF-PQ R=0.8317 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.8317; protocol B ids differ (0 rows as sets); B recall 0.8317
- arxiv-768 IVF-PQ R=0.8386 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1}: protocol B ids differ (0 rows as sets); B recall 0.8386
- arxiv-768 IVF-PQ R=0.8409 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.8409
- arxiv-768 IVF-PQ R=0.8461 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.8461
- arxiv-768 IVF-PQ R=0.8482 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 26 rows differ as sets, 95 only in order; last-call recall 0.8481; protocol B ids differ (20 rows as sets); B recall 0.8487
- arxiv-768 IVF-PQ R=0.8491 {"n_lists": 8192, "pq_dim": 192, "pq_bits": 8, "code_bytes_per_vec": 192, "kind": "ivfpq", "build_s": 11.465, "index_mib": 810.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 28 rows differ as sets, 98 only in order; last-call recall 0.8494; protocol B ids differ (24 rows as sets); B recall 0.8489
- arxiv-768 IVF-PQ R=0.9518 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9518
- arxiv-768 IVF-PQ R=0.9528 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9528; protocol B ids differ (0 rows as sets); B recall 0.9528
- arxiv-768 IVF-PQ R=0.9493 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 19 rows differ as sets, 100 only in order; last-call recall 0.9492; protocol B ids differ (23 rows as sets); B recall 0.9492
- arxiv-768 IVF-PQ R=0.9501 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 11.229, "index_mib": 1276.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 23 rows differ as sets, 99 only in order; last-call recall 0.9503; protocol B ids differ (27 rows as sets); B recall 0.9499
- arxiv-768 IVF-PQ R=0.9829 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1}: protocol B ids differ (0 rows as sets); B recall 0.9829
- arxiv-768 IVF-PQ R=0.9844 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 512, "lut": "f32", "refine_ratio": 1}: protocol B ids differ (0 rows as sets); B recall 0.9844
- arxiv-768 IVF-PQ R=0.9706 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 26 rows differ as sets, 95 only in order; last-call recall 0.9705; protocol B ids differ (22 rows as sets); B recall 0.9708
- arxiv-768 IVF-PQ R=0.9720 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 12.934, "index_mib": 2604.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 32 rows differ as sets, 102 only in order; last-call recall 0.9721; protocol B ids differ (24 rows as sets); B recall 0.9715
- openai3-1536 IVF-PQ R=0.8405 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.8405; protocol B ids differ (0 rows as sets); B recall 0.8405
- openai3-1536 IVF-PQ R=0.8514 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 1 rows differ as sets, 2 only in order; last-call recall 0.8513
- openai3-1536 IVF-PQ R=0.8630 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 1 rows differ as sets, 2 only in order; last-call recall 0.8631; protocol B ids differ (1 rows as sets); B recall 0.8631
- openai3-1536 IVF-PQ R=0.8696 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.8696
- openai3-1536 IVF-PQ R=0.8760 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 27 rows differ as sets, 70 only in order; last-call recall 0.8767; protocol B ids differ (26 rows as sets); B recall 0.8760
- openai3-1536 IVF-PQ R=0.8789 {"n_lists": 8192, "pq_dim": 384, "pq_bits": 8, "code_bytes_per_vec": 384, "kind": "ivfpq", "build_s": 9.187, "index_mib": 780.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 22 rows differ as sets, 64 only in order; last-call recall 0.8791; protocol B ids differ (22 rows as sets); B recall 0.8789
- openai3-1536 IVF-PQ R=0.9453 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 384, "lut": "f32", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9453
- openai3-1536 IVF-PQ R=0.9399 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 9 rows differ as sets, 66 only in order; last-call recall 0.9398; protocol B ids differ (10 rows as sets); B recall 0.9395
- openai3-1536 IVF-PQ R=0.9462 {"n_lists": 8192, "pq_dim": 768, "pq_bits": 8, "code_bytes_per_vec": 768, "kind": "ivfpq", "build_s": 17.78, "index_mib": 1174.0, "n_probes": 512, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 18 rows differ as sets, 71 only in order; last-call recall 0.9459; protocol B ids differ (15 rows as sets); B recall 0.9463
- openai3-1536 IVF-PQ R=0.8309 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 32, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.8309; protocol B ids differ (0 rows as sets); B recall 0.8309
- openai3-1536 IVF-PQ R=0.8768 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 64, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 2 only in order; last-call recall 0.8768; protocol B ids differ (0 rows as sets); B recall 0.8768
- openai3-1536 IVF-PQ R=0.8982 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 96, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.8982
- openai3-1536 IVF-PQ R=0.9118 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 128, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 1 only in order; last-call recall 0.9118; protocol B ids differ (1 rows as sets); B recall 0.9119
- openai3-1536 IVF-PQ R=0.9272 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 192, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 0 rows differ as sets, 4 only in order; last-call recall 0.9272; protocol B ids differ (0 rows as sets); B recall 0.9272
- openai3-1536 IVF-PQ R=0.9357 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 256, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 2 rows differ as sets, 5 only in order; last-call recall 0.9358; protocol B ids differ (1 rows as sets); B recall 0.9357
- openai3-1536 IVF-PQ R=0.9471 {"n_lists": 8192, "pq_dim": 1536, "pq_bits": 8, "code_bytes_per_vec": 1536, "kind": "ivfpq", "build_s": 25.128, "index_mib": 2254.0, "n_probes": 384, "lut": "f16", "refine_ratio": 1}: ids vary between calls: 22 rows differ as sets, 65 only in order; last-call recall 0.9469; protocol B ids differ (17 rows as sets); B recall 0.9472
