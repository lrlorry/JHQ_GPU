# JHQ-GPU 论文实验与系统评估完整计划

> **目标**：把当前 JHQ GPU 工程整理成一篇实验完整、比较公平、系统证据充分的 GPU ANN 论文。
>
> **论文实现基线**：[`jhq_v14_streaming_add`](./jhq_v14_streaming_add/)
>
> **CPU 参考实现**：JHQ CPU 论文及其官方实现
>
> **不作为本文主实现**：`hblock_v14` 及后续 HBlock/block-graph 探索。它们属于另一条研究路线，不应与当前 JHQ-GPU 论文混写。

---

## 0. 结论先行

当前项目已经证明了 **JHQ 在 GPU 上可以获得显著加速**，其中最有价值的系统优化是：

1. query batching；
2. CUDA Graph；
3. Byte LUT；
4. primary code 的 `[N,M] -> [M,N]` 转置布局；
5. memory-bounded streaming add。

仓库现有开发记录中，`jhq_v12_transposed` 在 Vogue-768 上达到约 `44,890 QPS`、`Recall@10 ≈ 0.9982`，相对官方单线程 JHQ CPU 约 `8.3x`。这些数字说明方向成立，但**还不能直接支撑投稿**。

当前最关键的问题不是单纯“再多跑几个数据集”，而是：

- GPU 实现是否完全忠实于 CPU JHQ；
- JQ-GPU 与 JHQ-GPU 是否齐全；
- small-batch latency 是否被正确测量；
- CPU/GPU/PCIe/HBM 的系统瓶颈是否被拆清楚；
- 是否与当前主流 GPU ANN 方法在相同 recall、相同 batch、相同显存约束下公平比较。

论文应当讲的不是：

> GPU 比单线程 CPU 快。

而是：

> JHQ 的 hierarchical quantization 在 GPU 上产生了特殊的 LUT、top-k、residual refinement 和不连续访存瓶颈；本文通过 JHQ-specific GPU execution、memory-coalesced code layout、selective refinement 和 memory-bounded construction 解决这些问题。

---

## 1. 标记约定

本文档采用以下标记，避免把代码事实、原论文要求和研究建议混在一起：

- **[代码事实]**：直接来自当前 `JHQ_GPU` 仓库实现或版本记录；
- **[原论文]**：来自 CPU JHQ 论文及其官方实现；
- **[实验建议]**：为了形成完整论文而提出的实验或实现要求；
- **[推算]**：根据当前 buffer 布局和参数做出的近似计算，必须用真实测量验证。

---

# Part I：论文版本冻结与 P0 阻塞问题

## 2. 冻结论文代码线

### 2.1 主版本

使用：

```text
jhq_v14_streaming_add
```

原因：

- 继承 v12 查询路径中的 Byte LUT、CUDA Graph 和转置 primary-code layout；
- 建库时只对 full-precision raw/rotated vectors 进行分批处理；
- 面向 10M–17.8M 高维数据，避免一次性保存两份完整 FP32 向量；
- 与 CPU JHQ 的 IVF + hierarchical quantization 路线直接对应。

### 2.2 不要把 `hblock_v14` 当作当前论文版本

`hblock_v14` 是另一套 hierarchical block routing / PQ / GPU top-k 架构，其数据结构、候选生成和精排逻辑都已经偏离 CPU JHQ。它可以成为未来另一篇论文或 appendix exploration，但不应作为当前 JHQ-GPU 的最终实现。

### 2.3 建议创建 paper branch

```text
paper/jhq-gpu-v1
```

该分支只保留：

- paper-faithful JQ-GPU；
- paper-faithful JHQ-GPU；
- benchmark harness；
- profiler instrumentation；
- result scripts；
- reproducibility configuration。

不要继续把 HBlock、block graph 和 JHQ GPU 的实验代码放在同一个执行入口中。

### 2.4 每次正式实验必须记录

- Git commit SHA；
- compiler version；
- CUDA version；
- driver version；
- cuVS/Faiss version；
- GPU 型号和显存；
- CPU 型号、线程数和内存；
- CMake flags；
- CUDA architecture；
- dataset checksum；
- random seed；
- 完整参数。

---

## 3. P0-1：Residual codebook 必须与 CPU JHQ 对齐

### 当前状态

**[代码事实]** 当前 GPU 版本只有一个全局 `res_c1d_`：

- 所有 subspace；
- 所有维度；
- 所有 residual sample；

被 flatten 后共同训练一个 global 1D k-means residual codebook。

### 原论文要求

**[原论文]** JHQ residual level 应针对每个 subspace 建立 scalar residual codebook，并在该 subspace 内共享。

### 风险

如果 GPU 与 CPU 使用不同 residual codebook，审稿人可以认为：

> GPU 加速的不是论文中定义的 JHQ，而是一个近似变体。

这会直接影响算法一致性、recall 和公平比较。

### 必须实现

```text
residual_codebook[m][Kr]
```

搜索阶段至少逻辑上支持：

```text
residual_lut[batch][M][Ds][Kr]
```

实际可以进一步压缩或重排，但算法语义必须一致。

### 必须做的消融

| 版本 | 含义 |
|---|---|
| Global residual codebook | 当前 GPU 简化实现 |
| Per-subspace residual codebook | CPU 论文一致版本 |

报告：

- Recall@10；
- QPS；
- residual LUT 时间；
- residual codebook memory；
- build time；
- quantization error。

### 验收标准

- GPU per-subspace codebook 与 CPU reference 在固定 seed、小数据集上的 codebook/重建误差趋势一致；
- GPU 与 CPU 使用相同参数时，Recall 差异可以解释；
- 主实验只使用 per-subspace 版本；
- global 版本仅作为消融。

---

## 4. P0-2：必须实现真正的 JQ-GPU

### 当前状态

**[代码事实]** 当前 v14 查询固定执行：

1. JL rotation；
2. IVF probe selection；
3. primary LUT；
4. primary code scan；
5. residual LUT；
6. residual refinement；
7. final top-k。

缺少一个干净的 JQ-only GPU 路径。

### 必须实现

提供：

```text
num_levels = 1
```

或：

```text
compute_residuals = false
```

JQ-GPU 与 JHQ-GPU 必须共用：

- 同一 JL transform；
- 同一 IVF coarse partition；
- 同一 primary code；
- 同一 primary LUT；
- 同一 code layout；
- 同一 top-k 实现。

唯一差别应当是是否保存、加载和计算 residual level。

### 为什么必须做

这是回答以下问题的唯一干净实验：

> JHQ 的 hierarchical residual refinement 在 GPU 上是否真的值得？

### 验收标准

主文必须同时包含：

- JQ CPU；
- JHQ CPU；
- JQ-GPU；
- JHQ-GPU。

并分别报告：

- recall-QPS；
- build time；
- index size；
- query memory traffic。

---

## 5. P0-3：修正 small-batch 测量路径

### 当前问题

**[代码事实]** 当前 CUDA Graph 按固定 `B_full = batch_cap` 捕获。默认 `batch_cap=256`。

当实际只有一个 query 时，当前实现仍可能：

- padding 255 个零 query；
- H2D 传输完整的 256-query buffer；
- 执行完整的 256-query CUDA Graph；
- 只返回第一个 query 的结果。

因此，当前 `batch=1` 不是一个真实的 single-query latency。

### 必须修改

推荐按 batch bucket 缓存多个 CUDA Graph：

```text
B in {1, 8, 32, 128, 256, 1024}
```

或者采用：

- small-batch dynamic kernel path；
- large-batch CUDA Graph path。

### 必须报告两种服务模式

#### Throughput mode

```text
batch = 256 / 1024
```

报告 QPS。

#### Latency mode

```text
batch = 1 / 8 / 32
```

报告：

- P50；
- P95；
- P99；
- average latency；
- kernel-only latency；
- host-to-host latency。

### 验收标准

- batch=1 时只处理一个 query；
- profiler timeline 中没有 255 个无效 query；
- CAGRA 的 small-batch/multi-CTA 结果与 JHQ-GPU 使用相同 batch 语义。

---

## 6. P0-4：修正 pinned-buffer 生命周期并实现真正 overlap

### 当前风险

**[代码事实]** 当前循环复用单个 pinned query buffer：

```text
CPU memcpy -> pinned buffer
cudaMemcpyAsync -> GPU
下一批再次覆盖同一个 pinned buffer
```

如果前一批 DMA 未结束，CPU 就开始写下一批，存在潜在 buffer overwrite 风险。

另外，如果 D2H 目标是普通 pageable `std::vector`，异步 D2H 不一定产生真正 overlap。

### 必须实现

至少双缓冲：

```text
host_query_buffer[2]
device_query_buffer[2]
host_result_buffer[2]
stream[2]
event[2]
```

完整流水：

```text
Batch t:     H2D
Batch t-1:   GPU search
Batch t-2:   D2H
```

同时提供两类 API：

```text
search_host_to_host(...)
search_device_to_device(...)
```

### 系统消融

| 版本 | 数据传输方式 |
|---|---|
| Sync | 同步 H2D/compute/D2H |
| Async single stream | 单 stream async |
| Double-buffer | 双缓冲、多 stream |
| D2D | query/result 全在 GPU |

### 验收标准

Nsight Systems 中能够直接看到：

- H2D 与 kernel overlap；
- kernel 与 D2H overlap；
- 没有不必要的 stream-wide stall；
- buffer 生命周期由 CUDA event 保护。

---

## 7. P0-5：验证 `K_LOCAL=4` 是否损失 top-ck

### 当前状态

**[代码事实]** primary scan 中每个线程只保留 local top-4，然后再从所有线程的 local candidates 中生成全局 top-ck。

### 风险

该方法不保证严格 exact top-ck。如果一个线程负责的候选中有超过 4 个属于真正的全局 top-ck，则第 5 个及之后会被提前丢弃。

风险在以下配置中更高：

- `k=100`；
- `alpha=8`；
- IVF list 高度不均衡；
- 线程候选分配不均衡。

### 必须做的 correctness ablation

```text
K_LOCAL in {4, 8, 16}
```

并增加一个 exact/reference top-ck：

- CUB select；
- exact bitonic/radix；
- 或 CPU reference，仅用于验证。

报告：

- candidate recall@ck；
- final Recall@k；
- scan kernel time；
- shared memory；
- registers/thread；
- occupancy。

### 验收标准

主配置下，近似 top-ck 与 exact top-ck 的 final recall 差异必须足够小并明确报告。若差异明显，必须修改主算法。

---

## 8. P0-6：支持 CPU 论文的完整参数空间

### 当前约束

**[代码事实]** 当前 GPU 参数约束包括：

```text
d % M == 0
B % (d/M) == 0
B <= 8
Br in {4, 8}
```

当前默认参数大致为：

```text
M=96
B=8
Br=4
alpha=4
nlist=1024
nprobe=8
batch_size=256
```

### 原论文参数范围

主实验应尽量覆盖：

```text
nlist = 4096
nprobe in {1,2,4,8,16,32,64,128}
alpha in {2,4,8}
[B,Br] in {[8,4], [4,8], [8,8]}
```

不同维度下的 M：

| Dimension | M candidates |
|---:|---|
| 768 | 48, 96, 192, 384 |
| 1024 | 64, 128, 256 |
| 1536 | 32, 64, 128, 192, 256 |
| 3072 | 32, 64, 128, 192, 256, 384 |

### 必须完成

- 对照官方 CPU code，统一 primary codeword 生成和 packing 语义；
- 支持论文参数中能够合法构造的全部组合；
- 无法支持的参数必须给出明确数学/工程原因；
- 主实验固定 `nlist=4096`；
- `nlist in {1024,4096,16384}` 放入 sensitivity。

### 验收标准

不能再只用 Vogue-768、`M=96,nlist=1024,nprobe=8` 的单点结果代表整篇论文。

---

## 9. P0-7：明确静态 bulk-build 边界

### 当前实现边界

**[代码事实]** 当前 v14：

- `add()` 只允许调用一次；
- 没有 delete；
- 没有真正 incremental update；
- IVF centroid accumulation 在 CPU；
- residual samples 会 D2H；
- residual 1D k-means 在 CPU。

### 当前可以声称

> GPU-accelerated JHQ search with memory-bounded bulk index construction.

### 当前不能声称

> Fully GPU-native dynamic JHQ index.

除非进一步实现：

- GPU residual training；
- GPU centroid accumulation/update；
- incremental add；
- delete/update；
- multi-add correctness。

---

## 10. P0-8：建立 CPU/GPU correctness test suite

至少包含：

### 10.1 Quantization correctness

- primary code comparison；
- residual code comparison；
- reconstruction error；
- correction term comparison。

### 10.2 Distance correctness

随机抽样 query-candidate 对，比较：

```text
CPU JHQ compressed distance
GPU JHQ compressed distance
```

报告：

- max absolute error；
- mean absolute error；
- relative error；
- rank disagreement。

### 10.3 Search correctness

小数据集上比较：

- exact Flat；
- CPU JQ/JHQ；
- GPU JQ/JHQ；
- exact top-ck 与 local top-ck。

### 10.4 Determinism

固定 seed 重复运行，验证：

- codebook；
- IVF assignment；
- recall；
- output IDs；
- timing variance。

---

## 11. P0-9：编译目标和数值精度

### 当前问题

**[代码事实]** CMake 默认 CUDA architecture 为 `70`，同时启用了：

```text
--use_fast_math
```

正式实验不能依赖默认值。

### 必须完成

- 为目标 GPU 设置准确 compute capability；
- 记录 PTX/SASS target；
- 检查是否发生运行时 JIT；
- `--use_fast_math` on/off 消融；
- 记录 cuBLAS 的 TF32 配置；
- 对 rotation GEMM 测试 FP32/TF32；
- 确认数值变化不会影响 recall 结论。

---

# Part II：论文要回答的研究问题

## 12. Research Questions

### RQ1：总体性能

在相同 Recall@10 下，JHQ-GPU 是否优于：

- JHQ CPU；
- JQ CPU；
- GPU IVF-PQ；
- GPU IVF-RaBitQ；
- CAGRA；
- GPU Flat reference？

### RQ2：加速来自哪里

性能提升来自：

- GPU 并行本身；
- query batching；
- CUDA Graph；
- Byte LUT；
- transposed code layout；
- selective residual refinement；
- streaming add；

各自贡献多少？

### RQ3：高维、大规模建库能力

10M–17.8M、1024–3072D 数据能否：

- 单 GPU 建库；
- 单 GPU 常驻搜索；
- 避免 OOM；
- 控制 peak VRAM；
- 获得稳定 build throughput？

### RQ4：真实瓶颈

瓶颈究竟是：

- PCIe；
- HBM bandwidth；
- L2 cache；
- shared-memory bank conflict；
- top-k；
- JL GEMM；
- IVF probe selection；
- residual random access；
- CPU-side training？

### RQ5：参数与 workload 稳定性

方法对以下变化是否稳定：

- dimension；
- N；
- batch size；
- k；
- M；
- B/Br；
- alpha；
- nlist/nprobe；
- add_batch；
- GPU architecture。

### RQ6：资源效率

相同 recall 下，JHQ-GPU 是否在以下指标上具有优势：

- bytes/vector；
- peak VRAM；
- QPS/GiB；
- QPS/W；
- build time；
- query-time bytes read？

---

# Part III：数据集与统一实验协议

## 13. 六个主数据集

必须复现 CPU JHQ 论文中的六个数据集：

| Dataset | N | D |
|---|---:|---:|
| OpenAI3-1536 | 999,000 | 1,536 |
| OpenAI3-3072 | 999,000 | 3,072 |
| Vogue-768 | 932,328 | 768 |
| Arxiv-Abstracts-768 | 2,253,198 | 768 |
| BGE-M3-1024 | 10,091,524 | 1,024 |
| Stella-TREC24-1024 | 17,776,615 | 1,024 |

## 14. 数据协议

所有算法统一：

- 相同 base vectors；
- 相同 query vectors；
- 相同 ground truth；
- 相同 metric；
- 相同 normalization；
- 相同 `k`；
- 相同 query batch；
- 相同硬件；
- 相同随机 seed；
- 相同是否 rerank。

建议：

```text
nq = 1000
k = 10
seed = 42
```

同时为稳定 latency/QPS 测量，重复 replay query batch，使测量时间足够长，而不是只执行一次 1,000-query batch。

## 15. Ground-truth 要求

- 使用 exact GPU Flat 或 CPU exact Flat；
- ground-truth 至少保存 top-100，便于测试 `k=1/10/100`；
- 记录 metric 和 normalization；
- ground-truth 文件保存 checksum。

## 16. Training sample

当前 demo 使用前 100K 条向量训练。正式实验应改为：

- 固定 seed；
- 全库均匀随机采样；
- 记录 sample index；
- CPU/GPU 使用完全相同 training sample。

敏感性：

```text
n_train in {20K, 50K, 100K, 200K}
```

---

# Part IV：Baseline 设计

## 17. 主文必须包含的 baseline

| Method | 作用 | 优先级 |
|---|---|---|
| JHQ CPU official, 1 thread | 延续 CPU 原论文 | 必须 |
| JHQ CPU official, all cores | 避免只打单线程的公平性质疑 | 必须 |
| JQ CPU official | CPU 层级消融 | 必须 |
| JQ-GPU | 证明 GPU hierarchy 的价值 | 必须 |
| JHQ-GPU | 本文方法 | 必须 |
| cuVS IVF-PQ | 最直接的 GPU quantization baseline | 必须 |
| GPU IVF-RaBitQ | 当前强 GPU cluster+quantization 对手 | 必须，若运行环境支持 |
| cuVS CAGRA | 主流 GPU graph baseline | 必须 |
| GPU Flat exact | accuracy/throughput reference | 必须 |

## 18. 次要或 appendix baseline

| Method | 建议 |
|---|---|
| Faiss GpuIndexIVFPQ | 作为经典实现对照，避免与 cuVS 主图重复 |
| SONG | 历史 GPU graph baseline；可运行则放 appendix |
| BANG In-memory | 中等规模可驻留 GPU 的补充对比 |
| BANG | 主要面向 billion-scale/out-of-GPU-memory，不是六数据集主对手 |
| Jasper | 若代码稳定且时间允许，作为更新型 GPU graph baseline |
| CPU PQ/OPQ/IRVQ/LSQ++/RaBitQ | 与 CPU 原论文保持连续性 |

## 19. 主图不要塞过多算法

建议主图保留：

```text
JQ-GPU
JHQ-GPU
cuVS IVF-PQ
GPU IVF-RaBitQ
CAGRA
GPU Flat
```

CPU 方法单独放表或第二张图。

## 20. Baseline 公平性规则

每个算法必须：

1. 使用同一 dataset/query/ground truth；
2. 使用同一 metric；
3. 使用同一 GPU；
4. 使用同一 batch；
5. 使用同一 host/device 输入语义；
6. 使用同一 reranking 规则；
7. 独立进行合理参数搜索；
8. 在相同 recall 下比较；
9. 报告 index memory；
10. 报告 build time；
11. OOM/timeout 不得静默删除；
12. 不允许只挑一个对本文最有利的参数点。

## 21. Reranking 规则

主图建议：

- JHQ-GPU：no exact reranking；
- IVF-PQ：no exact reranking；
- IVF-RaBitQ：按其标准实现，但必须明确是否访问 raw vectors；
- CAGRA：标准 full-vector graph search；
- GPU Flat：exact reference。

额外 appendix：

- IVF-PQ + exact refinement；
- JHQ-GPU + optional exact reranking；
- 统一 candidate budget 下的 reranking 对比。

不能把“带 full-vector rerank 的 baseline”与“无 rerank 的本文方法”混在一起却不说明。

## 22. 两种公平视角

### 22.1 Unlimited best frontier

每个算法使用自己的最佳参数，展示 Recall-QPS Pareto frontier。

### 22.2 Equal-memory frontier

设置相同显存预算，例如：

```text
16 GiB
24 GiB
32 GiB
80 GiB
```

比较：

- 能否建库；
- 能否常驻；
- QPS；
- recall；
- build time。

这对 JHQ-GPU 的压缩优势非常重要。

---

# Part V：完整主实验矩阵

## 23. 主实验一：Recall-QPS 曲线

对每个数据集、每个算法画：

```text
x-axis: Recall@10
y-axis: QPS, log scale 可选
```

通过以下参数生成曲线：

- JHQ/JQ：`nprobe`, `alpha`, `M`, `[B,Br]`；
- IVF-PQ：`n_probes`, code length, LUT dtype；
- IVF-RaBitQ：`n_probes`, quantization bits 等；
- CAGRA：`itopk_size`, search width, graph degree 等。

固定 recall 表：

| Method | QPS@0.90 | QPS@0.95 | QPS@0.99 | Max Recall |
|---|---:|---:|---:|---:|

不要用“不同 recall 的峰值 QPS”直接计算 speedup。

---

## 24. 主实验二：Batch size 与 latency/throughput

```text
batch in {1, 8, 32, 128, 256, 1024}
```

每个 batch 报告：

- QPS；
- average latency；
- P50/P95/P99；
- H2H latency；
- D2D latency；
- GPU utilization；
- PCIe bytes/query。

建议分别选择：

- Recall@10 ≈ 0.90；
- Recall@10 ≈ 0.95；
- Recall@10 ≈ 0.99。

---

## 25. 主实验三：建库性能

总建库时间：

```text
T_build = T_train + T_add
```

必须细分：

| Stage | 指标 |
|---|---|
| JL matrix generation | 时间、host memory |
| Training rotation | GPU time、H2D bytes |
| Primary codebook generation | 时间 |
| Primary encoding | vectors/s |
| Residual codebook training | CPU/GPU time |
| Residual encoding | vectors/s |
| IVF training | 每轮时间 |
| IVF assignment | 时间 |
| Sort by list ID | 时间 |
| Gather/reorder | 时间 |
| `[N,M] -> [M,N]` transpose | 时间 |
| Workspace allocation | 时间 |
| Total | build time、peak VRAM、peak RSS |

CPU baseline 同时报告：

- 1-thread query；
- all-core query；
- 32-thread 或标准配置 build。

---

## 26. 主实验四：规模扩展

使用 Stella 或同分布子集：

```text
N in {100K, 1M, 5M, 10M, 17.8M}
```

报告：

- Recall-QPS；
- build time；
- add throughput；
- persistent index size；
- peak VRAM；
- peak host RSS；
- bytes/vector；
- query-time bytes read；
- OOM/失败配置。

目标是验证：

> 随着 N 增大，JHQ 相比 JQ 的 hierarchical refinement 优势是否增强。

---

## 27. 主实验五：维度扩展

使用：

```text
D in {768, 1024, 1536, 3072}
```

重点观察：

- JL GEMM 占比；
- primary LUT 构建占比；
- residual LUT 占比；
- HBM scan bandwidth；
- index bytes/vector；
- QPS 随维度的下降趋势。

这是 JHQ-GPU 相对传统低维 ANN 的核心卖点之一。

---

## 28. 主实验六：不同 GPU

最低建议两张 GPU：

- 消费级高带宽 GPU，例如 RTX 5090；
- 数据中心 GPU，例如 A100/H100。

报告：

- absolute QPS；
- speedup over CPU；
- achieved bandwidth / peak bandwidth；
- achieved FLOPS / peak FLOPS；
- QPS/GiB；
- QPS/W。

如果只能使用一张 GPU，必须完整记录硬件并避免写“普遍适用于所有 GPU”。

---

## 29. 主参数网格

| Parameter | Values |
|---|---|
| `nprobe` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `nlist` | 1024, 4096, 16384；主实验 4096 |
| `alpha` | 2, 4, 8, alpha-max |
| `[B,Br]` | [8,4], [4,8], [8,8] |
| `k` | 1, 10, 100 |
| query batch | 1, 8, 32, 128, 256, 1024 |
| `add_batch` | 8K, 16K, 32K, 64K, 128K |
| `n_train` | 20K, 50K, 100K, 200K |

不需要在所有数据集做参数笛卡尔积。建议：

- 六数据集做完整主曲线；
- Vogue-768、OpenAI3-3072、Stella-1024 做深入 sensitivity；
- 其余数据集使用最佳/标准配置。

---

# Part VI：消融实验

## 30. 算法消融

| ID | Ablation | 证明内容 |
|---|---|---|
| A1 | JQ-GPU vs JHQ-GPU | hierarchy 是否必要 |
| A2 | JL vs no-JL | JL 对 accuracy/build 的作用 |
| A3 | global vs per-subspace residual codebook | 当前简化与论文实现差异 |
| A4 | primary only vs selective residual vs full residual | selective refinement 价值 |
| A5 | alpha = 2/4/8/max | candidate budget 权衡 |
| A6 | [B,Br] 三种配置 | primary/residual bit allocation |
| A7 | 不同 M | subspace size 的影响 |
| A8 | correction term on/off | 距离分解与精度 |
| A9 | K_LOCAL=4/8/16/exact | local top-k 是否丢候选 |
| A10 | CPU/GPU compressed distance | 数值正确性 |
| A11 | nlist/nprobe | coarse partition 与 scan 代价 |
| A12 | k=1/10/100 | top-k scalability |

---

## 31. GPU 系统消融

主文只保留有清晰因果关系的里程碑：

| Version | Core change |
|---|---|
| v1 | naive GPU full scan |
| v3 | + IVF |
| v4 | + batched query |
| v5 | + CUDA Graph |
| v10 | + Byte LUT |
| v12 | + `[M,N]` transposed primary layout |
| v14 | + memory-bounded streaming add |

### 31.1 Query waterfall

```text
v1 -> v3 -> v4 -> v5 -> v10 -> v12
```

每一步报告：

- Recall；
- QPS；
- total kernel time；
- HBM bytes；
- L2 hit rate；
- global-load efficiency；
- occupancy。

### 31.2 Build-memory waterfall

```text
v12 whole-vector add -> v14 streaming add
```

报告：

- peak VRAM；
- build time；
- H2D bandwidth；
- add throughput；
- maximum supported N。

### 31.3 独立系统消融

- CUDA Graph on/off；
- factored LUT vs Byte LUT；
- `[N,M]` vs `[M,N]`；
- sync copy vs async vs double-buffer；
- one stream vs two streams；
- pageable vs pinned output；
- `add_batch` sensitivity；
- `--use_fast_math` on/off；
- FP32 vs TF32 rotation；
- local top-k vs exact top-k。

---

# Part VII：系统指标与 Profiler 设计

## 32. PCIe 指标

教授提到的“PCI 使用率”应准确写成：

```text
PCIe utilization
```

必须测量：

| Metric | Definition |
|---|---|
| H2D bytes/query | total H2D bytes / #queries |
| D2H bytes/query | total D2H bytes / #queries |
| Effective H2D bandwidth | H2D bytes / H2D time |
| Effective D2H bandwidth | D2H bytes / D2H time |
| PCIe utilization | effective GB/s / measured sustained GB/s |
| Transfer-time ratio | transfer time / end-to-end time |
| Copy-compute overlap | overlapped copy time / copy time |
| GPU idle gap | GPU 等待数据的时间 |

### 32.1 先测机器真实 PCIe 上限

使用 CUDA bandwidth test 或等价 microbenchmark，分别测：

- pageable H2D；
- pinned H2D；
- pinned D2H；
- bidirectional throughput。

不要直接使用理论峰值作为 utilization 分母。

### 32.2 Search 理论 host traffic

当 index 常驻 GPU 时，理想 host traffic 约为：

```text
4*d + 8*k bytes/query
```

其中：

- query：FP32，`4*d`；
- result ID + distance：`8*k`。

例子：

| d, k | Ideal host traffic/query |
|---|---:|
| 1024, 10 | 4,176 bytes |
| 3072, 10 | 12,368 bytes |

因此 large-batch 常驻搜索中，PCIe 很可能不是主瓶颈；更可能的瓶颈是 HBM code scan、LUT 和 top-k。

但是如果 batch=1 仍执行 `B_full=256`，d=1024 时一次会复制约 1 MiB query buffer，因此必须先修 small-batch path 再报告 PCIe。

### 32.3 工具

- Nsight Systems：timeline 主证据；
- NVTX：阶段标记；
- CUPTI：精确 copy/kernel counter；
- NVML / `nvidia-smi dmon`：辅助采样；
- CUDA bandwidth test：机器上限。

---

## 33. GPU 显存指标

分别记录：

| Phase | Metrics |
|---|---|
| train | peak VRAM |
| add | peak VRAM |
| search | peak workspace |
| idle index | persistent VRAM |
| transient | temporary buffers |
| index | bytes/vector |
| compression | original bytes / index bytes |

### 33.1 测量方法

- 每个阶段前后调用 `cudaMemGetInfo`；
- NVML 高频采样；
- 包装 `cudaMalloc/cudaFree` 建 allocation logger；
- 单独记录 Thrust/CUB temporary storage；
- 输出 persistent、temporary、peak 三种数字。

只看程序结束后的 `nvidia-smi` 没有意义。

### 33.2 当前 v14 的近似内存模型

**[推算]** 忽略 codebook、centroid 和 workspace 后，persistent index 下界约为：

```text
N * (M + d*Br/8 + 8) bytes
```

其中：

- primary code：`M` bytes/vector；
- residual code：`d*Br/8` bytes/vector；
- ID：4 bytes；
- correction：4 bytes。

**[推算]** 当前 add 阶段显式 n-sized buffer 的峰值下界约为：

```text
N * (3*M + 2*d*Br/8 + 20) bytes
```

该式不包括：

- Thrust/CUB sort temporary storage；
- centroids/codebooks；
- batch raw/rotated buffers；
- query workspace；
- allocator fragmentation。

以 Stella `N=17,776,615,d=1024` 为例：

| M, Br | Persistent lower bound | Explicit add peak lower bound |
|---|---:|---:|
| M=128, Br=4 | ~10.73 GiB | ~23.64 GiB |
| M=128, Br=8 | ~19.20 GiB | ~40.59 GiB |
| M=256, Br=4 | ~12.85 GiB | ~30.00 GiB |

这些必须用真实 allocation trace 验证。

### 33.3 潜在下一步优化

v14 解决了 full-precision vectors 的 streaming，但 compressed arrays 在 gather/transpose 时仍可能多份共存。

可以进一步研究：

> Chunked or in-place list materialization without full duplicated compressed buffers.

这可能成为论文的额外系统贡献。

---

## 34. GPU 微架构指标

使用 Nsight Compute，对核心 kernel 报告：

| Metric | Purpose |
|---|---|
| Kernel time share | 找主瓶颈 |
| Achieved occupancy | register/shared-memory 限制 |
| SM active | 计算资源利用率 |
| DRAM throughput | 是否接近 HBM 上限 |
| L2 hit rate | LUT/code cache effectiveness |
| Global load efficiency | 访存合并程度 |
| Sectors/request | `[N,M]` vs `[M,N]` |
| Warp execution efficiency | divergence |
| Registers/thread | occupancy 约束 |
| Shared bank conflicts | Byte LUT 证据 |
| Warp stall reasons | memory/barrier/dependency |
| Instructions/byte | roofline 分析 |

### 34.1 Byte LUT 前后必须比较

- shared-memory bank conflicts；
- LUT load latency；
- scan time；
- QPS。

### 34.2 Transposed layout 前后必须比较

- global-load efficiency；
- sectors/request；
- L2 transactions；
- DRAM bytes；
- scan time；
- QPS。

这是解释“为什么快”的核心证据。

---

## 35. Kernel time breakdown

建议 NVTX 阶段：

```text
query_h2d
jl_rotation
centroid_gemm
probe_select
primary_lut
primary_scan
residual_lut
residual_refine
final_topk
result_d2h
```

建库阶段：

```text
train_sigma
jl_generate
train_rotation
primary_codebook
primary_encode
residual_train
residual_encode
ivf_train
ivf_assign
sort_by_list
list_gather
primary_transpose
workspace_alloc
```

每个阶段输出：

- average time；
- percentage；
- bytes read/write；
- achieved bandwidth；
- kernel launches。

---

## 36. CPU 与 host memory

必须记录：

- peak RSS；
- pinned host memory；
- CPU utilization；
- CPU thread count；
- CPU-side IVF update time；
- residual training time；
- data loading/mmap time；
- NUMA placement；
- CPU-GPU synchronization time。

工具：

- `getrusage`；
- `/usr/bin/time -v`；
- `/proc`/psutil；
- `numactl`；
- NVTX CPU ranges。

---

## 37. 能耗指标（加分项）

可选报告：

- average GPU power；
- Joules / 1,000 queries；
- QPS/W；
- build energy；
- QPS per dollar。

---

# Part VIII：实验运行规范

## 38. Timing protocol

### 38.1 Warm-up

- 第一次 search 用于 allocation/CUDA Graph capture，不计入结果；
- 至少 5–10 次 warm-up；
- 确认 GPU clocks 已稳定。

### 38.2 Repetition

每个配置至少：

```text
5 independent timing runs
```

每个 timing run 内 replay 足够多 query，使总时间建议大于 1 秒。

### 38.3 统计量

报告：

- median；
- mean；
- standard deviation；
- P95/P99 latency。

不要只报告最佳一次。

### 38.4 Cache state

主搜索结果使用 warm index/cache，符合在线 serving 场景。

如需 cold-start，单独报告：

- first query；
- first batch；
- graph capture；
- lazy initialization。

### 38.5 Clock 与环境

记录：

- persistence mode；
- power limit；
- GPU clocks；
- temperature；
- concurrent processes；
- MIG 状态；
- CPU governor；
- NUMA。

---

## 39. 结果 CSV schema

建议所有方法统一输出：

```text
run_id
commit_sha
timestamp
dataset
N
dimension
metric
method
method_version
gpu
cpu
cuda_version
driver_version
library_version
batch_size
k
nlist
nprobe
M
B
Br
alpha
add_batch
n_train
seed
rerank
input_mode
recall_at_k
qps
mean_latency_ms
p50_latency_ms
p95_latency_ms
p99_latency_ms
train_ms
add_ms
build_ms
persistent_vram_bytes
peak_vram_bytes
peak_host_rss_bytes
h2d_bytes
h2d_ms
d2h_bytes
d2h_ms
dram_bytes
l2_hit_rate
achieved_occupancy
sm_active
dram_throughput_gbs
avg_power_w
status
error_message
```

所有图表必须由 CSV 自动生成，不要人工复制数字。

---

## 40. 参数选择规则

为了避免 cherry-picking：

1. 先在 validation query 上调参；
2. test query 只用于最终报告；
3. 每种方法使用相同 recall target；
4. 报告完整 Pareto frontier；
5. 标明被 Pareto-dominated 的参数点；
6. 保存所有失败、OOM 和 timeout 记录。

---

# Part IX：论文必须产出的图表

## 41. 主文图

1. **JHQ-GPU system architecture**  
   Host query -> JL rotation -> IVF probe -> Byte LUT -> transposed primary scan -> selective residual refinement -> top-k。

2. **六个数据集 Recall@10-QPS 曲线**。

3. **固定 recall QPS 表**：0.90、0.95、0.99。

4. **Batch size vs QPS**。

5. **Batch size vs P50/P95/P99 latency**。

6. **GPU optimization waterfall**：v1 -> v3 -> v4 -> v5 -> v10 -> v12。

7. **Kernel time breakdown**。

8. **PCIe copy-compute timeline**。

9. **Build time + peak VRAM**。

10. **N scalability：100K -> 17.8M**。

11. **Index size / bytes-vector / compression ratio**。

12. **JQ vs JHQ / JL / alpha 核心消融**。

13. **Byte LUT profiler comparison**。

14. **Transposed-layout profiler comparison**。

## 42. Appendix 图表

- 所有 M/B/Br/nprobe 结果；
- global/per-subspace residual；
- K_LOCAL；
- fast-math/TF32；
- CPU baseline 全表；
- SONG/BANG/Faiss 补充结果；
- profiler counter 全表；
- OOM/timeout；
- 两张 GPU 的 portability；
- full raw CSV 索引。

---

# Part X：论文贡献与 claim 边界

## 43. 建议论文标题

```text
JHQ-GPU: Memory-Coalesced Hierarchical Quantization with
Memory-Bounded Construction for High-Dimensional ANN Search
```

## 44. 建议贡献点

### Contribution 1：Paper-faithful GPU JHQ

完整实现：

- JL transform；
- analytical primary quantization；
- hierarchical residual quantization；
- IVF search；
- selective residual refinement。

### Contribution 2：JHQ-specific GPU execution

- Byte LUT；
- memory-coalesced `[M,N]` primary-code layout；
- GPU top-k；
- fused/captured query pipeline。

### Contribution 3：Memory-bounded construction

- streaming raw/rotated vectors；
- 大规模高维建库；
- peak-memory analysis。

### Contribution 4：完整系统分析

- PCIe；
- HBM；
- L2；
- shared-memory conflicts；
- occupancy；
- build/query memory。

## 45. 当前不能过度声称

不要写：

- first GPU ANN method；
- fully GPU-native dynamic index；
- billion-scale，除非真实运行 1B；
- universally fastest GPU ANN；
- zero-copy；
- exact top-ck，除非已验证/修改；
- asynchronous overlap，除非 profiler 证明。

---

# Part XI：优先级与执行顺序

## 46. P0：正式实验前必须完成

- [ ] 从 `jhq_v14_streaming_add` 建 paper branch；
- [ ] 实现 per-subspace residual codebook；
- [ ] 实现 JQ-GPU；
- [ ] 修 small-batch fixed-B_full；
- [ ] 修 pinned-buffer 生命周期；
- [ ] 实现 double-buffer 或明确只做同步模式；
- [ ] 验证 K_LOCAL；
- [ ] 支持 CPU 论文参数；
- [ ] 主配置改为 `nlist=4096`；
- [ ] random training sample；
- [ ] CPU/GPU correctness tests；
- [ ] 设置准确 CUDA architecture；
- [ ] fast-math/TF32 数值验证；
- [ ] allocation logger；
- [ ] NVTX stage markers；
- [ ] 统一 CSV benchmark harness。

## 47. P1：构成论文主体

- [ ] 六个数据集；
- [ ] JQ/JHQ CPU；
- [ ] JQ/JHQ GPU；
- [ ] cuVS IVF-PQ；
- [ ] GPU IVF-RaBitQ；
- [ ] CAGRA；
- [ ] GPU Flat；
- [ ] Recall-QPS；
- [ ] batch latency；
- [ ] build time；
- [ ] VRAM/RSS；
- [ ] PCIe；
- [ ] optimization waterfall；
- [ ] Nsight Systems；
- [ ] Nsight Compute；
- [ ] scalability。

## 48. P2：增强论文

- [ ] 第二张 GPU；
- [ ] D2D API；
- [ ] double-buffer multi-stream；
- [ ] energy efficiency；
- [ ] SONG/BANG appendix；
- [ ] Jasper baseline；
- [ ] incremental add/delete；
- [ ] chunked compressed-buffer materialization；
- [ ] 1B/out-of-core extension。

---

# Part XII：最低可投稿实验包

## 49. Minimum publishable package

只有同时满足以下条件，才建议进入正式写作：

1. GPU 与 CPU JHQ 算法语义一致；
2. JQ-GPU 与 JHQ-GPU 均可运行；
3. 六个数据集完成；
4. 至少三个强 GPU baseline：IVF-PQ、IVF-RaBitQ、CAGRA；
5. 完整 Recall-QPS frontier；
6. build time、index size、peak VRAM 完整；
7. Byte LUT 和 transposed layout 有 profiler 证据；
8. small-batch 和 large-batch 均正确；
9. PCIe/HBM/occupancy 有定量分析；
10. 所有结果可由脚本和 CSV 复现。

---

# Part XIII：主要风险

## 50. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| GPU residual codebook 与 CPU 不一致 | Critical | 实现 per-subspace 版本 |
| small-batch 结果无效 | Critical | 多 graph bucket / dynamic path |
| K_LOCAL 导致 recall 损失 | High | exact top-ck 验证 |
| Stella [8,8] build OOM | High | allocation trace + chunked materialization |
| 只比单线程 CPU | High | 增加 all-core CPU |
| 缺失强 GPU baseline | High | IVF-PQ、IVF-RaBitQ、CAGRA |
| PCIe 指标无解释力 | Medium | Nsight timeline + measured bus peak |
| CUDA Graph 只对大 batch 有利 | Medium | 单独报告 small/large batch |
| build 仍含大量 CPU 工作 | Medium | 明确 claim；记录阶段时间 |
| `--use_fast_math` 改变 recall | Medium | on/off 数值消融 |
| 版本太多导致论文主线混乱 | High | 只保留 v1/v3/v4/v5/v10/v12/v14 |

---

# Part XIV：参考依据

## 51. 当前仓库

- [`README.md`](./README.md)
- [`VERSIONS.md`](./VERSIONS.md)
- [`jhq_v14_streaming_add/`](./jhq_v14_streaming_add/)
- [`examples/demo_jhq_v14_streaming_add.cu`](./examples/demo_jhq_v14_streaming_add.cu)
- [`CMakeLists.txt`](./CMakeLists.txt)

## 52. 主要论文

- JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization for High-Dimensional Approximate Nearest Neighbor Search.
- Billion-scale similarity search with GPUs.
- SONG: Approximate Nearest Neighbor Search on GPU.
- CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.
- BANG: Billion-Scale Approximate Nearest Neighbour Search Using a Single GPU.
- GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search.

---

# Final assessment

当前项目已经有一个可信的性能起点，但还处于：

```text
strong engineering prototype
```

而不是：

```text
submission-ready experimental system
```

真正决定论文能否成立的不是再增加一个新 kernel，而是完成以下闭环：

```text
算法一致性
-> 正确性验证
-> 强 baseline
-> 相同 recall 公平比较
-> build/query memory
-> PCIe/HBM/occupancy 证据
-> 六数据集可复现结果
```

完成 P0 和 P1 后，这个项目才具备完整 GPU ANN 系统论文的实验结构。
