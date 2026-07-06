# JHQ GPU 版本演进记录

数据集：Vogue-768（nb=932,328，nq=1,000，d=768，k=10）  
参数：M=96，B=8，Br=4，alpha=4.0，nlist=1024，nprobe=8，batch_size=1000

---

## 版本汇总表（nprobe=8，Recall@10≈0.998）

| 版本 | 核心改动 | QPS | 相对 v1 |
|------|---------|-----|---------|
| v1_plain | 基准 GPU 实现 | ~1,070 | 1× |
| v3_ivf | 引入 IVF，减少扫描量 | ~2,500 | 2.3× |
| v4_batched_query | 批量查询，提高 GPU 利用率 | ~9,460 | 8.8× |
| v5_cuda_graph | CUDA Graph 消除 kernel 启动开销 | ~12,700 | 11.9× |
| v6_async_h2d | 异步 H2D 传输（cudaMemcpyAsync + pinned） | — | — |
| v7_spin_sync | 自旋等待替代 cudaStreamSynchronize | — | — |
| v8_timing | 加 timing 工具（不改性能） | — | — |
| v9_step_timing | 每步 timing，定位瓶颈 | — | — |
| v10_bytelut | Byte LUT [B,M,256]，消除 16-way bank conflict | ~26,229 | 24.5× |
| v11_outerlut | 外层 m 循环 + 共享子表（失败） | ~19,364 | 18.1× |
| **v12_transposed** | **list_primary [M,N] 转置，32× 访问合并** | **~44,890** | **42×** |

---

## 各版本详细说明

### v1_plain — 基准 GPU 实现
- 直接将 CPU JHQ 逻辑搬到 GPU，逐 query 单独运行
- 全库暴力扫描，无 IVF
- QPS ≈ 1,070（nprobe 概念不适用，扫全库，alpha 代替 nprobe 控制精度）

### v3_ivf — 引入 IVF 索引
- 用 IVF 聚类（nlist=1024）缩小每次扫描范围
- 每 query 只扫 nprobe 个倒排列表，候选数 = nprobe × avg_list_size ≈ nprobe × 910
- QPS ≈ 2,500（nprobe=8）

### v4_batched_query — 批量查询
- 将 1000 个 query 打包成一个 batch 一次性送入 GPU
- 矩阵乘法（GEMM）的利用率大幅提升
- QPS ≈ 9,460（**3.8× vs v3**）

### v5_cuda_graph — CUDA Graph
- 将完整 pipeline（旋转→质心点积→select_probes→scan→残差→top-k）录制成 DAG
- 消除每次循环中多次 kernel launch 的 CPU 调度开销
- QPS ≈ 12,700（**1.3× vs v4**）

### v6_async_h2d — 异步 H2D
- query 数据通过 pinned host buffer + `cudaMemcpyAsync` 传输
- 传输与 GPU 计算 overlap，减少 host 阻塞时间

### v7_spin_sync — 自旋等待同步
- 将 `cudaStreamSynchronize`（驱动进入睡眠）替换为 `cudaStreamQuery` 自旋循环
- 消除线程唤醒延迟，稳定降低 P99 延迟

### v8_timing / v9_step_timing — 性能分析工具
- 插入 CUDA event 计时，精确测量每个 pipeline 步骤耗时
- 用于定位瓶颈（发现 scan_ivf 是主要瓶颈）

### v10_bytelut — Byte LUT，消除 bank conflict

**问题**：原始 scan 每步通过 `lut[m][k][j]` 查表，32 个线程同时访问同一 bank 的 16 个不同位置 → 16-way bank conflict，共享内存串行化。

**改动**：预计算 `byte_lut[B][M][256]`（float），将 M 维编码的所有可能 byte 值的距离预先算好。scan 时直接 `byte_lut[bqi][m][cm]` 一次查表，完全消除 bank conflict。

**代价**：`byte_lut` 显存占用 = 1000×96×256×4 = **94 MB**

**效果**：QPS ≈ 26,229（**2.1× vs v5**）

### v11_outerlut — 外层 m 循环（失败）

**想法**：把 scan 循环结构改为外 m 内 candidate：每次迭代 m 时，256 线程协作把 256-entry 子表（1KB）加载到 shared memory，所有候选查共享内存（~1 cycle）而非 L2（~30 cycle）。

**问题 1**：MAX_CANDS=32 静默丢弃  
IVF 列表大小不均匀（有的 >2000），n_my 超过 32 时 `break` 导致候选丢失 → Recall=0.7879。

**问题 2**：768 个 `__syncthreads()` 杀死 latency hiding  
修复后（chunked 方案）：4 chunks × 96 m-iterations × 2 syncs = **768 次硬屏障**。GPU 无法通过切换 warp 来隐藏内存延迟 → 比 v10 更慢（51ms vs 38ms）。

**结论**：外层 m 的正确思路需要从根本上消除 shared memory 的 sync 需求，即改变数据布局。

### v12_transposed — [M,N] 转置布局（当前最优）

**根本原因分析**：v10 的 `list_primary[N][M]`，32 个 warp 线程各自读第 m 个字节时，地址间隔为 M=96 字节，32 次读落在 32 条不同 cache line → **98% cache line 浪费**，每次 warp 读产生 32 条 HBM 访问。

**改动**：在 `add()` 时用 tiled 32×32 shared memory 转置 kernel，将 `list_primary[N][M]` 转置为 `list_primary_t[M][N]`（一次性，永久存储）。

scan kernel 中访问 `list_primary_t[m * N + abs_pos]`：32 个线程的 `abs_pos` 连续 → 32 个连续字节 = **1 条 cache line，利用率 100%**。

**内存影响**：转置后的临时 [N,M] buffer 在 add() 完成后即释放，净增显存 0。

**效果**：
- nprobe=4：QPS = **54,644**，Recall = 0.9907（峰值）
- nprobe=8：QPS = **44,890**，Recall = 0.9982
- 相对 v10：**1.71× 加速**
- 相对 JHQ CPU 官方版（nprobe=8）：**8.3× 加速**

**v12 vs CPU（nprobe=8，Recall≈0.998）**

| 方法 | QPS | 加速比 |
|------|-----|--------|
| JHQ CPU (Official) | ~5,400 | 1× |
| JQ CPU (Official) | ~11,300 | 2.1× |
| JHQ-GPU v12 | ~44,890 | **8.3×** |

---

## HBlock 系列

参数：K1=64，K2=128，ck1=8，ck2=32，ck3=256，leaf_size=128，bpv=384，batch_size=1024，k=10

### 版本汇总表

| 版本 | 核心改动 | QPS | Recall@10 |
|------|---------|-----|-----------|
| hblock_v1 | 三层层级索引基线（L1→L2→叶块），解析码本，GPU叶块kernel | — | — |
| hblock_v2 | 2D grid叶块kernel，100%占用率；sort_leaf_sel_kernel改善HBM局部性 | — | — |
| hblock_v3 | PCA级联路由投影 | — | — |
| hblock_v4 | 判别式S_B投影路由 | — | — |
| hblock_v5 | 转置LeafFine kernel；全局排序+L2残差复用，修复HBM随机读瓶颈 | — | — |
| hblock_v6 | 查询按L1中心预排序再派发（B2路由策略） | — | — |
| hblock_v7 | 每个unique叶块一个CUDA block | — | — |
| hblock_v8 | Flink风格micro-batch流式搜索，in-kernel top-p过滤 | — | — |
| hblock_v9 | 真正双缓冲流式派发 | — | — |
| hblock_v10 | 单次大kernel launch，充分利用GPU并行度 | — | — |
| hblock_v11 | 每个(查询, 叶块)对一个block；去掉codes的smem，100%占用率 | — | — |
| hblock_v12 | GPU端构建任务对 + CUB RadixSort(14-bit)按leaf_id排序；消除CPU排序 | 17,182 | 0.8862 |
| hblock_v13 | **转置codes布局[blk][bpv][leaf_size]**：128线程读1个cache line/step；kernel 50ms→7ms | 68,629 | 0.8056 |
| hblock_v14 | **GPU端top-k合并**：iota→qid RadixSort(10-bit)→分段top-k kernel；D2H 9MB→82KB | 131,277 | 0.8056 |

### hblock_v13 详细说明

**问题**：旧布局[blk][leaf_size][bpv]，128个线程各自读第b个字节时地址间隔bpv=384字节→128次cache miss/step。

**改动**：建图时按[blk][bpv][leaf_size]存储，搜索时`leaf_base[b * leaf_size + tid]`，128个线程读连续128字节=1个cache line。

**效果**：叶块kernel 50.4ms→6.9ms（7.3×），QPS 17K→69K。

### hblock_v14 详细说明

**问题**：v13叶块kernel后需回CPU合并结果（255K×4次heap push，7ms）+9MB D2H传输。

**改动**：全程在GPU完成——
1. iota fill d_pair_leaf_a [0..n_pairs)
2. CUB DeviceRadixSort按qid(10-bit)排序，得到qid有序置换数组
3. segmented_topk_kernel：每个query一个block，32线程共享内存堆，输出nq×k结果
4. D2H只传82KB（nq×k×8字节）

**关键bug修复**：输出stride必须用k而非K_MAX，否则D2H只覆盖前160条query的结果（84%数据读到未初始化内存，Recall从0.81跌到0.13）。

**效果**：CPU merge 7ms→0ms，D2H 9MB→82KB（110×），QPS 69K→131K。

### HBlock性能对比（Vogue-768，n=933K，RTX 5090）

| 方法 | Recall@10 | QPS | vs JHQ CPU |
|------|-----------|-----|-----------|
| JQ CPU | ~0.88 | ~24,500 | 0.19× |
| JHQ CPU | ~0.88 | ~17,000 | baseline |
| HBlock v12 | 0.8862 | 49,123 | 2.9× |
| HBlock v13 | 0.8056 | 68,629 | 4.0× |
| HBlock v14 | 0.8056 | 131,277 | **7.7×** |

---

## arXiv-Abstracts-768 扩展实验

数据集：arXiv-Abstracts-768（nb=2,253,198，nq=1,000，d=768，LID=31.8，RC=1.50）  
硬件：RTX 5090（SM120）  
参数：K1=64，K2=128，ck1=8，ck2=32，leaf_size=128，batch_size=1024，k=10

### ck3 扫描结果（HBlock v14）

| ck3 | QPS | Recall@10 | 覆盖率 |
|-----|-----|-----------|--------|
| 256 | 136,339 | 0.2572 | 1.5% |
| 600 | 87,884 | 0.3963 | 3.4% |
| 2048 | 35,657 | 0.7312 | 11.6% |

### 与原论文对比（arXiv-768）

| 方法 | QPS | Recall@10 | 硬件 |
|------|-----|-----------|------|
| JQ（论文）| ~2,289 | ~0.85 | CPU单线程 |
| JHQ（论文）| ~889 | ~0.90 | CPU单线程 |
| HBlock v14，ck3=2048 | 35,657 | 0.731 | RTX 5090 |

### 分析

**QPS**：ck3=2048时35K vs JHQ ~2K，领先约16×。

**Recall瓶颈**：同覆盖率下（ck3=600 ≈ Vogue ck3=256，均约3.5%），arXiv recall仅0.40 vs Vogue 0.81。根本原因是**路由质量**：arXiv为文本embedding（InstructorXL），RC=1.50更小、数据点更集中，JL旋转后near-Gaussian假设成立性较差，L1/L2路由选出的叶块精准度下降。

**QPS不随n下降**：n从933K→2.25M（2.4×），QPS从131K→136K（略升），印证路由开销O(batch×d)与n无关，GPU利用率随叶块数增多反而更充分。

**结论**：HBlock在QPS上有明显优势，但当前路由结构对RC小（近邻密集）的文本数据集泛化性不足，需提升K1/K2或引入数据自适应路由。

---

## 后续可能的优化方向

### 1. Byte LUT 访问优化（当前最大瓶颈）

v12 之后 `list_primary` 已经完全合并访问，瓶颈转移到 `byte_lut` 查表。

**问题**：`byte_lut[bqi][m][cm]` — 每个 candidate 做 96 次随机 L2 读（cm 不连续），LUT 总大小 94MB 远超 L2 容量（典型 32-40MB），cache 命中率低。

**方向 A — half/int8 LUT**  
将 LUT 从 float32 改为 float16 或 int16，LUT 大小减半（47MB），L2 压力减半。累加时用 `__half2float`。

**方向 B — 共享内存 LUT（逐 m 加载）**  
每次处理一个 m 时，256 线程协作把 `byte_lut[bqi][m][0..255]`（1KB）加载到 shared memory，然后所有候选从 shared 查。  
问题：需要 `__syncthreads()`，回到 v11 的困境。  
解法：把 list_primary 访问和 LUT 查表解耦（先把候选的 codeword 批量预取，再统一查表）。

**方向 C — 向量化读取 list_primary_t**  
将 M=96 个 byte 按 `uint4`（16字节）批量读取，减少 L1/L2 请求次数。需要对 M 做 16-byte 对齐 padding。

### 2. Warp-level 数据重用

相邻 warp 的候选点 `abs_pos` 连续，各自独立读 96 个 codeword。  
可以探索 warp shuffle 在相邻线程间传递 codeword，减少重复读取。

### 3. 多流 pipeline（Stream 并行）

目前 CUDA Graph 将整个 pipeline 串行化。  
可以将 batch 切成两半，用两个 stream 交叉执行（一个 stream 做 H2D + rotate，另一个同时做 scan），进一步隐藏 H2D 延迟。

### 4. select_probes 优化

当前 `select_probes_kernel` 用 O(nprobe × nlist) 的顺序 reduction，nprobe=128 时成本显著。  
可以改为 heap / bitonic sort 的一次性 top-nprobe 选取，复杂度 O(nlist × log(nprobe))。

### 5. 残差精化 (residual_refine) 加速

当前 `residual_refine_batched_kernel` 对 ck 个候选各做 d=768 维查表。  
类似 v12 的思路：将 `list_res` 也做转置，从 [N, bpv] 改为 [bpv, N]，改善 warp 访问合并。

### 6. 多 GPU

当前单 GPU 在 Recall=0.998 时 QPS≈45K。  
使用 NVLink 多 GPU 可线性扩展，但需要 query routing 和结果合并逻辑。

---

## 关键经验总结

| 教训 | 说明 |
|------|------|
| GPU 批量是前提 | v3→v4 的 3.8× 说明：单 query 打 GPU 利用率极低，必须打 batch |
| CUDA Graph 值得做 | 高频小 kernel pipeline 用 Graph 消除 CPU 调度开销，效果明显 |
| Bank conflict 很贵 | v10 的 2× 提升全来自消除 16-way bank conflict，shared memory 设计要慎重 |
| sync 比读内存更贵 | v11 的 768 个 `__syncthreads()` 比 32 倍 cache miss 代价更高，latency hiding 是 GPU 性能的核心 |
| 数据布局决定带宽 | v12 的 1.71× 提升只改了存储顺序，零算法变动；coalescing 是 HBM 带宽利用的关键 |
| 先 profile 再优化 | v11 的失败提醒：直觉上合理的优化（共享内存查表）可能因副作用（sync 开销）反而变慢 |
