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
| hblock_v1～v21 | 基础架构探索（IVF路由、JL路由、block graph雏形，见下文详细说明） | — | — |
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
| hblock_v17 | 3级JL路由+PQ粗排，当前IVF路线最佳 | 26,468 | 0.9914（ck=5） |
| hblock_v22～v27 | block graph架构探索：GPU beam search on block graph，bitonic sort，template beam | — | — |
| hblock_v28 | per-block exact L2 rerank，消除跨block PQ误差；oracle≈actual | ~200K | 0.9741（d=64,b=64） |
| hblock_v29 | balanced k-means（cluster=block），block centroid更精准 | ~200K | 0.9741（d=64,b=64） |
| **hblock_v30** | **修复visited-before-insert bug；depth=256可达0.9914** | **107K** | **0.9914（d=256,b=128）** |

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

### hblock_v17 — 3级 JL 路由 + PQ 粗排（当前最佳）

**架构：** 3级层级 JL 路由 + PQ coarse scan + 精确 L2 精排

- K1=K2=K3=16，共 4096 个 leaf cell，每个 leaf block 128 个向量
- 三级 JL 投影（d_proj=64）做近似质心路由；PQ codes（1 byte/向量）+ query-specific LUT 做粗排
- 关键 bug 修复：gather kernel 之前只写每个 cell 的第一个 leaf block，修复后 recall 从 0.70 升至 0.99+

**结果（RTX 5090）：**

| ck | Vogue recall | Vogue QPS | Arxiv recall | Arxiv QPS |
|----|-------------|-----------|-------------|-----------|
| 3  | 0.9342 |  91,463 | 0.9773 | 114,876 |
| 4  | 0.9657 |  64,769 | 0.9928 |  56,582 |
| 5  | 0.9914 |  26,468 | 0.9986 |  24,256 |
| 6  | 0.9937 |  19,160 | 0.9978 |  15,724 |

---

### hblock_v18 — JL 粗排替换 PQ（实验，已放弃）

**改动：** PQ coarse scan → JL coarse scan（d_proj=64，leaf 存 float32 投影向量，复用 L1 路由投影）

- K_MAX=64，JL_TOP_P=8
- recall 平台卡在 0.877（Vogue）/ 0.913（Arxiv）
- 根因：rerank_r=64 太小；JL 每 leaf block 读 32KB vs PQ 128B，带宽差 256 倍

---

### hblock_v19 — JL 粗排修复版（实验，已放弃）

**改动（基于 v18）：** 拆分 `d_proj_route=128`（路由）/ `d_proj_scan=64`（scan）；空质心 norm=1e30f；K_MAX 64→192，JL_TOP_P 8→16

**结果：** Vogue 0.979 @ 8K QPS；Arxiv 0.985 @ 8K QPS

**结论：** recall 平台打破，但 QPS 仍远不及 v17。JL 粗排整体不如 PQ 粗排路线。

---

### hblock_v20 — v17 + 路由升维 128D + 空质心修复

**改动：** 路由投影 64D → 128D，PQ 粗排不变；`upload_cents` 追踪 `h_valid` mask，空 cell norm=1e30f

**结果（RTX 5090）：**

| ck | Vogue recall | Vogue QPS | Arxiv recall | Arxiv QPS |
|----|-------------|-----------|-------------|-----------|
| 2  | 0.8872 | 101,646 | 0.9527 | 191,628 |
| 3  | 0.9657 |  67,778 | 0.9871 |  85,348 |
| 4  | 0.9803 |  42,163 | 0.9904 |  44,253 |
| 5  | 0.9963 |  20,855 | 0.9963 |  19,674 |
| 6  | 0.9971 |  15,091 | 0.9924 |  13,459 |

**分析：**

- **低-中 ck（2-5）：Pareto 改进明显。** 同等 recall 目标下需要更少的 probe：v17 ck=4 达到 0.9657，v20 ck=3 即可（67K vs 65K QPS，更快）；v17 ck=6 达到 0.9937，v20 ck=5 即可（21K vs 26K QPS，v20 慢但召回更高）。
- **高 ck（8-10）：recall 略有退步。** 根因：128D 路由改变了 k-means 分配，L3 残差 r3 分布改变影响 PQ 编码精度；加上 rerank_r=128 是瓶颈，大 ck 时 merge 截断吃掉了路由增益。
- **QPS 代价约 20-25%**：128D GEMM 比 64D 更贵。
- **结论：在 0.85-0.98 recall 区间，v20 Pareto 优于 v17；0.99+ 区间大体相当但 QPS 较低。**

---

### hblock_v22 — L1+L2 GPU beam + L3 CPU graph traversal

**改动：** 引入 block-level graph 搜索思路的雏形——L1/L2 级用 GPU beam 选候选 cell，L3 级在 CPU 上做 graph traversal over leaf cells。

**结论：** 架构验证，为后续全 GPU block graph 版本奠基。

---

### hblock_v23 — block-level graph with true block centroids

**改动：**
- 对每个 leaf block 计算真实质心（block centroid）
- 在 block-level 建 kNN 图，图节点 = block，边 = 最近邻 block
- 搜索时 beam search on block graph，而非枚举所有 leaf cell

**意义：** 从 IVF 式"枚举 cell"转向"图遍历"，图搜索天然支持跨 cell 导航，突破 ck 参数限制的 recall 天花板。

---

### hblock_v24 — oracle diagnostic + runtime sweep

**改动：**
- 引入 oracle recall 诊断：CPU 精确扫描 visited blocks，衡量"路由+图遍历"选到的 blocks 能给出多少 recall 上界
- TOP_P / beam_size 支持 runtime sweep（build 一次，搜索时调参）
- 去掉路由 D2H，减少 H2D/D2H 次数

---

### hblock_v21 — v17 + 稀疏子树 bitmask 路由

**改动：** train 时为每个父节点构建 16-bit valid children bitmask：
- `d_valid_c2_[c1]`：K2=16 bits，第 k 位置 1 表示 c1 的第 k 个 L2 子节点非空
- `d_valid_c3_[c1*K2+c2]`：K3=16 bits，同理
- 存储：16×2B + 256×2B = 544B，常驻 L1 cache
- `route_l2_beam_kernel` / `route_l3_beam_kernel` 对空子节点直接赋 INF，不竞争 beam 名额

**结果（RTX 5090）：**

| ck | Vogue recall | Vogue QPS | Arxiv recall | Arxiv QPS |
|----|-------------|-----------|-------------|-----------|
| 3  | 0.9511 |  97,544 | 0.9789 | 113,978 |
| 4  | 0.9675 |  63,542 | 0.9918 |  56,152 |
| 5  | 0.9938 |  26,685 | 0.9978 |  24,448 |
| 6  | 0.9937 |  18,879 | 0.9984 |  15,683 |

**分析：**

- **null result：与 v17 几乎无差别。** recall 和 QPS 差距均在噪声范围内（<0.003 recall，<2% QPS）。
- **根因：几乎没有空 cell。** n_km=200K 均匀分布在 K1×K2×K3=4096 个 cell，平均每 cell 约 49 个训练向量，L3 级别的空 cell 极少，bitmask 近乎全 1，对路由决策无影响。
- **bitmask 的适用条件：** K1×K2×K3 远大于 n/leaf_size（数据极度稀疏时）或数据分布高度不均匀时才能发挥作用。当前配置不满足。
- **结论：v21 在现有参数下无效，暂不作为后续基础版本。**

---

### hblock_v25 — mini k-means 语义 block 构造

**改动：** 在每个 L3 cell 内部跑 mini k-means（CPU，迭代 5 次），把 cell 内向量聚成 K 个子簇，每个子簇打包成一个 block。

**效果：** 同一 block 内的向量语义更接近，block centroid 更能代表 block 内容，block graph 的路由精度提升。

---

### hblock_v26 — template beam (32/64/128 slots) + global top-rerank_r

**改动：**
- beam size 通过 C++ template 参数 `<int SPT>` 实现（SPT=1/2/4 对应 32/64/128 slots），一次编译覆盖所有规格
- 去掉 per-block top_p 过滤，改为 global top-rerank_r 统一排名

---

### hblock_v27 — bitonic warp sort in leaf_flat

**改动：** leaf_flat kernel 内部用 bitonic warp sort 求 top-32（每 block 最佳候选），替换原来的线性扫描。保持 QPS，提升局部排序精度。

---

### hblock_v28 — per-block exact L2 rerank（消除跨 block PQ 误差）

**核心问题：** 原版在 global 层面用 PQ 距离比较跨 block 的候选，但不同 L3 cell 的 PQ 残差基点不同，PQ 距离存在系统性偏差，无法公平比较跨 block 候选。

**改动：**
- 每个 (query, block) 对先用 PQ 筛出 top `per_block_r=16` 候选
- 再对这 16 个候选做精确 L2（加载原始向量，128 线程并行）
- 最终合并各 block 的精确 top-`klocal=10`

**关键 bug 修复：** CUB SortPairs 参数顺序写反（keys_in/keys_out 与 values_in/values_out 对调），导致 recall 0.9235 → 0.0013。修复后 oracle ≈ actual（gap ≤ 0.002）。

**结果（RTX 5090，depth=64，beam=64）：**

| 数据集 | oracle recall | actual recall | QPS |
|--------|-------------|--------------|-----|
| Vogue | 0.9750 | 0.9741 | ~200K |
| Arxiv | 0.9701 | 0.9694 | ~199K |

---

### hblock_v29 — balanced k-means（cluster = block）

**问题：** v28 的 mini k-means 用标准 assignment，一个 cluster 可能超过 leaf_size=128 个向量，导致一个逻辑 cluster 被截断分到多个 block，block centroid 不再对应 k-means 质心，图搜索用质心导航时精度下降。

**改动：** 最终 assignment 改为 greedy sorted-pairs balanced assignment——按 (dist, vi, k) 排序后贪心分配，保证每个 cluster ≤ leaf_size 个向量，cluster 和 block 一一对应，block centroid 即 k-means 质心。

**结果（RTX 5090，depth=64，beam=64）：**

| 数据集 | oracle recall | actual recall | QPS |
|--------|-------------|--------------|-----|
| Vogue | 0.9750 | 0.9741 | ~200K |
| Arxiv | 0.9701 | 0.9694 | ~199K |

改善主要体现在 beam=128 路径比 v28 更稳定；oracle ≈ actual 的性质保持。

---

### hblock_v30 — 修复 visited-before-insert bug

**核心 bug：** beam search expansion 阶段，`try_visit`（atomicOr 标记 visited）在 `bmax` 检查之前执行。当一个 neighbor block 距离 ≥ bmax（无法进入 beam）时，它已被永久标记为 visited，后续通过其他图路径到达这个 block 时会被跳过——切断了通往 GT 近邻的图路径。

**表现：** beam=128 的 oracle recall 反常地低于 beam=64（beam 越大，bmax 在 entry 阶段保持 1e38f 越久，期间允许更多 distant block 进入并被 visited 标记，堵死更多路径）。

**修复：**
1. **entry 阶段**：先算 bmax，只有 mn < bmax 时才 atomicOr + insert
2. **expansion 阶段**：先用非原子读软检查 visited，再算距离，再检查 bmax，只有 nd < bmax 时才 atomicOr + insert

```cpp
// 修复后的 insert_if_better
auto insert_if_better = [&](float nd, int ni) {
    float lw; int ls; local_worst(lw, ls);
    float bmax = ...; // warp reduce
    if (nd >= bmax) return;  // 不够好 → 不标记 visited，路径保持开放
    bool fv = false;
    if (tid == 0) fv = try_visit(vis, ni, n_blks);
    if (!__shfl_sync(..., fv, 0)) return;
    // 插入 beam
};
```

**结果（RTX 5090，depth 和 beam sweep）：**

| depth | beam | Vogue oracle | Vogue actual | Vogue QPS |
|-------|------|-------------|-------------|-----------|
| 64 | 64 | 0.9745 | 0.9740 | 200K |
| 64 | 128 | 0.9706 | 0.9689 | 197K |
| 128 | 128 | **0.9889** | **0.9888** | 124K |
| 256 | 128 | **0.9916** | **0.9914** | 107K |

| depth | beam | Arxiv oracle | Arxiv actual | Arxiv QPS |
|-------|------|-------------|-------------|-----------|
| 64 | 64 | 0.9697 | 0.9692 | 192K |
| 128 | 128 | 0.9859 | 0.9857 | 119K |
| 256 | 128 | **0.9890** | **0.9889** | 111K |

**分析：**
- oracle ≈ actual（差距 <0.002）说明 per-block exact rerank 几乎无损，recall 瓶颈完全在 block 覆盖
- beam=32 在大 depth 下不涨：frontier 太窄，好候选被挤出去
- depth=256 vs depth=64：recall +0.017（Vogue），证明 graph 连通性足够，问题是 depth 不够
- 剩余 ~1% recall 缺口可通过增大 degree（32→64）或扩大建图 candidate 范围解决

**当前参数：** degree=32，n_c2_nbrs=4，n_c1_nbrs=2，max_cand_blocks=2048

---


### hblock_v32 — 参数统一（v30 thin wrapper）

**设计**：将 v30 的 `graph_depth` / `beam_size` 两参数统一为单一 `ef`，内部映射：

```
ef → depth = ef，beam = min(ef, 128)
```

routing、block search kernel、rerank 均不变（直接复用 route_gpu_v29 / gpu_block_search_v27）。

**结果**（Vogue-768，K1=K2=K3=16，graph_degree=32，ck1=2 ck2=2 ck3=4）：

| ef | recall@10 | QPS |
|----|-----------|-----|
| 32 | 0.9270 | 307K |
| 64 | 0.9707 | 208K |
| 128 | 0.9895 | 126K |
| 256 | 0.9897 | 123K |

---

### hblock_v33 — W=4 批量展开 + HNSW 早终止（实验版，已弃用）

**设计**：beam = ef（无 cap）；每轮展开 W=4 个 block 后检查 `best_unexp >= worst_beam`；**输出 final beam 内容**（非 expanded blocks）。

**问题**：final beam 丢弃"曾展开但被挤出 beam"的 block → 小 ef 时 recall 下降（ef=32: 0.9182 vs v32 的 0.9270）；ef=256 走 SPT=8，QPS 仅 72K；早终止收益极小（<2%）。

---

### hblock_v34 — 修正版（当前推荐）

**设计**：
- beam = min(ef, 128)（SPT ≤ 4）
- 保留 W=4 + HNSW 早终止
- **恢复 expanded-block 输出**（同 v30/v32 语义）
- 添加 warmup 调用消除冷启动干扰

| ef | recall@10 | QPS | 备注 |
|----|-----------|-----|------|
| 8 | 0.700 | 503K | warmup 后正常 |
| 32 | 0.921 | 311K | |
| 64 | 0.973 | 212K | |
| 128 | 0.991 | 130K | |
| 256 | 0.991 | 127K | ≈ ef=128（beam cap 截断实际展开约 128 次）|

---

### hblock_v35 — 天花板测试

**设计**：beam = ef（无 cap，ef=256 → SPT=8）；**去除 HNSW 早终止**；固定展开 ef 次；expanded-block 输出。

| ef | recall@10 | QPS | 备注 |
|----|-----------|-----|------|
| 128 | 0.990 | 130K | 与 v34 完全一致 |
| 256 | **0.997** | 72K | beam=256，完整 256 次展开 |

**结论**：v34 ef=256 的 0.991 被 beam cap 压制（非算法上限）。beam=256 完整展开后达 0.997，代价 QPS -43%。进一步提升需改 routing 覆盖率或 graph 结构。

---

### 各版本对比（ef=128 / ef=256）

| 版本 | beam | 早终止 | 输出 | ef=128 recall | ef=128 QPS | ef=256 recall | ef=256 QPS |
|------|------|--------|------|--------------|-----------|--------------|-----------|
| v32 | min(ef,128) | 无 | expanded | 0.9895 | 126K | 0.9897 | 123K |
| v33 | ef | 有 | final beam | 0.9897 | 129K | 0.9962 | 72K |
| v34 | min(ef,128) | 有 | expanded | 0.9907 | 130K | 0.9907 | 127K |
| v35 | ef | 无 | expanded | 0.9904 | 130K | **0.9968** | 72K |

**推荐**：速度优先用 v34（ef=128，0.991@130K）；精度优先用 v35（ef=256，0.997@72K）。

---

### hblock_v36 — GPU radix sort 替换 CPU stable_sort

**设计**：完全基于 v35，只改一处：建图阶段的全局排序从 CPU `std::stable_sort` 改为 GPU CUB `DeviceRadixSort`。

**改动位置**：`hblock_v36/jhq_gpu_index.cu`，`add()` 函数内 sort 段。
- 旧：在 host 上按 `(c1, c2, c3)` cell code 做 `std::stable_sort`，时间复杂度 O(n log n) 纯 CPU。
- 新：host 上预先计算 `h_keys[i] = c1*K2K3 + c2*K3 + c3`，上传至 GPU，调用 `cub::DeviceRadixSort::SortPairs` 完成排序，结果 `order[]` 下载回 host。搜索逻辑、block packing、非早停逻辑全部不变。

**实测效果**：排序时间从 CPU 103.9 ms → GPU 5.0 ms（**20× 加速**）；建图总时间 8941 ms（v35 约 9050 ms）；recall 与 v35 完全一致。

| ef | recall@10 | QPS | 备注 |
|----|-----------|-----|------|
| 128 | 0.9885 | 128K | 与 v35 一致 |
| 256 | **0.9957** | 72K | sort 加速不影响搜索质量 |

**更新版本对比表**：

| 版本 | beam | 早终止 | 输出 | ef=128 recall | ef=128 QPS | ef=256 recall | ef=256 QPS |
|------|------|--------|------|--------------|-----------|--------------|-----------|
| v32 | min(ef,128) | 无 | expanded | 0.9895 | 126K | 0.9897 | 123K |
| v33 | ef | 有 | final beam | 0.9897 | 129K | 0.9962 | 72K |
| v34 | min(ef,128) | 有 | expanded | 0.9907 | 130K | 0.9907 | 127K |
| v35 | ef | 无 | expanded | 0.9904 | 130K | **0.9968** | 72K |
| v36 | ef | 无 | expanded | 0.9885 | 128K | 0.9957 | 72K |

## 后续可能的优化方向

### 0. Block-level graph + super-block 物理布局（1B 方向）

**核心想法**：graph 节点保持为小粒度 logical block（如 128 vectors，GPU 扫描单元），但物理存储/传输单位可以是更大的 super-block/page，把若干空间上接近的 logical blocks 放在同一个连续物理块中。

```text
logical block      = graph node / GPU scan tile
physical page      = transfer unit
physical page      = 4 / 8 / 16 nearby logical blocks
```

**优势**：
- block-level graph 的节点数约为 `N / block_size`，相比 vector-level graph 小约 `block_size` 倍，因此图本身不会很大，建图也会比 vector graph 更快。
- 搜索 frontier 仍然按 `block_id` 做细粒度扩展，但加载时映射到 `page_id`，一次搬运可覆盖多个后续可能访问的 nearby blocks。
- 如果几个近距离 logical blocks 被放在同一个 physical page，图搜索的几次 hop 可能命中同一页，减少小随机传输和 PCIe/SSD 调度开销。
- 复杂度可从按 `visited_blocks * block_bytes` 进一步优化为按 `visited_pages * page_bytes`；只要布局好，`visited_pages << visited_blocks`。

**建议实验参数**：
- logical block 固定为 128 vectors；
- super-block/page 可扫 `4 / 8 / 16` 个 logical blocks；
- 初始建议 `8 blocks/page`，即 1024 vectors/page，在顺序读收益和无效搬运之间折中。

**关键约束**：super-block 只是物理 I/O 单元，不改变搜索粒度。graph node 仍然是 logical block，避免把算法剪枝粒度变粗。

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

### 6. 叶块内空间排序 + 早退

**现状**：`stable_sort` 保证同一 `(c1, c2)` 簇的向量进同一批叶块（簇级 locality ✓），但簇内 128 个向量按原始 ID 顺序填入，没有细粒度空间排序。

**思路**：建图时，对每个叶块内的向量按残差 `r2 = x - c1_center - c2_center` 到叶块中心的距离从近到远排列。搜索时若已累积足够好的 candidate，可在叶块中途跳出，不必扫满 128 个向量。

**前提**：需实现叶块内早退逻辑（当前 kernel 固定扫满整块）。排序本身在 CPU build 阶段完成，zero search overhead；早退减少有效扫描量，可在不增大 ck3 的前提下提升 recall/QPS 曲线。

### 7. 多 GPU

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
