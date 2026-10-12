# v62：List 并行 Primary Screening 与精确全局 top-ck

2026-10-12 · RTX 5090（AutoDL）· commits：Phase 0 `b3b3432`/`7723059`，v62 `862543a`，运行脚本 `e0d0518` · 原始数据：`raw/`

**决定：GO。**
- **正确性：** 每条查询、每个 batch、每个 G 都通过：non-tie 差异为 0，最终 top-k 逐位一致。
- **小 batch：** 每个 workload 在 b=8 时端到端快 ×2.06–2.76，b=32 时快 ×1.63–2.30。5 轮配对都稳定，min/max 相差不到 0.5%。阶段耗时可以解释这些收益。
- **大 batch：** b ≥ 256 时 v62 变慢，最低到 ×0.46（arxiv α=200，b=1000，G=8）。

所以它是一种小 batch 执行模式，不是 v61 扫描的替代品。两点都由 pipeline campaign 的 Experiment 2 和 4 跟进，见 `experiments/pipeline_campaign/MASTER_REPORT.md`。

**实验条件：**
- 每个数据集一份索引：arms6 trained-state cache，即论文那份。
- 同一配置的所有变体在同一进程里，搜索同一批查询。
- 构建：Release，`CMAKE_CUDA_ARCHITECTURES=80`，即 compute_80 PTX，由 driver 595.71.05 为 RTX 5090 JIT 编译；CUDA 13.0。
- 原始日志、构建参数、逐查询检查输出、sanitizer 日志和 nsys trace（gzip CSV）都在 `raw/`。

## 1. 执行图

```
v61（G = 0），每个 (B, ck, nprobe, lanes) 一个 CUDA graph：
  H2D q -> 旋转 (cuBLAS) -> 质心点积 (cuBLAS) -> select_probes [B 个 block]
        -> build_byte_lut -> scan_ivf_exact [B 个 block：每条查询一个，扫完它所有探测的 list，
           用 threshold-compaction 保留 top-ck] -> residual_refine L16 -> batched_topk_final [B 个 block] -> D2H

v62（G = 2, 4, 8）：
  ……LUT 之前完全相同……
        -> scan_ivf_group [B x G 个 block：block (q, g) 负责查询 q 的候选
           [g*T/G, (g+1)*T/G)，候选即按探针顺序首尾相接的探测 list；
           在 (distance, position) 全序下保留精确的局部 top-ck]
        -> merge_group_topk [B 个 block，log2 G 轮两两 merge-path，在全局内存里 ping-pong]
        -> residual_refine L16（v61 原样）-> batched_topk_final -> D2H
  G = 1：单组直接写 top-ck 数组，不需要 merge；工作量与 v61 相同，只多了全序比较。
```

## 2. CUDA 线程与 block 映射

- **分组扫描：** `grid = (B, G)`，`block = JHQ_BLOCK` 个线程（512；openai3-3072 为 1024）。
  - *候选切分：* 在设备端读取 T = `query_total[q]`，并在设备端计算切点 `T*g/G`，不需要逐查询的 host 同步。每个线程用二分查找 `probe_offsets`，找到自己的第一个 list。各组按候选数均衡，相差不超过 1 个；一个长 list 可能被分给两个组。
  - *每个线程：* 每个 chunk 处理一个候选。距离代码与 v61 相同，读同样的 word、做同样的加法，所以距离逐位一致（max |Δprimary| = 0）。
  - *shared memory：* 放 split LUT，加一个大小为 `cap = pow2 ≥ ck + BLOCK` 的 (distance, position) 缓冲区，在严格全序 `key_less = (distance, position)` 下做 bitonic 排序。
  - 不访问 residual 码。
- **合并：** 每条查询一个 256 线程的 block。
  - 第 r 轮把 list 两两合并。每对分到 `256/pairs` 个线程，每个线程先用 merge-path 二分查找定位自己负责的输出片段，再顺序合并。
  - 每轮保留 `min(ck, a + b)`。list 放在两个全局缓冲区 `grp [B][G][ck]` 和 `mrg [B][G/2][ck]` 中，所以无论 ck 多大，shared memory 里只放 2·G 个计数。
- **Scratch：** `B·G·ck·8 + B·(G/2)·ck·8` 字节，按需增长，在 capture 之前分配。batch_cap = 1024、G=8 时：vogue α=64 为 60 MiB，arxiv α=100 / 200 为 94 / 188 MiB，openai3 α=4 为 3.8 MiB。显存占用增加同样的量（例如 arxiv α=200：2653 → 2749 MiB）。

## 3. 全局 top-ck 精确性的证明

记 ≺ 为 `key_less`。因为 position 唯一，它是一条查询所有候选上的严格全序。设 S 为该查询在 ≺ 下的全局 top-ck。

1. 取 S 中位于 g 组的 x。整条查询中排在 x 之前的候选少于 ck 个，所以 g 组中排在它之前的也少于 ck 个，x 在 g 的局部 top-ck 里。threshold-compaction 缓冲区恰好在 ≺ 下保留局部 top-ck：
   - 只接纳排在当前第 ck 个 key 之前的候选；
   - 永不溢出（`cap ≥ ck + BLOCK`）；
   - 排序是全序排序。

   因此 ∪_g local(g) ⊇ S。
2. 合并两个有序 list 并保留前 `min(ck, a+b)` 个，得到的就是并集中最小的 `min(ck, a+b)` 个。S 中出现在任一 list 里的成员都在其中，因为并集里排在它之前的少于 ck 个。对轮数归纳，最终 list 就是按顺序排列的 S。
3. 空组或短组带着自己的计数（`grp_cnt`），有效前缀之后填 (INF, −1)，所以不会被悄悄截断。全序消除了所有并列，结果不依赖分组方式、block 大小或原子追加的顺序。

v61 只按距离排序：在恰好并列于第 ck 个距离的候选里，它保留扫描先追加的那个。v62 的并列选择是确定的，v61 的不是，并且会随 batch 大小变化（见下）。

## 4. 结果

### Phase 0：v61 L16 的阶段耗时（nsys，每次搜索的 GPU µs，中位数）

| workload | b | 延迟 ms | scan | refine | probe select | top-k | 其他 + host | scan 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv α100 | 8 | 1.076 | 849 | 111 | 56 | 23 | 38 | 79% |
| arxiv α200 | 8 | 1.277 | 935 | 209 | 57 | 31 | 45 | 73% |
| openai3 α4 | 8 | 0.973 | 723 | 24 | 78 | 16 | 132 | 74% |
| vogue α64 | 8 | 1.248 | 1057 | 72 | 53 | 22 | 45 | 85% |
| arxiv α100 | 1000 | 4.500 | 2914 | 714 | 227 | 64 | 581 | 65% |
| vogue α64 | 1000 | 4.583 | 3377 | 459 | 215 | 57 | 476 | 74% |

在所有 batch 下，扫描都占延迟的 61–85%，远高于 10–15% 的停止线。b=8 时它在 170 个 SM 上只跑 8 个 block。完整表格：`raw/phase0/stages.md`。

### Phase 2：正确性（每个 G 对比 G = 0；全部查询；b = 8, 32, 256, 1000）

- **覆盖：** 64 组比较（4 个 workload × 4 个 batch × G = 1, 2, 4, 8）。
- **探针：** 每条查询都相同。
- **Primary top-ck：** NON-TIE 为 0，max |Δ primary distance| = 0。
- **最终 top-k：** 100% 的查询，id 顺序与距离位都相同。
- **Recall@10：** 相同。
- **只在并列处的 top-ck 差异：** 每组比较 0–9 条查询，共 184 条。它们在 G=1/2/4/8 之间完全相同，却随 batch 大小变化，例如 arxiv α100 在 b=8 时为 0，b=1000 时为 1。这是 v61 依赖顺序的并列选择，不是 v62 的。
- **Sanitizer：**
  - compute-sanitizer memcheck：0 错误（arxiv、vogue、openai3；64 条查询）；
  - racecheck：0 hazard；synccheck：0 错误（vogue、openai3），检查对象为 `scan_ivf_group` 和 `merge_group_topk`；
  - racecheck 在 8 条查询、b=8 下运行，因为 64 条查询 10 分钟没跑完。每个 G 仍然都会运行，而 batch 大小只改变 grid.x。

### Phase 3：端到端，相对 G = 0 的配对加速（5 轮的 median [min, max]）

| workload | b | G0 ms | G1 | G2 | G4 | G8 |
|---|---:|---:|---|---|---|---|
| arxiv α100 | 8 | 1.078 | ×0.98 | ×1.37 | ×1.83 | **×2.32** [2.316, 2.320] |
| arxiv α100 | 32 | 1.159 | ×0.98 | ×1.37 | ×1.79 | **×1.82** |
| arxiv α100 | 256 | 1.719 | ×0.96 | ×0.93 | ×0.87 | ×0.75 |
| arxiv α100 | 1000 | 4.549 | ×0.96 | ×0.86 | ×0.71 | ×0.57 |
| arxiv α200 | 8 | 1.278 | ×0.97 | ×1.30 | ×1.67 | **×2.06** [2.058, 2.063] |
| arxiv α200 | 32 | 1.355 | ×0.97 | ×1.28 | ×1.62 | **×1.63** |
| arxiv α200 | 256 | 2.075 | ×0.93 | ×0.88 | ×0.79 | ×0.69 |
| arxiv α200 | 1000 | 5.778 | ×0.96 | ×0.81 | ×0.66 | ×0.46 |
| openai3 α4 | 8 | 0.974 | ×0.99 | ×1.49 | ×1.99 | **×2.37** [2.372, 2.377] |
| openai3 α4 | 32 | 1.020 | ×0.99 | ×1.46 | **×1.89** | ×1.70 |
| openai3 α4 | 256 | 2.179 | ×0.99 | ×1.00 | ×0.96 | ×0.81 |
| openai3 α4 | 1000 | 6.909 | ×0.99 | ×0.95 | ×0.87 | ×0.71 |
| vogue α64 | 8 | 1.249 | ×0.99 | ×1.53 | ×2.15 | **×2.76** [2.756, 2.761] |
| vogue α64 | 32 | 1.356 | ×1.01 | ×1.57 | ×2.23 | **×2.30** |
| vogue α64 | 256 | 2.172 | ×0.98 | ×1.21 | **×1.23** | ×1.11 |
| vogue α64 | 1000 | 4.584 | ×0.99 | ×0.99 | ×0.89 | ×0.73 |

每一行的 Recall@10 都相同。延迟取 batch 延迟的中位数；QPS 是同步客户端背靠背提交 batch 时的吞吐。两者对所有变体采用同一口径。

### 收益背后的阶段耗时（nsys，每次搜索的 GPU µs，中位数）

| workload | b | G | scan | merge | refine | 其他 GPU | host+间隔 | kernel 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv α100 | 8 | 0 | 849 | – | 111 | 100 | 19 | 8 |
| arxiv α100 | 8 | 8 | 216 | 16.7 | 111 | 100 | 20 | 9 |
| arxiv α200 | 8 | 0 | 939 | – | 210 | 110 | 19 | 8 |
| arxiv α200 | 8 | 8 | 225 | 36.5 | 210 | 111 | 20 | 9 |
| openai3 α4 | 8 | 0 | 728 | – | 25 | 194 | 27 | 9 |
| openai3 α4 | 8 | 8 | 155 | 3.7 | 25 | 194 | 31 | 10 |
| vogue α64 | 8 | 0 | 1059 | – | 72 | 90 | 28 | 8 |
| vogue α64 | 8 | 8 | 246 | 9.1 | 72 | 90 | 22 | 9 |
| arxiv α200 | 1000 | 0 | 3530 | – | 1349 | 632 | 268 | 7 |
| arxiv α200 | 1000 | 8 | 7652 | 2624 | 1361 | 632 | 270 | 8 |

- **小 batch：** b=8 时扫描缩短 3.9–4.7 倍，merge 增加 4–37 µs，其他部分不变。端到端收益就是扫描省下的时间减去 merge。
- **kernel 数：** G > 1 时多一个 kernel。graph 的 host+间隔几乎不变（15–31 µs）。
- **完整表格：** `raw/perf/stages.md`。

## 5. 小 batch 与大 batch

- **b=8（8 个 block）：** 分组把 block 数乘以 G，这是唯一的杠杆。即使 G=8 也只有 64 个 block，所以收益随 G 一直上升。
- **b=32：** G=4（128 个 block）和 G=8（256 个 block）相差在约 2% 以内。例外是 openai3：它的 block 有 1024 个线程，更早填满 GPU，G=8 反而慢 10%。此时 GPU 已经满了。
- **b ≥ 256：** v61 本身已有 ≥256 个 block，分组只会增加工作：
  - 每个组都从无穷大阈值开始填自己的缓冲区，接纳和 bitonic 排序大约随 G 增长；
  - LUT 每条查询被搬进 shared memory G 次；
  - merge 的线程级顺序读取不合并，arxiv α200 b=1000 G=8 时达 2.6 ms。

  因此扫描时间随 G 上升（arxiv α200 b=1000：3530 → 7652 µs）。
- **vogue 在 b=256 是例外**（G4 ×1.23）。vogue 的每查询候选数波动大得多：CV 为 0.51，其他为 0.08–0.11；batch 内最大值是均值的 2.2 倍，其他为 1.3 倍。每查询一个 block 时，整个 batch 要等最大的那条查询。

## 6. 失败案例与瓶颈

- **大 batch 退化：** 除 vogue 外的所有 workload，b ≥ 256 时为 ×0.46–0.96。原因见第 5 节：重复的选择工作、重复的 LUT 搬运、不合并的 merge。
- **G=1 在大 batch 下比 v61 慢 1–4%：** 这是全序的代价，即接纳和排序中多出的 position 比较。换来的是确定性。
- **缓冲区大小规则很脆弱：** `cap = pow2 ≥ ck + BLOCK` 在每次压缩后只剩 `cap − BLOCK − ck` 个空位。arxiv ck = 500 时只剩 12 个，几乎每个 chunk 都要压缩，扫描比 ck=1000 时慢 1.36 倍。v61 与 v62 都受影响；论文的配置不在此列（在 campaign 的 Experiment 2 中发现）。
- **b=8 / G=8 下剩下的耗时：**
  - 扫描，仍占端到端的 47–56%；
  - 精排（16–35%）；
  - probe selection（56 µs；B 个 block，同样缺并行度）；
  - 最终 top-k（16–31 µs；B 个 block）。

## 7. 决定

**GO。** 正确性通过；b=8 和 b=32 下每个 workload 都超过 ×1.15（×1.63–2.76），且稳定；阶段耗时能解释收益。只适用于小 batch，大 batch 需要配一个调度策略。

## 8. 当时的方案：Budget-Aware 动态局部配额（v62 中未实现）

**方案：** 给每个组一个配额 b_i < ck，并附一个证书：
- 设 u_i 为该组最好的未返回 key，τ 为返回并集中的第 ck 个 key；
- 如果没有任何 u_i 排在 τ 之前，证书成立；
- 否则放宽配额，或回退到完整的局部 top-ck。

目标是削减 G·ck 规模的中间结果和 merge。

**后来的结果**（campaign Experiment 2）：
- 在实现之前先做了 pilot 测量。设 n_i 为第 i 组在全局 top-ck 中的成员数，证书成立当且仅当每组 n_i ≤ b_i；而第 0 组（最近的 lists）持有 46–81% 的 top-ck。
- 所以等额配额（在 v62 的切分下等同于按候选数成比例的配额）几乎从不成立。校准过的配额能 ≥99% 成立，但缩不小第 0 组，而第 0 组决定关键路径。
- 任何一次失败都会让整个 batch 停下来，等一条查询做 fallback。
- 因此放弃了配额方案。取而代之的是 v63：同一查询的各组共享一个精确界，没有证书，也没有 fallback。实测 GO：相对 v62，b=8 快 ×1.03–1.07，b=1000 最多快 ×1.56，结果完全相同。
