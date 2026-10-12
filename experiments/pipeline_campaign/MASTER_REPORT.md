# JHQ-GPU GPU-Native Pipeline 研究总报告

2026-10-12/13 · RTX 5090（AutoDL，170 SM）· driver 595.71.05 · CUDA 13.0 · 分支 `exp/pipeline-campaign`

**实验条件**
- **构建：** Release；除非另行注明，`CMAKE_CUDA_ARCHITECTURES=80`，即以往所有 JHQ-GPU 数字所用的构建。
- **索引：** 论文那份，来自 arms6 trained-state cache，参数取自 `run_budget_arms6.sh`。
- **对比方式：** 同一配置的所有变体在同一进程里，搜索同一份索引和同一批查询。
- **计时：** 口径与论文相同，即 `search()` 外层的 wall clock，含 H2D、graph launch、D2H 和同步。每个配置跑 5 轮，变体顺序轮换；加速比在轮内配对，报告 median [min, max]。
- **GPU 时间：** 无人值守运行时段为 2026-10-12 16:44–22:02 UTC，在 6 小时预算内。

## 结论

- **最好的实际 pipeline 是 v65**（`jhq_v65_schedule`，设置 `JHQ_PRIMARY_GROUPS=-1`、`JHQ_REFINE_LANES=16`），由四部分组成：
  - v61 的 L16 精排；
  - v62 的 list 并行 primary screening；
  - v63 的精确共享界；
  - 按 batch 在 host 端选择 G 的调度。
- **结果与 v61 L16 完全一致：** non-tie primary 差异为 0；234,320 次逐查询比较里，最终 id 集合全部相同；Recall@10 相同。
- **端到端加速（相对 v61 L16）：** 6 个数据集 × np 128/512，论文 sampled-rule α，13 行几何平均（b=1000 为 11 行）。

  | b | v65 | 各行范围 | 与逐行 oracle（最佳固定 G）的差距 |
  |---:|---:|---|---|
  | 8 | **×2.73** | ×2.01–3.78 | ≤3.4% |
  | 32 | **×2.36** | ×1.67–3.13 | ≤0.4% |
  | 256 | ×1.06 | ×0.997–1.44 | 最多 22%（bge-m3、stella） |
  | 1000 | ×1.015 | ×0.99–1.10 | ≤1% |

- **Recall–QPS frontier**（α ×½ 与 ×2）：b=8 在每个 α 上都快 ×1.81–3.18；b=1000 不变，只有 vogue 快 6–14%。
- **成功的部分有三项：**
  - 小 batch 下的并行 primary screening（v62）；
  - 跨组精确剪枝（v63）。它让 list 并行路径在中等 batch 下也划算，并消除了 v62 在大 batch 下的大部分损失；
  - 两项规则的调度（v65）。任何一行都没有比 v61 慢超过 0.3%。
- **失败或跳过的部分：**
  - 基于配额的预算方案：pilot 判 NO-GO；
  - 用 Πᵀc 路由（v64）：NO-GO；
  - 新数据布局与选择性 kernel 融合：两者的前提条件都被测量否定，SKIPPED。
- **研究判断：** 加速真实、可复现，并且保持正确性。但它来自 cuVS 这类 IVF 搜索里常见的 GPU 调度手段：list 并行扫描加合并、阈值共享、按占用率切分工作。本工作把它们适配到了 JHQ 在这里唯一与众不同的性质上，即一个大而精确的全局 top-αk。这是系统和工程层面的贡献，不是新算法（见“新颖性审计”）。

## 实验决策表

| 版本 | 假设 | 正确性 | 最佳端到端加速（配对） | 退化 | 决定 |
|---|---|---|---|---|---|
| v61 L16（基线） | — | 见 v61 报告 | 本轮各行 L16 相对 L0：×1.01–2.37 | — | 基线 |
| sm_120 原生编译（Exp 0） | 原生编译比 compute_80 PTX JIT 快 | recall 差 ≤2e-4；NON-TIE 0 | L0 ×1.10–1.31（768 维） | b ≤ 32 时 v61 pipeline 慢 5–27%；openai3 ×0.92–0.95 | 单独报告，不采用 |
| v62 list 分组（Exp 1） | 把一条查询的 primary screening 拆给 G 个 block，能降低小 batch 延迟 | 64/64 比较 NON-TIE 0，最终结果逐位一致，sanitizer 无错 | ×2.76（vogue b=8）；b=8 为 ×2.06–2.76，b=32 为 ×1.63–2.30 | b ≥ 256 时 ×0.46–0.96 | **GO** |
| v63 动态局部配额（Exp 2 pilot） | 每组配额 b_i < ck 加证书，能削减中间结果 | 离线：当且仅当 n_i ≤ b_i 时证书成立 | 上限：b=8 时 ≤ 约 4%，只来自 merge | 整个 batch 因 fallback 停顿 | **NO-GO**（未实现） |
| v63 共享界（Exp 2） | 各组共享精确界：任何一组的第 ck 个 key 都是全局第 ck 个的上界 | 48/48 与 v62 完全相同（top-ck 与最终结果），NON-TIE 0，sanitizer 无错 | 相对 v62：b=8 为 ×1.03–1.07；×1.56（arxiv α200 b=1000 G8） | 相对 v62 无；openai3 在 b ≤ 32 为 ±0.5% | **GO** |
| v64 Πᵀc 路由（Exp 3） | 让质心打分改用 Πᵀc，路由就能与旋转并行 | 探针相同的查询 NON-TIE 0；最终 top-k 相同；但 768 维有 1–8% 查询、openai3 有 40% 查询的探针集合不同 | b ≤ 32 为 ×1.005–1.017；openai3 b=256 为 ×1.045 | 并发 kernel 变慢 8.5%（资源竞争） | **NO-GO** |
| v65 调度（Exp 4） | 一条固定规则能按 batch 选出接近最佳固定 G 的 G | 与 v61 L16 完全一致（见上） | b = 8/32/256/1000 几何平均 ×2.73 / ×2.36 / ×1.06 / ×1.015 | 最差一行 ×0.997；b=256 的 regret 最高 1.22 | **GO** |
| v66 数据布局（Exp 5） | primary 扫描受数据布局限制 | — | — | — | **SKIPPED**（证据见 Exp 5） |
| v67 kernel 融合（Exp 6） | merge/refine/top-k 之间的同步或数据传输有明显代价 | — | — | — | **SKIPPED**（证据见 Exp 6） |
| v65 = 最终集成（Exp 7） | 成功的模块能组合起来 | 同 v65 | 同 v65 | 同 v65 | **最终 pipeline**（没有其他可集成的模块，因此不另建 v68） |

## Experiment 0：基线审计与 workload 剖析

**v61 L16 的阶段耗时**（nsys，`experiments/pipeline_v62/raw/phase0/stages.md`）
- 在所有 batch 和所有 workload 下，primary 扫描都占延迟的 61–85%。
- b=8 时扫描是每条查询一个 block，即 170 个 SM 上只有 8 个 block，耗时 723–1057 µs，整次搜索 0.97–1.25 ms。
- probe selection（52–78 µs）和最终 top-k（16–31 µs）同样是每查询一个 block。
- 上限：扫描若为 0，端到端可达 ×2.6–6.5；拆成 8 份，可达 ×2.1–3.9。
- 结论：primary screening 是瓶颈，小 batch 的问题是并行度不足。

**原生 sm_120**（`exp0/EXP0_TABLES.md`）
- 可以编译和运行：CUDA 13.0 支持 compute_120，`cuobjdump` 确认是原生 SASS。
- 但并非处处更快：
  - v61 L0：768 维数据集快 ×1.10–1.31，openai3 为 ×0.95；
  - v61 L16：在 ±8% 以内，所以原生编译下 L16/L0 变小（arxiv α200：×1.74，而 compute_80 下是 ×2.39）；
  - 小 batch 的 v61 pipeline 在原生编译下更慢（arxiv b=8：1.364 ms 对 1.078 ms）；
  - v62 的相对加速保持甚至变大（b=8 为 ×2.33–2.78）。
- 因此本轮所有数字都用 compute_80，sm_120 只在单独的表里报告，从不混合加速比。

**单批延迟与稳态吞吐**
- harness 同时报告两者：
  - 单批延迟：每次调用的中位数；
  - 吞吐：同步客户端背靠背提交 batch 时，查询数除以总时间。
- 两者对所有变体采用同一口径。所有小 batch 结论在两种口径下都成立。

## Experiment 1：v62 并行 primary screening

完整报告见 `experiments/pipeline_v62/PIPELINE_V62_REPORT.md`，包含执行图、线程映射、精确性证明、各 G 的阶段和端到端表、大小 batch 的对比、失败案例，以及引出 Experiment 2 的方案。

- **改变了什么：** b=8 时扫描缩短 3.9–4.7×（例如 vogue 从 1059 降到 246 µs），merge 增加 4–37 µs。
- **b ≥ 256 为什么变慢：**
  - 每个组都从无穷大阈值开始重新装满缓冲区；
  - LUT 被搬进 shared memory G 次；
  - merge 的访存不合并（arxiv α200 b=1000 G=8 时达 2.6 ms）。
- **v61 的并列选择依赖顺序：** 在恰好并列的候选中，它保留扫描先追加的那个，其 tie-only 差异随 batch 大小变化。v62 及之后按 (distance, position) 全序，结果是确定的。

## Experiment 2：Budget-Aware 全局候选选择

**Pilot**（`exp2/raw/quota_study.md`、`exp2/raw/nsys/stages.md`；pilot 之前没有写任何实现）
- **判定条件：** 设 n_i 为第 i 组在全局 top-ck 中的成员数。配额 b_i 能证明返回的并集包含全局 top-ck，当且仅当每组都满足 n_i ≤ b_i。证明写在 `exp2_quota_study.py` 里，不需要在 GPU 上运行这个策略。
- **集中度：** 探针按距离从近到远排序，所以第 0 组（最近的 lists）占了全局 top-ck 的大部分：G=8 时为 46–81%。
- **等额配额**（在 v62 的均衡切分下等同于按候选数成比例的配额）几乎从不成立：G=8 时，slack 2 只有 0–2.1% 的查询成立；slack 4（即 v62 ck 的一半）只有 5–67%。
- **按组校准的配额**用一半查询校准、在另一半上测试：slack 1.5 时 99.2–100% 成立，返回量只有 v62 G·ck 的 30–35%。
- **但收益在小 batch 下出不来：**
  - 第 0 组的配额缩不下来；而 B·G 个 block 填不满 GPU 时，kernel 时间由最慢的 block 决定；
  - 用 v62 在 ck = b 下运行，相当于任何配额方案在证书零成本时的第一遍。它表明能省的只有 merge：≤27 µs，arxiv α200 b=8 约 4%，其他 < 2%。
- **每次证书失败都会让整个 batch 停下来**，等一条查询做 fallback，概率为 1 − (1−p)^B。
- 结论：**配额方案 NO-GO**，没有做自适应调参。

**留下来的优化：v63 共享界**（`exp2/v63_raw/summary.md`）
- **机制：**
  - 每个组用 `atomicMin` 把自己的第 ck 个 key 发布到该查询的一个 64-bit 字里。这个字是 distance 位左移 32 位再拼上 position，对非负距离保持顺序；
  - 组只接纳不晚于该界的候选；
  - 扫描结束时，组只写出不晚于该界的那段前缀。
- **对任何执行时序都精确：** 一个已发布的第 ck 个 key 之前（含）已有 ck 个候选，所以它之后的候选不可能进入全局 top-ck；而且这个界只会下降。没有证书，也没有 fallback。
- **实测精确：** 48/48 个配置与 v62 的 top-ck 和最终结果完全相同；memcheck 0 错误，racecheck 0 hazard，synccheck 0 错误。
- **v63 相对同 G 的 v62：**

  | b | 加速 |
  |---:|---|
  | 8 | ×1.03–1.07（openai3 的 ck 只有 40，×1.00） |
  | 32 | ×1.04–1.14 |
  | 256 | ×1.06–1.43 |
  | 1000 | ×1.02–1.56 |

  例如 arxiv α100 b=256 G4：扫描 1432 → 986 µs；arxiv α200 b=1000 G8：merge 2629 → 471 µs。
- **primary 距离计算量不变：** 每个候选仍然要算。省下的是接纳、排序和 merge 流量，所以 GPU 越满，效果越明显。
- **消融 Fixed Local ck（v62）vs Dynamic Local Budget：**
  - 配额形式由上面的上限否定；精确共享界的形式保留。
  - 至于“固定两阶段配额是否比多轮扩充更快”：这里两者都不值得做，因为共享界根本不需要多轮。

## Experiment 3：v64 用 Πᵀc 路由

详见 `exp3/raw/summary.md`。
- **恒等式：** ⟨Πq, c⟩ = ⟨q, Πᵀc⟩ 对任意 Π 成立，而 probe selection 只读这些点积和 ‖c‖²。因此在实数运算下，精确性不需要正交性。
- **实测正交性：** max |ΠᵀΠ − I| 在 768 维上为 6.0e-7，在 openai3-3072 上为 6.5e-5。
- **并发方式：** 第二条 stream，配独立的 cuBLAS handle，在 CUDA graph 内 fork/join。
- **正确性：** 探针相同的查询 NON-TIE 都是 0；最终 top-k 和 recall 处处相同。
- **探针漂移：** 浮点差异（点积相对误差 ≤8e-5）让 768 维上 1–8% 的查询、openai3 上 40% 的查询改变了探针集合。这在近似并列处偏离了论文语义，本次测量中无害。
- **模式 1 vs 0：** 单纯换坐标没有代价（×1.000）。
- **模式 2 vs 0：**
  - 并行省下的前端时间：768 维 b=8 为 5 µs，openai3 b=8 为 16 µs，openai3 b=1000 为 171 µs；
  - 端到端：b ≤ 32 为 ×1.003–1.017，最好也只有 ×1.045（openai3 b=256）；
  - 并发 kernel 自身变慢 8.5%（openai3 b=1000：kernel 时间总和 1403 → 1522 µs）。
- **结论：NO-GO。** 收益太小，探针漂移是真实存在的。
- 这个恒等式是教科书内容，本代码库在 assignment 中已经在用 Πᵀc（`ensure_assign_centroids`）。

## Experiment 4：v65 自适应执行调度

- **为什么要做：** v62/v63 与 v61 在不同场景各有优势。b ≤ 32 时 G=8 最好；b ≥ 256 时 v61 的扫描最好；而 vogue 直到 b=1000 仍然受益于 G > 1。
- **vogue 例外的原因：** 它的每查询候选数 CV 为 0.51，batch 内最大值是均值的 2.2 倍；其他数据集分别是 0.08–0.11 和 1.3。每查询一个 block 时，整个 batch 要等最大的那条查询。
- **规则：** 两个常数，只在 pipeline_v62 的 4 个 workload 上拟合，在任何其他配置运行之前就已固定。
  - *fill*：满足 B·G·BLOCK ≤ 0.6 ×（SM 数 × 每 SM 最大线程数）的最大 G；
  - *skew*：每查询候选数 CV ≥ 0.3 时为 4，否则为 1；
  - G = max(fill, skew)。G = 1 时运行 v61 的扫描。
- **CV 的测量：** 每个索引一次，用 512 个 base 向量作伪查询，从不使用评测查询。它与真实查询一致（vogue 0.495 对 0.509）。
- **没有逐次搜索的 host–device 同步：** 调度开销已经包含在测得的端到端时间里。
- **结果**（`exp47/BENCH_TABLES.md`；每个 b 有训练 4 行、held-out 9 行）：

  | b | v65 几何平均（训练 / held-out） | oracle | 最差 regret |
  |---:|---|---|---|
  | 8 | ×2.32 / ×2.93 | ×2.30 / ×2.94 | 1.034 |
  | 32 | ×1.96 / ×2.56 | ×1.96 / ×2.56 | 1.004 |
  | 256 | ×1.10 / ×1.04 | ×1.12 / ×1.15 | **1.22** |
  | 1000 | ×1.02 / ×1.01 | ×1.02 / ×1.02 | 1.010 |

- **失手的地方：** b=256 的 bge-m3 和 stella（nlist 32768）。G=8 在那里能达到 ×1.11–1.22，但它们的 CV 很低（0.08–0.14），规则选了 G0。可见每查询总候选数不能说明全部，下一个可以尝试的特征是 list 大小统计。调度在任何一行都不低于 v61 的 ×0.997。
- **oracle**（逐行最佳固定 G，从测试查询上读出）只作为上限报告。
- **对照固定 G=8：** 它单独在 b=8/32 能达到 ×2.73 / ×2.34，但在 b=1000 只有 ×0.89。调度保留了小 batch 的收益，又避免了大 batch 的损失。

## Experiment 5：Stage-Aware 数据布局 —— SKIPPED

前提条件，即剖析证明数据布局是瓶颈，并不成立。
- **大 batch 接近带宽上限：** b=1000 时，v61 的扫描在“无复用”假设下读取 primary 码的需求带宽为 1.34–1.64 TB/s，而 RTX 5090 的峰值是 1.79 TB/s：
  - vogue：48,000 候选 × 96 B，3.44 ms；
  - arxiv：47,143 × 96 B，2.92 ms；
  - openai3：18,316 × 384 B，4.30 ms。

  word layout 已经是合并访存。要再少读字节就得改码本身，而这是被禁止的。
- **小 batch 受并行度限制，**这一点由 v62/v63 解决。
- **没有计数器级证明：** Nsight Compute 计数器不可用（ERR_NVGPUCTRPERM），这里只有需求估算。
- 已有的布局工作（word layout、v50 `JHQ_REGROUP` / `JHQ_FUSE_REFINE`）没有被重新包装。

## Experiment 6：选择性 kernel 融合 —— SKIPPED

前提条件，即 merge/refine/top-k 边界上有可测的同步或数据传输代价，并不成立。来自 v62 的 trace：
- **graph 内空闲：** 每次搜索 GPU 在 captured graph 里总共只空闲 0.4–3.4 µs。
- **边界间隔：** scan→merge、merge→refine、refine→top-k 为 0.06–0.13 µs，已达 nsys 的分辨率。
- **中间数据：** composite distance 每条查询只有 ck 个 float（b=8、α=200 时 64 KB）。
- **融合去不掉的部分：** b=8 时 top-k kernel 的 16–31 µs 是它自己每查询一个 block 的选择工作，融合后依然存在。真正待解决的是让最终选择并行化，而不是融合。

## Experiment 7：最终集成与评估

- **最终版本：** v65 = v61 L16 + v62 分组 + v63 共享界 + v65 调度。不包含 v64（NO-GO）。没有其他可集成的模块，因此不另建 v68。
- **消融：** 相对 v61 L16 配对，全部行的几何平均，b = 8 / 32 / 256 / 1000。

  | 变体 | b=8 | b=32 | b=256 | b=1000 |
  |---|---:|---:|---:|---:|
  | v61 L0 | ×0.89 | ×0.90 | ×0.82 | ×0.78 |
  | v63 G2 | ×1.32 | ×1.32 | ×1.00 | ×0.96 |
  | v63 G4 | ×2.01 | ×1.97 | ×1.09 | ×0.95 |
  | v63 G8 | ×2.73 | ×2.34 | ×1.10 | ×0.89 |
  | **v65** | **×2.73** | **×2.36** | **×1.06** | **×1.015** |

  不带共享界的 v62 见 Experiment 1 的表。
- **v65 在 b=8 的各行：** vogue/512 ×3.78，openai3-3072/512 ×3.57，stella ×3.10–3.12，openai3-1536/512 ×2.97，vogue/128 ×2.83，arxiv/512 ×2.67，bge-m3 ×2.54–2.55，openai3-1536/128 ×2.37，openai3-3072/128 ×2.24，arxiv/128 ×2.01–2.27。
- **Recall–QPS frontier**（np 128，α ×½ 与 ×2，b = 8 和 1000；`exp47/raw/frontier`）：同等 recall 下，b=8 快 ×1.81–3.18；b=1000 为 ×0.99–1.01，只有 vogue 为 ×1.06–1.14。
- **校准：** 没有在评测查询上做任何校准。调度的统计量来自 base 向量；α 用论文的 sampled-rule 值，没有重新校准。

## 五个问题的回答

**Q1：v61 之后，Primary Screening 的瓶颈能否用更高的 GPU 并行度解决？**
小 batch 和中等 batch 可以。把每条查询的候选拆给 G 个 block 并精确合并，b=8 时 screening 缩短 4–5 倍，整次搜索快 ×2.0–3.8，recall 不变。大 batch 不行：GPU 已经满了，拆分只会增加选择和 merge 的工作。那里的办法是不拆，这正是调度在做的事。

**Q2：JHQ 的全局 Top-αk 是否允许更有效的 Budget-Aware 候选选择？**
配额形式不行。全局 top-αk 集中在最近的那几个 list 上，所以关键路径组的配额缩不下来，而证书失败会让整个 batch 停顿。但它允许精确共享界：任何一组的第 αk 个 key 都是整条查询的上界。共享界能剪掉远处组的接纳和 merge 输入，而且不需要 fallback；相对独立分组，b=8 快 ×1.03–1.07，b=1000 最多快 ×1.56。它省的是选择工作和数据流量，不是距离计算。

**Q3：新的执行 pipeline 是否利用了 JHQ 的表示结构，而不只是复制 RaBitQ/cuVS 的任务并行？**
- **大体上没有。** (query, list-group) block 加 merge 的分解方式，与 cuVS IVF-PQ、GPU IVF-RaBitQ 的任务并行相同。
- **JHQ 特有的部分：** 需要一个精确且很大的全局 top-αk（ck 最高 2000），而且它要送入 residual 精排。这让精确、确定的 merge 和共享界都值得做；共享界的效果来自 αk 与探针顺序的相互作用。
- **没有利用的部分：** 分层码本身（primary/residual）。精排沿用 v61，没有改动。

**Q4：哪些优化真正提高了端到端 Recall–QPS，哪些只有 kernel 级加速？**
- **端到端有效的：** v61 L16 精排、v62/v63 的小 batch screening（×2–3.8），以及 v65 调度。它们在每个测过的 α 上都推动了小 batch 的 frontier。
- **只有 kernel 或前端收益的：**
  - v64 的并行：最多省 171 µs，端到端最多 ×1.045；
  - v63 的大 batch 扫描节省（最多 −45%）：这是相对 v62 而言，相对 v61 大多只多出 ≤10%；
  - sm_120 原生编译下更快的 L0 kernel：没有带到小 batch pipeline 上。

**Q5：哪些可以合理写进论文？**
1. 一项测量刻画：加上 warp 精排之后，JHQ-GPU 的小 batch 延迟主要是每查询一个 block 的 primary 扫描（61–85%），b=8 时 170 个 SM 上只有 8 个 block。
2. 面向大全局 top-αk 的精确 list 并行 primary screening：全序 merge 及其证明、确定的并列处理，以及在 6 个数据集上 ×2.0–3.8 的小 batch 加速，recall 不变。
3. 跨组精确共享界，以及它为什么适用于 αk 量级的选择（探针顺序的集中性）；配额方案作为负面结果。
4. 两项规则的 batch 调度：保留收益而不在大 batch 损失，并报告 held-out regret。

这四项都应该表述为针对 JHQ 的 GPU 系统适配与测量，而不是算法上的首创。

## 新颖性审计

| 组成 | 已有工作（不完全列举） | 本工作新增的部分 |
|---|---|---|
| list 并行扫描 + merge（v62） | cuVS IVF-PQ / IVF-Flat 按 (query, probe) 并行，再跨 probe 选择；FAISS GPU 类似；GPU IVF-RaBitQ 按 probe 融合扫描与过滤 | 针对最高 2000 的 ck 的精确、确定的 merge，输出送入 JHQ 的 residual 精排；在 JHQ 的索引上实测 |
| 共享界（v63） | 并行 top-k 之间共享阈值是标准做法（Fagin 式 threshold algorithm、共享第 k 距离的 GPU kNN） | 应用到 JHQ αk 规模的精确选择上；实测它与探针顺序集中性的联系（第 0 组持有 46–81% 的 top-ck） |
| 调度（v65） | 按占用率切分工作、split-K 启发式（例如 cuBLAS） | 两特征规则（fill、每查询工作量偏斜），并做 held-out 评估 |
| Πᵀc 路由（v64） | 线性代数；本代码库的 assignment 已在使用 | 无，并且 NO-GO |
| 配额（Exp 2 pilot） | 带证书的预算化 top-k | 一个负面结果及其机制（关键路径组、batch 级 fallback） |

**以下内容没有实验支持，因此不主张：**
- 关于其他 GPU、关于整条 pipeline 的原生 sm_120、关于多 stream 并发 batch 的任何结论；
- 调度规则在这 12 个配置之外的泛化性；
- 在同等 recall 下相对 cuVS 或 GPU IVF-RaBitQ 的任何优势（本轮没有运行它们）。

## 复现与说明

- **提交**（分支 `exp/pipeline-campaign`）：

  | 内容 | commit |
  |---|---|
  | Phase 0 | `d4cd5d2`、`b3b3432`、`7723059` |
  | v62 | `862543a` |
  | v62 运行脚本修正 | `e0d0518` |
  | Exp 0 脚本 | `a8edba7` |
  | Exp 2 pilot | `41f6522` |
  | v64 | `eac8ff8` |
  | v63 | `e2ed15b` |
  | v65 + benchmark | `d425ab8` |
  | 结果与报告 | `8933c2a` 及之后 |

- **脚本：** `experiments/pipeline_v62/run_phase0.sh`、`run_v62.sh`，以及 `experiments/pipeline_campaign/{exp0_sm120,exp2_v63_pilot,exp2_v63,exp3_v64,exp47_bench}.sh`。每个脚本都记录构建参数（`raw/**/build_env.txt`），构建失败即中止。
- **原始数据：** 放在各报告旁的 `raw/`，nsys trace 为 gzip 压缩的 CSV。`.nsys-rep` 和配额 dump 留在服务器上（`/root/autodl-tmp/{pipeline_v62_nsys,campaign_*}`）。
- **运行位置：** 所有运行都在一个 detached worktree（`/root/autodl-tmp/jhq_pipeline_wt`）里。服务器主目录 `/root/JHQ_GPU` 有另一个会话未提交的修改（CMakeLists.txt、jhq_v21_cascade/jhq_gpu_index.cu），没有动。
- **与计划的偏差：**
  - racecheck 在 8 条查询、b=8 下运行，因为 64 条查询 10 分钟都没跑完；
  - 第一次 Phase 3 的 nsys 导出被 Phase 0 留下的旧 SQLite 文件挡住，之后从同一份报告重新导出；
  - 编写 v63 和 v65 期间 GPU 空闲了约一小时；
  - stella 只有 999 条查询，所以没有 b=1000 的行。
- **未修改：** v61、ADC 论文、已审计的历史实验，以及 trained-state cache。
