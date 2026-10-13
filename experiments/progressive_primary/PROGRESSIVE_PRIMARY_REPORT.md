# Query-Adaptive Progressive Primary Screening：Pilot 报告

分支 `exp/progressive-primary-pilot`。RTX 5090（AutoDL），共用约 1 分钟 GPU 时间，只用于两次只读导出，没有运行任何新的搜索 kernel。

## 结论：NO-GO

在 arxiv-768、vogue-768、openai3-3072 三个代表性工作负载上：

1. **离线筛选能做到 Recall 非劣，但代价太大。**
   - 两个 768 维数据集需要读一半的 word（p = 0.5）。
   - 同时全局 Top-L 必须保留 L = 8·ck，即 5120–8000 个候选，占全部 IVF 候选的 11–17%。
   - 理想有效字节因此能减少 41–44%。
   - 但 Stage 2 对这些候选做散乱补读，按 32 B sector 计有 3.9–4.6 倍放大。计入后，真实读取只减少 14–24%。
2. **成本模型显示，768 维上加速不了，而且结论不依赖不确定的参数。**
   - 在所有 test 非劣的工作点上，即使假设 Top-L 选择完全免费、散乱读取跑满 DRAM 峰值，arxiv 的端到端节省也 ≤ 8.7%。
   - vogue 只有这种上限情景达到 10–14%；乐观情景 ≤ 5.4%，中性情景为负。
3. **query-aware 和 IVF 条件策略没有稳定的增量。**
   - 唯一预测收益超过 10% 的是 openai3-3072。
   - 但在预设的工作点上，固定存储顺序前缀（A0，即 v21 前缀 cascade 的机制）与所有新策略打平，新颖性边际为 0。
   - 按 brief 第 11 节，这属于“复现已有固定 Prefix 筛选”，判 NO-GO。

**局部发现（不构成 GO）：** 在 openai3 上，若按真实字节选工作点（看到第一轮结果后补充的第二规则）：
- A1（query-aware 顺序）在 p = 0.25 时只需 L = 8·ck，A0 需要 16·ck。
- 推导的中性情景端到端节省，A1 为 +17%（b=8）/ +24%（b=1000），A0 为 +12% / +14%。

这是单数据集、事后规则、纯推导的结果。并且 v65 的 paper-faithful 构建明文禁止前缀 cascade（见 1.3 节），这类变体不能作为 JHQ-GPU 报告。第 7 节给出若要继续时最小的验证方式。

## 证据等级

| 标记 | 含义 |
|---|---|
| **[源码/数学]** | 从源码或码本逐位核验的事实，或代数恒等式 |
| **[离线实测]** | 在真实导出的主码、候选和 ground truth 上，用 FP64 离线计算的结果（CPU） |
| **[推导]** | `COST_MODEL.md` 的成本模型：实测阶段时间加明确写出的假设 |
| **[GPU 实测]** | 本轮没有。门槛未通过，没有 CUDA 原型 |

---

## 1. 源码审计

### 1.1 表示与恒等式 [源码/数学]

由 `demo_jhq_v66_export_primary` 从 v65 的索引（arms6 cache）只读导出并逐项检查：

- **量化幅值：** 768 维为 ±0.0243385509，3072 维为 ±0.0121692754。
  - 两个幅值逐位对称：`levels[0] == -levels[1]`。
  - M·K·Ds 个码本项全部等于其 digit 对应的 level，0 个不符。
  - 因此 ŷ_j = c·s_j，并且同一配置下**所有维度共用同一个 c**（c_j ≡ c）。
- **码字布局：**
  - word w = 维度 32w…32w+31 = 子空间 4w…4w+3；
  - word 的第 i 个字节是子空间 4w+i，字节的最高位是该子空间的第一维；
  - bit 1 表示 +c。
- **Primary 距离：** D_P = ‖q'‖² + d·c² − 2·Σ_w S_w，其中 S_w = Σ_{j∈w} c·q'_j·s_j。
  - 每个候选的 S_w 都从码字节重建。
  - 与导出的 FP64 D_prim 相比，最大相对偏差 5.8e-15 / 4.2e-15 / 7.9e-15（arxiv / vogue / openai3）。
  - 这同时验证了 word→维度映射、位序和符号约定。
- **基线复现：**
  - 离线用 FP64 复现的 full(ck) + refine，在 100% / 99.9% / 99.8% 的查询上与 v65 自己返回的 top-10 完全一致。
  - v65 的 top-ck 位置在 99.5% / 100% / 99.1% 的查询上与 FP64 top-ck 一致，其余差异来自 fp32 并列。
- **16+16 分解 LUT：**
  - 每个字节拆成高 4 位（子空间前 4 维）和低 4 位（后 4 维），各查一次 16 项表；
  - 每个 word 共 8 次 shared memory 查表。

### 1.2 GPU 物理布局 [源码]

- **读取方式：** primary 码按 `[W][N]` 的 uint32 word 行存储（`JHQ_WORD_LAYOUT=1`）。一个 warp 的 32 个 lane 读 32 个连续候选的同一个 word，正好是一次 128 B 事务。
- **Stage 1 仍然合并：**
  - 每个 (查询, list) 只读一部分 word 行，每一行在该 list 内是连续段，访存仍然合并。
  - 离线逐 sector 计数：Stage 1 的 sector 比例 = p ×（1.019–1.048），多出的部分只是 list 边界。
- **Stage 2 不合并：**
  - 被选中的候选在每个未读 word 行上各自落在一个 32 B sector 里，只有相邻候选同时被选中时才共享。
  - 这是 word 行布局固有的代价，见 3.4 节。
- **按查询 / 按 list 定 word 顺序的额外成本：**
  - A1 只需要 q'（建 LUT 时已在片上），每个查询对 W 个能量排序一次。
  - A2/A3 需要读每个 probed list 的 μ（int8），每个查询 nprobe·d 字节：768 维为 98 KB，占 B_old 的 2.2–2.5%；3072 维为 393 KB，占 5.4%。
  - 再加每个 (查询, list) 一个长度 |J| 的 word 索引表。

### 1.3 已有类似实现：`JHQ_PREFIX_NUM`、`JHQ_PREFIX_KEEP`、`JHQ_EXIT_BOUND`、`JHQ_WORD_LAYOUT` [源码]

| 开关 | 做了什么 | 现状 |
|---|---|---|
| `JHQ_PREFIX_NUM/DEN`（v21 cascade） | 第一遍只读前 M·NUM/DEN 个子空间（**固定存储顺序前缀**，默认 1/4），算部分和 | 只存在于旧的 `scan_ivf_coalesced_kernel`（`jhq_v65_schedule/search.cu:1098`）。<br>v65 的构建参数 `V61_DEFS` 固定 `JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=1`，即关闭。<br>paper-faithful 构建的 `jhq_faithful_guard` **禁止**运行时设置 `JHQ_PFX_NUM/DEN`，原文：“prefix cascade: prunes before the full primary distance exists (Algorithm 1 line 3) … do not report the result as JHQ-GPU”。 |
| `JHQ_PREFIX_KEEP` | 每个线程按部分和保留 KEEP 个候选，第二遍从部分和继续补完 | 同上。精确的全局 top-αk（`exact_topck()`）在 paper-faithful 构建里写死。 |
| `JHQ_EXIT_BOUND`（v48/v49） | 精确早停：用后缀最小表项作下界，按 warp 投票退出 | 默认 0；v62 起的分组扫描“by design”不实现。<br>NEGATIVES 记录：v48 损失 1–3%，v49 ≈ 0。 |
| `JHQ_WORD_LAYOUT`（v52） | `[M/4][N]` 的 32 位 word 布局，warp 内 128 B 合并 | v65 默认开启。本轮 Stage 1 的合并读取和 Stage 2 的不合并都由它决定。 |

**历史测量：** `results/v39_cascade_price/README.md` 在 v39 上给过 cascade 的价格：
- M/2 前缀在 vogue / bge-m3 / stella 上 QPS +110% / +38% / +23%，recall −0.0002 / −0.0003 / 0；
- M/4 在 vogue 上 recall −0.0086；
- 前缀越短，第二遍散乱补读越多，反而更慢；
- 只做每线程保留（PM = M）本身就贡献 +62% / +16% / +11%。

**这些数字不能拿来比 v65：**
- 当时的精确基线比 v65 慢得多：v39 vogue 约 3.7 万 QPS，v65 在 b=1000 约 22 万；
- 只靠保留就能拿到的那部分收益，此后已被精确选择路径吸收；
- 3072 维从未测过。

**本轮与旧 cascade 的区别：**
- (a) 全局 Top-L，不再是每线程 KEEP；
- (b) query-aware（A1）或 IVF 条件（A2）的 word 顺序；
- (c) 用 list 符号均值补全未读部分（A0c / A1c / A3）。

A0 加全局 Top-L 就是旧 cascade 的机制，只换了选择方式，这正是新颖性检验的对照。

**两条相关的旧结论：**
- v54（`results/front6/V54_SIGN_IP.md`）验证过同一个恒等式。不用表、直接算符号内积反而慢 30–52%，因为分解 LUT 已经是最优的预计算。所以本轮所有方案都保留 LUT。
- `results/front6/QDUP.md` 测过按 list 跨查询复用的上限：+4–26%，其中 nprobe=128 时最多 +11.6%；重复读取大部分已由 L2 承担。我此前口头建议过“list 级跨查询复用”，这其实已被测过，在此更正。

---

## 2. 实验设置 [离线实测]

- **查询与候选：**
  - 三个工作负载各 1000 条查询，与 Local-First 一轮相同（seed 20261013）。
  - 按 400 / 200 / 400 划分 calibration / validation / test。
  - 候选集合、probe 顺序、FP64 D_prim 和 D_H 来自 `/root/autodl-tmp/lf_pilot`；逐维主码、list offset 和 q' 由本轮新的只读导出补齐（`/root/autodl-tmp/pp_pilot`）。
  - **所有策略使用完全相同的候选集合和主码。**
- **策略：**

  | 策略 | 读取顺序 | 未读部分 |
  |---|---|---|
  | A0 | 存储顺序前缀 | 按 0 计 |
  | A1 | 按 I_w = Σ(c·q'_j)² 降序 | 按 0 计 |
  | A2 | 每个 probed list 按 Σ(c·q'_j)²(1 − λ²μ_ℓj²) 降序 | 按 0 计 |
  | A3 | A2 的 word | 补 Σ_{j∉J} c·q'_j·λ·μ_ℓj |
  | A0c、A1c | A0 / A1 的 word | 同 A3 的均值补全，用于把“顺序”与“均值估计”的贡献分开 |

  - μ_ℓj = E[s_j | ℓ] 按 list 全部成员统计后量化为 int8，与实际存储一致。
  - λ ∈ {0, .25, .5, .75, 1} 只在 calibration 上，按 (策略, p, L) 以 full-primary top-ck 覆盖率选取。
- **网格：** p ∈ {0.125, 0.25, 0.5, 0.75, 1}，按整 word 读；L/ck ∈ {1, 2, 4, 8, 16, 32}，不超过候选数。
- **Stage 2：**
  - Top-L 补读剩余 word，用精确 D_prim 取 top-T（T ∈ {ck, T*}）；
  - 再用原有残差 D_H 精排，取 top-10。并列都按 position 打破。
- **基线：**
  - full(ck) 即 v65；
  - full(T*) 即 α 优化后的 v65，T* 按 Local-First 一轮的规则在 validation 上重新选出：arxiv 750、vogue 640、openai3 30，与上一轮相同。
  - 主要比较都在 T_final = T* 下对 full(T*) 进行。
- **预设判定规则（写在 `analyze.py` docstring，第一次运行前固定）：**
  - 非劣：test 配对 bootstrap 95% CI 下端 ≥ −0.001（0.1 pp）。CI 包含 0 不等于相等。
  - 工作点：validation 平均差 ≥ −0.0005 的 (p, L) 中，理想字节最小者，结果在 test 上报告。
  - 严重长尾：test 上每查询差值的 P1 < −0.1 或最差值 < −0.2。
  - 新颖性：A0c/A1/A1c/A2/A3 中，test 非劣的最佳工作点要比 A0 的工作点至少少 0.05·B_old 理想字节。
  - **第二规则（看到第一轮结果后补充，单独标明）：** 同样的 validation 选择，但按真实字节（sector 级 + 元数据 + scratch），并做同样的新颖性检验。
- **字节核算（每查询，primary 码）：**
  - B_old = n·W·4；理想 B_new = n·|J|·4 + L·(W − |J|)·4。
  - Sector 级：
    - Stage 1 = 每个 probed list 每个已读 word 行连续段跨过的 32 B sector；
    - Stage 2 = 每个未读 word 行上被选中候选所在的不同 sector，按 word 行起点 (w·N) mod 8 对齐。
    - 在存储上相邻、又都被 probe 的两个 list，边界 sector 被计两次（B_old 也一样）。
  - 另计 μ 元数据（nprobe·d B）和 Top-L scratch（写入加读回 (pos, score)，16·L B）。
- **正确性检查：** `selftest.py` 在合成索引上用暴力实现逐项对照：Stage 2 sector 集合计数、A0 选择与 recall、p = 1 时与基线逐查询完全一致。服务器上两次运行都通过。

---

## 3. 结果

### 3.1 部分主码保留了多少重要候选 [离线实测]

test 集，覆盖率 = Stage 1 Top-L 中包含的比例。三个指标依次为：
- full-primary top-ck 的覆盖率；
- full-IVF top-10（全部候选按 D_H）的覆盖率；
- v65 最终 top-10 在 Stage 1 被丢掉的比例。

**A0（固定前缀）：**

| | p | L = ck | L = 8·ck | L = 32·ck |
|---|---|---|---|---|
| arxiv | 0.125 | 0.21 / 0.57 / 43% | 0.69 / 0.90 / 9.6% | 0.988 / 0.998 / 0.20% |
| | 0.25 | 0.34 / 0.80 / 20% | 0.86 / 0.986 / 1.35% | 0.999 / 1.000 / 0.03% |
| | 0.5 | 0.57 / 0.96 / 3.9% | 0.988 / 0.999 / 0.03% | 1 / 1 / 0 |
| vogue | 0.125 | 0.24 / 0.67 / 33% | 0.71 / 0.93 / 6.9% | 0.971 / 0.996 / 0.37% |
| | 0.25 | 0.38 / 0.87 / 13% | 0.88 / 0.992 / 0.85% | 0.997 / 1.000 / 0 |
| | 0.5 | 0.59 / 0.98 / 2.3% | 0.989 / 1.000 / 0.05% | 1 / 1 / 0 |
| openai3 | 0.125 | 0.54 / 0.83 / 17% | 0.93 / 0.985 / 1.5% | 0.993 / 1.000 / 0.03% |
| | 0.25 | 0.68 / 0.95 / 5.5% | 0.992 / 1.000 / 0.05% | 1 / 1 / 0 |
| | 0.5 | 0.81 / 0.99 / 0.9% | 1 / 1 / 0 | 1 / 1 / 0 |

- **12.5–25% 的 bit 可以保留重要候选，但要配很大的 L。** full-IVF top-10 覆盖率远高于 top-ck 覆盖率，说明 Stage 1 丢掉的多数是本来进不了最终结果的候选。
- **决定能否非劣的是长尾查询**，见 `WORD_IMPORTANCE_ANALYSIS.csv` 中的 L_need：
  - arxiv、p = 0.5 时，保住 v65 最终 top-10 所需的 L：中位数只有 0.4·ck，P90 是 2.1·ck，最坏 10·ck。
  - 少数查询决定了 L 必须取到 8·ck。

### 3.2 query-aware 顺序与 IVF 条件统计 [离线实测]

**query 能量在 word 间只是轻度集中：**
- 按 I_w 选出 25% 的 word，只拿到 33% 的 Σ(c·q'_j)²（均匀为 25%）；12.5% 的 word 拿到 18%。
- 这正是 JL 旋转的设计效果：它把能量摊平到各个维度。

**Stage 1 估计与完整分数的 Pearson 相关（p = 0.25，test）：**

| | A0 | A1 | A2 (λ=1) | A1c | A3 (λ=1) |
|---|---|---|---|---|---|
| arxiv | 0.724 | 0.717 | 0.601 | 0.760 | 0.796 |
| vogue | 0.784 | 0.813 | 0.743 | 0.846 | 0.826 |
| openai3 | 0.875 | 0.908 | 0.894 | 0.919（λ=0.25） | 0.862 |

**list 符号均值的作用：**
- arxiv 的符号整体严重偏置：|全局均值| = 0.63，E[μ²] = 0.62（vogue 0.32 / 0.40，openai3 0.18 / 0.25）。所以 A1 的“能量”高估了 arxiv 上那些几乎恒定的维度，A1 没有比 A0 更好。
- 均值补全（A3）对 arxiv 帮助最大：p = 0.25、L = ck 时覆盖率从 0.34 升到 0.46。
- 但补全量是每个 list 的常数，**只能调整 list 之间的相对位置，不能改变 list 内的排序**。

**λ 的选择随数据集变化：** calibration 选出的 λ 在 arxiv 为 1（A3），vogue 为 0.25，openai3 为 0。openai3 上 λ = 1 反而有害：p = 0.25 时所需 L 的中位数从 0.6·ck 涨到 5.8·ck。

**在非劣点上，新策略最多省一档 L（2 倍）：**

| 数据集 | 情形 |
|---|---|
| vogue p = 0.5 | A1/A2/A3 用 8·ck，A0 需要 16·ck |
| openai3 p = 0.25 | A1 用 8·ck，A0 需要 16·ck |
| openai3 p = 0.5 | A1 用 2·ck，A0 需要 4·ck |
| arxiv p = 0.75 | 均值补全用 2·ck，A0 需要 4·ck |
| arxiv p = 0.5 | 全部打平 |

### 3.3 维持 v65 Recall 的最小 p 与 L [离线实测]

条件：validation 平均差 ≥ −0.0005，且 test 非劣，T_final = T*。

| | A0 | 新策略中最好的 |
|---|---|---|
| arxiv | p = 0.5, L = 8·ck（8000 / 47.1K 候选） | 同 A0；A1c 可到 p = 0.25 但需 L = 16·ck |
| vogue | p = 0.5, L = 16·ck | p = 0.5, L = 8·ck（5120 / 48.0K，A1、A2） |
| openai3 | p = 0.125, L = 32·ck（1280 / 18.3K） | p = 0.125, L = 32·ck；p = 0.25 时 L = 8·ck（320，A1） |

p = 0.125 在 arxiv 上只有加均值补全、L = 32·ck 才非劣，此时真实字节反而超过 B_old（1.12）；在 vogue 上任何策略都不非劣。

### 3.4 理想字节与真实字节 [离线实测]

validation 选出的非劣工作点（T_final = T*）：

| | 工作点 | ΔRecall@10 [95% CI] | 理想字节 | Stage 1 sector | Stage 2 sector | 元数据 + scratch | 真实合计 | LUT 查表节省 |
|---|---|---|---|---|---|---|---|---|
| arxiv | A0 p=0.5 L=8ck | −0.0002 [−0.0007, 0] | 0.586 | 0.500 | 0.328 | 0.028 | **0.856** | 42% |
| arxiv | A1c p=0.25 L=16ck | −0.0002 [−0.0007, 0] | 0.508 | 0.250 | 0.633 | 0.078 | **0.961** | 50% |
| vogue | A1 p=0.5 L=8ck | 0 [0, 0] | 0.565 | 0.500 | 0.240 | 0.021 | **0.762** | 45% |
| openai3 | A0 p=0.125 L=32ck | 0 [0, 0] | 0.187 | 0.125 | 0.275 | 0.003 | **0.403** | 81% |
| openai3 | A1 p=0.25 L=8ck（第二规则） | 0 [0, 0] | 0.263 | 0.250 | 0.067 | 0.001 | **0.318** | 74% |

- 各列都是相对 B_old 的比例，只有 LUT 列是查表次数节省，= (1−p)(1−L/n)。
- **Stage 2 的读放大：** 相对每个 word 4 B 的理想值，sector 级实际是 3.9×（arxiv A0）、4.6×（vogue A1）、5.4×（openai3 A1）。
- **arxiv 上 query-aware 反而更差：** 理想字节最省的 A1c（0.508），真实字节最多（0.961），因为它要 L = 16·ck。

### 3.5 GPU 成本模型 [推导]

见 `COST_MODEL.md` 和 `figures/fig3_gpu_cost_breakdown.png`。

预测的端到端**时间节省**，相对 v65@T*，在 validation 选出的非劣工作点上，取 b=8 与 b=1000 中较好的一个：

| | 策略 / 工作点 | U 上限（Top-L 免费 + 峰值带宽） | O 乐观 | C 中性 |
|---|---|---|---|---|
| arxiv | A0 / A2 p=0.5 L=8ck | 7.7% | −25% | −72% |
| arxiv | A1c p=0.25 L=16ck | 8.7% | −34% | −155% |
| vogue | A1 / A2 p=0.5 L=8ck | 13.9% / 12.4%（b=1000） | 5.4% / 4.0% | −71% / −74% |
| openai3 | 全部，p=0.125 L=32ck（预设规则） | 33% | 33% | 6% |
| openai3 | A0 p=0.25 L=16ck（第二规则） | 36% | 36% | 14% |
| openai3 | A1 p=0.25 L=8ck（第二规则） | 40% | 40% | 24% |

**关于 C 情景的大负数：** 768 维的 Top-L 远超已测选择曲线的范围（ck ≤ 2000），线性外推得到的大负数只说明选择成本会主导，不当作点预测。真正稳健的结论来自 U 上限：arxiv 在任何非劣点都不到 10%，vogue 只有上限情景超过 10%。

### 3.6 Phase C 门槛

由 `summary.json` 从结果文件自动判定。

| | 1 非劣 | 2 理想字节 ≥30% | 3 无严重长尾 | 4 可行 | 5 成本 ≥10%（O / C） | 新颖性（预设 / 第二规则） |
|---|---|---|---|---|---|---|
| arxiv（A0、A1c、A2 的预设工作点） | ✓ | ✓ | ✓ | ✓ | ✗ / ✗ | ✓（只在理想字节上，真实字节更差）/ ✗ |
| vogue（A2；A1 为第二规则） | ✓ | ✓ | ✓ | ✓ | ✗ / ✗ | ✗ / ✗ |
| openai3（预设规则，所有策略） | ✓ | ✓ | ✓ | ✓ | ✓ / ✗ | **✗**（A0 打平）/ ✓（边际 0.058） |
| openai3（第二规则，A0 与 A1） | ✓ | ✓ | ✓ | ✓ | ✓ / ✓ | — |

第 4 项“可行”的定义：元数据加 scratch ≤ 0.10·B_old，且真实字节 < B_old。

---

## 4. 回答 brief 的 8 个问题

**1. 只读 12.5–50% 的 Primary Bits 能否保留重要候选？** [离线实测]
- **p = 0.5 可以：** L = 8·ck 时，full-primary top-ck 覆盖 0.99，v65 最终 top-10 只丢 0–0.05%。
- **p = 0.25 只在 3072 维可以：** L = 8·ck；768 维要 L = 16–32·ck。
- **p = 0.125 只在 3072 维、L = 32·ck 时可以。**
- 能保留，但代价是让 L 远大于 ck。L 由约 10% 的长尾查询决定，不是由典型查询决定。

**2. Query-aware word order 是否明显优于固定前缀？** 否。
- [离线实测] 在 768 维上：
  - arxiv 上没有收益，Pearson r 0.717 对 0.724；
  - vogue 上 Pearson r 提高 0.03，非劣点的 L 少一档。
- [离线实测] openai3 上 r 提高 0.03，p = 0.25 时 L 从 16·ck 降到 8·ck。
- [源码/数学] 原因是 JL 旋转把 query 能量摊平到各 word，可利用的不均匀性很小：25% 的 word 只含 33% 的能量。

**3. IVF 条件符号统计是否带来稳定增量？** 否。 [离线实测]
- 只换顺序（A2）不比 A1 好，arxiv 上更差（r 0.60）。
- 均值补全在小 L 时对 arxiv 帮助很大，但：
  - 到非劣点最多省一档 L，还要付 2–5% 的元数据；
  - 最优 λ 在三个数据集上分别是 1 / 0.25 / 0，对 openai3 有害；
  - list 常数补全不能改变 list 内的排序。

**4. 维持 v65 Recall 的最小 p 和 L？** [离线实测] 见 3.3 节。
- arxiv：p = 0.5，L = 8·ck；
- vogue：p = 0.5，L = 8·ck（A1/A2），A0 需要 16·ck；
- openai3：p = 0.125，L = 32·ck，或 p = 0.25，L = 8·ck（A1）/ 16·ck（A0）。

**5. 实际省的是 LUT、主码读取，还是两者？** 两者都省，但比例不同。
- [离线实测] Stage 1 的合并读取和查表都降到 p。
- [离线实测] 计入 Stage 2 后的净节省：
  - 查表：arxiv 42%，vogue 45%，openai3 74–81%；
  - 真实读取：arxiv 14%，vogue 21–24%，openai3 60–68%。
- [推导] 哪一项决定 v65 扫描时间，取决于 batch：
  - b=8 时受并行度限制；
  - b=1000 时，按“无复用”估算的需求带宽接近 DRAM 峰值；但 QDUP 显示重复读取大多由 L2 承担，所以是读取还是查表决定扫描时间，没有计数器证据（Nsight Compute 权限不可用）。

**6. 新增的 Global Top-L 和 Stage 2 不规则读取是否吃掉收益？** 在 768 维上是。
- [离线实测] Stage 2 的 sector 流量是 B_old 的 24–33%，而 Stage 1 只省 50%。
- [推导] L = 8·ck 的全局选择远超 v65 现有的选择规模：
  - v65 在 b=1000 时 ck 从 1000 到 2000，扫描已增加约 595 µs [M]；
  - b=8 时分组 merge 从 13 µs 增至 39 µs [M]。
- [推导] 即使假设选择免费，arxiv 也到不了 10%。
- [离线实测] openai3 的 L 只占 1.7–7% 的候选，Stage 2 流量是 B_old 的 7–28%，没有吃掉收益。

**7. 相同 Recall 下能否真实超过 α 优化后的 v65？** 没有 GPU 实测。
- [推导] arxiv：不能，所有情景下的非劣点都为负或 < 10%。
- [推导] vogue：不能，乐观情景 ≤ 5.4%，中性情景为负。
- [推导] openai3：模型预测可以（中性 +12–24%，乐观 +33–40%），但同样的收益由固定前缀 A0 取得，不归功于 query-adaptive 机制。

**8. 失败原因是什么？** 按影响大小排序：
1. **筛选质量不足以让 L 接近 ck** [离线实测]
   - 部分分数与完整分数的相关只有 0.72–0.91（p = 0.25）。
   - 对 top-ck 这种需要整体排序正确的目标，长尾查询迫使 L ≫ ck。
   - 这在 768 维上是根本限制：每个候选的码只有 96 B，而候选多达 4.7–4.8 万，L 一大，选择和补读就很贵。
2. **GPU 执行成本过高** [推导]：全局 Top-L 把 v65 已经付出的选择代价放大 8–16 倍。
3. **访存不合并** [离线实测]：Stage 2 在 word 行布局下每个有用的 4 B 占一个 32 B sector，放大 3.9–5.4 倍。
4. **统计预测不准** [离线实测]：μ 补全只在小 L 和偏置严重的数据（arxiv）上有用，而且依赖数据集，不是主要原因。

---

## 5. 局部发现：openai3-3072

3072 维的经济账与 768 维相反：
- 每个候选的码有 384 B，ck = 40，候选约 1.8 万；
- 即使 L = 8·ck 也只占候选的 1.7%；
- Stage 1 只读 25%，Stage 2 的补读很少（B_old 的 6.7%）。

在第二规则下，A0 和 A1 都通过全部五项门槛；A1 的 L 只有 A0 的一半，中性情景多 5–10 个百分点。

**它不构成 GO：**
1. 按预设规则（理想字节工作点）的新颖性边际为 0：所有策略都选 p = 0.125、L = 32·ck，结果打平。这正是 brief 第 11 节定义的“复现固定前缀筛选”。
2. 第二规则是看到第一轮结果后补充的，事后规则不能用来翻转判定。
3. 只有一个数据集，收益全是推导；选择成本的外推跨度达 3–5 个倍数（实测只到 ck = 40）。
4. 这是近似筛选。v65 的 paper-faithful 构建明文禁止前缀 cascade，必须以命名变体报告，不能叫 JHQ-GPU。

---

## 6. 与此前 pilot 的一致性

本轮与 Residual-Aware Bound、Bound-Aware Quantization、Local-First 三轮指向同一结论：
- 在当前架构下，对**全部** IVF 候选做“先粗后精”的两段处理，第二段的散乱访问和选择代价都会吃掉第一段的节省。
- 768 维上 primary 码很短（96 B）而候选很多，几乎没有可省的空间。

上一轮的两个附带线索仍然成立，且与本轮无关，都没有在 GPU 上实测：
- α 偏保守约 25%（T* = 0.75·ck）；
- refine kernel 每个查询只用一个 block。

## 7. 下一步建议

- **不建议**继续投入 query-adaptive progressive screening。
- 如果导师认为近似筛选在论文范围内，并想确认 openai3 的局部发现，最小的验证是：
  - 只做 openai3（b = 8 / 1000，A0 与 A1，p = 0.25，L = 8 / 16·ck），GPU 不超过 30 分钟；
  - 分别实测成本模型里最不确定的三项：Stage 1 在 p = 0.25 时相对 v65 扫描的时间、全局 Top-320/640 选择、Stage 2 散乱补读。
  - 这三项能否兑现 +12–24% 只有 GPU 实测能回答。

## 8. 复现

- **服务器（key 登录，脚本里没有任何凭据）：**
  ```sh
  /root/autodl-tmp/launch_b.sh exp/progressive-primary-pilot experiments/progressive_primary/run_pilot.sh progressive.log
  ```
  - `run_pilot.sh` 依次运行：构建 `demo_jhq_v66_export_primary`（构建失败即中止）→ 三个工作负载各一次只读导出 → `selftest.py` → `analyze.py`。
  - 输出到 `results/progressive_primary`。
  - 大文件留在服务器 `/root/autodl-tmp/pp_pilot/<tag>/`：primary.u8 和 qrot.f64；每查询数组为 perquery.npz。
- **本地（拿到结果目录后）：**
  ```sh
  python3 cost_model.py .
  python3 figures.py .
  python3 make_summary.py .
  ```
- **两次运行：**
  - 第一次运行（`d0fa5d3`）与第二次运行（`c95815f`）的导出和逐配置结果完全相同。
  - 第二次只增加了第二规则、修正了新颖性检验（第一次误把 test 不非劣的挑战者算进去），并保存逐查询数组。
  - 两次的日志都在 `raw/`。

## 9. 文件

| 文件 | 内容 |
|---|---|
| `RECALL_VS_BYTES.csv` | 每个 (策略, p, L, T_final) 在 test 上的 recall、配对 CI、长尾分位数、理想与 sector 级字节，以及 validation 上的差值与字节 |
| `COVERAGE_VS_READ_RATIO.csv` | Stage 1 Top-L 的两种覆盖率、遗漏率与读取比例 |
| `WORD_IMPORTANCE_ANALYSIS.csv` | 各策略、λ、p 的 query 能量占比、Pearson r、所需 L 的分布 |
| `PER_QUERY_OPERATING_POINTS.csv.gz` | 工作点的逐查询 recall，可重算 CI |
| `analysis_summary.json` | 检查项、μ 统计、两种规则的工作点、新颖性 |
| `COST_MODEL.md`、`cost_model.json`、`COST_MODEL_TABLE.csv` | 成本模型 |
| `summary.json` | 门槛与判定 |
| `figures/fig1_recall_vs_bytes.png` | Recall–字节，理想与真实各一行 |
| `figures/fig2_topL_coverage.png` | Top-L 覆盖率 |
| `figures/fig3_gpu_cost_breakdown.png` | GPU 成本分解 |
| `figures/fig4_word_importance.png` | word 重要性 |
| `analyze.py`、`selftest.py`、`run_pilot.sh`、`cost_model.py`、`figures.py`、`make_summary.py`、`../../examples/demo_jhq_v66_export_primary.cu` | 脚本与导出程序 |
| `raw/` | 构建、导出、selftest、analyze、成本模型的原始日志，两次运行各一份 |
