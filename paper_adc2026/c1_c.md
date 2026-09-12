# Section 6 压缩方案(待核对,论文未改动)

## 目标与现状

| | |
|---|---|
| 当前 | 14 页,正文(§6 + Conclusion + 致谢)结束于 **p13**,References 起于 p14 |
| ADC 限制 | **12 页不含参考文献** → 正文 13 页,**超一页** |
| Section 6 正文 | 约 **140 行** |
| 本方案能砍 | **23 行** |

**先说清楚:23 行拿不到一整页。** 缺口来源见文末。但版面是非线性的 —— p13 现在只被 Fig 5 + 8 行正文 + Conclusion + 致谢占着,不是满页,23 行有可能正好把后三者拉回 p12。**建议先做再看。**

---

## 两条标准

### 标准一 —— 你同学的要求(不可违反)

他的批评是"数字堆积、和三四章没联系"。所以每段必须同时保留:

- **测量**:数字 + 覆盖范围(几个格子 / 几个数据集)
- **机制**:为什么会这样,挂到 §3/§4 的设计上

**压缩只删复述,不删这两样。**

### 标准二 —— 审稿人视角

**每个断言只说一次:测量一次、机制一次、边界一次。说第二次的删掉。**

审稿人读一段只要三件事:测到了什么 → 为什么 → 它不能说明什么。Section 6 现在的通病是"边界"说两遍 —— 句中限定一次,段尾又总结一次。

---

## 逐处方案

### 1. RQ1「JHQ separates routing coverage…」 −4 行

**删**(三句都在复述已给出的 57%/70%):

> ~~Each retained candidate requires its residual code to be read and scored, so a large fixed survivor set remains costly even when primary screening is efficient.~~
> ~~Refinement is therefore a substantial remaining cost in these measured cells, and~~ the survivor budget controls both…

**改后**:

> JHQ separates routing coverage from second-level work: increasing $\mathrm{nprobe}$ visits more inverted lists, whereas $\alpha k$ bounds how many screened candidates receive residual scoring. Broader routing can therefore expand primary screening without requiring residual scoring of every additional candidate. At $\alpha=100$ and $\mathrm{nprobe}=32$, residual refinement accounts for approximately 57\% of measured work on Vogue and 70\% on OpenAI-3072. Reducing that set avoids residual work, but candidates discarded during screening cannot be recovered later, so the survivor budget controls both the available acceleration and the risk of losing useful neighbours.

**保留**:nprobe/αk 的分工(机制)、57%/70%(测量)、"丢掉的候选无法恢复"(机制)。

---

### 2.「Benefit and reuse.」 −4 行

**删**(payback 的定义 §4.3 已给,arxiv 前一句已说):

> ~~Calibration is paid once for a reusable budget, whereas savings accumulate over subsequent batches; short workloads may end before recovering this cost, and the slower arxiv selections never repay through throughput.~~

**改后**:

> At $S=128$, mean selected-arm QPS is 1.05--2.77$\times$ fixed-$\alpha=100$ QPS in ten of twelve configurations. The arxiv configurations instead sacrifice 12--14\% throughput for higher mean recall because matching a larger reference can require increasing the budget above the fixed comparison point; those selections never repay through throughput. Calibration costs 5.2--30.8\,ms, yielding an estimated payback of 1.4--36.4 batches of 500-query equivalents in configurations with throughput gains (Figure~\ref{fig:budget}(b)). Selected-arm QPS averages previously measured fixed-arm throughputs; taking the reciprocal of that average underestimates mean batch time when selections differ, so the aggregate payback is optimistic. Calibration includes reference search and agreement checks but excludes sample gathering, and reuse assumes a stable workload.

（"never repay" 并进 arxiv 那句,不单独成句。）

---

### 3.「Protocol and workload variation.」 −3 行

**删段尾**（下一整段就是讲这个）:

> ~~Median agreement alone does not establish quality for every calibration draw.~~

并压缩 oracle 的描述:

> ~~Across probes 128 and 512, the oracle chooses the fastest grid point within $10^{-3}$ of the best measured held-out recall. Its budgets range from 4 to 200, and the sampled rule's median agrees in ten of twelve configurations at $S=128$~~
> → The oracle chooses the fastest grid point within $10^{-3}$ of the best measured held-out recall; its budgets range from 4 to 200, and the sampled rule's median agrees in ten of twelve configurations at $S=128$

**保留**:`This variation motivates workload-specific control of $c_k=\alpha k$…` 整句 —— 这是本段唯一的机制句。

---

### 4.「Output stability and held-out quality.」 −3 行

**删**（同一条边界说了两遍）:

> ~~Output stability is therefore a label-free heuristic rather than a ground-truth recall guarantee.~~

段尾的 `zero failures … do not establish zero risk` 已经把这条边界说清楚了,而且更具体。

---

### 5.「Packed access.」 −2 行

**删**:

> ~~; differences near 1\% require stronger timing evidence~~

**理由很硬**:这句依赖的 **621-config 计时方差统计已被 t2 删掉**,现在文中没有任何数据支撑"1% 需要更强证据"这个判断。留着是悬空断言。

---

### 6.「Alternative execution choices.」 −2 行

**删**（复述前一句）:

> ~~Avoiding that small setup cost can therefore increase work repeated throughout the scan.~~

前一句已经说了"移除表格 = 把可复用的距离换成每候选重算,而建表只占 0.20–2.30%",结论不言自明。

---

### 7.「Table grouping.」 −2 行

**删**（G4/G8 那句已经说完）:

> ~~The extra lookups recur throughout the scan without reducing table footprint, showing why~~ the smallest tables need not give the fastest search.
> → 改为并入前句:`…is observed to be 3.4--38.5\% slower, so the smallest tables need not give the fastest search.`

---

### 8.「Residual codes remain the largest…」 −2 行

**删**（§2 Preliminaries 已有）:

> ~~Each vector retains eight residual bits per dimension alongside one primary bit.~~

§2 原文:"With $B_r = 8$ residual bits per dimension, the logical representation stores one primary and eight residual bits per dimension."

---

### 9.「Role of the hierarchy.」 −1 行

**删**（同义反复 —— "补充了丢失的信息""改善排序"说的是一回事）:

> ~~Residual codes supply information omitted by the compact primary representation, improving the ranking among retained candidates.~~
> → 并入:`Primary-only recall peaks at 0.5135--0.8642 across the six datasets, and residual refinement extends every range by supplying information the compact primary code omits.`

---

## 合计

| 处 | 段落 | 省 |
|---|---|---|
| 1 | RQ1 机制段 | 4 |
| 2 | Benefit and reuse | 4 |
| 3 | Protocol and workload variation | 3 |
| 4 | Output stability | 3 |
| 5 | Packed access | 2 |
| 6 | Alternative execution choices | 2 |
| 7 | Table grouping | 2 |
| 8 | Residual codes | 2 |
| 9 | Role of the hierarchy | 1 |
| | **合计** | **23** |

**零信息损失** —— 删掉的每一句,其内容都在同段别处、别节、或图表标题里存在。

---

## 如果 23 行不够,缺口从哪出(都不在 6_1.md 范围内,需你另行授权)

**A. Table 2 改单行单元格 —— 约 −13 行**
现在每格是 `\shortstack{α; recall \\ kQPS}`,12 个数据行 = 24 行表格。改单行需要腾列宽,办法是删掉 `Full pool` 列（它与 `Oracle` 在 12 行中有 8 行完全相同）。**代价:少一个 arm。**

**B. Conclusion —— 约 −14 行**
整段在复述 §6 的数字（23/24、1.05–2.77×、40/64),这些在摘要里全都有。但 Conclusion 不在 6_1.md 覆盖范围。

**C. 图高各收 —— 约 −9 行**
Fig 2 从 2.05in 收到 1.8,其余按比例。**我不建议** —— 你已因图被拉伸/放大抱怨过两次,再压会重蹈覆辙。

---

## 建议执行顺序

1. 先做上面 9 处(23 行),编译看实际页数
2. 若仍在 13 页,再从 A 或 B 二选一
3. **C 留作最后手段**

确认后我改、编译、跑校验(页数、正文/文献边界、未定义引用、float 是否劈开句子),再报结果。
