# 成本模型：下界筛选 pipeline 与 v65 的对比

**本文件中没有任何一个数字来自新实现的 pipeline。** 按 brief 的规定，只有在存活候选数具有竞争力时才做 GPU microbenchmark（Pilot E）。实测存活数约等于全部 probe 候选（ck 的 24–1774 倍），条件不满足，所以 Pilot E 的新 kernel 实测为 **SKIPPED**。下表用的是 pipeline campaign 已有的实测值，加上本次 pilot 实测的存活数。

各项的来源标记：

| 标记 | 含义 |
|---|---|
| **[M]** | 实测：pipeline campaign 的 nsys GPU 时间（`../pipeline_campaign/exp2/v63_raw/perf`、`../pipeline_campaign/exp2/raw/nsys`） |
| **[D]** | 推导：实测速率 × 本次实测的存活数 |
| **[T]** | 理论：一个写明的假设 |

由 `cost_model.py` 生成，结果在 `cost_model.json`，图为 `figures/fig7_cost_model.png`。

## 模型

```
T_new = T_front + T_bound_LUT + T_bound_scan + T_seed + T_survivor_refine
        + T_final_selection + T_coordination + T_host
```

| 项 | 取值 | 来源 |
|---|---|---|
| T_front | 旋转 + 质心 GEMM + probe 选择，与 v65 相同 | [M] |
| T_bound_LUT | 与 primary LUT 形状相同（每子空间 16+16 项），取 primary LUT 实测时间 | [M] |
| T_bound_scan | 选择开销最小的扫描：v62 在最小 ck 下的扫描（arxiv α=12.5、vogue α=8、openai3 α=1；b=8 用 G8，b=1000 用 G0）。这是乐观的代理，因为存活者几乎是全部候选，实际还要写出全部候选 | [M]（代理） |
| T_seed | 每条查询 64 个种子 × 本 batch 的精排速率 | [D] |
| T_survivor_refine | 精排数 × batch=1000 时实测的每候选精排时间（实测中最高效的速率：768 维 0.72 ns、3072 维 3.84 ns）。带宽下限：每候选读 d 字节残差码 ÷ 1.79 TB/s | [D] / [T] |
| T_final_selection | 乐观：0；按实测速率：top-k kernel 时间 ×（精排数 / ck）。该 kernel 每查询一个 block，耗时随输入线性增长 | [T] / [D] |
| T_coordination | 多两个 graph 节点，加一次存活者压缩：5 µs | [T] |
| T_host | 与 v65 相同 | [M] |

## 结果（np = 128，中位数查询）

| workload | b | ck | 需精排数（中位数） | v65 实测 µs | 新方法：乐观 µs | 新方法：实测速率 µs | 相对 v65（乐观 … 实测速率） |
|---|---:|---:|---:|---:|---:|---:|---|
| arxiv-768 α100 | 8 | 1000 | 47,205 | 480 | 517 | 1,586 | ×0.93 … ×0.30 |
| vogue-768 α64 | 8 | 640 | 40,568 | 422 | 511 | 1,874 | ×0.83 … ×0.22 |
| openai3-3072 α4 | 8 | 40 | 18,100 | 433 | 971 | 8,298 | ×0.45 … ×0.05 |
| arxiv-768 α100 | 1000 | 1000 | 47,205 | 4,556 | 37,280 | 40,356 | ×0.12 … ×0.11 |
| vogue-768 α64 | 1000 | 640 | 40,568 | 4,611 | 33,084 | 36,672 | ×0.14 … ×0.13 |
| openai3-3072 α4 | 1000 | 40 | 18,100 | 6,982 | 76,916 | 120,892 | ×0.09 … ×0.06 |

![成本模型](figures/fig7_cost_model.png)

## 结论

- **最乐观的情况下也慢：** 最终选择零成本、扫描不含选择开销，新方法在 b=8 时仍比 v65 慢 7–55%，b=1000 时慢 7–11 倍。
- **瓶颈不在下界表或扫描，而在需要精排的候选数：**
  - 下界几乎剪不掉候选，残差精排从 ck 个变成全部 probe 候选；
  - 每个候选的残差码有 d 字节，是 primary 码（M 字节）的 8 倍；
  - b=1000 时，仅读残差码的带宽下限就是 arxiv 20 ms、openai3 31 ms，而 v65 整次搜索只要 4.6–7.0 ms。
- **要追平 v65：** 精排数必须回到 ck 量级，也就是剪掉 97–99.9% 的候选。按 `tightness.json`，这需要 LB/D_H ≥ 约 0.75–0.83，而实测最大只有 0.30。
