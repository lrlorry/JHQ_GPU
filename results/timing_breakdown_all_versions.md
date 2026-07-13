# HBlock 各版本耗时统计

数据来源：
- v2/v3/v4：`results/hblock_v2v3v4_bench_20260706.txt`（服务器实测，RTX 5090，vogue-768）
- v14/v15/v16：服务器 arxiv-abstracts-768 实测 QPS + 代码推算

硬件：RTX 5090 (SM120, HBM 3.6 TB/s, L2=64MB, L1=128KB/SM)
数据集参数：K1=64, K2=128, ck1=8, ck2=32, ck3=256, k=10, batch_size=1024, leaf_size=128, bpv=384 (Br=4, Kr=16)

---

## 一、v2 步骤耗时（vogue-768，932K×768，13140 leaf blocks）

稳定 batch 均值（warm-up 第1次 batch 有 cuBLAS 初始化开销，已排除）：

| 步骤                | 耗时 (ms) | 占比  |
|---------------------|-----------|-------|
| H2D                 | 0.06      | 0.1%  |
| Rotate (JL GEMM)    | 0.03      | 0.1%  |
| L1 GEMM+topk        | 0.03      | 0.1%  |
| L1 residual         | 0.01      | 0.0%  |
| L2 GEMM+topk        | 0.06      | 0.1%  |
| L2 residual         | 0.01      | 0.0%  |
| GatherLeaf          | 0.10      | 0.2%  |
| SortLeaf            | 0.01      | 0.0%  |
| FineLUT             | 0.05      | 0.1%  |
| **LeafFine**        | **42.08** | **97.3%** |
| FinalTopk           | 0.80      | 1.9%  |
| D2H                 | 0.03      | 0.1%  |
| **TOTAL**           | **43.26** | 100%  |

**Recall@10=0.806，QPS=22,980**

备注：第1个 batch 因 cuBLAS 初始化 L1 GEMM 耗时 16.8ms（后续 0.03ms）。

---

## 二、v3 步骤耗时（vogue-768，PCA 路由，11772 leaf blocks）

| 步骤                  | 耗时 (ms) | 占比  |
|-----------------------|-----------|-------|
| H2D                   | 0.06      | 0.2%  |
| L1 Proj+GEMM+topk     | 0.03      | 0.1%  |
| L1 residual           | 0.01      | 0.0%  |
| L2 Proj+GEMM+topk     | 0.06      | 0.1%  |
| L2 residual           | 0.01      | 0.0%  |
| GatherLeaf            | 0.10      | 0.2%  |
| SortLeaf              | 0.01      | 0.0%  |
| FineLUT               | 0.05      | 0.1%  |
| **LeafFine**          | **39.54** | **97.2%** |
| FinalTopk             | 0.80      | 2.0%  |
| D2H                   | 0.03      | 0.1%  |
| **TOTAL**             | **40.69** | 100%  |

**Recall@10=0.504，QPS=24,432**

---

## 三、v3 ck3 sweep（vogue-768，稳定 batch 均值）

| ck3 | LeafFine (ms) | TOTAL (ms) | Recall@10 | QPS    |
|-----|---------------|------------|-----------|--------|
| 64  | 10.96         | 11.30      | 0.555     | 86,565 |
| 128 | 20.54         | 20.97      | 0.533     | 47,132 |
| 256 | 39.54         | 40.69      | 0.504     | 24,440 |

规律：LeafFine ∝ ck3（线性），其余步骤 ~0.35ms 固定开销。

---

## 四、v4 步骤耗时（vogue-768，S_B判别路由，11906 leaf blocks）

| 步骤                  | 耗时 (ms) | 占比  |
|-----------------------|-----------|-------|
| H2D                   | 0.06      | 0.1%  |
| L1 Proj+GEMM+topk     | 0.03      | 0.1%  |
| L2 Proj+GEMM+topk     | 0.06      | 0.1%  |
| GatherLeaf            | 0.10      | 0.2%  |
| FineLUT               | 0.05      | 0.1%  |
| **LeafFine**          | **39.32** | **97.2%** |
| FinalTopk             | 0.80      | 2.0%  |
| **TOTAL**             | **40.49** | 100%  |

**Recall@10=0.504，QPS=24,558**

---

## 五、v14 推算（arxiv-abstracts-768，2.253M×768，20335 leaf blocks）

v14 timing 格式：Route / GPUSort / Kernel+Merge+DMA / Extract（4段）

实测 QPS=136,000 → batch latency ≈ 1024/136000 × 1000 ≈ **7.5 ms**

| 阶段                 | 估算耗时 (ms) | 说明                                    |
|----------------------|---------------|-----------------------------------------|
| Route                | ~0.5          | JL GEMM + L1/L2 GEMM + LUT 构建，已 warm |
| GPUSort              | ~0.1          | CUB RadixSort，n_pairs=262144           |
| **Kernel+Merge+DMA** | **~6.5**      | leaf PQ scan + top-k merge + D2H        |
| Extract              | ~0.1          | CPU partial_sort 1024×10                |
| **TOTAL**            | **~7.5**      |                                         |

关键改进 vs v2：
- 转置 codes 布局 `[blk][bpv][leaf_size]` → coalesced 读
- global LUT（50MB）常驻 L2（L2=64MB）→ LUT 访存命中率高
- arxiv 比 vogue leaf 数多（20335 vs 13140），但 n_pairs 相同（1024×256=262144）

**Recall@10≈0.257（arxiv），QPS=136,000**

---

## 六、v15 汇总（arxiv，key-by-c1 路由，compact LUT）

设计：64 次顺序 c1 迭代，每次用 compact LUT 减少 LUT 大小

**Recall@10=0.257，QPS=77,000**（比 v14 慢 43%，compact LUT 反而引入串行化开销）

---

## 七、v16 设计目标（arxiv，multi-query leaf kernel）

| 指标                   | v14        | v16 设计目标         |
|------------------------|------------|----------------------|
| leaf codes HBM 读      | ~13× 重复  | 每块只读 1 次（L1 命中）|
| GPU blocks per batch   | 262,144    | 20,335               |
| codes 访问方式          | 直接 HBM   | `__ldg` L1 只读缓存  |
| shared memory          | 无（flat） | static 544B only     |
| 预期 leaf kernel 耗时  | ~6.5 ms    | ~0.5 ms（13× 减少）  |
| 预期 QPS               | 136K       | >500K（待实测）      |

---

## 八、各阶段固定开销汇总（所有版本共有，warm batch）

| 步骤               | 典型耗时 (ms) |
|--------------------|---------------|
| H2D (queries)      | 0.06          |
| JL Rotate GEMM     | 0.03          |
| L1/L2 GEMM+topk    | 0.09          |
| GatherLeaf+Sort    | 0.11          |
| FineLUT            | 0.05          |
| FinalTopk (merge)  | 0.80          |
| D2H                | 0.03          |
| **非 LeafFine 合计** | **~1.17**   |

**结论：LeafFine（leaf PQ scan kernel）是唯一瓶颈，占 97%+ 时间。优化方向只有一个：减少 leaf kernel 的 HBM 带宽消耗。**
