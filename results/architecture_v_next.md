# HBlock 下一代架构设计

## 核心认识

### 瓶颈本质
```
LeafFine = HBM 带宽瓶颈（读 codes），占 97% 时间
路由 GEMM = 计算瓶颈，GPU TFLOPS 富余，几乎免费

→ 多花计算（路由）换省带宽（LeafFine）是最优交换
```

### 当前问题
```
v14：路由太草率（6D JL）→ 必须扫 256 leaf 补召回 → HBM 打满
理想：路由足够精准      → 只扫 10-20 leaf            → HBM 几乎闲置
```

---

## 三个叠加优化

### 1. 多级高质量路由（减少每个 query 的 leaf 扫描量）

**目标**：用 3 级路由把候选从 20335 块精准缩小到 10-20 块

```
L1（PCA 64D，K1=512，select ck1=4）：20335 → ~160 块
L2（PCA 64D，K2=256，select ck2=4）：160   → ~16 块
L3（PCA 64D，K3=64， select ck3=2）：16    → ~8  块
LeafFine：只扫 8 块 × 128 vectors = 1024 vectors/query
```

**路由中心训练（解析解，无 k-means）**：
- PCA SVD top-64 特征向量（LAPACK，<1s）
- 每级递归二分，cell 均值为中心
- 路由 GEMM：B×64×K，最大 1024×64×512=33M FLOPs，≈0.1ms

**LeafFine 扫描量**：
```
当前：256 blocks × 128 = 32,768 vectors/query
改后：8   blocks × 128 = 1,024  vectors/query  → 32× 减少
```

---

### 2. Batch 内 Query 排序（增大每个 leaf block 的 query 复用）

**原理**：空间相近的 query 路由到相同 leaf block，聚在一起处理可最大化 codes 的 L1 复用。

**实现**：batch 内按 L1 路由结果（c1）做 RadixSort，一次 GPU sort，~0.1ms。

```
随机 batch：  每个 leaf block 被 ~13 个 query 共享（v16 现状）
排序后 batch：同一 L1 cell 的 query 聚集，每个 leaf block 被 ~50-200 个 query 共享
```

这与 v16 的 CSR 多 query 内核天然兼容，segment 更长，codes 在 L1 驻留更久。

---

### 3. v16 CSR 多 Query Kernel（一次 HBM 读服务多个 query）

已在 v16 实现，与上述两项叠加后效果放大：

```
codes 从 HBM 读一次 → L1 缓存 → 服务 N 个 query

当前 v16（随机 batch）：N ≈ 13
加 query 排序后：       N ≈ 50-200（取决于 batch 中 query 的空间聚集度）
```

---

## 三项叠加后的带宽估算

| 项目 | 当前 v14 | 目标架构 | 减少倍数 |
|------|---------|---------|---------|
| leaf blocks/query | 256 | 8 | 32× |
| queries/leaf block | 1（各自独立）| 100（聚集+CSR）| 100× |
| codes HBM 流量/batch | 12.88 GB | ~0.04 GB | ~320× |
| LeafFine 耗时 | 6.5 ms | <0.1 ms | >60× |
| 路由耗时 | 0.1 ms | ~0.5 ms | -5× |
| **总耗时** | **7.5 ms** | **~1 ms** | **~7×** |

---

## 完整流程对比

### 当前 v14
```
H2D → JL(6D) → L1 GEMM → L2 GEMM → 256 leaf → LeafFine(6.5ms) → Merge → D2H
```

### 目标架构
```
H2D
→ PCA(64D) → L1 GEMM(0.1ms) → L2 GEMM(0.1ms) → L3 GEMM(0.1ms)
→ query RadixSort by c1（0.1ms）
→ 8 leaf blocks/query
→ v16 CSR kernel：一次读 codes，服务 ~100 queries（<0.1ms）
→ Merge → D2H
总：~1ms
```

---

## 各优化项的 ROI 排序

| 优先级 | 改动 | 收益 | 难度 | 独立性 |
|--------|------|------|------|--------|
| 1 | **PCA 路由替换 JL** | recall 0.257→0.7+ | 低 | 独立 |
| 2 | **三级路由** | leaf 256→8，LeafFine 32× | 中 | 依赖1 |
| 3 | **Query 排序** | codes 复用 13→100×，免费加速 | 低 | 依赖 v16 |
| 4 | **LUT INT8** | LUT L2 流量 4×，recall≈不变 | 低 | 独立 |
| 5 | **Re-rank** | recall →0.9+ | 中 | 独立 |

---

## 为什么这是正确的架构方向

GPU 资源不对等：
```
计算（TFLOPS）：远超需求，大量闲置
带宽（HBM TB/s）：紧张，是真正瓶颈
```

当前 v14 把大量时间花在带宽受限的 LeafFine 上，而路由（计算密集）几乎不花时间。

目标架构把时间从"带宽受限操作"转移到"计算密集操作"，充分利用 GPU 的计算优势，同时把 HBM 带宽需求压到极低。这是在 GPU 上做 ANN 搜索的正确设计哲学。
