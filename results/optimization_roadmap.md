# HBlock 优化路线图

## 背景：LeafFine 带宽分析

### v14 LeafFine HBM 流量
```
codes:  262144 pairs × 49152 B = 12.88 GB  (HBM, 主瓶颈)
LUT:    1024 queries × 48KB unique, 256× 复用 → 48MB unique，L2 全程命中
→ HBM 带宽需求: 12.88 GB / 3.6 TB/s = 3.58 ms（观测 6.5 ms，调度开销约 2ms）
```

### v16 LeafFine HBM 流量（__ldg + L1 codes 复用）
```
codes HBM:  20335 × 49152 B = 950 MB → 0.26 ms
LUT L2:     20335 × 13 queries × 48KB = 12.3 GB → 12.3G / 8 TB/s ≈ 1.5 ms
→ 瓶颈从 codes-HBM 转移到 LUT-L2
预期 LeafFine: ~1.8 ms，3-4× vs v14
```

---

## 方案一：v16 + LUT INT8 量化（优先级最高）

### 原理
```
LUT 当前: float32 = 768 × 16 × 4B = 48KB / query
量化为 uint8: 768 × 16 × 1B = 12KB / query（4× 压缩）
```

### 带宽影响
```
LUT L2: 12.3 GB → 3.1 GB → 0.4 ms
codes HBM: 0.26 ms（同 v16）
→ LeafFine 总计: ~0.7 ms，vs v14 的 6.5 ms ≈ 9× 加速
```

### 实现
- 训练时对每个 query 的 LUT slice 做 per-slice min-max 量化
- 存储 scale + offset（float32 per query per subspace）
- 累加时将 uint8 结果还原（or 累加 int32 最后 scale）
- 精度损失: recall 预计下降 <1%

### 代码改动
- `search.cuh`: d_lut_fine 改为 uint8*，增加 d_lut_scale / d_lut_zero
- `search.cu` FineLUT kernel: 量化输出
- `multi_query_leaf_kernel`: uint8 读取 + int32 累加 + 最后 scale

---

## 方案二：PCA 降维（d=768 → d'=256）

### 原理
```
PCA top-256 解释 ~85% 方差
bpv: 768/2 = 384 → 256/2 = 128
codes: 128 × 128 B = 16KB / leaf block（vs 49KB，3× 压缩）
LUT:  256 × 16 × 4B = 16KB / query（vs 48KB，3× 压缩）
```

### 带宽影响（在 v16 基础上）
```
codes HBM:  20335 × 16384 B = 313 MB → 0.09 ms
LUT L2:     20335 × 13 × 16KB = 4.1 GB → 0.5 ms
→ LeafFine: ~0.6 ms，进一步 3× vs v16-INT8
```

### 额外好处
- 16KB codes 允许 L1 同时驻留 8 个 leaf block（vs 当前 2-3 个）
- 路由 GEMM 变小: 768 → 256 维
- Recall 损失取决于 PCA 保留信息量（需实测）

---

## 方案三：Re-rank 精排（提升 Recall）

### 原理
```
PQ scan 输出 top-N 候选（N=256）→ 精确内积排序 → 真 top-10
```

### 实现（全 GPU）
```
d_base_vecs 常驻 HBM: 2.253M × 768 × 4B = 6.5GB（RTX 5090 有 32GB）

步骤:
1. PQ scan → top-256 候选 ID per query (d_cand_ids [B, 256])
2. Sort candidates by vector ID (减少 random access 局部性)
3. Gather: d_base_vecs[cand_ids] → d_cand_vecs [B, 256, 768]
4. cuBLAS SGEMM: d_q_batch [B, 768] × d_cand_vecs^T → d_cand_dists [B, 256]
5. Top-10 from d_cand_dists

瓶颈: gather 步骤 random HBM reads
  batch × N × d × 4B = 1024 × 256 × 768 × 4 = 786 MB random
  → ~1 ms（worst case random，sorted by ID 后改善 3-5×）
GEMM: 1024 × 768 × 256 = 200M FLOPs → < 0.1 ms
```

### Recall 提升
```
ck3=256 候选 → re-rank top-256 → Recall@10: 0.257 → ~0.85+
```

---

## 方案对比总结

| 版本          | 数据集  | LeafFine (ms) | Total (ms) | QPS     | Recall@10 | 状态     |
|---------------|---------|---------------|------------|---------|-----------|----------|
| v2            | vogue   | 42.1          | 43.3       | 22,980  | 0.806     | 实测     |
| v14           | arxiv   | ~6.5          | ~7.5       | 136,000 | 0.257     | 实测     |
| v15           | arxiv   | ?             | ~13.3      | 77,000  | 0.257     | 实测     |
| v16           | arxiv   | ~1.8 (估)    | ~3.0       | ~340K   | 0.257     | 待测     |
| v16+INT8      | arxiv   | ~0.7 (估)    | ~2.0       | ~500K   | ~0.255    | 待实现   |
| v16+PCA256    | arxiv   | ~0.6 (估)    | ~1.5       | ~680K   | 待测      | 待实现   |
| v16+rerank    | arxiv   | ~1.8+1.0     | ~5.0       | ~200K   | ~0.85+    | 待实现   |

---

## 带宽分析：各版本 LeafFine HBM/L2 流量

| 版本     | codes HBM | LUT L2   | 总 HBM  | 预期时间 |
|----------|-----------|----------|---------|----------|
| v14      | 12.88 GB  | 48 MB    | ~13 GB  | 6.5 ms   |
| v16      | 0.95 GB   | 12.3 GB  | ~1 GB   | ~1.8 ms  |
| v16+INT8 | 0.95 GB   | 3.1 GB   | ~1 GB   | ~0.7 ms  |
| v16+PCA  | 0.31 GB   | 4.1 GB   | ~0.3 GB | ~0.6 ms  |

关键认识：
- v14 → v16: 瓶颈从 codes HBM 转到 LUT L2
- v16 → INT8: 解决 LUT L2 瓶颈
- v16 → PCA: 同时压缩 codes 和 LUT
