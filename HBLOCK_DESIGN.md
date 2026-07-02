# HBlock 设计文档

> HBlock = Hierarchical Block Index
> 面向海量向量数据集（十亿~万亿级），GPU 原生分层物理块索引。

---

## 1. 核心思想

传统 IVF 方案在 GPU 上将所有数据常驻显存，规模上限受限于 GPU 内存。

HBlock 的思路来自 HBase：**数据按量化码分层组织成物理块，搜索时按层路由、按需 copy，最终将叶子块加载到 GPU 做精细计算。**

每一层逻辑相同：
```
计算重建距离 → 找 top-k 目标块 → copy 物理块 → 下探到下一层
```

---

## 2. 实现架构（hblock_v1）

### 2.1 确认的设计决策

| 维度 | 决策 |
|------|------|
| 层级结构 | 3路由层 + 叶子层（L1 常驻 GPU L2，L2/L3 按需载入 SM） |
| 路由残差 | **每层用上一层的残差**（primary → coarse residual → fine residual） |
| 路由表项格式 | **B 方案：2字节质心ID + 4字节块指针 = 6字节/项** |
| 路由距离计算 | GEMM（第一版），未来可替换为 byte_lut 扫描 |
| 叶子 fine code | Kr=16, Br=4 标量 PQ（与 JHQ v12 残差编码相同） |
| 叶子块大小 | 128 vectors × 384B fine code = 48KB（恰好填满一个 SM） |

### 2.2 hblock_v1 实现（两层路由 + 叶子，用于 Vogue-768 验证）

```
L1 routing  K1=64 个质心（GEMM，全局路由）
  └─ L2 routing  K2=128 个质心（GEMM，L1残差空间路由）
        └─ Leaf block（128 vectors × 384B fine code）
```

**编码（add）：**
```
x_rot = Pi @ x
L1_code  = argmin_j ||x_rot - C1[j]||²
r1       = x_rot - C1[L1_code]
L2_code  = argmin_j ||r1 - C2[j]||²
r2       = r1 - C2[L2_code]
fine_code = scalar_PQ(r2, Kr=16, Br=4)  # 384 bytes

按 (L1_code, L2_code) 排序，每128个向量组成一个叶子块
```

**搜索（search）：**
```
1. q_rot = Pi @ q
2. dots1 = q_rot @ C1^T → top-ck1 个 L1 质心
3. q_r1  = q_rot - C1[best_L1]
4. dots2 = q_r1  @ C2^T → top-ck2 个 L2 质心
5. q_r2  = q_r1  - C2[best_L2]
6. 路由表查询：(L1_code, L2_code) → 叶子块范围
7. 构建 fine LUT from q_r2 → [d, Kr]
8. 对选中的 ck3 个叶子块：GPU SM 计算 fine 距离（128 vectors/块）
9. 全局 top-k
```

---

## 3. GPU Memory Hierarchy 分析

| 层级 | 大小 | 访问粒度 |
|------|------|---------|
| L2 cache | 6–40 MB | 128 B cache line |
| Shared memory / SM | 48 KB | 可配置 |
| Warp | — | 32 threads |
| Global memory (HBM) | GB | 128 B cache line |

关键约束：
- **L1 路由表常驻 GPU L2**：每次 query 必访问，对小数据集（Vogue-768）完全满足
- **L2/L3 路由块按需载入 SM**：32KB 刚好
- **叶子块用 SM 做精细计算**：48KB = 128 vectors（Br=4，d=768）

---

## 4. 覆盖规模（6字节/路由项）

```
路由块大小：32KB / 6B = 5461 项/块

1 层路由：5461 × 128         ≈ 700K vectors
2 层路由：5461² × 128        ≈ 3.8B vectors
3 层路由：5461³ × 128        ≈ 21T vectors
```

对于 Vogue-768（N=932K）：2层路由已覆盖（甚至 1 层即可），架构在此验证后扩展至海量数据集。

---

## 5. JL 解析解

JL 旋转后，子空间每个维度 $x_j \sim \mathcal{N}(0, \sigma^2)$。

对于 K1D=2（1-bit 量化器），最优质心有闭合解：

$$c_0 = -\sigma\sqrt{\frac{2}{\pi}}, \quad c_1 = +\sigma\sqrt{\frac{2}{\pi}}$$

hblock_v1 路由质心使用标准 k-means（CPU），后续版本可换用解析解。

---

## 6. 与 JHQ-GPU v12 的对比

| 维度 | JHQ-GPU v12 | HBlock-v1 |
|------|-------------|-----------|
| 数据组织 | IVF cluster（按质心） | 按量化码分层物理块 |
| 搜索入口 | centroid GEMM → primary scan → residual refine | L1 GEMM → L2 GEMM → 叶子块 fine 计算 |
| 精度控制 | nprobe + alpha | ck1/ck2/ck3（每层可调） |
| GPU 内存需求 | 全量数据常驻 GPU | 仅叶子块按需 copy |
| 适用规模 | ~百万级（受 GPU 内存限制） | 十亿~万亿级 |
| Vogue-768 性能 | 基准 | 验证架构正确性（不追求性能） |

---

## 7. 文件结构

```
hblock_v1/
  encode.cuh    - Params + CPU k-means + GPU 编码声明
  encode.cu     - CPU k-means、assign、subtract、fine encode GPU 核函数
  search.cuh    - SearchWorkspace + search_hblock 声明
  search.cu     - 所有 GPU 搜索核函数 + search_hblock 实现
  jhq_gpu_index.cuh  - HBlockIndex 类声明
  jhq_gpu_index.cu   - HBlockIndex 类实现（train/add/search）

examples/
  demo_hblock_v1.cu  - 命令行演示
```

### CLI 参数（demo_hblock_v1）
```
demo_hblock_v1 <base> <query> <gt> [K1=64] [K2=128] [ck1=4] [ck2=16] [ck3=64] [k=10] [route_iters=20] [batch_size=256]
```

---

## 8. 后续扩展

1. **byte_lut 路由**：用 Mr=2 子空间（2字节路由码）替换 GEMM，显著降低路由开销
2. **3层路由**：L1 表常驻 L2，L2/L3 按需载入，覆盖 TB 级数据集
3. **异步 copy**：叶子块从 host/NVMe 异步预取
4. **CUDA Graph**：捕获完整搜索 pipeline 为 CUDA Graph

---

## 9. LeafFine 性能瓶颈分析（hblock_v2）

### 现状

| 参数 | 值 |
|------|----|
| B (batch) | 256 |
| ck3 | 256 |
| leaf_size | 128 |
| bpv | 384 bytes |
| LeafFine 耗时 | ~43ms |
| 有效 HBM 带宽 | ~74 GB/s（峰值 2TB/s 的 3.7%）|

**根因**：B×ck3 = 65536 个 CUDA block 同时对 403MB leaf_codes 发出随机读请求。
每个 leaf block 平均被 ~8 个 query 各读一次，但 A100 L2=40MB << 403MB，全部是 HBM miss。
数据总量 3.2GB，全随机，实际有效带宽极低。

已尝试：
- `sort_leaf_sel`（query 内按 leaf_idx 排序）→ **无效**，因跨 query 的 inter-query 随机性仍然存在

### 方案对比

#### 方案 A：全局跨 query 排序 + 转置 kernel（推荐）

原理：把 leaf_sel[B, ck3] 扁平化为 65536 个 `(leaf_idx, bqi, slot)` 三元组，
GPU radix sort 按 leaf_idx 全局排序，然后 1D kernel 顺序处理——
同一 leaf_idx 的 ~8 个 query 请求连续，leaf block 进 shared memory 只读一次，
对 8 个 query 用各自 LUT 串行计算。

数量：
- leaf_codes 读：8192 blocks × 49KB = **403MB 顺序读**（vs 3.2GB 随机）
- LUT 读：256 queries × 49KB = **12MB，L2 常驻**
- 理论 LeafFine：403MB / 2TB/s ≈ 0.2ms + 计算 ≈ **1–3ms**

实现要点：
1. CUB DeviceRadixSort 排 65536 条 (key=leaf_idx, val=packed(bqi,slot))
2. CUB DeviceRunLengthEncode 找 unique leaf_idx 及各自 count
3. CUB DeviceExclusiveScan 得到 group offset
4. 新 kernel：grid=n_unique_leaf_blocks, blockDim=leaf_size；shared memory 存 leaf codes (48KB)，对每个命中的 query 串行计算距离

预期收益：LeafFine 43ms → 1–3ms，约 **20–40x 加速**

#### 方案 B：减少 ck3（参数调整，立即可试）

ck3 线性缩减读量，有效带宽不变（仍是随机读）：

| ck3 | 总读量 | 预期 LeafFine | 说明 |
|-----|--------|--------------|------|
| 256 | 3.2GB | 43ms | 当前 |
| 128 | 1.6GB | ~21ms | Recall 会下降 |
| 64  | 0.8GB | ~11ms | Recall 下降更多 |

治标不治本，适合快速测 Recall/QPS trade-off 曲线。

#### 方案 C：减少 leaf_size（结论：不值得）

减小 leaf_size 会降低每次随机读的粒度，但：
- 若 ck3 不变 → candidates 减少，Recall 掉
- 若 ck3 同比增加 → 随机请求数同比增加，总延迟不变甚至更差
- GPU 随机读的代价是 latency × 请求数，不是 bandwidth × 数据量

leaf block 平均 fill rate ≈ 932K / (n_blocks × 128) ≈ **89%**（k-means 大致均衡），
11% 是零填充，影响 GPU compute（INF slot）但不影响 HBM 瓶颈。

### Recall 优化方向（独立于延迟）

**Per-block fine codebook**：当前 fine PQ 的 1D centroids `d_fine_c1d` 是全局共享的（在所有向量的 L2 残差上训练）。
可以为每个 (c1,c2) 对单独训练一套 fine 码本，更贴合该 block 的局部残差分布。

代价：
- 存储：8192 blocks × 768 dims × Kr × 4 bytes = 8192 × 49KB ≈ **402MB**
- 搜索时 LUT 需 per-block 构建（不再是全局一张表）

收益：相同 ck3 下 Recall 提升，不影响 LeafFine 延迟。

**注意**：per-block 随机旋转矩阵（768×768）不可行——
① 存储 8192×2.25MB = 18.4GB；② 随机旋转不改变各向同性 Gaussian 的统计性质，PQ 质量不提升。
轻量替代（sign-flip + permutation，192B/block）存储可行但同样对质量无帮助。

---

## 10. 优化思路 Backlog

### 思路 A：全局跨 query 排序 + 转置 kernel

见方案 A（第 9 节）。核心：把访问维度从 (query, slot) → (leaf_block, query_list)，
每个 leaf block 进 shared memory 只读一次，对命中它的所有 query 串行计算。

实现步骤：
1. CUB DeviceRadixSort：65536 条 `(leaf_idx, bqi<<8|slot)` 按 leaf_idx 排序
2. CUB DeviceRunLengthEncode：找 unique leaf_idx 和每组 count
3. CUB DeviceExclusiveScan：得到每组起始 offset
4. 转置 kernel：grid=n_unique_leaf_blocks, blockDim=leaf_size
   - 48KB shared memory 存当前 leaf block 的 codes（装得下：leaf_size×bpv=48KB）
   - 对该 block 命中的每个 query，用 __ldg 读其 LUT，串行计算 128 个距离
5. 结果写回 fine_dists[bqi][slot]

预期：LeafFine 43ms → 1–3ms（20–40x）

### 思路 B：相似 query 队列聚合（Query Batching by Similarity）

**核心思想**：相似的 query 在 embedding 空间里是邻居，路由后会命中大量相同的 leaf block。
如果把相似 query 打包进同一个 batch，这些 leaf block 第一次读入 L2 后对后续 query 全部命中——
本质上是用 query 调度来提高 GPU L2 cache 利用率。

**极限情况估算**：
- 若 256 个 query 完全相似，全部路由到同一组 256 个 leaf block
- 唯一数据量：256 × 49KB = **12.5MB，全部装入 A100 L2（40MB）**
- 65536 次 block 访问中，首次冷读 256 次（HBM），其余 65280 次全是 L2 hit
- L2 带宽 ~4TB/s，LeafFine 理论降到 **< 1ms**

**与方案 A 的关系**：
- 方案 A 解决"同一 batch 内每个 leaf block 重复读"的问题
- 思路 B 解决"batch 之间相同 leaf block 反复从 HBM 冷读"的问题
- 两者正交，可以叠加

**实现方案**：

```
在线队列（serving 场景）：
  incoming queries → buffer（大小 N=1024）
  → 计算每个 query 的 routing key（top-1 L1 centroid index，1 次 GEMM）
  → 按 routing key 排序
  → 每 256 个相邻 query 打一个 batch → GPU 处理

离线批量（benchmark 场景）：
  对整个 query 集合，先跑 L1 GEMM → 得到每个 query 的 top-1 c1
  → 按 c1 排序 query 集合
  → 顺序按 B=256 切 batch
```

**代价**：
- 在线场景：增加等待时间（攒够 N 个 query 才处理），引入 tail latency
- 离线场景：额外一次全量 L1 GEMM（很快），排序 nq 个 query（nq=1000 时可忽略）
- Benchmark 里直接加一行排序即可验证

**预期收益**：
取决于 query 集合本身的 embedding 分布。若 query 分布集中（如推荐系统同一主题），
收益极大；若完全随机（如 benchmark），收益较小（random → ~8x overlap，排序后 →更高 overlap）。

**与 HBase 的对比**：
- HBase BlockCache：热 HFile block 常驻内存，被多个 client 命中
- 思路 B：热 leaf block 常驻 GPU L2，被同 batch 内相似 query 命中
- 区别在于 HBase 是被动缓存（LRU），思路 B 是主动调度（排序聚合）

### 思路 B2：利用路由层级本身做 query 分组（Hierarchical Query Routing Queue）

**核心洞察**：不需要单独计算 query 相似度——HBlock 的层级路由码本身就是现成的 similarity key。
- L1 code（top-1 c1）相同的 query，必然共享大量 (c1, ?) leaf block
- L1+L2 code（top-1 c1, top-1 c2）都相同的 query，leaf block 集合几乎完全重叠

**机制**：
```
query 到来
  → 只做 L1 GEMM（64 个 centroid，很便宜）→ 得到 top-1 c1
  → 投入对应的 L1 queue（共 K1=64 个队列）
  → 某个 queue 满 B=256 时，弹出这批 query → GPU batch 处理
```

**优势**：
- 分组 key 直接复用 L1 路由，无额外开销
- 同 queue 内 256 个 query 共享 top-1 c1 → ck1×ck2=256 个 (c1,c2) 对大量重叠
- Leaf block 重叠率从随机 batch 的 ~8x 升至 **接近 256x**（极限：同 queue 的 query 访问完全相同的 256 个 leaf block）
- 256 blocks × 49KB = **12.5MB 唯一数据，全部装入 A100 L2（40MB）**
- 与思路 A（转置 kernel）叠加：首批冷读 12.5MB sequential，后续 batch L2 全命中 → **LeafFine 接近 0ms**

**两级分组**（更激进）：
```
先按 top-1 c1 分 64 个队列
每个队列内再按 top-1 c2 分 128 个子队列
→ 共 8192 个 (c1,c2) 队列
→ 同一子队列的 query leaf block 几乎完全一致
```

**在线 serving 的挑战**：
- K1=64 队列，某些 c1 热点队列快速填满，冷门队列等待时间长
- 需要超时机制（等待超过 T ms 就强制发射，哪怕不满 256）
- Tail latency 会上升（等待攒队列）

**离线 benchmark 验证（5 分钟实现）**：
```cpp
// search 前加这几行，其他不变
// 1. 跑 L1 GEMM 得到每个 query 的 top-1 c1
// 2. 按 c1 对 query 排序
// 3. 顺序切 batch → search
```
不用改任何 GPU kernel，直接验证 leaf block overlap 的收益。

### 思路 C：GPU 专用 Leaf Cache 模块（解耦设计）

**动机**：Leaf block 全部等大（leaf_size × bpv = 48KB），是最理想的 fixed-size buffer pool 场景，
不需要处理内存碎片，管理极简单。可以把 leaf 存取完全解耦成一个独立模块，路由模块只负责输出需要哪些 leaf_idx，Leaf Cache 模块负责"给数据"。

**整体架构**：
```
┌─────────────────────────────────────────────────────┐
│                  HBlock Search Pipeline               │
├─────────────┬───────────────────────────────────────┤
│ Routing     │ L1/L2 GEMM → gather leaf_sel           │
│ Module      │ 输出：leaf_idx 列表                    │
├─────────────┼───────────────────────────────────────┤
│ Leaf Cache  │ Cuckoo filter（presence check）        │
│ Module      │ Hot set buffer（HBM，~50-100MB）       │
│ (NEW)       │ Eviction policy（FIFO / LRU）          │
│             │ Async prefetch stream                  │
├─────────────┼───────────────────────────────────────┤
│ Leaf Compute│ 转置 kernel：leaf block → 所有 query   │
│ Module      │ 输入来自 Leaf Cache，不直接读 leaf_codes│
└─────────────┴───────────────────────────────────────┘
```

**Cuckoo Filter 做 presence check**：
- 8192 个 leaf block，fingerprint 8-bit，bucket size 4
- 总大小：~16KB，全程 L2 命中
- 每次 lookup：2 次内存访问，高度并行
- 支持删除（比 Bloom filter 优越的地方），eviction 时可以同步删 filter 里的记录
- false positive rate ~1%：误判为"在缓存"，实际取数据时发现不对，fallback 到 HBM fetch

**Hot set buffer**：
- 在 GPU HBM 里划一块专用区域（如 50MB ≈ 1000 个 leaf block）
- 直接映射（leaf_idx % n_slots → slot）或组相联
- 50MB < A100 L2（40MB）…实际需要更小，约 30MB ≈ 600 个 leaf block 才能稳定 L2 驻留
- 与 similarity-aware batching 叠加：同一 L1 queue 的连续 batch 访问同 256 个 leaf block（12.5MB），
  全部 L2 驻留，后续 batch 几乎零 HBM 读

**Async Prefetch**：
```
batch N routing 完成 → 得到 leaf_sel
→ 查 cuckoo filter → 找出 miss 的 leaf_idx
→ 在 prefetch stream 上异步发起 HBM → hot buffer copy
→ batch N compute 在 compute stream 上开始（与 prefetch overlap）
→ batch N+1 routing 同时进行
```

**为什么完全解耦是对的**：
- Leaf block 等大 → buffer pool 管理零碎片，trivial free list
- 路由模块只输出 leaf_idx 列表，不关心数据从哪来
- Leaf Cache 可以单独演进（换 eviction 策略、换 prefetch 算法）
- Leaf Compute 只消费数据，不关心 cache 细节
- 未来扩展到 NVMe/Host memory 时，只需改 Leaf Cache 模块
