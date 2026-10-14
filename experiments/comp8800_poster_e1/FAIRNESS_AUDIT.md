# E1 公平性审计（FAIRNESS_AUDIT）

本审计在实验开始前完成。每一项都写明：现有代码怎么做、发现了什么问题、本轮怎么改、哪些条件仍无法完全对齐。所有改动都只加新文件（`experiments/comp8800_poster_e1/`），没有改动任何 `jhq_vNN_*` 版本目录、论文 CSV 或历史结果。

## 0. 环境与代码版本

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32 607 MiB，驱动 595.71.05（新克隆的 AutoDL 实例，bjb2） |
| 工具链 | CUDA 13.0（nvcc V13.0.88），g++-12（RaBitQ 程序），cuVS 26.08.01，CuPy 14.2.0，NumPy 2.4.6 |
| JHQ 程序 | `demo_jhq_v69_batch`（`examples/demo_jhq_batch_pipeline.cu` + `jhq_v69_probesel/`），Release，`CMAKE_CUDA_ARCHITECTURES=80`（compute_80 PTX 在 sm_120 上 JIT），与 10/14 practical_opt 的 v69 数字是同一构建配置 |
| RaBitQ 程序 | `bench_rabitq_sweep.cu`，`-arch=sm_120`，链接 wheel 自带的 `libcuvs.so` |
| 代码 | 服务器工作区 `/root/autodl-tmp/jhq_pipeline_wt`（我之前实验用的独立 worktree，detached），通过 git bundle 切到本地 main 的提交；tree hash 与本地一致。`/root/JHQ_GPU`（有其他会话未提交的改动）完全没有碰 |
| 主机 | 容器 cgroup 内存上限 90 GiB；宿主机负载 40+（其他租户，容器内看得到但无法控制），容器内没有其他计算任务 |
| GPU 空闲读数 | 空闲时 nvidia-smi 显示利用率 99–100%、117 W，但显存只用 2 MiB、没有任何计算进程。用 v69 的历史锚点核对性能（见 FINAL_RESULTS_CN.md 第 1 节） |

查询与 GT：两个数据集都是 1 000 条查询，GT 宽度 vogue 100、arxiv 100。所有方法使用同一个 `query.fvecs`、同一个 `groundtruth.ivecs`，Recall@10 = |返回的前 10 个 ∩ GT 的前 10 个| / 10 的平均（`common/recall.cuh` 的定义；Python 和 C++ 程序各自实现了同一公式）。距离一律平方 L2。

## 1. 计时边界（逐方法）

主口径 A（主机到主机）：从主机内存里的原始 FP32 查询开始，到 top-10 的 id 和距离回到主机内存、流同步完成为止。每次调用回答全部 1 000 条查询（batch = 1 000）。

| 方法 | A 口径实际包含 | 现有代码的问题 | 本轮处理 |
|---|---|---|---|
| JHQ v69 | `idx.search(host 查询指针, …)`：memcpy 到 pinned 缓冲 → H2D → CUDA graph（旋转 GEMM、质心 GEMM、probe 选择、建表、扫描、精排、top-k）→ D2H id 与距离（到调用方的 pageable 数组）→ 自旋等待流完成 | 无 | 原样使用，不改代码 |
| CAGRA fp32 | `bench_all.py` 的 `call_xfer`：`cp.asarray`（pageable H2D）→ search → `cp.asnumpy` | ① 每次调用新建 cuVS 资源句柄、新分配输出（对 cuVS 不利）；② 查询从 pageable 内存上传，而 JHQ 用 pinned（对 cuVS 不利） | 复用一个 Resources 句柄和一条流、预分配输出、查询经 pinned 缓冲上传 |
| CAGRA int8 | `call_xfer` 里的查询是**预先量化好的** int8（`scalar.transform` 在计时区外做完） | ③ 查询量化不在计时区内，所谓"xfer"并不是从原始 FP32 查询开始（对 cuVS 有利、对 JHQ 不公平）；若 transform 返回设备数组，还会连 H2D 都跳过 | 计时区内：pinned 上传 FP32 查询 → 设备上 `scalar.transform` 量化 → search → D2H。量化器（数据集的属性）在计时区外训练一次 |
| IVF-PQ | 同 CAGRA fp32 | 同 ①② | 同 CAGRA fp32 |
| IVF-PQ + refine | 本轮新增（IVF-PQ 取 k·r 个候选，再用设备上的 FP32 原始数据精确重排） | — | 重排在计时区内；原始 FP32 数据算进该方法的显存 |
| IVF-RaBitQ | `bench_ivf_rabitq.cu`：H2D（pageable）→ search → D2H id 与距离 → 同步，主机到主机 | ② pageable 上传；每个点都重新 build + serialize + deserialize（只影响耗时，不影响搜索数字） | pinned 上传；见第 2 节 |
| GPU 暴力搜索 | 本轮新增参考线（cuVS brute_force，FP32） | — | 同 A 口径 |

辅助口径 B（设备常驻）：查询已经在 GPU 上（int8 已经量化好），只计 search（和 refine）+ 同步。所有 cuVS 方法和 RaBitQ 都测了 B；**JHQ 的 API 只接受主机指针，没有 B**。JHQ 的主机端开销可以从 10/14 practical_opt 的 nsys 分阶段数据估计（vogue b=1000：端到端 4 042 µs，GPU 合计 3 762 µs，差约 7%）——仅作参考，不进主图。

主图和所有倍数只用 A。

## 2. 构建与缓存

| 方法 | 现有做法 | 本轮 |
|---|---|---|
| JHQ v69 | 训练状态缓存 `arms6_cache/<ds>`（以数据和参数为键）；每个进程加载缓存、重新编码 add（秒级）；一个进程内扫多个 α | 不变；每个 nprobe 一个进程，α 在进程内扫 |
| CAGRA | `bench_all.py` 把 build 放在每个搜索参数的循环里：每个 itopk/search_width 都重建一次图 | 每个 graph_degree 只建一次，所有 itopk × search_width 在同一个索引上扫 |
| IVF-PQ | 同上，每个 n_probes 重建 | 每个 pq_dim 只建一次 |
| IVF-RaBitQ | 每个 (n_probes, mode) 一个进程：上传 + build + serialize + deserialize | 每个 (nlist, bits) 只建一次，round-trip 一次，然后扫全部 mode × n_probes。serialize/deserialize 是必要步骤（不 round-trip 的话自检召回 0/20），保留并单独计时；文件放在数据盘 `/root/autodl-tmp`（系统盘只剩 5 GB） |

所有方法的搜索 QPS 都不含一次性构建；构建时间、索引显存单独记录。

## 3. 每个点的测量方式

| | JHQ（harness 原样） | cuVS 与 RaBitQ（本轮新程序） |
|---|---|---|
| 预热 | 每个变体每轮先跑 1 次不计时（共 5 次），首次还包括 graph capture 和 JIT | 3 次不计时 |
| 计时 | 5 轮 × 24 次调用，每轮报告 QPS = 24 000 / 24 次总时间、中位与 p95 延迟 | 5 轮 × 24 次调用，**保存每一次调用的时间**；单次超过 40 ms 的慢点每轮减少调用次数（记录里有实际次数） |
| 报告 | 5 轮 QPS 的中位数（另记最小、最大） | 同左 |
| 检查 | G = −1（v69 调度）与 G = 0（v61 路径）逐查询对比 probe、top-ck、最终结果；recall 由 `evaluate_recall` 计算 | id 越界（含哨兵值）、行内重复 id、各次调用 id 是否相同（确定性）、B 与 A 的 id 是否相同 |

## 4. 存储预算

| 方法 | 每向量编码 | 说明 |
|---|---|---|
| JHQ v69（Br=8） | 主码 d/8 B + 残差 d B + 校正项 | 768 维约 868 B |
| IVF-RaBitQ bits=9 | 9d/8 B | 与 JHQ Br=8 预算相当（bits 含 1-bit 码） |
| IVF-RaBitQ bits=8 | d B | 历史数据用的就是 8，比 JHQ 少约 11% |
| IVF-RaBitQ bits=4 | d/2 B | 更小更快，召回上限更低 |
| IVF-PQ | pq_dim B（pq_bits=8） | 192/384/768 B |
| IVF-PQ + refine | pq_dim B + 设备上的 FP32 原始数据（4d B） | 精确重排需要原始数据 |
| CAGRA fp32 | 4d B + 4·degree B | 搜索时数据集常驻显存 |
| CAGRA int8 | d B + 4·degree B | |

不同 bits/pq_dim 的结果不当作相同存储预算比较；海报只在同一张图上按方法画前沿，显存另列。显存口径：所有方法都记录"搜索时整卡已用显存"（与 JHQ harness 的 `used_mib` 同一口径，含 CUDA context），另记各自运行时只开 context 的基线和索引构建前后的差值。

## 5. 参数选择原则

- 各方法独立扫自己的参数，取各自的 Pareto 前沿；不为 JHQ 挑弱的基线配置。
- CAGRA：graph_degree ∈ {32, 64}，itopk ∈ {32, 64, 96, 128, 192, 256, 384, 512}，search_width ∈ {1, 2, 4, 8}；构建用 cuVS 默认（intermediate 128、ivf_pq 建图）。
- IVF-RaBitQ：bits ∈ {9, 8, 4}，nlist 与 JHQ 相同（vogue 4096、arxiv 8192），四种搜索模式全扫，n_probes 16–1024 的密网格（历史上 768 维最好的模式是 LUT32）。
- IVF-PQ：nlist 与 JHQ 相同，pq_dim ∈ {192, 384, 768}，LUT/内部距离 fp32 与 fp16 两种，n_probes 16–512。
- JHQ：nprobe ∈ {64, 128, 256} × α ∈ {½, 1, 2}×论文 α（vogue 64、arxiv 100）；只有覆盖不到目标 recall 才扩展。

## 6. 冒烟测试中发现并处理的问题（正式运行前）

三轮冒烟测试都只在 vogue 上跑小网格，结果放在 `SMOKE*/`，不进入正式结果。

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | 第一次冒烟时 JHQ 阶段一直卡住 | 冒烟脚本已持有 `/root/.gpu_lock`，子脚本又申请同一把锁，等待父进程 | `run_e1.sh` 增加 `E1_LOCK_HELD=1`：调用方已持锁时不再重复申请 |
| 2 | cuVS 的 Python 进程写完元数据就退出，没有错误输出 | 把绑定在 CuPy 流上的 `Resources(stream=...)` 传给 `cagra.build`、`ivf_pq.search` 等会**段错误**（阻塞流和非阻塞流都一样，用 20 000×64 的小数据单独复现过） | 构建用 cuVS 默认资源；搜索用一个默认 `Resources()` 句柄反复复用。用 C API `cuvsStreamGet` 查到它的流是 per-thread 默认流（2），所以 CuPy 的上传和回传也放在这条流上，每次调用只在最后同步一次 |
| 3 | CAGRA 搜索报 `dtype int64 not supported` | cuVS 26.08 的 CAGRA 只接受 uint32 邻居输出（文档写的是 int64） | CAGRA 用 uint32 输出，其他方法用 int64；检查 id 前统一转 int64（uint32 哨兵值会被判为越界 id） |
| 4 | RaBitQ 的 A 口径偶尔出现 29–68 ms 的单次调用（平时 5 ms） | 分段计时：memcpy p99 0.63 ms、H2D 0.07 ms、D2H 0.02 ms，**尖峰全部在 search 本身**（p99 33.5 ms，最大 39 ms），与容器 CPU 限流无关（限流计数前后不变） | 冒烟 1 的 70K QPS 是 2 轮 × 5 次的短测量被尖峰拖低；3 轮 × 24 次时 A = 200 543，与 9 月一致。正式运行给 RaBitQ 的 RAFT workspace 配 2 GiB 内存池（`RQ_POOL=2048`）：A 从 200 652 升到 206 380（+3%），p99 从 5.98 降到 5.28 ms。偶发尖峰仍有，靠 5 轮取中位数抵消 |
| 5 | 构建阶段容器被 CPU 限流（`nr_throttled` 42 → 116） | `nproc` 报宿主机的 208 核，OpenMP 默认开 208 个线程，超过容器 25 核配额 | 所有进程固定 `OMP_NUM_THREADS=16`；之后限流计数不再增加 |
| 6 | IVF-PQ 不开内存池时比开池慢 7%（79 437 对 85 354 QPS） | 每次搜索都临时 cudaMalloc | 正式运行 cuVS 方法统一开 RMM 内存池（`CUVS_POOL=1`），只在构建并测完显存之后才开。CAGRA 开不开池没有区别（264K 对 266K） |
| 7 | RaBitQ 每次调用返回的 id 不完全相同 | 1 000 行中两次调用之间约 8–10 行 id 集合不同、40–50 行只是顺序不同；recall 在第 4 位小数波动（0.9579 / 0.9586）。暴力搜索、IVF-PQ 只有并列项的顺序差异；CAGRA 完全确定 | 不算缺陷，点保留；每个点同时记录首次调用和最后一次调用的 recall |
| 8 | IVF-PQ + refine 很慢（pq192、np64、比例 2：R = 0.901，14 521 QPS；单独的 IVF-PQ 在相同 nprobe 下是 79K） | 未查明；refine 一步每次约 56 ms | 本身是可选项，E1-A 不运行；冒烟数据保留在 `SMOKE3/` |

## 7. 仍无法完全对齐的条件

1. JHQ 的结果回传到 pageable 数组，基线回传到 pinned 数组（每次 80 KB，影响可忽略，且偏向基线）。
2. JHQ 是 C++ 直接调用；CAGRA、IVF-PQ 经 cuVS 的 Python 封装，每次调用多出 Python 开销（量级几十 µs，相对 b=1000 的 1–10 ms 调用约 1–3%）。RaBitQ 是 C++。如果有影响，也偏向 JHQ，报告里会用 B 口径和调用时间分布说明。
3. JHQ 使用 compute_80 PTX（JIT 到 sm_120），基线是原生 sm_120 库：这是两边各自最快的已知配置（campaign Exp 0 测过 JHQ 原生编译在小 batch 更慢）。
4. 宿主机负载来自其他租户，无法控制；计时包含主机端，所以每个点保留 5 轮的离散度，异常离散会在结果里标出。
5. 只有一张卡、一个 batch（1 000）。不推广到其他 GPU 或 batch。
