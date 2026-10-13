# JHQ-CPU：残差精排按 list 分组（存档）

**对象：** 官方 CPU 实现 `jiabhan/JHQ`，基于 commit `1636e19`（serialization fix）。这份工作不是本仓库的代码，所以以 `git format-patch` 导出的 patch 系列存放在这里。

**内容（`patches/`，按顺序共 7 个 commit）：**

| # | 内容 |
|---|---|
| 0001 | 在 Linux 上对 faiss `5702caf` 编译所需的修补（与之前服务器上的 `JHQ_patched` 相同），以及基准程序 `bench_refine_locality`（不带新 API 时作为原版对照） |
| 0002 | 优化本体：全局两阶段搜索的残差精排按 list 分组（`IndexIVFJHQSearch.cpp`、`IndexIVFJHQ.h`） |
| 0003 | 服务器上的构建、运行、分析脚本 |
| 0004 | 按实测数据设定自动模式（T ≥ 256 时分组，并在 list 内按 offset 升序）；分发函数改为 noinline |
| 0005 | 计时与计数改为编译开关 `JHQ_REFINE_PROFILE` |
| 0006 | 插桩默认关闭（实测会让批量 QPS 下降 2–4%） |
| 0007 | 结果（`jhq/benchmarks/refine_locality/results/`）与报告 |

`run_logs.tar.gz` 里是每次运行的标准输出日志，共 53 个文件。`jiabhan/JHQ` 的 `.gitignore` 忽略了 `*.log`，所以它们不在 patch 里，解压到 apply 后的仓库根目录即可放回原位。

**还原：** 已验证 apply 后的 tree 与原 commit 完全相同（`5e68d12f`）。

```sh
git clone https://github.com/jiabhan/JHQ.git && cd JHQ
git switch -c opt/list-grouped-refine 1636e19
git am /path/to/JHQ_GPU/experiments/jhq_cpu_grouped_refine/patches/*.patch
tar xzf /path/to/JHQ_GPU/experiments/jhq_cpu_grouped_refine/run_logs.tar.gz
```

**结论**（详见 `REPORT_CN.md`，与 0007 中 `jhq/benchmarks/refine_locality/REPORT_CN.md` 相同）：

- **正确性：** 在三种存储、两个入口、L2 与 IP 下，结果与原实现逐位相同，并列时的结果也一样。
- **精排阶段：** 只有按 list 预解码的存储在 T ≥ 400 时快 1.04–1.26 倍；separated 和 packed 存储没有收益。
- **端到端：** 0–5%。主阶段（primary 扫描）占查询时间的 67–98%。
