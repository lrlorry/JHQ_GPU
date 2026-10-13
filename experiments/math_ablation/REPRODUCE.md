# 复现步骤

所有步骤都在 AutoDL 上运行（RTX 5090，208 核，容器内存上限 90 GiB）。SSH 只用密钥登录，仓库里不含凭据。

## 0. 前提数据（服务器上，不进仓库）

复用 rank-optimal 研究的只读导出，见 `experiments/rank_optimal_quantization/REPRODUCE.md`：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/lf_pilot/<tag>/rb_q*.bin` | 论文索引对 1000 条查询的探查候选、FP64 D_prim / D_H、ground truth |
| `/root/autodl-tmp/pp_pilot/<tag>/` | 论文索引的 primary 码、list 偏移、旋转后的查询 |
| `/root/autodl-tmp/roq/<tag>/` | 旋转矩阵、残差导出、list_ids、y = Πx（xpos.f32）、list 均值（lmean.f32），以及 JHQ 的离线复现 var_jhq_re |
| `/root/autodl-tmp/arms6_cache/<ds>` | 论文索引的训练缓存（B0 与 v68 原型都从它构建） |

本研究自己的产物写在 `/root/autodl-tmp/math_ablation/<tag>/`：
- `v1_e8[_grouped]/`：E8 码、重训的残差、导出给 GPU 的文件；
- `v3_rd/<alloc>/`：变位宽打包的残差。

这些都是几 GB 的数据，不提交。

## 1. 服务器上的步骤

先 push 分支，再在服务器上运行：

```sh
# 正确性测试 + 离线三变体 + B0 前沿（一次跑完；offline1 内部并行，B0 在 CPU 空闲后才测）
/root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation \
    "experiments/math_ablation/run_stage.sh e8test rdtest offline1 offline2 b0" ma_stage1.log
# 收尾：等上一步释放锁后，依次运行 V1 的 E8 原型前沿（v68）、B0 的 nsys 分阶段时间、
# V3 不同分配的评估，然后只读一次测试集（E8、RD、EVT）
/root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation experiments/math_ablation/finish.sh ma_finish.log
```

各阶段做的事：
- `e8test` / `rdtest`：码本与编码器的一致性、变位宽打包的往返测试；
- `offline1`：并行运行 E8 编码（全局 scale 与按子空间 scale）、残差统计与码率分配、EVT 模拟；
- `offline2`：E8 评估，以及 RD 重新编码（评估移到 `finish.sh`，只评互不相同的分配）；
- `b0`：v65 的 α × nprobe × batch 前沿（`b0_frontier.sh`）。

## 2. 本地汇总（不需要数据）

```sh
python3 make_tables.py <结果目录> .     # ABLATION_RESULTS / RECALL_QPS_FRONTIER / STAGE_BREAKDOWN / INDEX_MEMORY
python3 figures_ma.py  .            figures/
```

## 3. 版本

- v65（`jhq_v65_schedule`）不变；
- v68（`jhq_v68_e8`）是 V1 的 GPU 原型：v67 加完整 256 项查表，再加码本级别的 import_state；
- 每次运行的 commit 写在 EXPERIMENT_LOG.md 和各阶段日志的第一行。
