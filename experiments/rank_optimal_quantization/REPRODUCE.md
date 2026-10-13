# 复现步骤

所有离线计算都在 AutoDL 上的 CPU 进行，GPU 只用于导出、旋转和 rank-loss 训练。服务器只用密钥登录，仓库里不含任何凭据。

## 0. 前提数据（服务器上，不进仓库）

| 路径 | 来源 | 内容 |
|---|---|---|
| `/root/autodl-tmp/arms6_cache/<ds>` | 论文索引缓存 | v65 训练好的索引 |
| `/root/autodl-tmp/pp_pilot/<tag>/` | `experiments/progressive_primary/run_pilot.sh` 的导出 | primary.u8、list_offsets.i32、qrot.f64、qids.i32、meta.json |
| `/root/autodl-tmp/lf_pilot/<tag>/rb_q*.bin` | `experiments/local_first_refinement` 的导出 | 每条查询的探查候选、FP64 D_prim / D_H、ground truth |
| `/root/autodl-tmp/roq/<tag>/` | 本研究第 1 步 | Pi.f32、res_c1d.f32、res_codes.u8、corr.f32、list_ids.i32、meta.json、xpos.f32，以及 var_*/、learned_*.npz |

tag：`arxiv-768_np128_a100`、`vogue-768_np128_a64`、`openai3-3072_np128_a4`。

原始候选与 xpos 共数 GB，不提交。导出的正确性由 `prep.py` 逐位核验，结果在 `prep_<tag>.json`，raw/ 下有日志。

## 1. 服务器上的步骤

每一步都先 push 分支，再用 `launch_b.sh` 把服务器工作树切到该 commit 并后台运行。不要手工拷文件。

```sh
# 0. 只读导出（编译 demo_jhq_v66_export_quant，GPU 约 30 s）
/root/autodl-tmp/launch_b.sh exp/rank-optimal-quantization experiments/rank_optimal_quantization/run_export.sh roq_export.log
# 1. 核验 + 理论数值 + 第 1 轮
/root/autodl-tmp/launch_b.sh exp/rank-optimal-quantization "experiments/rank_optimal_quantization/run_stage.sh prep theory iter1" roq_stage1.log
# 2. 第 2 轮 + 误排序界 + LMMSE 核实
... "run_stage.sh iter2 rankerr" ；评估重跑用 iter2eval
# 3. 第 3 轮：rank-loss 训练（GPU，run_rankloss.sh <tags>），然后 iter3build
# 4. 第 4、5 轮
... "run_stage.sh iter4" ; "run_stage.sh iter5"
# 5. 第 5 轮诊断、v67 编译检查、测试集确认（只运行一次）
... "run_stage.sh diaglist build67 confirm=median" ; "run_stage.sh diagnorm"
```

**运行时的两个注意点：**
- **内存：** 容器内存上限 90 GiB，超过 memory.high 不会被 OOM 杀掉，而是会卡死。评估脚本已按块计算，并限制了进程数。
- **阶段脚本：** 正在运行时不要在服务器上切换 commit。bash 按需读脚本，中途替换会读到新文件。

输出写在服务器的 `results/rank_optimal_quantization/`，阶段日志为 `results/roq_stage*.log`。本目录中的 CSV、JSON、NPZ 和 raw/ 是从那里拷回的副本。

## 2. 本地分析（不需要数据）

```sh
python3 analyze_perquery.py . iter5 calib val          # 配对 bootstrap（任意轮次）
python3 analyze_perquery.py . final test
python3 roq_cost_model.py . eval_iter2.json eval_iter5.json eval_final_test.json
python3 make_deliverables.py .                          # METHOD_COMPARISON.csv、RECALL_VS_BUDGET.csv
python3 figures_roq.py . val iter2 iter3 iter4 iter5 --methods=JHQ,median,lloydmax,learned_rl_tl_zero,lmmse@JHQ,list_median,LLR-iso
python3 figures_roq.py . test final
```

## 3. 关键版本

各轮运行时的 commit 记在 ITERATION_LOG.md 每节的“结果（commit … 运行）”中。测试集只在 `5c0f563` 读过一次。
