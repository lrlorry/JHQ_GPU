# 分支存档

2026-10-14 起，仓库只保留 `main` 一个分支。原来的实验分支都已合并或存档进 `main`，然后删除。

## 已并入 main 历史的分支

这些分支的全部 commit 都在 `main` 的历史里，代码和结果都在 `main` 上，没有任何丢失。需要原来的分支名时，用下面的 commit 重建：

```sh
git branch exp/practical-opt 36a79c3a63a5f1a1b666a10126365dd9ef3a6da1
```

| 分支 | 最后一个 commit | 日期 | 对应目录 |
|---|---|---|---|
| `fix/recall-eval-v15` | `862543adc405f45cf8e1c9dc5253c339c5de5888` | 2026-10-13 | |
| `exp/pipeline-campaign` | `0d87dc98fb4da520d4e9c1d93e6797009fd1fc51` | 2026-10-13 | `experiments/pipeline_campaign`，`jhq_v62`–`jhq_v65` |
| `exp/residual-aware-bound-pilot` | `e17b23f53cd8e8cebf276587406e79de37a25e49` | 2026-10-13 | `experiments/residual_aware_bound`，`jhq_v66` |
| `exp/bound-aware-quant-pilot` | `765afae522cf7ace5b2915f91408157c69bea577` | 2026-10-13 | `experiments/bound_aware_quant` |
| `exp/local-first-refinement-pilot` | `72cab00e1ce012a25b5ed5e35d7579df3c524943` | 2026-10-13 | `experiments/local_first_refinement` |
| `exp/progressive-primary-pilot` | `5699d34a7483c11b7e23b89053bd8a6332f4fa55` | 2026-10-13 | `experiments/progressive_primary` |
| `exp/rank-optimal-quantization` | `6fda5a7d0aad61956554bf78797870ada7572a49` | 2026-10-13 | `experiments/rank_optimal_quantization`，`jhq_v67` |
| `exp/math-algorithm-ablation` | `3fa267cd76a198cccfff04395f933087d0a98ba7` | 2026-10-13 | `experiments/math_ablation`，`jhq_v68` |
| `exp/anytime-certified-jhq` | `36d86cde84f0865f937e5a67006cc94ecd08c451` | 2026-10-14 | `experiments/anytime_certified_jhq` |
| `exp/practical-opt` | `36a79c3a63a5f1a1b666a10126365dd9ef3a6da1` | 2026-10-14 | `experiments/practical_opt`，`jhq_v69`，`jhq_v70` |

合并前 `main` 停在 `7b59478`（论文 PDF），它同样在历史里。

## 没有合并的分支：以 patch 存档

| 分支 | 原 commit | 基于 | 存档 |
|---|---|---|---|
| `feat/residual-level2` | `911b5d5a198bf87a062e86dac546dc924f76d2ec`（2026-09-02，WIP：第二级残差的编码与搜索路径） | `c1a704a` | `feat-residual-level2/0001-*.patch` |

它直接改了旧版本目录 `jhq_v21_cascade`，和现在的 `main` 有冲突（`jhq_gpu_index.cu`、`jhq_gpu_index.cuh`、`search.cu`），所以没有合并。

patch 由 `git format-patch` 导出，包含作者、日期、说明和完整 diff。已验证在 `c1a704a` 上 apply 后，tree 与原 commit 完全相同（`c77a5455`）。还原方法：

```sh
git switch -c feat/residual-level2 c1a704a
git am archive/feat-residual-level2/0001-*.patch
```

## 另一个仓库的工作

JHQ-CPU（`jiabhan/JHQ`）的残差精排优化不属于本仓库的代码。它以 patch 系列存放在 `experiments/jhq_cpu_grouped_refine/`，说明见那里的 README。
