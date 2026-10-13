#!/bin/bash
# GPU phase 2: waits for the offline stages and B0 (the study's lock) so nothing else loads the CPU while timing,
# then the V1 E8 prototype frontier (v1_gpu.sh) and the B0 stage breakdown (b0_stages.sh).
#   /root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation "experiments/math_ablation/gpu_phase2.sh v1_e8" ma_gpu2.log
set -u
exec 9>/root/.lock_ma; flock 9                         # blocking: after run_stage.sh (offline + b0) is done
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
R=results/math_ablation
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) gpu phase 2 ($*) ==="
bash experiments/math_ablation/v1_gpu.sh $R/v1_gpu ${1:-v1_e8}
echo "v1_gpu rc=$?"
bash experiments/math_ablation/b0_stages.sh $R/b0_stages
echo "b0_stages rc=$?"
echo "=== MA_GPU2_DONE $(date -u +%FT%TZ) ==="
