#!/bin/bash
# B0 frontier, run again after finish.sh (the first attempt passed JHQ_BP_GROUPS=-1 alone; the harness needs the
# v61 reference 0 first). Waits for the study's lock (blocking) so the CPU is idle while the GPU is timed.
#   /root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation experiments/math_ablation/b0_rerun.sh ma_b0.log
set -u
exec 9>/root/.lock_ma; flock 9
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) b0 rerun ==="
bash experiments/math_ablation/b0_frontier.sh results/math_ablation/b0 > results/math_ablation/raw/b0.txt 2>&1
echo "b0 rc=$? $(grep -c 'ok ' results/math_ablation/raw/b0.txt) configs"
grep "^B0" results/math_ablation/raw/b0.txt
echo "=== MA_B0_DONE $(date -u +%FT%TZ) ==="
