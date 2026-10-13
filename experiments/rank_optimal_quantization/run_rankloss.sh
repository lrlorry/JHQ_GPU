#!/bin/bash
# Round 3 training only (GPU): rankloss.py for the three workloads. Separate from run_stage.sh so it can run
# while a CPU evaluation holds that script's lock; it takes the GPU lock itself.
#   /root/autodl-tmp/launch_b.sh exp/rank-optimal-quantization experiments/rank_optimal_quantization/run_rankloss.sh roq_rankloss.log
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/rank_optimal_quantization
R=results/rank_optimal_quantization
mkdir -p $R/raw
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) rankloss ==="
exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
  t0=$(date +%s)
  python3 $E/rankloss.py $tag $R > $R/raw/rankloss_$tag.txt 2>&1
  echo "rankloss $tag rc=$? $(( $(date +%s) - t0 )) s"
done
flock -u 8
echo "=== ROQ""_RANKLOSS_DONE $(date -u +%FT%TZ) ==="
