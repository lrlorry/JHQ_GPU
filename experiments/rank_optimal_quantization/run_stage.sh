#!/bin/bash
# Rank-optimal quantization study: run named offline stages on the box (CPU, plus the GPU for the
# rotation in prep). Stages: prep, theory, iter1, ... (see the case below). Logs to results/.
#   /root/autodl-tmp/launch_b.sh exp/rank-optimal-quantization \
#       "experiments/rank_optimal_quantization/run_stage.sh prep theory iter1" roq_stage.log
set -u
exec 9>/root/.lock_roq; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/rank_optimal_quantization
R=results/rank_optimal_quantization
mkdir -p $R/raw
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) stages: $* ==="
for st in "$@"; do
  t0=$(date +%s)
  case $st in
    prep)
      exec 8>/root/.gpu_lock; flock 8
      for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
        python3 $E/prep.py $tag $R > $R/raw/prep_$tag.txt 2>&1; echo "prep $tag rc=$?"
      done
      flock -u 8 ;;
    theory) python3 $E/theory_numerics.py > $R/raw/theory_numerics.txt 2>&1; echo "theory rc=$?" ;;
    iter1)  python3 $E/iter1_llr.py $R 64 > $R/raw/iter1.txt 2>&1; echo "iter1 rc=$? $(tail -1 $R/raw/iter1.txt | cut -c1-300)" ;;
    *) echo "unknown stage $st" ;;
  esac
  echo "stage $st took $(( $(date +%s) - t0 )) s"
done
echo "=== ROQ""_STAGE_DONE $(date -u +%FT%TZ) ==="
