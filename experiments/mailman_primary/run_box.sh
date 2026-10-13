#!/bin/bash
# Mailman primary scan, stages 0-1: structure statistics of the real primary codes (CPU) and the isolated
# scoring microbenchmark (GPU) on the three paper indexes' exports (read only: progressive_primary's
# pp_pilot and residual_aware_bound's lf_pilot).
#   bash experiments/mailman_primary/run_box.sh <smoke|stats|bench|all> [profile]    (from the repository root)
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
MODE=${1:-all}; PROF=${2:-}
R=results/mailman_primary; mkdir -p $R build
PP=/root/autodl-tmp/pp_pilot; LF=/root/autodl-tmp/lf_pilot
TAGS="vogue-768_np128_a64 arxiv-768_np128_a100 openai3-3072_np128_a4"
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) mode=$MODE $PROF"
timeout 900 nvcc -O3 --use_fast_math -Xcompiler -fopenmp "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
    -o build/mailman_bench experiments/mailman_primary/mailman_bench.cu > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== DONE build failed, nothing run ==="; exit 1; }

if [ $MODE = smoke ]; then
  exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
  env MB_ROUNDS=2 MB_REPS=3 OMP_NUM_THREADS=24 timeout 1800 build/mailman_bench $PP/vogue-768_np128_a64 \
      $LF/vogue-768_np128_a64 $R/smoke_vogue.json 8 > $R/smoke_vogue.txt 2>&1
  echo "smoke rc=$? $(tail -1 $R/smoke_vogue.txt)"; cat $R/smoke_vogue.txt
  echo "=== MAILMAN_DONE $(date -u +%FT%TZ)"; exit 0
fi
if [ $MODE = stats ] || [ $MODE = all ]; then
  for t in $TAGS; do
    env OMP_NUM_THREADS=8 timeout 7200 python3 experiments/mailman_primary/stats.py $PP/$t $LF/$t $R/stats_$t.json \
        --lists 512 > $R/stats_$t.txt 2>&1
    echo "stats $t rc=$?  $(date -u +%T)"
  done
fi
if [ $MODE = bench ] || [ $MODE = all ]; then
  exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > $R/gpu.txt 2>&1
  PF=0; SUF=""; [ -n "$PROF" ] && { PF=1; SUF=_profile; }
  for t in $TAGS; do
    env MB_PROFILE=$PF OMP_NUM_THREADS=24 timeout 3600 build/mailman_bench $PP/$t $LF/$t \
        $R/bench_$t$SUF.json 8,1000 > $R/bench_$t$SUF.txt 2>&1
    echo "bench $t rc=$? $(tail -1 $R/bench_$t$SUF.txt)  $(date -u +%T)"
  done
fi
echo "=== MAILMAN_DONE $(date -u +%FT%TZ)"
