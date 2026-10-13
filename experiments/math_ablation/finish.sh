#!/bin/bash
# Everything after run_stage.sh (offline stages + B0), in order, under the study's lock (blocking, so nothing else
# loads the CPU while the GPU is timed): the V1 E8 prototype frontier (v1_gpu.sh), the B0 stage breakdown
# (b0_stages.sh), the V3 evaluation of the distinct allocations, then the single read of the test split.
#   /root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation experiments/math_ablation/finish.sh ma_finish.log
set -u
exec 9>/root/.lock_ma; flock 9
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/math_ablation; R=results/math_ablation
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) finish ==="
bash $E/v1_gpu.sh $R/v1_gpu v1_e8 > $R/raw/v1_gpu.txt 2>&1
echo "v1_gpu rc=$? $(grep -c 'ok ' $R/raw/v1_gpu.txt) runs  $(date -u +%T)"
bash $E/b0_stages.sh $R/b0_stages > $R/raw/b0_stages.txt 2>&1
echo "b0_stages rc=$? $(tail -1 $R/raw/b0_stages.txt)  $(date -u +%T)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py eval $R/v3_rd > $R/raw/rdeval.txt 2>&1
echo "rdeval rc=$?  $(date -u +%T)"
echo "--- test split, read once"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v1_e8/implementation/eval_e8.py $R/v1_e8 --test > $R/raw/e8eval_test.txt 2>&1
echo "e8 test rc=$?  $(date -u +%T)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py eval $R/v3_rd --test > $R/raw/rdeval_test.txt 2>&1
echo "rd test rc=$?  $(date -u +%T)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v2_evt/implementation/evt_sim.py $R/v2_evt --test > $R/raw/evt_test.txt 2>&1
echo "evt test rc=$?  $(date -u +%T)"
echo "=== MA_FINISH_DONE $(date -u +%FT%TZ) ==="
