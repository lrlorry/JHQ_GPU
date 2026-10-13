#!/bin/bash
# The single read of the test split (400 queries per workload), after everything else (the study's lock, blocking):
# V1 E8 against the paper index and its re-implementation, V3 allocations against uniform, V2 policies against the
# nprobe cut. What is compared was fixed on calibration + validation (EXPERIMENT_LOG.md, "最终确认").
#   /root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation experiments/math_ablation/confirm_test.sh ma_confirm.log
set -u
exec 9>/root/.lock_ma; flock 9
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/math_ablation; R=results/math_ablation
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) test confirmation ==="
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v1_e8/implementation/eval_e8.py $R/v1_e8 --test > $R/raw/e8eval_test.txt 2>&1
echo "e8 test rc=$?"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py eval $R/v3_rd --test > $R/raw/rdeval_test.txt 2>&1
echo "rd test rc=$?"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v2_evt/implementation/evt_sim.py $R/v2_evt --test > $R/raw/evt_test.txt 2>&1
echo "evt test rc=$?"
echo "=== MA_CONFIRM_DONE $(date -u +%FT%TZ) ==="
