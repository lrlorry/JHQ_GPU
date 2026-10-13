#!/bin/bash
# Math-ablation study: named stages on the box. Offline stages use the CPU (and the data of the rank-optimal study);
# b0 measures v65 (variant -1, the scheduled pipeline) on the GPU. Logs to results/math_ablation/.
#   /root/autodl-tmp/launch_b.sh exp/math-algorithm-ablation "experiments/math_ablation/run_stage.sh e8test rdtest" ma_stage1.log
set -u
[ -z "${MA_NESTED:-}" ] && { exec 9>/root/.lock_ma; flock -n 9 || { echo "already running"; exit 0; }; }
export MA_NESTED=1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/math_ablation
R=results/math_ablation
mkdir -p $R/v1_e8 $R/v2_evt $R/v3_rd $R/b0 $R/raw /root/autodl-tmp/math_ablation
TAGS="arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4"
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) stages: $* ==="
for st in "$@"; do
  t0=$(date +%s)
  case $st in
    e8test) python3 $E/v1_e8/implementation/e8.py test $R/v1_e8 > $R/raw/e8test.txt 2>&1; echo "e8test rc=$? $(tail -1 $R/raw/e8test.txt)" ;;
    e8build)
      pids=""
      for t in $TAGS; do
        OMP_NUM_THREADS=4 python3 $E/v1_e8/implementation/e8.py build $R/v1_e8 $t > $R/raw/e8build_$t.txt 2>&1 & pids="$pids $!"
        OMP_NUM_THREADS=4 python3 $E/v1_e8/implementation/e8.py build $R/v1_e8 $t grouped > $R/raw/e8build_${t}_grouped.txt 2>&1 & pids="$pids $!"
      done
      rc=0; for p in $pids; do wait $p || rc=1; done; echo "e8build rc=$rc" ;;
    e8eval) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v1_e8/implementation/eval_e8.py $R/v1_e8 > $R/raw/e8eval.txt 2>&1; echo "e8eval rc=$? $(tail -2 $R/raw/e8eval.txt | cut -c1-400)" ;;
    rdtest) python3 $E/v3_rd/implementation/rd.py test $R/v3_rd > $R/raw/rdtest.txt 2>&1; echo "rdtest rc=$? $(tail -1 $R/raw/rdtest.txt)" ;;
    rdstats)
      for t in $TAGS; do OPENBLAS_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py stats $R/v3_rd $t > $R/raw/rdstats_$t.txt 2>&1; echo "rdstats $t rc=$?"; done ;;
    rdbuild)
      pids=""
      for t in $TAGS; do for a in uniform waterfill empirical queryweighted; do
        OMP_NUM_THREADS=4 python3 $E/v3_rd/implementation/rd.py build $R/v3_rd $t $a > $R/raw/rdbuild_${t}_$a.txt 2>&1 & pids="$pids $!"
      done; done
      rc=0; for p in $pids; do wait $p || rc=1; done; echo "rdbuild rc=$rc" ;;
    rdeval) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py eval $R/v3_rd > $R/raw/rdeval.txt 2>&1; echo "rdeval rc=$? $(tail -2 $R/raw/rdeval.txt | cut -c1-400)" ;;
    evt) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v2_evt/implementation/evt_sim.py $R/v2_evt > $R/raw/evt.txt 2>&1; echo "evt rc=$? $(tail -1 $R/raw/evt.txt | cut -c1-300)" ;;
    b0) bash $E/b0_frontier.sh $R/b0 > $R/raw/b0.txt 2>&1; echo "b0 rc=$? $(tail -1 $R/raw/b0.txt)" ;;
    offline1)
      # independent CPU work in parallel: E8 builds, residual statistics, the EVT simulation
      bash $0 e8build > $R/raw/offline1_e8.txt 2>&1 &
      p1=$!
      ( for t in $TAGS; do OPENBLAS_NUM_THREADS=1 python3 $E/v3_rd/implementation/rd.py stats $R/v3_rd $t > $R/raw/rdstats_$t.txt 2>&1; echo "rdstats $t rc=$?"; done ) > $R/raw/offline1_rd.txt 2>&1 &
      p2=$!
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/v2_evt/implementation/evt_sim.py $R/v2_evt > $R/raw/evt.txt 2>&1 &
      p3=$!
      wait $p1; echo "offline1 e8: $(grep 'rc=' $R/raw/offline1_e8.txt | tr '\n' ' ')"
      wait $p2; echo "offline1 rd: $(cat $R/raw/offline1_rd.txt | tr '\n' ' ')"
      wait $p3; echo "offline1 evt rc=$? $(tail -1 $R/raw/evt.txt | cut -c1-300)" ;;
    offline2)
      bash $0 e8eval > $R/raw/offline2_e8.txt 2>&1 &
      p1=$!
      bash $0 rdbuild rdeval > $R/raw/offline2_rd.txt 2>&1 &
      p2=$!
      wait $p1; wait $p2
      echo "offline2: $(grep -h 'rc=' $R/raw/offline2_e8.txt $R/raw/offline2_rd.txt | cut -c1-300 | tr '\n' ' ')" ;;
    *) echo "unknown stage $st" ;;
  esac
  echo "stage $st took $(( $(date +%s) - t0 )) s"
done
echo "=== MA""_STAGE_DONE $(date -u +%FT%TZ) ==="
