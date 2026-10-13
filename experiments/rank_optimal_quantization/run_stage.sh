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
    iter2)
      # the four round-2 variants of every workload, built in parallel (CPU), then evaluated (calibration + validation)
      pids=""
      for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
        for v in jhq_re lloydmax median mean; do
          OMP_NUM_THREADS=8 python3 $E/quantize.py $tag $v > $R/raw/quantize_${tag}_$v.txt 2>&1 &
          pids="$pids $!"
        done
      done
      rc=0; for p in $pids; do wait $p || rc=1; done; echo "iter2 quantize rc=$rc"
      python3 $E/eval_variants.py $R iter2 jhq_re lloydmax median mean > $R/raw/eval_iter2.txt 2>&1
      echo "iter2 eval rc=$? $(tail -1 $R/raw/eval_iter2.txt | cut -c1-300)" ;;
    rankerr) python3 $E/rank_error.py $R median lloydmax > $R/raw/rank_error.txt 2>&1; echo "rankerr rc=$? $(tail -1 $R/raw/rank_error.txt | cut -c1-300)" ;;
    iter3)
      # rank-loss thresholds (GPU training on database pseudo-queries), then the four learned variants
      exec 8>/root/.gpu_lock; flock 8
      for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
        python3 $E/rankloss.py $tag $R > $R/raw/rankloss_$tag.txt 2>&1; echo "rankloss $tag rc=$?"
      done
      flock -u 8
      pids=""
      for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
        for v in learned_rl_t_zero learned_rl_t_median learned_rl_tl_zero learned_rl_tl_median; do
          OMP_NUM_THREADS=8 python3 $E/quantize.py $tag $v > $R/raw/quantize_${tag}_$v.txt 2>&1 &
          pids="$pids $!"
        done
      done
      rc=0; for p in $pids; do wait $p || rc=1; done; echo "iter3 quantize rc=$rc"
      python3 $E/eval_variants.py $R iter3 median learned_rl_t_zero learned_rl_t_median learned_rl_tl_zero learned_rl_tl_median > $R/raw/eval_iter3.txt 2>&1
      echo "iter3 eval rc=$? $(tail -1 $R/raw/eval_iter3.txt | cut -c1-300)" ;;
    iter4)
      pids=""
      for tag in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
        for v in list_mean list_median; do
          OMP_NUM_THREADS=8 python3 $E/quantize.py $tag $v > $R/raw/quantize_${tag}_$v.txt 2>&1 &
          pids="$pids $!"
        done
      done
      rc=0; for p in $pids; do wait $p || rc=1; done; echo "iter4 quantize rc=$rc"
      python3 $E/eval_variants.py $R iter4 median list_mean list_median > $R/raw/eval_iter4.txt 2>&1
      echo "iter4 eval rc=$? $(tail -1 $R/raw/eval_iter4.txt | cut -c1-300)" ;;
    lmmse) python3 $E/diag_lmmse.py $R median > $R/raw/lmmse.txt 2>&1; echo "lmmse rc=$? $(tail -1 $R/raw/lmmse.txt | cut -c1-200)" ;;
    confirm=*)
      # the final, single read of the test split, for the variants named after "confirm=" (comma separated)
      vs="${st#confirm=}"
      python3 $E/eval_variants.py $R final ${vs//,/ } --test > $R/raw/eval_final_test.txt 2>&1
      echo "confirm rc=$? $(tail -1 $R/raw/eval_final_test.txt | cut -c1-300)" ;;
    *) echo "unknown stage $st" ;;
  esac
  echo "stage $st took $(( $(date +%s) - t0 )) s"
done
echo "=== ROQ""_STAGE_DONE $(date -u +%FT%TZ) ==="
