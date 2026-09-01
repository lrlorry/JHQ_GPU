#!/bin/bash
# Everything v16 still owed, in the order that decides the most:
#   A. residual codebook layout (gap #3) -- the one ablation never run
#   B. v16 vs v15 across datasets -- does the PQ primary hold up beyond Vogue
#   C. v16 M-sweep on arXiv -- confirm the Vogue frontier generalises
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
LOG=$R/all.log; : > $LOG
V=/root/data/vogue-768_
A=/root/autodl-tmp/arxiv-abstracts-768/
O15=/root/autodl-tmp/openai3-1536/
S=/root/autodl-tmp/stella-trec24/

# Clocks cannot be locked in this container; one discarded pass so the first
# measured run does not pay the ramp.
./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs \
    96 8 4 4.0 10 1024 8 8 256 "" 5 >/dev/null 2>&1

one () {  # bin ds prefix M alpha nprobe km
  printf "%-22s %-14s M=%-4s a=%-5s np=%-4s km=%-2s " "$1" "$2" "$4" "$5" "$6" "$7" | tee -a $LOG
  timeout 900 ./build/demo_jhq_$1 "${3}base.fvecs" "${3}query.fvecs" "${3}groundtruth.ivecs" \
      $4 8 4 $5 10 1024 $6 8 256 "" $7 2>&1 \
    | grep -E "^Recall@|^QPS" | sed 's/  */ /g' | tr '\n' ' ' | tee -a $LOG
  echo | tee -a $LOG
}

echo "##### A. gap#3: per-subspace vs global residual codebook #####" | tee -a $LOG
for spec in "vogue $V" "arxiv $A"; do
  set -- $spec
  for M in 192 96 48; do
    one v16_pq_primary "$1 per-sub" $2 $M 4.0 128 5
    one v16_globalres  "$1 global"  $2 $M 4.0 128 5
  done
done

echo "" | tee -a $LOG
echo "##### B. v16 vs v15 across datasets (M=d/8, alpha=4) #####" | tee -a $LOG
one v15_eval_fix  "vogue v15"  $V   96  4.0 128 -
one v16_pq_primary "vogue v16"  $V   96  4.0 128 5
one v15_eval_fix  "arxiv v15"  $A   96  4.0 128 -
one v16_pq_primary "arxiv v16"  $A   96  4.0 128 5
one v15_eval_fix  "oai1536 v15" $O15 192 4.0 128 -
one v16_pq_primary "oai1536 v16" $O15 192 4.0 128 5
one v15_eval_fix  "stella v15"  $S   128 4.0 128 -
one v16_pq_primary "stella v16"  $S   128 4.0 128 5

echo "" | tee -a $LOG
echo "##### C. arXiv M-sweep: does the Vogue frontier generalise? #####" | tee -a $LOG
for M in 192 96 48 24; do one v16_pq_primary "arxiv" $A $M 4.0  128 5; done
for M in 96 48; do        one v16_pq_primary "arxiv" $A $M 64.0 128 5; done

echo "" | tee -a $LOG
echo "=========== ALL DONE $(date +%H:%M:%S) ===========" | tee -a $LOG
