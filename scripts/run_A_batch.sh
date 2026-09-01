#!/bin/bash
# A: fill the gaps that need no new code.
#   A1  v16 on the two datasets it has not seen
#   A2  FAISS IVF-PQ at the code lengths the GPU path refuses, on CPU. Recall
#       does not depend on which device runs the same index, so this is what
#       fills the equal-total-bytes column that GPU FAISS could not reach
#       (M=96 already needs 98304 B of shared memory against 49152 available).
#   A3  alpha sweep on arXiv, currently only alpha in {4, 64}
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
LOG=$R/A.log; : > $LOG
O3=/root/autodl-tmp/openai3-3072/
BGE=/root/autodl-tmp/bge-m3/
A=/root/autodl-tmp/arxiv-abstracts-768/
V=/root/data/vogue-768_

./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs \
    96 8 4 4.0 10 1024 8 8 256 "" 5 >/dev/null 2>&1   # warm-up, discarded

one () {  # bin label prefix M alpha nprobe km
  printf "%-20s %-14s M=%-4s a=%-5s np=%-4s " "$1" "$2" "$4" "$5" "$6" | tee -a $LOG
  timeout 1800 ./build/demo_jhq_$1 "${3}base.fvecs" "${3}query.fvecs" "${3}groundtruth.ivecs" \
      $4 8 4 $5 10 1024 $6 8 256 "" $7 2>&1 \
    | grep -E "^Recall@|^QPS" | sed 's/  */ /g' | tr '\n' ' ' | tee -a $LOG
  echo | tee -a $LOG
}

echo "##### A1. v16 vs v15 on the remaining two datasets #####" | tee -a $LOG
one v15_eval_fix   "oai3072 v15" $O3  384 4.0 128 -
one v16_pq_primary "oai3072 v16" $O3  384 4.0 128 5
one v15_eval_fix   "bgem3 v15"   $BGE 128 4.0 128 -
one v16_pq_primary "bgem3 v16"   $BGE 128 4.0 128 5

echo "" | tee -a $LOG
echo "##### A3. arXiv alpha sweep at M=48 and M=96 #####" | tee -a $LOG
for a in 8.0 16.0 32.0 100.0; do one v16_pq_primary "arxiv" $A 48 $a 128 5; done
for a in 8.0 16.0 32.0;        do one v16_pq_primary "arxiv" $A 96 $a 128 5; done

echo "" | tee -a $LOG
echo "=========== A DONE $(date +%H:%M:%S) ===========" | tee -a $LOG
