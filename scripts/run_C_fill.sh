#!/bin/bash
# Work that does not wait on cuVS, so the GPU is not idle during the install.
#   C1  alpha past 64 on Vogue -- arXiv was still climbing at 100, Vogue was
#       only ever measured to 64
#   C2  latency against batch size. batch_size is fixed at construction, so
#       each point is a fresh index; that is why this had not been run.
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
LOG=$R/C.log; : > $LOG
V=/root/data/vogue-768_
A=/root/autodl-tmp/arxiv-abstracts-768/

./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs \
    96 8 4 4.0 10 1024 8 8 256 "" 5 >/dev/null 2>&1     # warm-up, discarded

one () {  # label prefix M alpha nprobe batch
  printf "%-16s M=%-4s a=%-6s np=%-4s b=%-5s " "$1" "$3" "$4" "$5" "$6" | tee -a $LOG
  timeout 1200 ./build/demo_jhq_v16_pq_primary "${2}base.fvecs" "${2}query.fvecs" \
      "${2}groundtruth.ivecs" $3 8 4 $4 10 1024 $5 8 $6 "" 5 2>&1 \
    | grep -E "^Recall@|^QPS|^Latency" | sed 's/  */ /g' | tr '\n' ' ' | tee -a $LOG
  echo | tee -a $LOG
}

echo "##### C1. alpha past 64 on Vogue #####" | tee -a $LOG
for a in 100.0 200.0; do one "vogue M=48" $V 48 $a 128 256; done
for a in 100.0;        do one "vogue M=96" $V 96 $a 128 256; done

echo "" | tee -a $LOG
echo "##### C2. latency vs batch size (fresh index per point) #####" | tee -a $LOG
for b in 1 8 32 128 256 1024; do one "vogue b=$b" $V 96 4.0 8 $b; done
for b in 1 32 256;             do one "arxiv b=$b" $A 96 4.0 8 $b; done

echo "" | tee -a $LOG
echo "=========== C DONE $(date +%H:%M:%S) ===========" | tee -a $LOG
