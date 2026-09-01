#!/bin/bash
# A2, which run_A_batch.sh defined but never called: FAISS IVF-PQ at the code
# lengths GpuIndexIVFPQ refuses. Fills the equal-total-bytes column -- JHQ at
# M=48 on 768-d occupies 48 + 384 + 8 = 440 B/vector, so comparing it against
# FAISS at 48 B flatters it. Only recall is used from this run.
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:$PATH
R=results/v16; mkdir -p $R
P=/root/miniconda3/bin/python3
$P scripts/bench_faiss_cpu_largeM.py --name vogue-768 \
   --base /root/data/vogue-768_base.fvecs --query /root/data/vogue-768_query.fvecs \
   --gt /root/data/vogue-768_groundtruth.ivecs --ms 192,384 --nprobes 8,32,128 \
   --out $R/faiss_cpu_largeM_vogue.csv 2>&1 | tee $R/faiss_cpu_vogue.log
$P scripts/bench_faiss_cpu_largeM.py --name arxiv-768 \
   --base /root/autodl-tmp/arxiv-abstracts-768/base.fvecs \
   --query /root/autodl-tmp/arxiv-abstracts-768/query.fvecs \
   --gt /root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs --ms 192,384 \
   --nprobes 8,32,128 --out $R/faiss_cpu_largeM_arxiv.csv 2>&1 | tee $R/faiss_cpu_arxiv.log
echo "=========== A2 DONE $(date +%H:%M:%S) ==========="
