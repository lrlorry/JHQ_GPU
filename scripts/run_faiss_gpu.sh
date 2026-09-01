#!/bin/bash
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:$PATH
mkdir -p results/v16
P=/root/miniconda3/bin/python3
$P scripts/bench_faiss_gpu.py --name vogue-768 \
   --base /root/data/vogue-768_base.fvecs --query /root/data/vogue-768_query.fvecs \
   --gt /root/data/vogue-768_groundtruth.ivecs --ms 48,96,192,384 \
   --out results/v16/faiss_gpu_vogue.csv 2>&1 | tee results/v16/faiss_vogue.log
$P scripts/bench_faiss_gpu.py --name arxiv-768 \
   --base /root/autodl-tmp/arxiv-abstracts-768/base.fvecs \
   --query /root/autodl-tmp/arxiv-abstracts-768/query.fvecs \
   --gt /root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs --ms 48,96,192,384 \
   --out results/v16/faiss_gpu_arxiv.csv 2>&1 | tee results/v16/faiss_arxiv.log
echo "=========== FAISS DONE $(date +%H:%M:%S) ==========="
