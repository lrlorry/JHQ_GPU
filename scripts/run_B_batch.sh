#!/bin/bash
# B: the external GPU baselines. CAGRA is the one the paper cannot omit;
# cuVS IVF-PQ is the stronger version of the FAISS comparison already run;
# HNSW is the reference point readers carry in their heads.
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
P=/root/miniconda3/bin/python3
for spec in "vogue-768 /root/data/vogue-768_" \
            "arxiv-768 /root/autodl-tmp/arxiv-abstracts-768/"; do
  set -- $spec; name=$1; p=$2
  $P scripts/bench_cuvs_hnsw.py --name $name \
     --base ${p}base.fvecs --query ${p}query.fvecs --gt ${p}groundtruth.ivecs \
     --out $R/baselines_$name.csv 2>&1 | tee $R/baselines_$name.log
done
echo "=========== B DONE $(date +%H:%M:%S) ==========="
