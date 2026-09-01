#!/bin/bash
# The one number the paper is currently missing. Everything else in batch B can
# wait behind it.
#
# Equal stored bytes says JHQ has no recall advantage: IVF-PQ at 384 B reaches
# 0.9269 on Vogue and 0.9516 on arXiv against JHQ's 0.9182 / 0.9436 at 440 B.
# The claim that survives is about the hot path rather than the footprint --
# JHQ reads its 48 B primary code for every candidate and the 384 B residual
# for only ck of them, roughly 49 B per candidate against IVF-PQ's full 384.
# If that 7.8x shows up as throughput at matched recall, the design holds; if
# it does not, "stores more, reads less" is not a claim either.
#
# So: cuVS IVF-PQ at pq_dim 384, the configuration GpuIndexIVFPQ cannot reach
# because its lookup table sits in shared memory. CAGRA and HNSW follow.
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
P=/root/miniconda3/bin/python3

for spec in "vogue-768 /root/data/vogue-768_" \
            "arxiv-768 /root/autodl-tmp/arxiv-abstracts-768/"; do
  set -- $spec; name=$1; p=$2
  echo ""
  echo "########## $name — cuVS IVF-PQ (the matched-bytes throughput question) ##########"
  $P scripts/bench_cuvs_hnsw.py --name $name --which ivfpq \
     --base ${p}base.fvecs --query ${p}query.fvecs --gt ${p}groundtruth.ivecs \
     --out $R/cuvs_ivfpq_$name.csv 2>&1 | tee $R/cuvs_ivfpq_$name.log
done
echo "=========== B-CORE DONE $(date +%H:%M:%S) ==========="

for spec in "vogue-768 /root/data/vogue-768_" \
            "arxiv-768 /root/autodl-tmp/arxiv-abstracts-768/"; do
  set -- $spec; name=$1; p=$2
  echo ""
  echo "########## $name — CAGRA + HNSW ##########"
  $P scripts/bench_cuvs_hnsw.py --name $name --which cagra,hnsw \
     --base ${p}base.fvecs --query ${p}query.fvecs --gt ${p}groundtruth.ivecs \
     --out $R/graph_$name.csv 2>&1 | tee $R/graph_$name.log
done
echo "=========== B-REST DONE $(date +%H:%M:%S) ==========="
