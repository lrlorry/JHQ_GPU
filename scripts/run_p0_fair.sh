#!/bin/bash
# Give IVF-PQ a training fraction it can fit before reporting that it cannot
# build at 17.8M vectors. Its failure was allocating 36.4 GiB, which is half the
# dataset in fp32 -- cuVS's default kmeans_trainset_fraction of 0.5 -- not the
# index itself, whose payload at pq_dim=128 is about 2.3 GiB.
exec 9>/root/.lock_fair
flock -n 9 || { echo "another run holds the lock"; exit 0; }
exec >/root/p0_fair.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=/root/jhq_cache
R=results/final
for frac in 0.05 0.02; do
  echo "###### stella-trec24, IVF-PQ, kmeans_trainset_fraction=$frac"
  python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 \
    --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
    --trainset-fraction $frac --out $R/fair_stella_ivfpq_f${frac}.csv
done
echo "###### bge-m3, IVF-PQ at the same fractions, for the same reason"
for frac in 0.10 0.05; do
  python3 scripts/bench_all.py --dataset bge-m3 --reps 3 --nlist 8192 \
    --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
    --trainset-fraction $frac --out $R/fair_bge_ivfpq_f${frac}.csv
done
echo "### P0_FAIR_DONE"
