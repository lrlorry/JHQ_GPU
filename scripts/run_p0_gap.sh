#!/bin/bash
# The four cells the six-by-five grid is still missing: every baseline on
# stella-trec24, and fp32 CAGRA on bge-m3. The last is expected to run out of
# memory -- 10.1M x 1024 fp32 is 41.3 GiB against a 31.4 GiB card -- but that
# has to be recorded from a run rather than inferred from the arithmetic.
exec 9>/root/.lock_gap
flock -n 9 || { echo "another run holds the lock"; exit 0; }
exec >/root/p0_gap.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=/root/jhq_cache; mkdir -p /root/jhq_cache
R=results/final
nvidia-smi --query-gpu=memory.total --format=csv,noheader

echo "############ stella-trec24, 17.8M x 1024 -- the three missing baselines"
B="python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384"
$B --method ivfpq --pq-dims 128,256,512 --nprobe 8,32,128,256 --out $R/p0_stella-trec24_ivfpq.csv
$B --method cagra-int8 --graph-degrees 32 --itopk 64,128,256,512 --search-width 1 --out $R/p0_stella-trec24_cagra_int8.csv
$B --method cagra --graph-degrees 32 --itopk 64,128 --search-width 1 --out $R/p0_stella-trec24_cagra.csv

echo "############ bge-m3, fp32 CAGRA -- measured, not assumed"
python3 scripts/bench_all.py --dataset bge-m3 --reps 3 --nlist 8192 \
  --method cagra --graph-degrees 32 --itopk 64,128 --search-width 1 --out $R/p0_bge-m3_cagra.csv

echo "############ JHQ Br=8 on stella, so both residual settings exist everywhere"
python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 \
  --method jhq --binary demo_jhq_v22_s2b1 --M 128 --Br 8 --nprobe 32,128,256 \
  --out $R/p0_stella-trec24_jhq_Br8.csv
echo "### P0_GAP_DONE"
