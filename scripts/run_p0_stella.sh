#!/bin/bash
# The stella-trec24 baselines, retried with the single-allocation fvecs reader.
# The first attempt was killed by the host OOM killer: the old reader
# materialised the 67 GiB file twice before cuVS asked for anything.
exec 9>/root/.lock_stella
flock -n 9 || { echo "another stella run holds the lock"; exit 0; }
exec >/root/p0_stella.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_p0_gap.sh" >/dev/null; do sleep 30; done
export JHQ_INDEX_CACHE=/root/jhq_cache
R=results/final
B="python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384"
free -g | awk '/^Mem/{print "host mem before: "$3" used of "$2" GB"}'
echo "###### IVF-PQ"
$B --method ivfpq --pq-dims 128,256 --nprobe 8,32,128,256 --out $R/p0_stella-trec24_ivfpq.csv
free -g | awk '/^Mem/{print "host mem: "$3" used of "$2" GB"}'
echo "###### CAGRA int8"
$B --method cagra-int8 --graph-degrees 32 --itopk 64,128,256 --search-width 1 \
   --out $R/p0_stella-trec24_cagra_int8.csv
free -g | awk '/^Mem/{print "host mem: "$3" used of "$2" GB"}'
echo "###### CAGRA fp32 -- 17.8M x 1024 fp32 is 72.8 GiB against a 31.4 GiB card,"
echo "       so this is expected to fail; the point is to record it from a run."
$B --method cagra --graph-degrees 32 --itopk 64 --search-width 1 \
   --out $R/p0_stella-trec24_cagra.csv
echo "### P0_STELLA_DONE"
