#!/bin/bash
# The cuVS baselines again, on a protocol that matches what JHQ's number
# measures.
#
# JHQ's timed region is idx.search(host_queries, ...): the queries go up, the
# search runs, the ids and distances come back. cuVS's was the kernel alone --
# xq_d already resident, cp.asnumpy() after the clock stopped. On a 50 ms
# search that gap is under 1%. On CAGRA at 618,859 QPS the whole call is 1.6 ms
# and a 4 MB query upload is 10% of it, every bit of it in cuVS's favour.
#
# Both timings are recorded in the params column of every row; qps_mean carries
# the matching one. All six datasets, all three baselines.
set -u
exec 9>/root/.lock_cuvs; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
L=${LOG:-/root/cuvs.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
OUT=${OUT:-/root/cuvs_rerun}; mkdir -p $OUT
for ds in vogue-768 arxiv-768 bge-m3 stella-trec24 openai3-1536 openai3-3072; do
  for m in cagra cagra-int8 ivfpq; do
    say "--- $ds $m ---"
    CUVS_TIMING=xfer timeout 5400 python3 scripts/bench_all.py \
        --method $m --dataset $ds --out $OUT/${ds}_${m}.csv >>$L 2>&1
    say "    rc=$? rows=$(grep -c . $OUT/${ds}_${m}.csv 2>/dev/null || echo 0)"
  done
done
say "=== CUVS""_DONE ==="
