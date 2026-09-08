#!/bin/bash
# nlist against the throughput gap.
#
# At matched recall JHQ trails int8 CAGRA by 2.5x on stella, and the reason is
# candidates touched: an IVF scan reads N*nprobe/nlist of them where a graph
# walks a few thousand. More, smaller lists is the one knob that reduces that
# without changing the method.
#
# nlist=65536 failed in results/pending/: n_train is 100,000 and 65536
# centroids leaves 1.5 training points each, against the ~39 FAISS wants. So
# the training set has to grow with nlist, which is what this sweeps.
set -u
exec 9>/root/.lock_nlist; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}
L=${LOG:-/root/nlist.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
# The fetch fails often enough -- GnuTLS, or the proxy not taking -- that
# letting it fail quietly means the box runs an older commit and produces
# numbers nothing explains. It cost one launch of this very script, which ran
# eight commits behind and could not find itself. Abort instead.
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v39_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== NLIST""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # paths M nlist nprobe n_train tag
  env $E JHQ_TILE_M_RT=$2 JHQ_N_TRAIN=$5 timeout 4000 \
      build/demo_jhq_v39_exp $1 $2 8 8 100.0 10 $3 $4 8 1024 "" 5 >/tmp/n.$$ 2>/dev/null
  printf "  %-8s nlist=%-6s np=%-4s n_train=%-9s cand=%-9s recall=%-8s qps=%-9s train=%-9s add=%s\n" \
    "$6" "$3" "$4" "$5" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/n.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/n.$$)" \
    "$(awk '/^  train:/{print $2;exit}' /tmp/n.$$)" \
    "$(awk '/^  add:/{print $2;exit}' /tmp/n.$$)" >> $L; rm -f /tmp/n.$$; }

# 39 training points per centroid is the FAISS convention; take that as the floor
# and keep 100K where it already suffices, so the training set is the only thing
# that moves with nlist.
say "########## stella 17.8M, M=128 ##########"
for spec in "8192 100000" "16384 100000" "32768 1300000" "65536 2600000" "131072 5200000"; do
  set -- $spec; nl=$1; nt=$2
  for np in 8 32 128 256; do
    run "$ST" 128 $nl $np $nt stella $((17776615*np/nl))
  done
done
say "########## bge-m3 10.1M, M=128 ##########"
for spec in "8192 100000" "32768 1300000" "65536 2600000"; do
  set -- $spec; nl=$1; nt=$2
  for np in 8 32 128 256; do
    run "$BG" 128 $nl $np $nt bge $((10091524*np/nl))
  done
done
say "=== NLIST""_DONE ==="
