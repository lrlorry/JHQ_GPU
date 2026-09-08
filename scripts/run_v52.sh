#!/bin/bash
# The word layout, against the byte layout it replaces.
#
# Same codes in a different arrangement, so recall must be identical to the
# digit -- that is the first check. The second is whether the gather moves
# toward its roofline: it sits at 47-50% of 1792 GB/s now, and a full 128-byte
# transaction should take it most of the way.
#
# Swept across nprobe because the gather's share grows with it.
set -u
exec 9>/root/.lock_v52; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v52_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v52.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v52_byte demo_jhq_v52_word >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L|head -20 >>$L; say "=== V52""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # bin paths M nlist nt np blk tag
  local C=$CACHE/${1}_${8}_${4}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$7 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/$1 $2 $3 8 8 100.0 10 $4 $6 8 1024 "" 3 >/tmp/v52.$$ 2>/tmp/v52e.$$
  local rc=$?
  printf "  %-16s %-14s nlist=%-7s np=%-5s blk=%-5s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "$8" "$4" "$6" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v52.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v52.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/v52e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v52.$$ /tmp/v52e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
say "########## byte vs word, nprobe 32 -> 1024 ##########"
for np in 32 128 512 1024; do
  for b in v52_byte v52_word; do
    go demo_jhq_$b "$VG"  96  4096  159744  $np 512  vogue
    go demo_jhq_$b "$BG"  128 32768 1277952 $np 512  bge
    go demo_jhq_$b "$ST"  128 32768 1277952 $np 512  stella
    go demo_jhq_$b "$O30" 384 4096  159744  $np 1024 openai3-3072
  done
done
say "=== V52""_DONE ==="
