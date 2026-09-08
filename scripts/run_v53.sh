#!/bin/bash
# How much does a bigger selection buffer buy?
#
# x1 is what v45 through v52 have been running. x2 and x4 grow cap while it and
# the table still fit shared memory, which makes the bitonic sort fire eighty
# and six thousand times less often -- each one twice or four times as long.
# Whether that trade pays is the question.
#
# Identical selection, so recall must match x1 to the digit.
#
# The same run finally times the v50 regroup and fusion, written earlier and
# never measured.
set -u
exec 9>/root/.lock_v53; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v53_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v53.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v53_x1 demo_jhq_v53_x2 demo_jhq_v53_x4 \
  demo_jhq_v50_base demo_jhq_v50_group demo_jhq_v50_fuse demo_jhq_v50_both >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L|head -20 >>$L; say "=== V53""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # bin paths M nlist nt np blk tag
  local C=$CACHE/${8}_${4}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$7 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/$1 $2 $3 8 8 100.0 10 $4 $6 8 1024 "" 3 >/tmp/v53.$$ 2>/tmp/v53e.$$
  local rc=$?
  printf "  %-16s %-14s nlist=%-7s np=%-5s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "$8" "$4" "$6" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v53.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v53.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/v53e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v53.$$ /tmp/v53e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
say "########## 1. selection buffer size ##########"
for np in 128 512 1024; do
  for b in v53_x1 v53_x2 v53_x4; do
    go demo_jhq_$b "$VG"  96  4096  159744  $np 512  vogue
    go demo_jhq_$b "$BG"  128 32768 1277952 $np 512  bge
    go demo_jhq_$b "$ST"  128 32768 1277952 $np 512  stella
    go demo_jhq_$b "$O30" 384 4096  159744  $np 1024 openai3-3072
  done
done
say "########## 2. regroup and fusion, finally measured ##########"
for np in 128 512; do
  for b in v50_base v50_group v50_fuse v50_both; do
    go demo_jhq_$b "$VG" 96  4096  159744  $np 512 vogue
    go demo_jhq_$b "$ST" 128 32768 1277952 $np 512 stella
  done
done
say "=== V53""_DONE ==="
