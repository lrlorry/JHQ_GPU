#!/bin/bash
# The probe cursor, on and off, at the nprobe where it should matter most.
#
# The wasted work is about nprobe/2 boundary tests per candidate, so the effect
# has to grow with nprobe. If it does not, the instructions were hidden behind
# the gather's memory latency and the 30% instruction share was never 30% of
# the time -- which is the thing this run is for.
#
# Recall must be identical to the last digit: same candidates, same distances,
# same order.
set -u
exec 9>/root/.lock_v51; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v51_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v51.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v51_off demo_jhq_v51_on >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== V51""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # bin paths M nlist nt np blk tag
  local C=$CACHE/${8}_${4}_${5}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$7 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/$1 $2 $3 8 8 100.0 10 $4 $6 8 1024 "" 3 >/tmp/v51.$$ 2>/tmp/v51e.$$
  local rc=$?
  printf "  %-16s %-14s nlist=%-7s np=%-5s blk=%-5s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "$8" "$4" "$6" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v51.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v51.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/v51e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v51.$$ /tmp/v51e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

# nprobe is swept wide because the waste is proportional to it.
say "########## probe cursor off vs on, nprobe 8 -> 1024 ##########"
for np in 8 32 128 512 1024; do
  for b in v51_off v51_on; do
    go demo_jhq_$b "$VG"  96  4096  159744  $np 512  vogue
    go demo_jhq_$b "$BG"  128 32768 1277952 $np 512  bge
    go demo_jhq_$b "$ST"  128 32768 1277952 $np 512  stella
    go demo_jhq_$b "$O30" 384 4096  159744  $np 1024 openai3-3072
  done
done
say "=== V51""_DONE ==="
