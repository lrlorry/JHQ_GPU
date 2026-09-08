#!/bin/bash
# v48: the guaranteed early exit, and its one parameter.
#
# v48_off sets JHQ_EXIT_EVERY past any M, so the exit never fires; it must
# reproduce v47 bit for bit and is the control that says the restructuring
# itself cost nothing.
set -u
exec 9>/root/.lock_v48; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE_ROOT=${CACHE_ROOT:-/root/autodl-tmp/jhq_cache_by_bin}; mkdir -p $CACHE_ROOT
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v48.log}; : > $L
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
cmake --build build -j 16 --target \
  demo_jhq_v48_e8 demo_jhq_v48_e16 demo_jhq_v48_e32 demo_jhq_v48_off demo_jhq_v47_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -30 >> $L; say "=== V48""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # bin paths M nlist nprobe ntrain tag alpha
  mkdir -p $CACHE_ROOT/$1
  env $E JHQ_INDEX_CACHE=$CACHE_ROOT/$1 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 timeout 5000 \
      build/$1 $2 $3 8 8 ${8:-100.0} 10 $4 $5 8 1024 "" 3 >/tmp/v48.$$ 2>/dev/null
  printf "  %-16s %-8s nlist=%-7s np=%-4s a=%-6s recall=%-8s qps=%s\n" \
    "$1" "$7" "$4" "$5" "${8:-100.0}" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v48.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v48.$$)" >> $L
  rm -f /tmp/v48.$$
}

for a in 100.0 8.0; do
  say "########## alpha=$a ##########"
  for np in 32 128 256; do
    for b in v47_exp v48_off v48_e8 v48_e16 v48_e32; do
      run demo_jhq_$b "$VG" 96  1024  $np 100000 vogue $a
      run demo_jhq_$b "$BG" 128 8192  $np 100000 bge   $a
      run demo_jhq_$b "$ST" 128 16384 $np 100000 stella $a
    done
  done
done
say "########## at the new front's configuration ##########"
for np in 32 128 256; do
  for b in v47_exp v48_e8 v48_e16 v48_e32; do
    run demo_jhq_$b "$ST" 128 32768 $np 1300000 stella
  done
done
say "=== V48""_DONE ==="
