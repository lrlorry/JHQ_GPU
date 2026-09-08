#!/bin/bash
# v46: is the collapse the int overflow, and what does a properly trained
# coarse quantizer actually buy?
#
# 1. The threshold is sharp and falsifiable. n * d_ wraps at 2^31, so on
#    d=1024 the last good training size is 2,097,152 rows. v45 must break
#    between 2,000,000 and 2,200,000 and v46 must not.
# 2. With sigma fixed, re-run the nlist ladder that could not be read before.
# 3. The training-set ladder at fixed nlist, which is where the QPS was.
set -u
exec 9>/root/.lock_v46; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
# One cache per binary. cache_path() keys on the data and the parameters, not
# on the code, so a version that changes what training produces gets a hit on
# its predecessor's state and the fix never runs. That is exactly what happened
# on the first attempt: v46 loaded v45's sigma=0 codebook and reported the same
# 0.477, which reads as "the fix does not work".
CACHE_ROOT=${CACHE_ROOT:-/root/autodl-tmp/jhq_cache_by_bin}
mkdir -p $CACHE_ROOT
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v46.log}; : > $L
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
cmake --build build -j 16 --target demo_jhq_v46_exp demo_jhq_v46_diag demo_jhq_v45_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -30 >> $L; say "=== V46""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # bin paths M nlist nprobe ntrain diag tag
  mkdir -p $CACHE_ROOT/$1
  env $E JHQ_INDEX_CACHE=$CACHE_ROOT/$1 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${7:+JHQ_DIAG=1} timeout 5000 \
      build/$1 $2 $3 8 8 100.0 10 $4 $5 8 1024 "" 3 >/tmp/v46.$$ 2>/tmp/v46e.$$
  local rc=$?
  printf "  %-18s %-8s nlist=%-7s np=%-4s nt=%-8s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s rc=%s\n" \
    "$1" "$8" "$4" "$5" "$6" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v46.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v46.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/v46.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/v46.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/v46e.$$ | sed 's/^/      ! /' >> $L
  rm -f /tmp/v46.$$ /tmp/v46e.$$
}

say "########## 1. the 2^31 threshold: 2,097,152 rows at d=1024 ##########"
for nt in 1800000 2000000 2097152 2200000 2600000; do
  run demo_jhq_v45_diag "$ST" 128 16384 128 $nt 1 stella
  run demo_jhq_v46_diag "$ST" 128 16384 128 $nt 1 stella
done
say "########## 2. nlist with sigma fixed ##########"
for spec in "16384 640000" "32768 1300000" "65536 2600000"; do
  set -- $spec; nl=$1; nt=$2
  for np in 8 32 128 256; do
    run demo_jhq_v46_diag "$ST" 128 $nl $np $nt 1 stella
  done
done
say "########## 3. the coarse training ladder at nlist=16384 ##########"
for nt in 100000 320000 640000 1300000 2600000; do
  for np in 32 128 256; do
    run demo_jhq_v46_diag "$ST" 128 16384 $np $nt 1 stella
  done
done
say "########## 4. bge, same two questions ##########"
for nt in 100000 640000 1300000; do
  for np in 32 128 256; do
    run demo_jhq_v46_diag "$BG" 128 8192 $np $nt 1 bge
  done
done
say "=== V46""_DONE ==="
