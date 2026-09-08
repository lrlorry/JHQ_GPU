#!/bin/bash
# v44 against v43: same probes, and what nlist can now reach.
#
# Correctness first. The probe set is exactly what cand_mean and ivf_recall
# measure, so running both binaries with JHQ_DIAG=1 on one configuration and
# comparing those two numbers is a direct check that the streaming selection
# picks the same lists as the nprobe-pass argmin it replaces. Recall is the
# weaker check and is reported beside it.
#
# Then the point of the rewrite: nlist=32768/65536/131072, which the old kernel
# could not launch.
set -u
exec 9>/root/.lock_v44; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v44.log}; : > $L
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
cmake --build build -j 16 --target demo_jhq_v44_exp demo_jhq_v44_diag demo_jhq_v43_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -30 >> $L; say "=== V44""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # bin paths M nlist nprobe ntrain diag tag
  env $E JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${7:+JHQ_DIAG=1} timeout 4000 \
      build/$1 $2 $3 8 8 100.0 10 $4 $5 8 1024 "" 3 >/tmp/v44.$$ 2>/tmp/v44e.$$
  local rc=$?
  printf "  %-18s %-8s nlist=%-7s np=%-4s nt=%-8s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s rc=%s\n" \
    "$1" "$8" "$4" "$5" "$6" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v44.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v44.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/v44.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/v44.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/v44e.$$ | sed 's/^/      ! /' >> $L
  rm -f /tmp/v44.$$ /tmp/v44e.$$
}

say "########## 1. v43 vs v44: same probes? ##########"
for np in 8 32 128; do
  run demo_jhq_v43_exp  "$VG" 96  1024  $np 100000 1 vogue
  run demo_jhq_v44_diag "$VG" 96  1024  $np 100000 1 vogue
  run demo_jhq_v43_exp  "$BG" 128 8192  $np 100000 1 bge
  run demo_jhq_v44_diag "$BG" 128 8192  $np 100000 1 bge
  run demo_jhq_v43_exp  "$ST" 128 16384 $np 100000 1 stella
  run demo_jhq_v44_diag "$ST" 128 16384 $np 100000 1 stella
done

say "########## 2. probe selection speed, no diag ##########"
for np in 8 32 128 256; do
  run demo_jhq_v43_exp "$ST" 128 16384 $np 100000 "" stella
  run demo_jhq_v44_exp "$ST" 128 16384 $np 100000 "" stella
done

say "########## 3. nlist the old kernel could not launch ##########"
# ~39 training points per centroid, the FAISS convention.
for spec in "16384 640000" "32768 1300000" "65536 2600000" "131072 5200000"; do
  set -- $spec; nl=$1; nt=$2
  for np in 8 32 128 256; do
    run demo_jhq_v44_diag "$ST" 128 $nl $np $nt 1 stella
  done
done
for spec in "8192 320000" "32768 1300000" "65536 2600000"; do
  set -- $spec; nl=$1; nt=$2
  for np in 8 32 128 256; do
    run demo_jhq_v44_diag "$BG" 128 $nl $np $nt 1 bge
  done
done
say "=== V44""_DONE ==="
