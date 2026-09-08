#!/bin/bash
# v49: attribute the refine stage's 9%, and retry the early exit with a bound
# that is not the loosest one available.
#
# Four things, kept apart:
#   JHQ_REFINE_BLOCK=0 vs unset   the hard-coded 256 against JHQ_BLOCK
#   v49_base vs v47_exp           the Br template alone (same launch width)
#   v49_e8/e16/e32                the suffix-minimum exit, three granularities
#   JHQ_RESID_LUT=0 vs 1          the materialised residual table, which has
#                                 only ever been checked for numerical
#                                 agreement (1.192e-07), never for speed
set -u
exec 9>/root/.lock_v49; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v49_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v49.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target \
  demo_jhq_v49_base demo_jhq_v49_e8 demo_jhq_v49_e16 demo_jhq_v49_e32 demo_jhq_v47_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -30 >>$L; say "=== V49""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"
go(){ # bin paths M Br nlist nt np tag extra
  local C=$CACHE/${1}_${4}_${5}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${9:-JHQ_UNUSED=1} timeout 9000 \
      build/$1 $2 $3 8 $4 100.0 10 $5 $7 8 1024 "" 3 >/tmp/v49.$$ 2>/tmp/v49e.$$
  local rc=$?
  printf "  %-14s %-9s Br=%-3s np=%-5s %-22s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "$8" "$4" "$7" "${9:-default}" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v49.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v49.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/v49e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v49.$$ /tmp/v49e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"

say "########## 1. refine width: 256 (as it was) vs JHQ_BLOCK ##########"
for np in 32 128 512; do
  go demo_jhq_v49_base "$VG" 96  8 4096  159744  $np vogue  JHQ_REFINE_BLOCK=0
  go demo_jhq_v49_base "$VG" 96  8 4096  159744  $np vogue
  go demo_jhq_v49_base "$ST" 128 8 32768 1277952 $np stella JHQ_REFINE_BLOCK=0
  go demo_jhq_v49_base "$ST" 128 8 32768 1277952 $np stella
done
say "########## 2. the Br template alone: v47 vs v49_base at 256 ##########"
for np in 32 128 512; do
  go demo_jhq_v47_exp  "$VG" 96  8 4096  159744  $np vogue
  go demo_jhq_v49_base "$VG" 96  8 4096  159744  $np vogue  JHQ_REFINE_BLOCK=0
  go demo_jhq_v47_exp  "$ST" 128 8 32768 1277952 $np stella
  go demo_jhq_v49_base "$ST" 128 8 32768 1277952 $np stella JHQ_REFINE_BLOCK=0
  go demo_jhq_v47_exp  "$VG" 96  4 4096  159744  $np vogue-br4
  go demo_jhq_v49_base "$VG" 96  4 4096  159744  $np vogue-br4 JHQ_REFINE_BLOCK=0
done
say "########## 3. the suffix-minimum exit ##########"
for np in 32 128 512; do
  for b in v49_base v49_e8 v49_e16 v49_e32; do
    go demo_jhq_$b "$VG" 96  8 4096  159744  $np vogue
    go demo_jhq_$b "$BG" 128 8 32768 1277952 $np bge
    go demo_jhq_$b "$ST" 128 8 32768 1277952 $np stella
  done
done
say "########## 4. the materialised residual table, timed at last ##########"
for np in 32 128; do
  go demo_jhq_v49_base "$VG" 96  8 4096 159744 $np vogue JHQ_RESID_LUT=0
  go demo_jhq_v49_base "$VG" 96  8 4096 159744 $np vogue JHQ_RESID_LUT=1
  go demo_jhq_v49_base "$VG" 96  4 4096 159744 $np vogue-br4 JHQ_RESID_LUT=0
  go demo_jhq_v49_base "$VG" 96  4 4096 159744 $np vogue-br4 JHQ_RESID_LUT=1
done
say "=== V49""_DONE ==="
