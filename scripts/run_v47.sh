#!/bin/bash
# v47 against v46: the factorised LUT.
#
# The identity is exact in real arithmetic but not bit-for-bit -- the two
# halves are summed separately -- so recall is expected to agree to about 1e-4
# and not beyond. cand and ivf_recall must be identical, since neither depends
# on the primary distance.
set -u
exec 9>/root/.lock_v47; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE_ROOT=${CACHE_ROOT:-/root/jhq_cache_by_bin}; mkdir -p $CACHE_ROOT
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v47.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_exp demo_jhq_v47_diag demo_jhq_v46_exp demo_jhq_v46_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -30 >> $L; say "=== V47""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # bin paths M nlist nprobe ntrain diag tag alpha
  mkdir -p $CACHE_ROOT/$1
  env $E JHQ_INDEX_CACHE=$CACHE_ROOT/$1 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${7:+JHQ_DIAG=1} timeout 5000 \
      build/$1 $2 $3 8 8 ${9:-100.0} 10 $4 $5 8 1024 "" 3 >/tmp/v47.$$ 2>/tmp/v47e.$$
  local rc=$?
  printf "  %-18s %-8s nlist=%-7s np=%-4s nt=%-8s a=%-6s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s rc=%s\n" \
    "$1" "$8" "$4" "$5" "$6" "${9:-100.0}" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v47.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v47.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/v47.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/v47.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/v47e.$$ | sed 's/^/      ! /' >> $L
  rm -f /tmp/v47.$$ /tmp/v47e.$$
}

say "########## 1. same answers? ##########"
for np in 32 128 256; do
  run demo_jhq_v46_diag "$VG" 96  1024  $np 100000 1 vogue
  run demo_jhq_v47_diag "$VG" 96  1024  $np 100000 1 vogue
  run demo_jhq_v46_diag "$ST" 128 16384 $np 100000 1 stella
  run demo_jhq_v47_diag "$ST" 128 16384 $np 100000 1 stella
done
say "########## 2. speed, three datasets, two alphas ##########"
for a in 100.0 8.0; do
  for np in 8 32 128 256; do
    for b in v46_exp v47_exp; do
      run demo_jhq_$b "$VG" 96  1024  $np 100000 "" vogue $a
      run demo_jhq_$b "$BG" 128 8192  $np 100000 "" bge $a
      run demo_jhq_$b "$ST" 128 16384 $np 100000 "" stella $a
    done
  done
done
say "########## 3. at the configuration the new front uses ##########"
for np in 8 32 128 256; do
  run demo_jhq_v46_exp "$ST" 128 32768 $np 1300000 "" stella
  run demo_jhq_v47_exp "$ST" 128 32768 $np 1300000 "" stella
done
say "=== V47""_DONE ==="
