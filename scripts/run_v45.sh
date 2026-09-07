#!/bin/bash
# v45 against v44 and v43, and the confound the v44 run left open.
#
# 1. v45 must return exactly what v44 does -- the change is when the buffer is
#    sorted, not what survives -- so recall, cand and ivf_recall are the check.
# 2. Speed at both ends of nprobe: v44 lost 18% at nprobe=8 against v43 and
#    broke even at 128, which is the signature of a sort per chunk.
# 3. stella at nlist=65536 came back with recall 0.733 at every nprobe while
#    ivf_recall was 0.99 -- routing fine, everything after it broken. That run
#    also had JHQ_N_TRAIN=2600000, and nlist and n_train have never been moved
#    independently. Four cells separate them.
set -u
exec 9>/root/.lock_v45; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v45.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target \
  demo_jhq_v45_exp demo_jhq_v45_diag demo_jhq_v44_exp demo_jhq_v44_diag demo_jhq_v43_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -30 >> $L; say "=== V45""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # bin paths M nlist nprobe ntrain diag tag alpha
  env $E JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${7:+JHQ_DIAG=1} timeout 4000 \
      build/$1 $2 $3 8 8 ${9:-100.0} 10 $4 $5 8 1024 "" 3 >/tmp/v45.$$ 2>/tmp/v45e.$$
  local rc=$?
  printf "  %-18s %-8s nlist=%-7s np=%-4s nt=%-8s a=%-6s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s rc=%s\n" \
    "$1" "$8" "$4" "$5" "$6" "${9:-100.0}" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v45.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v45.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/v45.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/v45.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/v45e.$$ | sed 's/^/      ! /' >> $L
  rm -f /tmp/v45.$$ /tmp/v45e.$$
}

say "########## 1. v43 / v44 / v45 identical results, and speed ##########"
for np in 8 32 128 256; do
  for b in v43_exp v44_diag v45_diag; do
    run demo_jhq_$b "$ST" 128 16384 $np 100000 1 stella
  done
done
say "########## 2. speed without diag, three datasets ##########"
for np in 8 32 128 256; do
  for b in v43_exp v44_exp v45_exp; do
    run demo_jhq_$b "$VG" 96  1024  $np 100000 "" vogue
    run demo_jhq_$b "$BG" 128 8192  $np 100000 "" bge
    run demo_jhq_$b "$ST" 128 16384 $np 100000 "" stella
  done
done
say "########## 3. is the 65536 collapse nlist or n_train? ##########"
run demo_jhq_v45_diag "$ST" 128 32768 128 1300000 1 stella
run demo_jhq_v45_diag "$ST" 128 32768 128 2600000 1 stella
run demo_jhq_v45_diag "$ST" 128 65536 128 1300000 1 stella
run demo_jhq_v45_diag "$ST" 128 65536 128 2600000 1 stella
say "########## 4. the coarse training set, at fixed nlist ##########"
for nt in 100000 320000 640000 1300000; do
  for np in 32 128 256; do
    run demo_jhq_v45_diag "$ST" 128 16384 $np $nt 1 stella
  done
done
say "=== V45""_DONE ==="
