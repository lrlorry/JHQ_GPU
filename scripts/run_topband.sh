#!/bin/bash
# How wide is the front's top band under build-to-build variation?
#
# At the top the front is nearly vertical: on vogue at nlist=1024 and
# nprobe=512, the published front records 0.9950 at 11,536 QPS and this one
# 0.9940 at 14,398 -- the same configuration, one training run apart. 1.0e-3 of
# recall is the measured build-to-build floor and it is worth 2-3x of apparent
# QPS there, so any single-build comparison in that band is reading noise.
#
# Three independent builds per configuration, top nprobe only. The spread is
# the answer; the median front is what should be plotted.
set -u
exec 9>/root/.lock_topband; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/topband.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== TOPBAND""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"
run(){ # tag paths M nlist nt np rep
  local C=/root/autodl-tmp/tb_$1_$4_$7; rm -rf $C; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/demo_jhq_v47_diag $2 $3 8 8 100.0 10 $4 $6 8 1024 "" 3 >/tmp/tb.$$ 2>/dev/null
  printf "  %-14s nlist=%-7s np=%-5s build=%s  recall=%-8s qps=%s\n" "$1" "$4" "$6" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/tb.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/tb.$$)" >> $L
  rm -rf $C /tmp/tb.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
for r in 1 2 3; do
  say "### build $r"
  run vogue-768    "$VG"  96  1024  100000 512  $r
  run vogue-768    "$VG"  96  1024  100000 1024 $r
  run arxiv-768    "$AX"  96  2048  100000 512  $r
  run arxiv-768    "$AX"  96  2048  100000 1024 $r
  run stella       "$ST"  128 16384 100000 512  $r
  run openai3-1536 "$O15" 192 1024  100000 512  $r
  run openai3-1536 "$O15" 192 1024  100000 1024 $r
done
say "=== TOPBAND""_DONE ==="
