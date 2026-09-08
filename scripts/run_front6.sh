#!/bin/bash
# The six-dataset front, re-measured, and the build cost in the same run.
#
# Two things at once because they come from the same process: the demo prints
# train and add alongside recall and QPS, so a cold-cache run gives the front
# and what it cost to build. Every timing in results/ so far was a cache hit
# reporting 35 ms of training, which is not a number.
#
# Each dataset is measured twice over: once at the configuration the published
# fronts used (nlist as it was, JHQ_N_TRAIN=100000) and once at ~39 training
# points per centroid with nlist doubled where the coarse quantizer has room.
# results/v46_sigma/ found +50 to +96% from the training set alone on stella
# and nothing outside stella and bge has been checked.
#
# Caches live on /root/autodl-tmp. They were on the system disk, one per
# binary, and filled it: 30 GB overlay, 7.2 GB of trained state.
set -u
exec 9>/root/.lock_front6; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/front6_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/front6.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== FRONT6""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"

# name paths M nlist n_train nprobe-list
sweep(){ # tag paths M nlist ntrain probes...
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5; shift 5
  local C=$CACHE/${tag}_${nl}_${nt}; mkdir -p $C
  for np in "$@"; do
    env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt timeout 6000 \
        build/demo_jhq_v47_diag $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 \
        >/tmp/f6.$$ 2>/tmp/f6e.$$
    local rc=$?
    printf "  %-14s M=%-4s nlist=%-7s nt=%-8s np=%-4s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s train=%-10s add=%-10s vram=%-8s rc=%s\n" \
      "$tag" "$M" "$nl" "$nt" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/f6.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/f6.$$)" \
      "$(awk '/^cand_mean/{print $3;exit}' /tmp/f6.$$)" \
      "$(awk '/^ivf_recall/{print $2;exit}' /tmp/f6.$$)" \
      "$(awk '/^  train:/{print $2;exit}' /tmp/f6.$$)" \
      "$(awk '/^  add:/{print $2;exit}' /tmp/f6.$$)" \
      "$(awk '/^VRAM used/{print $4;exit}' /tmp/f6.$$)" "$rc" >> $L
    [ $rc -ne 0 ] && head -2 /tmp/f6e.$$ | sed 's/^/      ! /' >> $L
    rm -f /tmp/f6.$$ /tmp/f6e.$$
  done
  rm -rf $C
}

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## as published: nlist as it was, JHQ_N_TRAIN=100000 ##########"
sweep vogue-768    "$VG"  96  1024  100000 8 32 128 256
sweep arxiv-768    "$AX"  96  2048  100000 8 32 128 256
sweep bge-m3       "$BG"  128 8192  100000 8 32 128 256
sweep stella       "$ST"  128 16384 100000 8 32 128 256
sweep openai3-1536 "$O15" 192 1024  100000 8 32 128 256
sweep openai3-3072 "$O30" 384 1024  100000 8 32 128 256

say "########## trained to ~39 points a centroid, nlist raised ##########"
# n_train is 39*nlist, capped at N. d=3072 wraps sigma above 699,050 rows, and
# d=1536 above 1,398,101 -- v47 carries the i64 fix, so these are safe now, but
# the caps are what the published runs would have hit.
sweep vogue-768    "$VG"  96  4096   159744  8 32 128 256
sweep arxiv-768    "$AX"  96  8192   319488  8 32 128 256
sweep bge-m3       "$BG"  128 16384  638976  8 32 128 256
sweep stella       "$ST"  128 32768  1277952 8 32 128 256
sweep openai3-1536 "$O15" 192 4096   159744  8 32 128 256
sweep openai3-3072 "$O30" 384 4096   159744  8 32 128 256
say "=== FRONT6""_DONE ==="
