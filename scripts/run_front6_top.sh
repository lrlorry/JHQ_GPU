#!/bin/bash
# Extend the new fronts past the published ceilings.
#
# run_front6.sh swept nprobe 8/32/128/256 for both configurations. That is not
# a matched sweep: the new configuration raises nlist 2-4x, so at the same
# nprobe it reads 0.27-0.36x the candidates and its front stops far short --
# bge-m3 at 0.9575 against the published 0.9932. The ceiling did not move, the
# sweep did.
#
# nprobe scaled by the nlist ratio and then past it: 512, 1024, 2048.
set -u
exec 9>/root/.lock_f6top2; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/f6top2_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/f6top2.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== F6TOP""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
sweep(){ # tag paths M nlist ntrain probes...
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5; shift 5
  local C=$CACHE/${tag}; mkdir -p $C
  for np in "$@"; do
    env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt timeout 9000 \
        build/demo_jhq_v47_diag $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 \
        >/tmp/ft.$$ 2>/tmp/fte.$$
    local rc=$?
    printf "  %-14s M=%-4s nlist=%-7s nt=%-8s np=%-5s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s rc=%s\n" \
      "$tag" "$M" "$nl" "$nt" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/ft.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/ft.$$)" \
      "$(awk '/^cand_mean/{print $3;exit}' /tmp/ft.$$)" \
      "$(awk '/^ivf_recall/{print $2;exit}' /tmp/ft.$$)" "$rc" >> $L
    [ $rc -ne 0 ] && head -2 /tmp/fte.$$ | sed 's/^/      ! /' >> $L
    rm -f /tmp/ft.$$ /tmp/fte.$$
  done
  rm -rf $C
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## nprobe past 256, new configurations ##########"
sweep vogue-768    "$VG"  96  4096   159744  512 1024 2048
sweep arxiv-768    "$AX"  96  8192   319488  512 1024 2048
sweep bge-m3       "$BG"  128 16384  638976  512 1024 2048
sweep stella       "$ST"  128 32768  1277952 512 1024 2048
sweep openai3-1536 "$O15" 192 4096   159744  512 1024 2048
sweep openai3-3072 "$O30" 384 4096   159744  512 1024 2048

# And the same for the published configuration. Extending only one arm leaves
# the envelope missing its better leg exactly where the new configuration is
# weakest -- at the top, where a larger nlist buys nothing and costs a coarse
# search over four to thirty-two times as many centroids.
say "########## nprobe past 256, published configurations ##########"
sweep vogue-768-old    "$VG"  96  1024  100000 512 1024 2048
sweep arxiv-768-old    "$AX"  96  2048  100000 512 1024 2048
sweep bge-m3-old       "$BG"  128 8192  100000 512 1024 2048
sweep stella-old       "$ST"  128 16384 100000 512 1024 2048
sweep openai3-1536-old "$O15" 192 1024  100000 512 1024 2048
sweep openai3-3072-old "$O30" 384 1024  100000 512 1024 2048
say "=== F6TOP""_DONE ==="
