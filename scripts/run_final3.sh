#!/bin/bash
# Three things left, in one pass.
#
# 1. bge-m3's nlist optimum was not bracketed: 32768 won at every recall and
#    was the largest tried. 65536 and 131072 bracket it or move it.
# 2. The fronts plotted use one nlist per dataset and it is not the best one.
#    The nlist sweep took nprobe 8/32/128/512/1024; adding 64 and 256 on the
#    two best values per dataset gives a front dense enough to plot.
# 3. n_train and nlist are separated only on stella. Everywhere else they moved
#    together, so "+50 to +96% from the training set" is one dataset's result
#    attributed to six. Fixed nlist, three training sizes.
set -u
exec 9>/root/.lock_final3; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/final3_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/final3.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== FINAL3""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
go(){ # tag paths M nlist nt probes...
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5; shift 5
  local C=$CACHE/${tag}_${nl}_${nt}; mkdir -p $C
  for np in "$@"; do
    [ $np -gt $nl ] && continue
    env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt timeout 9000 \
        build/demo_jhq_v47_diag $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 >/tmp/f3.$$ 2>/tmp/f3e.$$
    local rc=$?
    printf "  %-14s M=%-4s nlist=%-7s nt=%-8s np=%-5s recall=%-8s qps=%-9s cand=%-9s train=%-9s rc=%s\n" \
      "$tag" "$M" "$nl" "$nt" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/f3.$$)" "$(awk '/^QPS/{print $3;exit}' /tmp/f3.$$)" \
      "$(awk '/^cand_mean/{print $3;exit}' /tmp/f3.$$)" "$(awk '/^  train:/{print $2;exit}' /tmp/f3.$$)" "$rc" >> $L
    [ $rc -ne 0 ] && head -2 /tmp/f3e.$$|sed 's/^/      ! /' >>$L
    rm -f /tmp/f3.$$ /tmp/f3e.$$
  done
  rm -rf $C
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. bge-m3 nlist ceiling ##########"
go bge-m3 "$BG" 128 65536  2555904 8 32 128 512 1024
go bge-m3 "$BG" 128 131072 5111808 8 32 128 512 1024

say "########## 2. denser nprobe on the two best nlist ##########"
go vogue-768    "$VG"  96  2048   100000  64 256
go vogue-768    "$VG"  96  4096   159744  64 256
go arxiv-768    "$AX"  96  8192   319488  64 256
go arxiv-768    "$AX"  96  16384  638976  64 256
go bge-m3       "$BG"  128 32768  1277952 64 256
go stella       "$ST"  128 32768  1277952 64 256
go stella       "$ST"  128 65536  2555904 64 256
go openai3-1536 "$O15" 192 8192   319488  64 256
go openai3-3072 "$O30" 384 4096   159744  64 256
go openai3-3072 "$O30" 384 8192   319488  64 256

say "########## 3. n_train at fixed nlist, the other five ##########"
for nt in 100000 320000 640000; do go vogue-768    "$VG"  96  4096  $nt 32 128 512; done
for nt in 100000 320000 640000; do go arxiv-768    "$AX"  96  8192  $nt 32 128 512; done
for nt in 100000 320000 640000; do go bge-m3       "$BG"  128 16384 $nt 32 128 512; done
for nt in 100000 320000 640000; do go openai3-1536 "$O15" 192 4096  $nt 32 128 512; done
for nt in 100000 320000 640000; do go openai3-3072 "$O30" 384 4096  $nt 32 128 512; done
say "=== FINAL3""_DONE ==="
