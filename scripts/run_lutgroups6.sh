#!/bin/bash
# The granularity sweep on six datasets, not two.
#
# Figure 4(a) is the evidence for G=2, which is the first contribution's core
# claim, and it rests on vogue-768 and openai3-3072 alone -- eight cells.  The
# same gap in the budget experiment turned out to hide a dataset where the
# rule loses, so two datasets is not enough to call a design rule.
#
# Cost was never the reason it stopped at two: the original phase 2 ran 32
# configurations in 2 minutes 17 seconds.  Six datasets is 96, and the two
# largest dominate the bill.
#
# nlist follows the frontier per dataset.  That changes openai3-3072, which
# the original swept at 4096 while the frontier now reports 8192, so the
# granularity result and the frontier will describe the same index.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_lutgroups6.sh /root/g6.sh \
#     && setsid nohup bash /root/g6.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_g6; flock -n 9 || { echo running; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
L=/root/g6.log; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
CACHE=/root/autodl-tmp/g6_cache; mkdir -p $CACHE
D=/root/autodl-tmp; V=/root/data
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v59_g1 demo_jhq_v59_g2 demo_jhq_v59_g4 \
      demo_jhq_v59_g8 demo_jhq_v59_b_g1 demo_jhq_v59_b_g2 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E 'error' $L | head -8 | cut -c1-170 >>$L; say "=== G6""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){ # tag paths M nlist blk np G [bin-prefix]
  local tag=$1 paths=$2 M=$3 nl=$4 blk=$5 np=$6 G=$7 pre=${8:-g}
  local C=$CACHE/${tag}_${nl}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$blk JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$((39*nl)) \
      timeout 6000 build/demo_jhq_v59_${pre}$G $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 \
      >/tmp/g6.$$ 2>/tmp/g6e.$$
  local rc=$?
  printf "  %-14s M=%-4s layout=%-5s G=%-2s np=%-5s recall=%-8s qps=%-9s smem=%-8s rc=%s\n" \
    "$tag" "$M" "$([ "$pre" = g ] && echo word || echo byte)" "$G" "$np" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/g6.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/g6.$$)" \
    "$(awk '/lut_smem|smem/{print $NF;exit}' /tmp/g6.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -4 /tmp/g6e.$$ | cut -c1-140 >> $L
  rm -f /tmp/g6.$$ /tmp/g6e.$$
  return 0
}

# The smaller four first: if G=2 stops winning anywhere, it shows up there
# before the two largest have spent twenty minutes.
say "########## phase 2: the sweep, six datasets ##########"
for np in 8 32 128 512; do
  for G in 1 2 4 8; do
    one vogue-768    "$VG"  96  4096  512  $np $G
    one arxiv-768    "$AX"  96  8192  512  $np $G
    one openai3-1536 "$O15" 192 8192  512  $np $G
    one openai3-3072 "$O30" 384 8192  1024 $np $G
  done
done
say "=== G6""_DONE small four ==="
for np in 8 32 128 512; do
  for G in 1 2 4 8; do
    one bge-m3 "$BG" 128 32768 512 $np $G
    one stella "$ST" 128 32768 512 $np $G
  done
done
say "=== G6""_DONE ==="
