#!/bin/bash
# Two questions GPT raised, both answerable inside the paper's own parameters.
#
# 1. alpha has never been tuned. results/front6/alpha6.log measured rank_lost
#    against alpha and it saturates at a per-dataset value that spans 25x:
#    openai3-3072 is flat from alpha=4, arxiv-768 still improving past 200.
#    But that sweep ran with JHQ_DIAG=1 (readbacks on the search stream) at
#    BLOCK=1024, before v51 and v52, so its QPS column is not a result. This
#    re-measures it clean on today's default.
#
# 2. "the primary code should carry more information." Equation 4's
#    admissibility is Ds | B, so at B=8 the legal Ds are 1, 2, 4, 8 and
#    L = 2^(B/Ds) gives 8, 4, 2, 1 bits a dimension. Every run this project
#    has ever made used Ds=8, one bit a dimension. Ds=4 is two bits, M = d/4,
#    and it is a parameter -- M is argv[4] -- not an algorithm change. The
#    residual dominates the byte count at Br=8, so it costs about +11% of
#    total bytes.
#
# The question is whether the better primary ordering lets alpha fall further
# than it costs. Both parts report recall and QPS on the same binary.
set -u
exec 9>/root/.lock_v55; flock -n 9 || { echo running; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
CACHE=/root/autodl-tmp/v55_cache; mkdir -p $CACHE
D=/root/autodl-tmp; V=/root/data
L=/root/v55.log; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 || { say "=== fetch failed ==="; exit 1; }
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake --build build -j 16 --target demo_jhq_v53_x1 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L|head -20 >>$L; say "=== V55""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # tag paths M nlist nt np alpha blk
  local C=$CACHE/${1}_M${3}_${4}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$8 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/demo_jhq_v53_x1 $2 $3 8 8 $7 10 $4 $6 8 1024 "" 3 >/tmp/v55.$$ 2>/tmp/v55e.$$
  local rc=$?
  printf "  %-14s M=%-4s bpd=%-4s np=%-5s a=%-6s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "$3" "$9" "$6" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v55.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v55.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/v55e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v55.$$ /tmp/v55e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. alpha, measured clean at Ds=8 (1 bit/dim) ##########"
for a in 4.0 8.0 16.0 32.0 100.0; do
  for np in 128 512; do
    go vogue-768    "$VG"  96  4096  159744  $np $a 512  1
    go arxiv-768    "$AX"  96  8192  319488  $np $a 512  1
    go stella       "$ST"  128 32768 1277952 $np $a 512  1
    go openai3-3072 "$O30" 384 4096  159744  $np $a 1024 1
  done
done
say "########## 2. Ds=4 (2 bit/dim), M = d/4 ##########"
for a in 4.0 8.0 16.0 32.0 100.0; do
  for np in 128 512; do
    go vogue-768    "$VG"  192 4096  159744  $np $a 512 2
    go arxiv-768    "$AX"  192 8192  319488  $np $a 512 2
    go stella       "$ST"  256 32768 1277952 $np $a 512 2
  done
done
say "=== V55""_DONE ==="
