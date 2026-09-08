#!/bin/bash
# v50: regroup and fusion, alone and together.
#
# Both leave the answer alone -- same ck, same composite distances, same
# summation order -- so recall must match v50_base to the last digit off one
# trained state. That is the first thing checked; the speed is the second.
set -u
exec 9>/root/.lock_v50; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v50_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v50.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target \
  demo_jhq_v50_base demo_jhq_v50_group demo_jhq_v50_fuse demo_jhq_v50_both >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -30 >>$L; say "=== V50""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
# One cache per configuration, shared by the four variants: they differ only in
# the search kernel, so the trained state must be identical or the recall
# comparison measures training noise instead.
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # bin paths M Br nlist nt np alpha block tag
  local C=$CACHE/${10}_${4}_${5}_${6}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$9 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 timeout 9000 \
      build/$1 $2 $3 8 $4 $8 10 $5 $7 8 1024 "" 3 >/tmp/v50.$$ 2>/tmp/v50e.$$
  local rc=$?
  printf "  %-16s %-12s Br=%-3s np=%-5s a=%-6s blk=%-5s recall=%-8s qps=%-9s rc=%s\n" \
    "$1" "${10}" "$4" "$7" "$8" "$9" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/v50.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/v50.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/v50e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/v50.$$ /tmp/v50e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

# BLOCK=512 is what the sweep in results/front6/rest2.log put on top in 20 of
# 24 cells, so that is where these are measured rather than at the old 1024.
say "########## the four variants, BLOCK=512 ##########"
for np in 32 128 512; do
  for b in v50_base v50_group v50_fuse v50_both; do
    go demo_jhq_$b "$VG"  96  8 4096  159744  $np 100.0 512 vogue
    go demo_jhq_$b "$BG"  128 8 32768 1277952 $np 100.0 512 bge
    go demo_jhq_$b "$ST"  128 8 32768 1277952 $np 100.0 512 stella
    go demo_jhq_$b "$O30" 384 8 4096  159744  $np 100.0 1024 o30
  done
done
say "########## and at the paper's alpha ##########"
for np in 128 512; do
  for b in v50_base v50_group v50_fuse v50_both; do
    go demo_jhq_$b "$VG" 96  8 4096  159744  $np 8.0 512 vogue
    go demo_jhq_$b "$ST" 128 8 32768 1277952 $np 8.0 512 stella
  done
done
say "=== V50""_DONE ==="
