#!/bin/bash
# Is the cross-query reuse the cluster-centric rewrite would capture already
# being served by L2?
#
# Grouping tasks by IVF list so one list's codes are loaded once and scored
# against every query that probes it is a large rewrite. Its premise is that
# those reads are repeated and that the repeats cost DRAM traffic. At
# nprobe=128 and nlist=4096 a list is probed by ~31 of a 1000-query batch, so
# they are certainly repeated -- but this card has 96 MB of L2.
#
# JHQ_QUERY_DUP=D keeps nq/D queries and repeats each D times. Same query
# count, same blocks, same candidates, same selection; only the overlap
# between probe lists changes, and it changes by exactly D. Rising QPS says
# the repeats reach DRAM and the rewrite has something to win; flat QPS says
# L2 already has them and it does not.
#
# Recall will move with D -- a different query set -- and is printed only to
# confirm the run is the same search, not to compare across D.
set -u
exec 9>/root/.lock_qdup; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
# The script is copied out of the tree before launching -- it resets the tree
# below, and bash reads a script incrementally, so a script that rewrites
# itself mid-run corrupts. The copy cannot find the repo from $0, so it is
# named explicitly.
cd "${JHQ_REPO:-$(dirname "$0")/..}"
CACHE=${CACHE:-/root/autodl-tmp/qdup_cache}; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/qdup.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_qdup >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L|head -20 >>$L; say "=== QDUP""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # paths M nlist nt np blk tag dup
  local C=$CACHE/${7}_${3}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$6 JHQ_TILE_M_RT=$2 JHQ_N_TRAIN=$4 \
      JHQ_QUERY_DUP=$8 timeout 9000 \
      build/demo_jhq_qdup $1 $2 8 8 100.0 10 $3 $5 8 1024 "" 3 \
      >/tmp/qd.$$ 2>/tmp/qde.$$
  local rc=$?
  printf "  %-14s nlist=%-7s np=%-5s dup=%-3s recall=%-8s qps=%-9s rc=%s\n" \
    "$7" "$3" "$5" "$8" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/qd.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/qd.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/qde.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/qd.$$ /tmp/qde.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
# nlist=4096 gives ~31 queries a list at nprobe=128; nlist=32768 gives ~4. The
# two are kept side by side because the premise is weaker at the higher nlist,
# and stella and bge are where the fronts are actually reported.
for np in 128 1024; do
  for du in 1 2 4 8; do
    go "$VG"  96  4096  159744  $np 512  vogue        $du
    go "$BG"  128 32768 1277952 $np 512  bge          $du
    go "$ST"  128 32768 1277952 $np 512  stella       $du
    go "$O30" 384 4096  159744  $np 1024 openai3-3072 $du
  done
done
say "=== QDUP""_DONE ==="
