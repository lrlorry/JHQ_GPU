#!/bin/bash
# nlist per dataset, and per operating point.
#
# The envelope in results/front6/ has the two configurations crossing: the
# larger nlist wins by 1.4-2.1x over the middle of the front and loses at the
# top, where it buys no candidates and pays a coarse search over four to
# thirty-two times as many centroids. Two values do not locate an optimum and
# do not say whether the optimum moves with recall. Four per dataset, each with
# a training set at ~39 points a centroid, against a common nprobe ladder.
set -u
exec 9>/root/.lock_nlist6; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/nlist6_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/nlist6.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
# The fetch fails often enough -- GnuTLS, or the proxy not taking -- that
# letting it fail quietly means the box runs an older commit and produces
# numbers nothing explains. It cost one launch of this very script, which ran
# eight commits behind and could not find itself. Abort instead.
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== NLIST6""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
sweep(){ # tag paths M N nlist
  local tag=$1 paths=$2 M=$3 N=$4 nl=$5
  local nt=$((39*nl)); [ $nt -lt 100000 ] && nt=100000; [ $nt -gt $N ] && nt=$N
  local C=$CACHE/${tag}_${nl}; mkdir -p $C
  for np in 8 32 128 512 1024; do
    [ $np -gt $nl ] && continue
    env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt timeout 9000 \
        build/demo_jhq_v47_diag $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 \
        >/tmp/n6.$$ 2>/tmp/n6e.$$
    local rc=$?
    printf "  %-14s M=%-4s nlist=%-7s nt=%-8s np=%-5s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s train=%-9s rc=%s\n" \
      "$tag" "$M" "$nl" "$nt" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/n6.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/n6.$$)" \
      "$(awk '/^cand_mean/{print $3;exit}' /tmp/n6.$$)" \
      "$(awk '/^ivf_recall/{print $2;exit}' /tmp/n6.$$)" \
      "$(awk '/^  train:/{print $2;exit}' /tmp/n6.$$)" "$rc" >> $L
    [ $rc -ne 0 ] && head -2 /tmp/n6e.$$ | sed 's/^/      ! /' >> $L
    rm -f /tmp/n6.$$ /tmp/n6e.$$
  done
  rm -rf $C
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

for nl in 1024 2048 4096 8192;      do say "### vogue nlist=$nl";  sweep vogue-768    "$VG"  96  1000000  $nl; done
for nl in 2048 4096 8192 16384;     do say "### arxiv nlist=$nl";  sweep arxiv-768    "$AX"  96  2253000  $nl; done
for nl in 4096 8192 16384 32768;    do say "### bge nlist=$nl";    sweep bge-m3       "$BG"  128 10091524 $nl; done
for nl in 8192 16384 32768 65536;   do say "### stella nlist=$nl"; sweep stella       "$ST"  128 17776615 $nl; done
for nl in 1024 2048 4096 8192;      do say "### o15 nlist=$nl";    sweep openai3-1536 "$O15" 192 999000   $nl; done
for nl in 1024 2048 4096 8192;      do say "### o30 nlist=$nl";    sweep openai3-3072 "$O30" 384 999000   $nl; done
say "=== NLIST6""_DONE ==="
