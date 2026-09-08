#!/bin/bash
# alpha per dataset, and the bge-m3 nlist ceiling at a Br that fits.
#
# 1. results/v43_diag/ showed alpha is the only knob that moves the ranking
#    loss -- everything downstream of routing loses 0.13 to 0.51% and alpha is
#    what buys it back. It saturates at 20 on stella and 50 on vogue; the other
#    four have never been swept. ck = alpha*k, so this is a front parameter,
#    not a tuning detail: at alpha=8 the paper's setting costs vogue 4.2 points.
#
# 2. bge-m3's nlist optimum was not bracketed: 32768 won everywhere and 65536
#    OOMs in add(), where the n*bpv_ residual buffer is 10.3 GB at Br=8. At
#    Br=4 it is 5.1 GB and fits, so the ceiling is reachable there.
set -u
exec 9>/root/.lock_alpha6; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/alpha6_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/alpha6.log}; : > $L
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
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== ALPHA6""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
go(){ # tag paths M Br nlist nt np alpha
  local C=$CACHE/${1}_${4}_${5}_${6}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 timeout 9000 \
      build/demo_jhq_v47_diag $2 $3 8 $4 $8 10 $5 $7 8 1024 "" 3 >/tmp/a6.$$ 2>/tmp/a6e.$$
  local rc=$?
  printf "  %-14s Br=%-3s nlist=%-7s np=%-5s a=%-6s recall=%-8s ivf=%-8s rank_lost=%-8s qps=%-9s rc=%s\n" \
    "$1" "$4" "$5" "$7" "$8" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/a6.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/a6.$$)" \
    "$(awk '/^lost_route/{print $4;exit}' /tmp/a6.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/a6.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/a6e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/a6.$$ /tmp/a6e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. alpha saturation, six datasets ##########"
for a in 4.0 8.0 16.0 32.0 64.0 100.0 200.0; do
  for np in 32 128; do
    go vogue-768    "$VG"  96  8 4096   159744  $np $a
    go arxiv-768    "$AX"  96  8 8192   319488  $np $a
    go bge-m3       "$BG"  128 8 32768  1277952 $np $a
    go stella       "$ST"  128 8 32768  1277952 $np $a
    go openai3-1536 "$O15" 192 8 8192   319488  $np $a
    go openai3-3072 "$O30" 384 8 4096   159744  $np $a
  done
done
say "########## 2. bge-m3 nlist ceiling at Br=4 ##########"
for nl in 16384 32768 65536 131072; do
  nt=$((39*nl)); [ $nt -gt 10091524 ] && nt=10091524
  for np in 32 128 512; do
    [ $np -gt $nl ] && continue
    go bge-m3 "$BG" 128 4 $nl $nt $np 100.0
  done
done
say "=== ALPHA6""_DONE ==="
