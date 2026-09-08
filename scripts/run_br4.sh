#!/bin/bash
# Br=4, where JHQ's memory claim actually lives.
#
# bpv_ = (d*Br+7)/8 -- the residual is a byte per DIMENSION, not per subspace.
# At Br=8 and d=1024 that is 1024 B/vec on top of M=128 for the primary, so
# JHQ is 1160 B/vec against int8 CAGRA's 1152. Dead even, and every number
# measured on 8 September was taken there. Br=4 halves the residual to 648
# B/vec, 1.8x smaller than the graph, and that is the operating point the
# recall-at-equal-bytes claim rests on.
#
# The grid matches what the published fronts swept -- alpha=100, nprobe 32 to
# 1024 -- so the comparison is against f_*_M*Br4.csv point for point, plus
# nprobe=8 below it. nprobe never exceeds nlist: past that the probe list
# repeats and the extra points are the same measurement at a higher price,
# which is what made vogue's top look like a 3x regression when it was a wash.
set -u
exec 9>/root/.lock_br4; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/br4_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/br4.log}; : > $L
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
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== BR4""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
sweep(){ # tag paths M Br nlist ntrain
  local tag=$1 paths=$2 M=$3 Br=$4 nl=$5 nt=$6
  local C=$CACHE/${tag}_${Br}_${nl}; mkdir -p $C
  for np in 8 32 64 128 256 512 1024; do
    [ $np -gt $nl ] && continue
    env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt timeout 9000 \
        build/demo_jhq_v47_diag $paths $M 8 $Br 100.0 10 $nl $np 8 1024 "" 3 \
        >/tmp/b4.$$ 2>/tmp/b4e.$$
    local rc=$?
    printf "  %-14s M=%-4s Br=%-3s nlist=%-7s nt=%-8s np=%-5s recall=%-8s qps=%-9s cand=%-9s ivf=%-8s vram=%-9s bpv=%s rc=%s\n" \
      "$tag" "$M" "$Br" "$nl" "$nt" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/b4.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/b4.$$)" \
      "$(awk '/^cand_mean/{print $3;exit}' /tmp/b4.$$)" \
      "$(awk '/^ivf_recall/{print $2;exit}' /tmp/b4.$$)" \
      "$(awk '/^VRAM used/{print $4;exit}' /tmp/b4.$$)" \
      "$(awk '/^primary:/{print $0;exit}' /tmp/b4.$$ | grep -oE 'code=[0-9]+' | cut -d= -f2)" "$rc" >> $L
    [ $rc -ne 0 ] && head -2 /tmp/b4e.$$ | sed 's/^/      ! /' >> $L
    rm -f /tmp/b4.$$ /tmp/b4e.$$
  done
  rm -rf $C
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## Br=4, published configuration ##########"
sweep vogue-768-old    "$VG"  96  4 1024  100000
sweep arxiv-768-old    "$AX"  96  4 2048  100000
sweep bge-m3-old       "$BG"  128 4 8192  100000
sweep stella-old       "$ST"  128 4 16384 100000
sweep openai3-1536-old "$O15" 192 4 1024  100000
sweep openai3-3072-old "$O30" 384 4 1024  100000

say "########## Br=4, retuned configuration ##########"
sweep vogue-768        "$VG"  96  4 4096   159744
sweep arxiv-768        "$AX"  96  4 8192   319488
sweep bge-m3           "$BG"  128 4 16384  638976
sweep stella           "$ST"  128 4 32768  1277952
sweep openai3-1536     "$O15" 192 4 4096   159744
sweep openai3-3072     "$O30" 384 4 4096   159744

say "########## Br=2, retuned configuration -- the small end ##########"
sweep vogue-768        "$VG"  96  2 4096   159744
sweep bge-m3           "$BG"  128 2 16384  638976
sweep stella           "$ST"  128 2 32768  1277952
sweep openai3-3072     "$O30" 384 2 4096   159744
say "=== BR4""_DONE ==="
