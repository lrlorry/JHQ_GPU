#!/bin/bash
# Everything left that is a measurement, queued behind whatever holds the GPU.
#
#   1. M per dataset      one value each, inherited from the published choice,
#                         never re-swept under the current code or config.
#   2. BLOCK at alpha=8   results/lut_policy/ found BLOCK=512 with an fp32
#                         global table the fastest cell in that quadrant on
#                         vogue, and nothing followed it up.
#   3. bge nlist ceiling  65536 OOMs in add() at Br=8 (10.3 GB residual buffer)
#                         and at Br=4 too, which the arithmetic does not
#                         predict -- 5.1 GB should fit. Find out where.
#   4. Br=4 fronts        at the best nlist, which the Br=4 sweep predates.
set -u
exec 9>/root/.lock_rest; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/rest_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/rest.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v49_base demo_jhq_v47_diag >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== REST""_DONE build failed ==="; exit 1; }
say "waiting for the GPU"
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"
go(){ # bin paths M Br nlist nt np alpha block tag
  local C=$CACHE/${10}_${3}_${4}_${5}_${6}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 ${9:+JHQ_BLOCK=$9} timeout 9000 \
      build/$1 $2 $3 8 $4 $8 10 $5 $7 8 1024 "" 3 >/tmp/rs.$$ 2>/tmp/rse.$$
  local rc=$?
  printf "  %-13s M=%-4s Br=%-3s nlist=%-7s np=%-5s a=%-6s blk=%-5s recall=%-8s qps=%-9s cand=%-9s vram=%-9s rc=%s\n" \
    "${10}" "$3" "$4" "$5" "$7" "$8" "${9:-1024}" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/rs.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/rs.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/rs.$$)" \
    "$(awk '/^VRAM used/{print $4;exit}' /tmp/rs.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/rse.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/rs.$$ /tmp/rse.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. M per dataset -- equation 4 needs Ds | B, so M is a multiple of d/8 ##########"
for np in 32 128; do
  for M in 96 192 384; do go demo_jhq_v49_base "$VG"  $M 8 4096  159744  $np 100.0 "" vogue;   done
  for M in 96 192 384; do go demo_jhq_v49_base "$AX"  $M 8 8192  319488  $np 100.0 "" arxiv;   done
  for M in 128 256 512; do go demo_jhq_v49_base "$BG" $M 8 32768 1277952 $np 100.0 "" bge;     done
  for M in 128 256 512; do go demo_jhq_v49_base "$ST" $M 8 32768 1277952 $np 100.0 "" stella;  done
  for M in 192 384;     do go demo_jhq_v49_base "$O15" $M 8 8192 319488  $np 100.0 "" o15;     done
  for M in 384;         do go demo_jhq_v49_base "$O30" $M 8 4096 159744  $np 100.0 "" o30;     done
done
say "########## 2. BLOCK at the paper's alpha ##########"
for b in 256 512 1024; do
  for np in 128 512; do
    go demo_jhq_v49_base "$VG" 96  8 4096  159744  $np 8.0 $b vogue
    go demo_jhq_v49_base "$ST" 128 8 32768 1277952 $np 8.0 $b stella
  done
done
say "########## 3. bge nlist ceiling, and where it actually stops ##########"
for spec in "4 65536 2555904" "4 131072 5111808" "8 65536 2555904"; do
  set -- $spec
  go demo_jhq_v49_base "$BG" 128 $1 $2 $3 128 100.0 "" bge-ceiling
done
say "########## 4. Br=4 fronts at the best nlist ##########"
for np in 8 32 128 512; do
  go demo_jhq_v49_base "$VG"  96  4 4096  159744  $np 100.0 "" vogue-br4
  go demo_jhq_v49_base "$AX"  96  4 8192  319488  $np 100.0 "" arxiv-br4
  go demo_jhq_v49_base "$BG"  128 4 32768 1277952 $np 100.0 "" bge-br4
  go demo_jhq_v49_base "$ST"  128 4 32768 1277952 $np 100.0 "" stella-br4
  go demo_jhq_v49_base "$O15" 192 4 8192  319488  $np 100.0 "" o15-br4
  go demo_jhq_v49_base "$O30" 384 4 4096  159744  $np 100.0 "" o30-br4
done
say "=== REST""_DONE ==="
