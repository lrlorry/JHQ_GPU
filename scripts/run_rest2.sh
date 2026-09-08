#!/bin/bash
# Two re-runs the last pass invalidated, and one sweep it only sampled.
#
# 1. M again. The first sweep ran at JHQ_BLOCK=1024, and M=96 is exactly the
#    configuration whose residual codebook fits in shared memory
#    (d*4 + M*Kr*4 = 101,376 at d=768, the opt-in limit) -- which is the one
#    place a 1024-thread refine launch costs 24-38%. So M=96 was measured
#    handicapped and M=192 came out "57% faster with twice the primary code",
#    which cannot be true. Every M row here uses the width its own cb_smem
#    wants: 256 when the codebook is staged in shared, JHQ_BLOCK otherwise.
#
# 2. BLOCK at both alphas, all six. At alpha=8, 512 beat 1024 by 2.6% on vogue
#    and 19% on stella, with recall identical -- and alpha=8 is the paper's
#    setting. Two datasets is not a policy.
set -u
exec 9>/root/.lock_rest2; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/rest2_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/rest2.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v49_base >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -20 >>$L; say "=== REST2""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_DIAG=1"

# The codebook is staged in shared when d*4 + M*256*4 fits the 101,376 opt-in.
# There the refine launch keeps 256 threads; everywhere else it takes JHQ_BLOCK.
refine_width(){ [ $(( $1 * 4 + $2 * 256 * 4 )) -le 101376 ] && echo 0 || echo 1024; }

go(){ # paths d M nlist nt np alpha block tag
  local rw; rw=$(refine_width $2 $3)
  local C=$CACHE/${9}_${3}_${4}_${5}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$8 JHQ_REFINE_BLOCK=$rw \
      JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/demo_jhq_v49_base $1 $3 8 8 $7 10 $4 $6 8 1024 "" 3 >/tmp/r2.$$ 2>/tmp/r2e.$$
  local rc=$?
  printf "  %-13s M=%-4s nlist=%-7s np=%-5s a=%-6s blk=%-5s rw=%-5s recall=%-8s qps=%-9s rc=%s\n" \
    "$9" "$3" "$4" "$6" "$7" "$8" "$rw" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/r2.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/r2.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/r2e.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/r2.$$ /tmp/r2e.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. M, with the refine width its cb_smem wants ##########"
for np in 32 128 512; do
  for M in 96 192 384;   do go "$VG"  768  $M 4096  159744  $np 100.0 1024 vogue;  done
  for M in 96 192 384;   do go "$AX"  768  $M 8192  319488  $np 100.0 1024 arxiv;  done
  for M in 128 256 512;  do go "$BG"  1024 $M 32768 1277952 $np 100.0 1024 bge;    done
  for M in 128 256 512;  do go "$ST"  1024 $M 32768 1277952 $np 100.0 1024 stella; done
  for M in 192 384;      do go "$O15" 1536 $M 8192  319488  $np 100.0 1024 o15;    done
  for M in 384 768;      do go "$O30" 3072 $M 4096  159744  $np 100.0 1024 o30;    done
done
say "########## 2. BLOCK, six datasets, both alphas ##########"
for a in 8.0 100.0; do
  for b in 256 512 1024; do
    for np in 128 512; do
      go "$VG"  768  96  4096  159744  $np $a $b vogue
      go "$AX"  768  96  8192  319488  $np $a $b arxiv
      go "$BG"  1024 128 32768 1277952 $np $a $b bge
      go "$ST"  1024 128 32768 1277952 $np $a $b stella
      go "$O15" 1536 192 8192  319488  $np $a $b o15
      go "$O30" 3072 384 4096  159744  $np $a $b o30
    done
  done
done
say "=== REST2""_DONE ==="
