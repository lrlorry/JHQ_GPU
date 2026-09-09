#!/bin/bash
# Close the gaps in the fp32 comparison, which are sampling gaps, not results.
#
#  * openai3-1536 has never been re-measured on v51+v52. Before those it sat at
#    0.58-0.96x of CAGRA fp32 -- the closest of the six to parity -- so it is
#    the most likely fourth win and the one dataset that can still change the
#    headline.
#  * vogue and arxiv were measured at nprobe 128 and 512 only, so their fronts
#    jump a decade of QPS between two points and the interpolation against fp32
#    dips below 1x in the gap. nprobe 256 and 1024 fill it.
#  * openai3-3072 has four front points; 256 makes five.
#
# alpha is swept where the saturation point is known to be low, fixed at 100
# where it is not: results/front6/alpha_ds.log has the per-dataset picture.
set -u
exec 9>/root/.lock_fp32g; flock -n 9 || { echo running; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
CACHE=/root/autodl-tmp/fp32g_cache; mkdir -p $CACHE
D=/root/autodl-tmp; V=/root/data
L=/root/fp32g.log; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
# The proxy returns a transient 503 often enough to lose a run to it. Retry,
# then still refuse rather than run against a stale tree.
ok=0
for try in 1 2 3 4; do
  if git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then ok=1; break; fi
  say "fetch attempt $try failed"; sleep 15
done
[ $ok -eq 1 ] || { say "=== fetch failed 4x; refusing to run against a stale tree ==="; exit 1; }
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake --build build -j 16 --target demo_jhq_v53_x1 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L|head -20 >>$L; say "=== FPG""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
go(){ # tag paths M nlist nt np alpha blk
  local C=$CACHE/${1}_M${3}_${4}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$8 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 timeout 9000 \
      build/demo_jhq_v53_x1 $2 $3 8 8 $7 10 $4 $6 8 1024 "" 3 >/tmp/fg.$$ 2>/tmp/fge.$$
  local rc=$?
  printf "  %-14s M=%-4s np=%-5s a=%-6s recall=%-8s qps=%-9s rc=%s\n" "$1" "$3" "$6" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/fg.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/fg.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -3 /tmp/fge.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/fg.$$ /tmp/fge.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

say "########## 1. openai3-1536, never re-measured ##########"
for np in 32 128 256 512 1024; do
  for a in 8.0 32.0 100.0; do go openai3-1536 "$O15" 192 8192 319488 $np $a 512; done
done
say "########## 2. fill the nprobe gaps ##########"
for a in 16.0 32.0 100.0; do
  go vogue-768 "$VG" 96 4096 159744 256  $a 512
  go vogue-768 "$VG" 96 4096 159744 1024 $a 512
done
for a in 32.0 100.0; do
  go arxiv-768 "$AX" 96 8192 319488 256  $a 512
  go arxiv-768 "$AX" 96 8192 319488 1024 $a 512
done
for a in 8.0 32.0 100.0; do
  go openai3-3072 "$O30" 384 4096 159744 256 $a 1024
done
say "=== FPG""_DONE ==="
