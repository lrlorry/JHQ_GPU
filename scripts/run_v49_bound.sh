#!/bin/bash
# Section 3 of run_v49.sh again, with the pruned-candidate fix.
#
# The first attempt wrote a pruned candidate's partial sum into the buffer.
# v48 could do that -- its bound was `a > thr` alone, so the value was already
# past the threshold -- but with `a + s_suf[mhi] > thr` the partial sum may sit
# well under thr, and those candidates were admitted at a distance that
# understated them. Recall came back 0.08 against 0.98, which is the failure
# looking exactly like a correctness bug should.
#
# v49_base is the control: same binary, bound compiled out.
set -u
exec 9>/root/.lock_v49b; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/root/autodl-tmp/v49b_cache}; rm -rf $CACHE; mkdir -p $CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/v49b.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target \
  demo_jhq_v49_base demo_jhq_v49_e8 demo_jhq_v49_e16 demo_jhq_v49_e32 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -30 >>$L; say "=== V49B""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"
go(){ # bin paths M Br nlist nt np tag
  local C=$CACHE/${4}_${5}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 timeout 9000 \
      build/$1 $2 $3 8 $4 100.0 10 $5 $7 8 1024 "" 3 >/tmp/vb.$$ 2>/dev/null
  printf "  %-16s %-10s Br=%-3s np=%-5s recall=%-8s qps=%s\n" "$1" "$8" "$4" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/vb.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/vb.$$)" >> $L
  rm -f /tmp/vb.$$
}
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
say "########## suffix-minimum exit, with the fix ##########"
for np in 32 128 512; do
  for b in v49_base v49_e8 v49_e16 v49_e32; do
    go demo_jhq_$b "$VG" 96  8 4096  159744  $np vogue
    go demo_jhq_$b "$BG" 128 8 32768 1277952 $np bge
    go demo_jhq_$b "$ST" 128 8 32768 1277952 $np stella
  done
done
say "########## and at the paper's alpha, where ck is small ##########"
for b in v49_base v49_e8 v49_e16; do
  for np in 128 512; do
    env $E JHQ_INDEX_CACHE=$CACHE/8_4096 JHQ_TILE_M_RT=96 JHQ_N_TRAIN=159744 timeout 9000 \
        build/demo_jhq_$b $VG 96 8 8 8.0 10 4096 $np 8 1024 "" 3 >/tmp/vb.$$ 2>/dev/null
    printf "  %-16s %-10s a=8.0  np=%-5s recall=%-8s qps=%s\n" "demo_jhq_$b" vogue "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/vb.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/vb.$$)" >> $L
    rm -f /tmp/vb.$$
  done
done
say "########## the materialised residual table, all six ##########"
# +7.3% to +21.4% on vogue, at bit-identical recall, from a switch that has
# existed since v22 and was only ever checked for numerical agreement. vogue
# is one dataset and the buffer is B*d*Kr*4 -- 805 MB there, 1.07 GB at
# d=1024, 3.2 GB at d=3072 -- so both the gain and the cost need the rest.
rl(){ # bin paths M Br nlist nt np tag lut
  local C=$CACHE/rl_${8}_${4}_${9}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_RESID_LUT=$9 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$6 timeout 9000 \
      build/$1 $2 $3 8 $4 100.0 10 $5 $7 8 1024 "" 3 >/tmp/rl.$$ 2>/tmp/rle.$$
  local rc=$?
  printf "  RESID_LUT=%s %-12s Br=%-3s np=%-5s recall=%-8s qps=%-9s vram=%-9s rc=%s\n" \
    "$9" "$8" "$4" "$7" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/rl.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/rl.$$)" \
    "$(awk '/^VRAM used/{print $4;exit}' /tmp/rl.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -2 /tmp/rle.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/rl.$$ /tmp/rle.$$
}
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
for lut in 0 1; do
  for np in 32 128; do
    rl demo_jhq_v49_base "$VG"  96  8 4096  159744  $np vogue        $lut
    rl demo_jhq_v49_base "$AX"  96  8 8192  319488  $np arxiv        $lut
    rl demo_jhq_v49_base "$BG"  128 8 32768 1277952 $np bge          $lut
    rl demo_jhq_v49_base "$ST"  128 8 32768 1277952 $np stella       $lut
    rl demo_jhq_v49_base "$O15" 192 8 8192  319488  $np openai3-1536 $lut
    rl demo_jhq_v49_base "$O30" 384 8 4096  159744  $np openai3-3072 $lut
  done
done

say "########## refine width against cb_smem, all six ##########"
# The 256 -> 1024 change was +3.3% on stella and -25.3% on vogue. vogue is the
# only configuration whose residual codebook fits in shared: M*Kr*4 + d*4 is
# 101,376 at M=96 and exactly the opt-in limit, while M>=128 exceeds it and
# falls back to global. One data point for each side is not a policy.
rw(){ # paths M nlist nt np tag width
  local C=$CACHE/rw_${6}; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_REFINE_BLOCK=$7 JHQ_TILE_M_RT=$2 JHQ_N_TRAIN=$4 timeout 9000 \
      build/demo_jhq_v49_base $1 $2 8 8 100.0 10 $3 $5 8 1024 "" 3 >/tmp/rw.$$ 2>/dev/null
  printf "  REFINE_BLOCK=%-5s %-12s M=%-4s np=%-5s recall=%-8s qps=%s\n" \
    "$7" "$6" "$2" "$5" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/rw.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/rw.$$)" >> $L
  rm -f /tmp/rw.$$
}
for w in 0 1024; do
  for np in 32 128; do
    rw "$VG"  96  4096  159744  $np vogue        $w
    rw "$AX"  96  8192  319488  $np arxiv        $w
    rw "$BG"  128 32768 1277952 $np bge          $w
    rw "$ST"  128 32768 1277952 $np stella       $w
    rw "$O15" 192 8192  319488  $np openai3-1536 $w
    rw "$O30" 384 4096  159744  $np openai3-3072 $w
  done
done
say "=== V49B""_DONE ==="
