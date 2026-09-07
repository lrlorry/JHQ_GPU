#!/bin/bash
# Re-measure at BLOCK=1024 everything that was taken at the default 256.
#
# The block sweep in results/pending/ makes this necessary: 256 -> 1024 is
# 1.8-1.9x on the scan kernel alone (vogue 21,679 -> 36,712 QPS, stella
# 11,845 -> 19,422). Each A/B below was internally fair -- both arms at 256 --
# but three of them have a mechanism that moves the answer:
#
#   refine share     residual_refine_fused_kernel launches with a hard-coded
#                    256 threads, so it does not speed up while the scan does;
#                    its share must rise.
#   compaction share cap is 2048 at ck=1000 for every block size, so the same
#                    bitonic sort gets four times the threads; its share must fall.
#   LUT in shared    82 KB a block leaves one block resident either way, so the
#                    threads per SM are the block size: 17% occupancy at 256
#                    against 67% at 1024. The verdict was taken where shared
#                    memory costs the most.
set -u
exec 9>/root/.lock_b1024; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/b1024.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target \
  demo_jhq_v39_exp demo_jhq_v37_nores demo_jhq_v39_lut16 \
  demo_jhq_v40_exp demo_jhq_v40_lut16 \
  demo_jhq_v42_full demo_jhq_v42_nocomp demo_jhq_v42_neither >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -5 >>$L; say "=== B1024""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"

run(){ # bin paths M nlist nprobe BLOCK tag
  env $E JHQ_TILE_M_RT=$3 JHQ_BLOCK=$6 timeout 3000 build/$1 $2 $3 8 8 100.0 10 $4 $5 8 1024 "" 5 \
      >/tmp/b.$$ 2>/dev/null
  printf "  %-24s %-8s BLOCK=%-5s np=%-4s lat=%-9s qps=%-9s recall=%s\n" \
    "$1" "$7" "$6" "$5" \
    "$(awk '/^Latency/{print $3;exit}' /tmp/b.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/b.$$)" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/b.$$)" >> $L; rm -f /tmp/b.$$; }

for B in 256 1024; do
  say "########## BLOCK=$B ##########"
  say "--- refine share: full vs JHQ_NO_RESIDUAL ---"
  for r in 1 2; do
    run demo_jhq_v39_exp   "$VG" 96  1024  128 $B vogue
    run demo_jhq_v37_nores "$VG" 96  1024  128 $B vogue
    run demo_jhq_v39_exp   "$BG" 128 8192  128 $B bge
    run demo_jhq_v37_nores "$BG" 128 8192  128 $B bge
    run demo_jhq_v39_exp   "$ST" 128 16384 128 $B stella
    run demo_jhq_v37_nores "$ST" 128 16384 128 $B stella
  done
  say "--- scan split: full / nocomp / neither ---"
  for r in 1 2; do
    for v in full nocomp neither; do
      run demo_jhq_v42_$v "$VG" 96  1024  128 $B vogue
      run demo_jhq_v42_$v "$BG" 128 8192  128 $B bge
      run demo_jhq_v42_$v "$ST" 128 16384 128 $B stella
    done
  done
  say "--- LUT variants: fp32-global / half-shared / carveout ---"
  for r in 1 2; do
    for v in v39_exp v39_lut16 v40_exp v40_lut16; do
      run demo_jhq_$v "$VG" 96  1024  128 $B vogue
      run demo_jhq_$v "$ST" 128 16384 128 $B stella
    done
  done
done
say "=== B1024""_DONE ==="
