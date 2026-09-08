#!/bin/bash
# What should the LUT policy be, and at which block size?
#
# results/pending/b1024.log reversed the v40 verdict: a __half table in shared
# memory is 78-85% SLOWER at BLOCK=256 and 8% FASTER at BLOCK=1024. Two points
# do not make a policy -- they only say the crossover is between them -- and
# both were taken at alpha=100, where cap=2048 makes scan_base 16 KB for every
# block size. At the paper's alpha=8, ck=80 and cap tracks BLOCK: 512 at 256
# threads (4 KB) against 2048 at 1024 (16 KB), so the shared budget left for
# the table changes with the block size as well. The policy has to be decided
# on both axes or it is guessed.
#
# 128/256/512/1024 x {fp32-global, half-shared, carveout, both} x {vogue,
# stella} x {alpha=100, alpha=8}.
set -u
exec 9>/root/.lock_lutpol; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/lutpol.log}; : > $L
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
cmake --build build -j 16 --target \
  demo_jhq_v39_exp demo_jhq_v39_lut16 demo_jhq_v40_exp demo_jhq_v40_lut16 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L|head -5 >>$L; say "=== LUTPOL""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"

run(){ # bin paths M nlist nprobe BLOCK alpha tag
  env $E JHQ_TILE_M_RT=$3 JHQ_BLOCK=$6 timeout 3000 \
      build/$1 $2 $3 8 8 $7 10 $4 $5 8 1024 "" 5 >/tmp/lp.$$ 2>/dev/null
  printf "  %-18s %-8s a=%-6s BLOCK=%-5s lat=%-9s qps=%-9s recall=%s\n" \
    "$1" "$8" "$7" "$6" \
    "$(awk '/^Latency/{print $3;exit}' /tmp/lp.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/lp.$$)" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/lp.$$)" >> $L; rm -f /tmp/lp.$$; }

for A in 100.0 8.0; do
  say "########## alpha=$A ##########"
  for B in 128 256 512 1024; do
    say "--- BLOCK=$B ---"
    for r in 1 2; do
      for v in v39_exp v39_lut16 v40_exp v40_lut16; do
        run demo_jhq_$v "$VG" 96  1024  128 $B $A vogue
        run demo_jhq_$v "$ST" 128 16384 128 $B $A stella
      done
    done
  done
done
say "=== LUTPOL""_DONE ==="
