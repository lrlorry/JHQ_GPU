#!/bin/bash
# Is training reproducible? Everything since v46 has compared binaries with a
# cache dir each, which means a training run each. v48_off is a control that
# cannot differ from v47 -- the kernels produce the same distances and the
# training sources are byte-identical -- and it differed in 17 of 21 rows by
# about 3e-4. Either the control is wrong or training is not deterministic.
#
# Same binary, three cold caches. If the three disagree, every cross-binary
# recall comparison since v46 has a noise floor and has to be quoted with it.
set -u
exec 9>/root/.lock_det; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/det.log}; : > $L
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
[ $rc -ne 0 ] && { say "=== DET""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024 JHQ_DIAG=1"

run(){ # tag paths M nlist nprobe rep
  local C=/root/autodl-tmp/det_cache_$1_$6; rm -rf $C; mkdir -p $C
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=100000 timeout 5000 \
      build/demo_jhq_v47_diag $2 $3 8 8 100.0 10 $4 $5 8 1024 "" 3 >/tmp/det.$$ 2>/dev/null
  printf "  %-8s nlist=%-7s np=%-4s rep=%s  recall=%-8s cand=%-9s ivf=%-8s qps=%s\n" \
    "$1" "$4" "$5" "$6" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/det.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/det.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/det.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/det.$$)" >> $L
  rm -rf $C /tmp/det.$$
}

say "########## same binary, three cold caches ##########"
for r in 1 2 3; do
  run vogue  "$VG" 96  1024  128 $r
  run stella "$ST" 128 16384 128 $r
done
say "########## and three times off ONE warm cache ##########"
C=/root/autodl-tmp/det_warm; rm -rf $C; mkdir -p $C
for r in 1 2 3; do
  env $E JHQ_INDEX_CACHE=$C JHQ_TILE_M_RT=128 JHQ_N_TRAIN=100000 timeout 5000 \
      build/demo_jhq_v47_diag $ST 128 8 8 100.0 10 16384 128 8 1024 "" 3 >/tmp/dw.$$ 2>/dev/null
  printf "  warm     stella  rep=%s  recall=%-8s cand=%-9s qps=%s\n" "$r" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/dw.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/dw.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/dw.$$)" >> $L
  rm -f /tmp/dw.$$
done
rm -rf $C
say "=== DET""_DONE ==="
