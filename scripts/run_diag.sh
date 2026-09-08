#!/bin/bash
# v43: what does the scan actually read, and where is recall lost?
#
# Two numbers, both at several nprobe so the trend is visible:
#   cand_mean   candidates the scan read, against the N*nprobe/nlist estimate
#               every table in results/ has been quoting
#   ivf_recall  share of the true top-k whose list was opened at all
#
# ivf_recall - recall is what routing delivered and the scan then lost; more
# nprobe cannot recover that part.
set -u
exec 9>/root/.lock_diag; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
L=${LOG:-/root/diag.log}; : > $L
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
cmake --build build -j 16 --target demo_jhq_v43_exp >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error" $L | head -20 >> $L; say "=== DIAG""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BLOCK=1024"

run(){ # paths M nlist nprobe alpha tag
  env $E JHQ_DIAG=1 JHQ_TILE_M_RT=$2 timeout 3000 \
      build/demo_jhq_v43_exp $1 $2 8 8 $5 10 $3 $4 8 1024 "" 3 >/tmp/dg.$$ 2>/dev/null
  printf "  %-8s nlist=%-6s np=%-4s a=%-6s recall=%-8s ivf=%-8s route_lost=%-8s rank_lost=%-8s cand=%-9s est=%-9s x=%s\n" \
    "$6" "$3" "$4" "$5" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/dg.$$)" \
    "$(awk '/^ivf_recall/{print $2;exit}' /tmp/dg.$$)" \
    "$(awk '/^lost_route/{print $2;exit}' /tmp/dg.$$)" \
    "$(awk '/^lost_route/{print $4;exit}' /tmp/dg.$$)" \
    "$(awk '/^cand_mean/{print $3;exit}' /tmp/dg.$$)" \
    "$(awk '/^cand_mean/{for(i=1;i<=NF;i++)if($i=="=")print $(i+1);exit}' /tmp/dg.$$)" \
    "$(awk '/^cand_mean/{for(i=1;i<=NF;i++)if($i=="ratio")print $(i+1);exit}' /tmp/dg.$$)" >> $L
  grep -E "^cand_mean|^ivf_recall|^lost_route" /tmp/dg.$$ | sed 's/^/      | /' >> $L
  rm -f /tmp/dg.$$
}

for np in 8 32 128 256; do
  say "########## nprobe=$np ##########"
  run "$VG" 96  1024  $np 100.0 vogue
  run "$BG" 128 8192  $np 100.0 bge
  run "$ST" 128 16384 $np 100.0 stella
done
say "########## alpha sweep at nprobe=128 ##########"
for a in 8.0 20.0 50.0 100.0; do
  run "$VG" 96  1024  128 $a vogue
  run "$ST" 128 16384 128 $a stella
done
say "=== DIAG""_DONE ==="
