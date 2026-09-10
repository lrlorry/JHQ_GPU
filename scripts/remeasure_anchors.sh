#!/bin/bash
# Repeat the two timing points the paper leans on hardest, five times each.
#
# Both are quoted from a single measurement, and both are the kind of number a
# reviewer checks first:
#
#   vogue-768, CPU, nprobe=1024, 32 threads.  The 90x speedup at Recall@10
#   0.9905 interpolates between this cell's alpha=32 and alpha=100 points, and
#   the six alpha settings in it span 792 to 1,072 QPS -- 35% -- so one
#   measurement cannot support the figure.
#
#   vogue-768, batch 512, matched Recall@10 0.95.  The paper reports 1.01x
#   against IVF-RaBitQ. Claiming a one percent win from one run is not a claim.
#
# Runs are interleaved rather than blocked: five passes over both arms, so a
# thermal or contention drift shows up as spread within a pass rather than as a
# clean difference between arms. First pass of each arm is a warm-up and is
# labelled as such.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && (setsid bash scripts/remeasure_anchors.sh </dev/null >/dev/null 2>&1 &)
set -u
L=/root/anchors.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

R=/root/JHQ_paper2                       # the CPU reference build
timeout 1800 cmake --build "$R/build" --target bench_vogue768 -j 16 >/dev/null 2>&1
echo "cpu_build_rc=$?"
timeout 2400 cmake -S . -B build >/dev/null 2>&1
timeout 2400 cmake --build build -j 16 --target demo_jhq_v57 bench_rq_batch2 >/dev/null 2>&1
rc=$?; echo "gpu_build_rc=$rc"
[ "$rc" -ne 0 ] && { echo "=== ANCHORS_DONE build failed, nothing run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
C=$D/paper_cache/vogue-768

cpu_pass(){   # $1 = pass label
  echo "--- CPU pass=$1 $(date -u +%T)"
  OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores \
  JHQ_NLIST=4096 JHQ_M=96 JHQ_BR=8 JHQ_ALPHA=32.0,100.0 JHQ_NPROBE_MAX=1024 \
  timeout 7200 "$R/build/examples/bench_vogue768" $VG 2>&1 | grep -E "^(32|100)\.0 +1024"
  echo "cpu_rc=$? pass=$1"
}
gpu_pass(){   # $1 = pass label
  echo "--- GPU pass=$1 $(date -u +%T)"
  for np in 128 256 512; do
    env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=512 JHQ_TILE_M_RT=96 JHQ_N_TRAIN=159744 \
        timeout 9000 build/demo_jhq_v57 $VG 96 8 8 64.0 10 4096 $np 8 512 "" 3 \
      2>/dev/null | awk -v n=$np -v p="$1" \
        '/^Recall@10/{r=$3} /^QPS/{q=$3} END{printf "JHQ  np=%-5s pass=%-4s recall=%s qps=%s\n",n,p,r,q}'
    env JHQ_RQ_BATCH=512 timeout 9000 build/bench_rq_batch2 $VG 96 8 4096 10 2 3 \
      2>/dev/null | awk -v n=$np -v p="$1" '/np='"$np"'/{print "RQ   np="n" pass="p" "$0}'
  done
  echo "gpu_rc=$? pass=$1"
}

for pass in warmup 1 2 3 4 5; do
  cpu_pass "$pass"
  gpu_pass "$pass"
done
echo "=== ANCHORS_DONE $(date -u +%FT%TZ) ==="
