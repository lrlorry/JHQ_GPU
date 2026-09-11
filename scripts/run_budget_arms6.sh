#!/bin/bash
# The five-arm budget comparison on all six datasets, not two.
#
# Table 2 covers vogue-768 and openai3-3072 at two probe depths.  The script
# that produced it named those two and gave no reason; the binary costs about
# a second a cell -- the 64 repeats are calibration draws resampled over
# results already computed, not 64 searches -- so the whole original run took
# 55 seconds.  Four datasets were left out of the paper's second contribution
# for no cost that was ever paid.
#
# Settings follow the frontier's, per dataset, rather than the original
# script's.  That matters for openai3-3072, which the original ran at
# nlist=4096 and BLOCK=512 while the frontier uses 8192 and 1024; aligning
# them removes the caveat that the budget experiment and the frontier are
# separate protocols on that dataset.
#
# Serial behind whatever holds the GPU: these rows carry QPS, and a
# concurrent benchmark would perturb exactly the quantity the payback
# estimate divides by.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_budget_arms6.sh /root/arms6.sh \
#     && setsid nohup bash /root/arms6.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_arms6; flock -n 9 || { echo "already running"; exit 0; }
L=/root/arms6.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git -C /root/JHQ_GPU log --oneline -1) ==="
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_budget_arms >/dev/null 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { echo "=== ARMS6""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
CACHE=/root/autodl-tmp/arms6_cache
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths M nlist n_train BLOCK nprobe
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 blk=$6 np=$7
  local C=$CACHE/${tag}; mkdir -p "$C"
  echo "### $tag M=$M nlist=$nl np=$np $(date -u +%T)"
  # JHQ_RES_TRAIN_N=100000 is the frontier's setting (/root/_pa.sh).  The old
  # arms.sh omitted it and got away with a warm cache; without one the residual
  # codebook trains on the whole set, which is both slow and a different index
  # from the one the frontier reports -- defeating the point of aligning them.
  env JHQ_INDEX_CACHE="$C" JHQ_BLOCK=$blk JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt \
      JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 \
      JHQ_AS_GRID=200,100,64,32,16,8,4,2 JHQ_AS_SLOTS=1 \
      JHQ_ARM_S=32,64,128 JHQ_ARM_REPS=64 JHQ_ORACLE_TAU=0.001 \
      timeout 9000 build/demo_jhq_budget_arms $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3
  echo "run_rc=$? tag=$tag np=$np"
}

# tag paths M nlist n_train BLOCK -- the frontier's settings for each dataset
for np in 128 512; do
  one vogue-768    "$VG"  96  4096  159744  512  $np
  one arxiv-768    "$AX"  96  8192  319488  512  $np
  one openai3-1536 "$O15" 192 8192  319488  512  $np
  one openai3-3072 "$O30" 384 8192  319488  1024 $np
  one bge-m3       "$BG"  128 32768 1277952 512  $np
  one stella       "$ST"  128 32768 1277952 512  $np
done
echo "=== ARMS6""_DONE $(date -u +%FT%TZ) ==="
