#!/bin/bash
# B0: the v65 Recall-QPS frontier on the paper's indexes (arms6 cache), alpha and nprobe re-tuned around the paper's
# configuration. demo_jhq_v65_batch, variants 0 (v61, the reference the harness requires first) and -1 (v65, the scheduled G; the one reported), batches 8, 32, 256, 1000, 3 rounds, all 1000
# queries; the same build and timing scope as experiments/pipeline_campaign/exp47.
#   bash b0_frontier.sh <out dir>      (run from the repository root; called by run_stage.sh b0)
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
OUT=$1; mkdir -p $OUT
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
timeout 2400 cmake --build build -j 16 --target demo_jhq_v65_batch > $OUT/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $OUT/build.log | head; echo "B0 build failed, nothing run"; exit 1; }
exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv > $OUT/gpu.txt 2>&1
D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
for c in arxiv-768:128:25,50,75,100,150,200 vogue-768:128:16,32,48,64,96,128 openai3-3072:128:2,3,4,6,8 \
         arxiv-768:512:50,100,200,400 vogue-768:512:32,64,128 openai3-3072:512:2,4,8 \
         arxiv-768:32:50,100,200 arxiv-768:64:50,100,200 vogue-768:32:32,64,128 vogue-768:64:32,64,128 openai3-3072:32:2,4,8 openai3-3072:64:2,4,8; do
  IFS=: read ds np al <<< "$c"
  tag=${ds}_np${np}
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=3 JHQ_BP_CHECK=0 \
      JHQ_BP_LANES=16 JHQ_BP_GROUPS=0,-1 JHQ_BP_BATCHES=8,32,256,1000 \
      timeout 3600 build/demo_jhq_v65_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} $np 8 1024 $OUT/$tag 3 \
      > $OUT/$tag.txt 2>&1
  echo "B0 $tag rc=$? $(grep -c BP_OK $OUT/$tag.txt) ok  $(date -u +%T)"
done
flock -u 8
echo "B0_DONE"
