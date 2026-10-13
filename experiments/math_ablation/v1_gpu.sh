#!/bin/bash
# V1 E8-JHQ on the GPU: demo_jhq_v68_e8_bench (v65's search through the full 256-entry primary table) on the paper
# index (A) and on the imported E8 representation (B), alpha swept for a recall-QPS frontier, batches 8, 32, 256,
# 1000, 3 rounds, nprobe 128 and 512. B0 (v65 itself, factorised table) is b0_frontier.sh.
#   bash v1_gpu.sh <out dir> <e8 variant: v1_e8 | v1_e8_grouped>      (run from the repository root)
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
OUT=$1; VAR=${2:-v1_e8}; mkdir -p $OUT
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
timeout 2400 cmake --build build -j 16 --target demo_jhq_v68_e8_bench > $OUT/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $OUT/build.log | head -20; echo "V1GPU build failed, nothing run"; exit 1; }
for t in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
  OPENBLAS_NUM_THREADS=8 python3 experiments/math_ablation/v1_e8/implementation/export_e8.py $t $([ "$VAR" = v1_e8_grouped ] && echo grouped) > $OUT/export_$t.txt 2>&1
  echo "export $t rc=$? $(tail -1 $OUT/export_$t.txt | cut -c1-200)"
done
exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK TAG
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
TAG[arxiv-768]=arxiv-768_np128_a100; TAG[vogue-768]=vogue-768_np128_a64; TAG[openai3-3072]=openai3-3072_np128_a4
for c in arxiv-768:128:50,75,100,150 vogue-768:128:32,48,64,96 openai3-3072:128:2,3,4,6 \
         arxiv-768:512:100,200 vogue-768:512:32,64 openai3-3072:512:2,4; do
  IFS=: read ds np al <<< "$c"
  tag=${ds}_np${np}
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_E8_VARDIR=/root/autodl-tmp/math_ablation/${TAG[$ds]}/$VAR \
      JHQ_E8_ALPHAS_A="$al" JHQ_E8_ALPHAS_B="$al" JHQ_BP_ROUNDS=3 JHQ_BP_BATCHES=8,32,256,1000 OMP_NUM_THREADS=64 \
      timeout 3600 build/demo_jhq_v68_e8_bench ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} $np 8 1024 $OUT/$tag 3 \
      > $OUT/$tag.txt 2>&1
  echo "V1GPU $tag rc=$? $(grep -c E8_BENCH_OK $OUT/$tag.txt) ok  $(date -u +%T)"
done
flock -u 8
echo "V1GPU_DONE"
