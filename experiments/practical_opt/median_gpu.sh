#!/bin/bash
# Item 1: the rank-optimal study's per-dimension median thresholds on the GPU. demo_jhq_v67_repr_bench runs v65's
# search on the paper index (A) and on the imported median representation (B: median thresholds, conditional-mean
# levels, retrained residual, built offline in experiments/rank_optimal_quantization) side by side in one process:
# same rotation, lists, positions, kernels and timing scope; alpha swept for each so the recall-QPS frontiers can be
# compared at matched recall. Batches 8, 32, 256, 1000; 3 rounds, configuration order rotated.
#   /root/autodl-tmp/launch_b.sh exp/practical-opt experiments/practical_opt/median_gpu.sh po_median.log
set -u
exec 9>/root/.lock_po; flock 9
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
R=results/practical_opt/median; mkdir -p $R
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) median on the GPU ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
timeout 2400 cmake --build build -j 16 --target demo_jhq_v67_repr_bench > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head; echo "=== PO_MEDIAN_DONE build failed, nothing run ==="; exit 1; }
for t in arxiv-768_np128_a100 vogue-768_np128_a64 openai3-3072_np128_a4; do
  python3 experiments/rank_optimal_quantization/export_variant.py $t median > $R/export_$t.txt 2>&1
  echo "export $t rc=$? $(tail -1 $R/export_$t.txt)"
done
exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK TAG AA AB
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
TAG[arxiv-768]=arxiv-768_np128_a100; TAG[vogue-768]=vogue-768_np128_a64; TAG[openai3-3072]=openai3-3072_np128_a4
AA[arxiv-768]=50,75,100,150;  AB[arxiv-768]=25,32,40,50,75,100
AA[vogue-768]=32,48,64,96;    AB[vogue-768]=32,48,64,96,128
AA[openai3-3072]=2,3,4,6;     AB[openai3-3072]=2,3,4,6
for ds in arxiv-768 vogue-768 openai3-3072; do
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_RQ_VARDIR=$D/roq/${TAG[$ds]}/var_median \
      JHQ_RQ_ALPHAS_A=${AA[$ds]} JHQ_RQ_ALPHAS_B=${AB[$ds]} JHQ_BP_ROUNDS=3 JHQ_BP_BATCHES=8,32,256,1000 OMP_NUM_THREADS=64 \
      timeout 3600 build/demo_jhq_v67_repr_bench ${P[$ds]} ${MM[$ds]} 8 8 100 10 ${NL[$ds]} 128 8 1024 $R/${ds}_np128 3 \
      > $R/${ds}_np128.txt 2>&1
  echo "median $ds rc=$? $(grep -c RQ_BENCH_OK $R/${ds}_np128.txt) ok  $(date -u +%T)"
done
flock -u 8
echo "=== PO_MEDIAN_DONE $(date -u +%FT%TZ) ==="
