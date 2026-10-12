#!/bin/bash
# Phase 0 of experiments/pipeline_v62: freeze the v61 baseline (refinement
# lanes 16) and find where a search's time goes at batch sizes 8, 32, 256 and
# 1000, before writing any v62 code.
#
# Configurations (the paper's index: run_budget_arms6.sh settings, arms6
# cache): arxiv/128 at alpha 100 (the sampled rule's) and 200 (full pool),
# vogue/128 at 64, openai3-3072/128 at 4.
#
# Two runs of demo_jhq_v61_batch per dataset:
#   plain  5 rounds, no profiler -- the end-to-end latency and throughput;
#   nsys   1 round under nsys with graph-node tracing -- every kernel's duration
#          under its NVTX range, for the per-stage breakdown.
# stage_times.py joins them. Nothing here reads or writes credentials.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v61_batch -j8
#   nohup bash experiments/pipeline_v62/run_phase0.sh > results/pipeline_v62_phase0.log 2>&1 & tail -f /root/JHQ_GPU/results/pipeline_v62_phase0.log
# (or the same from a worktree: git worktree add --detach <dir> origin/<branch>)
set -u
exec 9>/root/.lock_pipeline_v62; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
# The repository this script lives in -- /root/JHQ_GPU, or a separate worktree
# when that one holds someone else's uncommitted work (it did, 2026-10-13).
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_v62/phase0
NS=/root/autodl-tmp/pipeline_v62_nsys
mkdir -p $R $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/GPU: /'
# Release and compute_80, as every earlier build on this box: a fresh worktree
# would otherwise get CMakeLists' default architecture.
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v61_batch >/dev/null 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { echo "=== PHASE0""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/demo_jhq_v61_batch.dir/flags.make; nvcc --version | tail -2; nsys --version; } > $R/build_env.txt 2>&1

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024

# h <dataset> <alphas> <out prefix> <rounds> [command prefix...]
h() {
  local ds=$1 al=$2 pre=$3 ro=$4; shift 4
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$ro JHQ_BP_LANES=16 \
      timeout 7200 "$@" build/demo_jhq_v61_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 "$pre" 3
}

for c in arxiv-768:100,200 vogue-768:64 openai3-3072:4; do
  IFS=: read ds al <<< "$c"
  [ -d "$D/arms6_cache/$ds" ] || echo "WARNING: no arms6 cache for $ds -- retrained index, not the paper's"
  h $ds $al $R/$ds 5 > $R/$ds.txt 2>&1
  echo "plain $ds rc=$? $(grep -c BP_OK $R/$ds.txt) ok"
  h $ds $al $NS/$ds 1 nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
      --force-overwrite true -o $NS/$ds > $R/${ds}_nsys.txt 2>&1
  rc=$?
  nsys stats -q -r cuda_gpu_trace:nvtx-name -f csv -o $R/${ds}_trace $NS/$ds.nsys-rep > /dev/null 2>&1
  echo "nsys $ds rc=$rc $(ls $R/${ds}_trace*.csv 2>/dev/null | wc -l) csv"
done

python3 experiments/pipeline_v62/stage_times.py $R --md $R/stages.md
echo "=== PHASE0""_DONE $(date -u +%FT%TZ) ==="
