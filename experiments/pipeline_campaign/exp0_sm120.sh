#!/bin/bash
# Campaign Experiment 0, the part Phase 0 of pipeline_v62 did not cover: does a
# native sm_120 build behave like the compute_80 PTX build every earlier number
# came from (the driver JIT-compiles that PTX for the RTX 5090)?
#
#   A  demo_jhq_v61_ab, refinement lanes 0 and 16 paired in one process (the
#      v61 report's harness and timing scope: all queries, batches of 1024,
#      5 rounds x 3 searches, lanes order rotated), built twice from the same
#      commit: build/ (compute_80) and build_sm120/ (sm_120). The two builds
#      run in alternating order per workload. Paired L16/L0 within a process;
#      sm_120/compute_80 across processes.
#   B  demo_jhq_v62_batch built for sm_120, the pipeline_v62 Phase 3 sweep
#      (G = 0, 1, 2, 4, 8; b = 8, 32, 256, 1000; 5 rounds), so the v62 speedups
#      can be read for both builds -- each against its own G = 0, never mixed.
#
# Workloads: pipeline_v62's (arxiv/128 at alpha 100 and 200, vogue/128 at 64,
# openai3-3072/128 at 4), the arms6 index. Nothing here reads or writes
# credentials.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build_sm120 -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build_sm120 --target demo_jhq_v61_ab demo_jhq_v62_batch -j8
#   nohup bash experiments/pipeline_campaign/exp0_sm120.sh > results/pipeline_campaign_exp0.log 2>&1 & tail -f results/pipeline_campaign_exp0.log
set -u
exec 9>/root/.lock_campaign_exp0; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_campaign/exp0
mkdir -p $R/ab $R/batch
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
for spec in build:80 build_sm120:120; do
  bd=${spec%%:*}; arch=${spec##*:}
  timeout 600 nice -n 10 cmake -S . -B $bd -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$arch >/dev/null 2>&1
  echo "configure $bd rc=$?"
  timeout 2400 nice -n 10 cmake --build $bd -j 16 --target demo_jhq_v61_ab demo_jhq_v62_batch > $R/build_$arch.log 2>&1
  rc=$?; echo "build $bd (arch $arch) rc=$rc"
  [ $rc -ne 0 ] && { grep -E "error" $R/build_$arch.log | head -20; echo "=== EXP0""_DONE build failed, nothing run ==="; exit 1; }
  { echo "--- $bd"; grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" $bd/CMakeCache.txt
    cat $bd/CMakeFiles/jhq_v61_exp.dir/flags.make $bd/CMakeFiles/jhq_v62_exp.dir/flags.make
    cuobjdump --list-elf $bd/demo_jhq_v62_batch 2>/dev/null | head -5
    cuobjdump --list-ptx $bd/demo_jhq_v62_batch 2>/dev/null | head -5; } >> $R/build_env.txt 2>&1
done
{ nvcc --version | tail -2; nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; } >> $R/build_env.txt 2>&1

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
CONFIGS="arxiv-768:100,200 vogue-768:64 openai3-3072:4"
envs() {  # dataset alphas
  echo JHQ_INDEX_CACHE="$D/arms6_cache/$1" JHQ_BLOCK=${BLK[$1]} JHQ_TILE_M_RT=${MM[$1]} \
       JHQ_N_TRAIN=${NT[$1]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
       JHQ_RES_TRAIN_N=100000
}

echo "### A: v61 lanes 0 vs 16, compute_80 vs sm_120 $(date -u +%T)"
i=0
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  order="build:80 build_sm120:120"; [ $((i % 2)) -eq 1 ] && order="build_sm120:120 build:80"; i=$((i + 1))
  for spec in $order; do
    bd=${spec%%:*}; arch=${spec##*:}
    env $(envs $ds) JHQ_AB_ALPHAS="$al" JHQ_AB_LANES=0,16 JHQ_AB_ROUNDS=5 JHQ_AB_REPS=3 \
        timeout 3600 $bd/demo_jhq_v61_ab ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 \
        $R/ab/${ds}_sm$arch 3 > $R/ab/${ds}_sm$arch.txt 2>&1
    echo "A $ds arch=$arch rc=$? $(grep -c AB_OK $R/ab/${ds}_sm$arch.txt) ok"
  done
done
rm -f $R/ab/*.ivecs $R/ab/*.fvecs   # per-query results: compared in-process, not needed here

echo "### B: v62 batch sweep, sm_120 build $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  env $(envs $ds) JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=5 JHQ_BP_CHECK=1 JHQ_BP_LANES=16 \
      JHQ_BP_GROUPS=0,1,2,4,8 JHQ_BP_BATCHES=8,32,256,1000 \
      timeout 3600 build_sm120/demo_jhq_v62_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 \
      $R/batch/${ds}_sm120 3 > $R/batch/${ds}_sm120.txt 2>&1
  echo "B $ds rc=$? $(grep -c BP_OK $R/batch/${ds}_sm120.txt) ok; NON-TIE sum: $(grep -o 'NON-TIE [0-9]*' $R/batch/${ds}_sm120.txt | awk '{s += $2} END {print s + 0}')"
done
echo "=== EXP0""_DONE $(date -u +%FT%TZ) ==="
