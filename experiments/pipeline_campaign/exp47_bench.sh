#!/bin/bash
# Campaign Experiments 4 and 7: the scheduled pipeline (v65) against every
# fixed choice, on the six datasets at np = 128 and 512 and the paper's
# sampled-rule alpha (run_warp_refine.sh's CONFIGS), batch sizes 8, 32, 256,
# 1000; one process per (dataset, np), so every variant searches the same
# index and queries with the same timing scope (wall clock around search():
# H2D, graph launch, D2H, sync), 5 rounds, variant order rotated, paired.
#
# Variants (demo_jhq_v65_batch, compute_80 PTX build as every earlier number):
#   0     v61, refinement lanes 16 (the campaign's baseline)
#   1000  v61, lanes 0 (the historical ablation)
#   102, 104, 108  v63: G = 2, 4, 8 list groups with the shared bound
#   -1    v65: G scheduled per batch (fixed rule, fitted on pipeline_v62's four
#         workloads only; its statistic calibrated on base-vector
#         pseudo-queries, never on the evaluation queries)
# The four pipeline_v62 workloads (vogue/128 a64, arxiv/128 a100 and a200,
# openai3-3072/128 a4) trained the rule; the other configurations are held out.
#
#   A  the benchmark, with per-query checks of every variant against 0
#   F  (if time) recall-QPS frontier: np = 128, alpha x1/2 and x2, variants 0
#      and -1, b = 8 and 1000, 3 rounds
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v65_batch -j8
#   nohup bash experiments/pipeline_campaign/exp47_bench.sh > results/pipeline_campaign_exp47.log 2>&1 & tail -f results/pipeline_campaign_exp47.log
set -u
exec 9>/root/.lock_campaign_exp47; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_campaign/exp47
mkdir -p $R/bench $R/frontier
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v65_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== EXP47""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/jhq_v65_exp.dir/flags.make
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; } > $R/build_env.txt 2>&1

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-1536]="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
P[bge-m3]="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
P[stella]="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-1536]=192; NL[openai3-1536]=8192; NT[openai3-1536]=319488; BLK[openai3-1536]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
MM[bge-m3]=128;       NL[bge-m3]=32768;      NT[bge-m3]=1277952;      BLK[bge-m3]=512
MM[stella]=128;       NL[stella]=32768;      NT[stella]=1277952;      BLK[stella]=512
# dataset:np:alphas -- the sampled-rule alpha of Table budget_current (arxiv/128 also 200)
CONFIGS="vogue-768:128:64 arxiv-768:128:100,200 openai3-3072:128:4 openai3-1536:128:8 bge-m3:128:32 stella:128:16
vogue-768:512:64 arxiv-768:512:200 openai3-3072:512:4 openai3-1536:512:8 bge-m3:512:64 stella:512:16"
# h <dataset> <np> <alphas> <prefix> <rounds> <check> <variants> <batches>
h() {
  local ds=$1 np=$2 al=$3 pre=$4 ro=$5 ck=$6 va=$7 bs=$8
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$ro JHQ_BP_CHECK=$ck \
      JHQ_BP_LANES=16 JHQ_BP_GROUPS=$va JHQ_BP_BATCHES=$bs \
      timeout 3600 build/demo_jhq_v65_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} $np 8 1024 "$pre" 3
}

echo "### A: benchmark $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds np al <<< "$c"
  [ -d "$D/arms6_cache/$ds" ] || echo "WARNING: no arms6 cache for $ds -- retrained index, not the paper's"
  tag=${ds}_np${np}
  h $ds $np $al $R/bench/$tag 5 1 0,1000,102,104,108,-1 8,32,256,1000 > $R/bench/$tag.txt 2>&1
  echo "A $tag rc=$? $(grep -c BP_OK $R/bench/$tag.txt) ok; $(grep -m1 'schedule:' $R/bench/$tag.txt); NON-TIE sum $(grep -o 'NON-TIE [0-9]*' $R/bench/$tag.txt | awk '{s += $2} END {print s + 0}')  $(date -u +%T)"
done
python3 experiments/pipeline_campaign/summarize_bench.py $R/bench $R/bench_summary.md > /dev/null
echo "summary rc=$?"

echo "### F: frontier $(date -u +%T)"
for c in vogue-768:128:32,128 arxiv-768:128:50,400 openai3-3072:128:2,8 openai3-1536:128:4,16 bge-m3:128:16,64 stella:128:8,32; do
  IFS=: read ds np al <<< "$c"
  tag=${ds}_np${np}
  h $ds $np $al $R/frontier/$tag 3 0 0,-1 8,1000 > $R/frontier/$tag.txt 2>&1
  echo "F $tag rc=$? $(grep -c BP_OK $R/frontier/$tag.txt) ok  $(date -u +%T)"
done
echo "=== EXP47""_DONE $(date -u +%FT%TZ) ==="
