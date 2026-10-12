#!/bin/bash
# Bound-aware residual quantization: offline pilot (no new search system).
# The three workloads of experiments/residual_aware_bound at np = 128, its 128
# seeded queries, the paper's index (arms6 cache). analyze.py then writes the
# tables and figures; io_model.py the memory/IO and cost model.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v66_bound_aware_quant -j8
#   nohup bash experiments/bound_aware_quant/run_pilot.sh > results/bound_aware_quant.log 2>&1 & tail -f results/bound_aware_quant.log
set -u
exec 9>/root/.lock_bound_aware; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/bound_aware_quant
mkdir -p $R/raw
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v66_bound_aware_quant > $R/raw/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/raw/build.log | head -20; echo "=== BQ""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/jhq_v66_exp.dir/flags.make; } > $R/raw/build_env.txt 2>&1

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
for c in arxiv-768:128:100 vogue-768:128:64 openai3-3072:128:4; do
  IFS=: read ds np a <<< "$c"
  tag=${ds}_np${np}_a${a}
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_RB_NQ=128 JHQ_RB_SEED=20261013 OMP_NUM_THREADS=192 \
      timeout 5400 build/demo_jhq_v66_bound_aware_quant ${P[$ds]} ${MM[$ds]} 8 8 $a 10 ${NL[$ds]} $np 8 1024 $R/raw/$tag 3 \
      > $R/raw/$tag.txt 2>&1
  echo "$tag rc=$? $(grep -c BQ_OK $R/raw/$tag.txt) ok | $(grep -m1 '^queries:' $R/raw/$tag.txt | cut -c1-400)  $(date -u +%T)"
  grep -E "^coarse" $R/raw/$tag.txt
done
python3 experiments/bound_aware_quant/analyze.py $R > $R/raw/analyze.txt 2>&1
echo "analyze rc=$? $(tail -1 $R/raw/analyze.txt)"
echo "=== BQ""_DONE $(date -u +%FT%TZ) ==="
