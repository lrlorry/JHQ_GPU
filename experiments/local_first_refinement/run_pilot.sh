#!/bin/bash
# Local-first refinement pilot, Phases A and B (offline).
#
# 1  export: demo_jhq_v66_bound_pilot (experiments/residual_aware_bound's
#    read-only exporter on the paper's index, v65's own search) for ALL 1000
#    queries of each workload: per probed candidate, in v65's probe order, the
#    FP64 primary distance D_prim and reconstructed distance D_H, plus v65's
#    final top-10 and the ground truth. No retraining; nothing new on the GPU
#    beyond v65's search of these queries.
# 2  analyze.py: group ranks (Phase A), recall against budget for global and
#    local selection (Phase B) on calibration / validation / test splits.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v66_bound_pilot -j8
#   nohup bash experiments/local_first_refinement/run_pilot.sh > results/local_first.log 2>&1 & tail -f results/local_first.log
set -u
exec 9>/root/.lock_local_first; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/local_first_refinement
W=/root/autodl-tmp/lf_pilot
mkdir -p $R/raw $W
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v66_bound_pilot > $R/raw/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/raw/build.log | head -20; echo "=== LF""_DONE build failed, nothing run ==="; exit 1; }

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
  mkdir -p $W/$tag
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_RB_NQ=1000 JHQ_RB_SEED=20261013 OMP_NUM_THREADS=128 \
      timeout 3600 build/demo_jhq_v66_bound_pilot ${P[$ds]} ${MM[$ds]} 8 8 $a 10 ${NL[$ds]} $np 8 1024 $W/$tag/rb 3 \
      > $R/raw/export_$tag.txt 2>&1
  echo "export $tag rc=$? $(grep -c RB_OK $R/raw/export_$tag.txt) ok | $(grep -m1 '^v65:' $R/raw/export_$tag.txt) | $(ls $W/$tag/rb_q*.bin | wc -l) query files  $(date -u +%T)"
done
flock -u 8
python3 experiments/local_first_refinement/analyze.py $W $R > $R/raw/analyze.txt 2>&1
echo "analyze rc=$? $(tail -1 $R/raw/analyze.txt)"
echo "=== LF""_DONE $(date -u +%FT%TZ) ==="
