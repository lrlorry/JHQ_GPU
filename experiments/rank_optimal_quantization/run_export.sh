#!/bin/bash
# Rank-optimal quantization study, step 0: read-only export of what an offline re-encoding of the
# paper index needs (demo_jhq_v66_export_quant: the rotation, residual codebooks and codes, the
# stored correction, list ids), for the three workloads of experiments/progressive_primary, whose
# export (primary codes, list offsets, rotated queries) and whose candidate files
# (/root/autodl-tmp/lf_pilot) the study reuses. The GPU only loads the cached index (arms6 cache)
# and encodes the base: no search, no training.
#
# On the box (detached): /root/autodl-tmp/launch_b.sh exp/rank-optimal-quantization \
#     experiments/rank_optimal_quantization/run_export.sh roq_export.log
set -u
exec 9>/root/.lock_roq; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/rank_optimal_quantization
W=/root/autodl-tmp/roq
mkdir -p $R/raw $W
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v66_export_quant > $R/raw/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/raw/build.log | head -20; echo "=== ROQ""_EXPORT_DONE build failed, nothing run ==="; exit 1; }

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
      JHQ_RES_TRAIN_N=100000 OMP_NUM_THREADS=128 \
      timeout 3600 build/demo_jhq_v66_export_quant ${P[$ds]} ${MM[$ds]} 8 8 $a 10 ${NL[$ds]} $np 8 1024 $W/$tag 3 \
      > $R/raw/export_$tag.txt 2>&1
  echo "export $tag rc=$? $(grep -c EQ_OK $R/raw/export_$tag.txt) ok | $(grep -m1 '^wrote' $R/raw/export_$tag.txt) $(date -u +%T)"
  cp $W/$tag/meta.json $R/raw/meta_quant_$tag.json 2>/dev/null
done
flock -u 8
echo "=== ROQ""_EXPORT_DONE $(date -u +%FT%TZ) ==="
