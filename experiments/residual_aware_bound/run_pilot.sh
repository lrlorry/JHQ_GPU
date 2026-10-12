#!/bin/bash
# Residual-aware factorized screening: the offline pilot (no new search system).
#
#   1  technical validation: 16 seeded queries per workload; audit of the
#      representation and LB <= D_H on every candidate of the probed lists.
#      Any audit mismatch or bound violation stops here.
#   2  the study: 128 seeded queries (the 16 are its first 16) per workload,
#      then analyze.py: bounds, survivors, progressive refinement, recall,
#      figures.
#
# Workloads: arxiv-768/128 alpha 100 (ck 1000), vogue-768/128 alpha 64 (640),
# openai3-3072/128 alpha 4 (40); then arxiv/128 alpha 200 and np = 512 for the
# three. The paper's index (arms6 cache), never retrained. Per-query candidate
# files (hundreds of MB) stay in /root/autodl-tmp/rb_pilot.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v66_bound_pilot -j8
#   nohup bash experiments/residual_aware_bound/run_pilot.sh > results/residual_bound.log 2>&1 & tail -f results/residual_bound.log
set -u
exec 9>/root/.lock_residual_bound; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/residual_aware_bound
W=/root/autodl-tmp/rb_pilot
mkdir -p $R/raw $W
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v66_bound_pilot > $R/raw/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/raw/build.log | head -20; echo "=== RB""_DONE build failed, nothing run ==="; exit 1; }
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
# r <dataset> <np> <alpha> <nq> <tag>
r() {
  local ds=$1 np=$2 a=$3 nq=$4 tag=$5
  mkdir -p $W/$tag
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_RB_NQ=$nq JHQ_RB_SEED=20261013 OMP_NUM_THREADS=128 \
      timeout 3600 build/demo_jhq_v66_bound_pilot ${P[$ds]} ${MM[$ds]} 8 8 $a 10 ${NL[$ds]} $np 8 1024 $W/$tag/rb 3 \
      > $R/raw/$tag.txt 2>&1
  local rc=$?
  cp $W/$tag/rb_summary.json $R/raw/${tag}_summary.json 2>/dev/null
  echo "$tag rc=$rc $(grep -c RB_OK $R/raw/$tag.txt) ok | $(grep -m1 '^audit:' $R/raw/$tag.txt) | $(grep -m1 '^queries:' $R/raw/$tag.txt)  $(date -u +%T)"
}

echo "### 1: technical validation, 16 queries $(date -u +%T)"
for c in arxiv-768:128:100 vogue-768:128:64 openai3-3072:128:4; do
  IFS=: read ds np a <<< "$c"
  r $ds $np $a 16 v16_${ds}_np${np}_a${a}
done
BAD=$(python3 - <<'EOF'
import json, glob
bad = 0
for f in glob.glob("results/residual_aware_bound/raw/v16_*_summary.json"):
    J = json.load(open(f)); a = J["audit"]; v = J["validation"]
    bad += (a["separability_mismatches"] > 0) + (a["level_mismatches"] > 0) + (v["violations_cond"] > 0) \
         + (v["violations_glob"] > 0) + (v["nan_bounds"] > 0) + (a["comp_minus_DH_minus_qnorm2_max_rel"] > 1e-4)
print(bad)
EOF
)
echo "validation problems: $BAD"
if [ "$BAD" != "0" ]; then echo "=== RB""_DONE validation FAILED: stopping before any pruning study ==="; exit 1; fi

echo "### 2: study, 128 queries $(date -u +%T)"
for c in arxiv-768:128:100 vogue-768:128:64 openai3-3072:128:4 arxiv-768:128:200 \
         arxiv-768:512:100 vogue-768:512:64 openai3-3072:512:4; do
  IFS=: read ds np a <<< "$c"
  r $ds $np $a 128 ${ds}_np${np}_a${a}
done
python3 experiments/residual_aware_bound/analyze.py $W $R > $R/raw/analyze.txt 2>&1
echo "analyze rc=$? $(tail -1 $R/raw/analyze.txt)"
echo "=== RB""_DONE $(date -u +%FT%TZ) ==="
