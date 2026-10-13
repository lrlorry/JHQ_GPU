#!/bin/bash
# Query-adaptive progressive primary screening pilot (offline).
#
# 1  export: demo_jhq_v66_export_primary on the paper's index (the arms6 cache, built exactly as
#    experiments/local_first_refinement built it): the primary codes [N][M] decoded from the word
#    layout, the list offsets, the levels with a bitwise symmetry check and a check of every
#    codebook entry, and q' for the same 1000 seeded queries. The GPU only loads the cached index
#    and encodes the base: no search, no training.
# 2  selftest.py: analyze.py on a synthetic index against independent re-implementations.
# 3  analyze.py: policies A0..A3 (and A0c/A1c) against full(ck) = v65 and full(T*) on 40/20/40
#    splits, with the per-candidate files of experiments/local_first_refinement
#    (/root/autodl-tmp/lf_pilot: v65's probe order, FP64 D_prim and D_H, ground truth).
#
# On the box (detached): /root/autodl-tmp/launch_b.sh exp/progressive-primary-pilot \
#     experiments/progressive_primary/run_pilot.sh progressive.log
set -u
exec 9>/root/.lock_progressive; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/progressive_primary
W=/root/autodl-tmp/pp_pilot
LF=/root/autodl-tmp/lf_pilot
mkdir -p $R/raw $W
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v66_export_primary > $R/raw/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/raw/build.log | head -20; echo "=== PP""_DONE build failed, nothing run ==="; exit 1; }

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
      timeout 3600 build/demo_jhq_v66_export_primary ${P[$ds]} ${MM[$ds]} 8 8 $a 10 ${NL[$ds]} $np 8 1024 $W/$tag 3 \
      > $R/raw/export_$tag.txt 2>&1
  echo "export $tag rc=$? $(grep -c EP_OK $R/raw/export_$tag.txt) ok | $(grep -m1 '^levels' $R/raw/export_$tag.txt) $(date -u +%T)"
  cp $W/$tag/meta.json $R/raw/meta_$tag.json 2>/dev/null
done
flock -u 8

rm -rf $D/pp_selftest && mkdir -p $D/pp_selftest
python3 experiments/progressive_primary/selftest.py $D/pp_selftest > $R/raw/selftest.txt 2>&1
rc=$?; echo "selftest rc=$rc $(tail -1 $R/raw/selftest.txt)"
[ $rc -ne 0 ] && { echo "=== PP""_DONE selftest failed, nothing analysed ==="; exit 1; }
python3 experiments/progressive_primary/analyze.py $LF $W $R 96 > $R/raw/analyze.txt 2>&1
echo "analyze rc=$? $(tail -1 $R/raw/analyze.txt)"
echo "=== PP""_DONE $(date -u +%FT%TZ) ==="
