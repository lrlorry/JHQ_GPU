#!/bin/bash
# v61 warp-cooperative refinement: correctness, then the six-dataset comparison.
#
# Every configuration is the paper's: run_budget_arms6.sh's index settings and
# arms6 trained-state cache (so the index is the one Table budget_current
# measured), np = 128 and 512, and the sampled-rule alpha that table reports.
# arxiv/128 also runs alpha=200, which is the full-pool/oracle budget there,
# not the rule's.
#
#   A  demo_jhq_v61_ab: one process per (dataset, np); every kernel -- lanes 0
#      (v58's), 1 (32-bit loads only), 8, 16, 32 -- on the same index, queries
#      and timing scope. Per-query ids and distances against lanes 0, recall,
#      5 rounds x 3 timed searches, kernel order rotated per round.
#   B  nsys, one process per (dataset, np, alpha), 1 round x 5 searches: the
#      refinement kernel's own GPU time per launch, by kernel instantiation.
#   C  compute-sanitizer memcheck on the refine and top-k kernels for all six
#      datasets (np=128), racecheck and synccheck on two; 64 queries.
#   D  summarize_warp.py -> results/warp_refine/summary.md
#
# Outputs go to results/warp_refine/ (small) and /root/autodl-tmp/warp_refine_nsys/
# (the .nsys-rep files). Nothing here reads or writes credentials.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v61_ab -j8
#   nohup bash experiments/warp_refine/run_warp_refine.sh > results/warp_refine.log 2>&1 & tail -f /root/JHQ_GPU/results/warp_refine.log
set -u
exec 9>/root/.lock_warp_refine; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd /root/JHQ_GPU
R=results/warp_refine
NS=/root/autodl-tmp/warp_refine_nsys
mkdir -p $R/ab $R/nsys $R/sanitizer $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v61_ab >/dev/null 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { echo "=== WARP_REFINE""_DONE build failed, nothing run ==="; exit 1; }

{
  echo "git $(git rev-parse HEAD)"; echo "--- untracked/modified on the box"; git status --short | head -20
  echo "--- GPU"; nvidia-smi --query-gpu=name,driver_version,vbios_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,persistence_mode --format=csv
  nvidia-smi -q -d CLOCK | sed -n 1,40p
  echo "--- toolchain"; nvcc --version | tail -2; nsys --version; compute-sanitizer --version | tail -1
  cmake --version | head -1; g++ --version | head -1
  echo "--- CMake cache"; grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS|CXX_FLAGS|CUDA_COMPILER)" build/CMakeCache.txt
  echo "--- jhq_v61_exp compile flags"; cat build/CMakeFiles/jhq_v61_exp.dir/flags.make
  echo "--- demo_jhq_v61_ab compile flags"; cat build/CMakeFiles/demo_jhq_v61_ab.dir/flags.make
  echo "--- host"; uname -sr; nproc; free -g | head -2
} > $R/env.txt 2>&1
echo "environment -> $R/env.txt"

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-1536]="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
P[bge-m3]="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
P[stella]="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
# M, nlist, n_train, BLOCK: run_budget_arms6.sh's (the frontier's) per dataset
MM[vogue-768]=96;    NL[vogue-768]=4096;  NT[vogue-768]=159744;  BLK[vogue-768]=512
MM[arxiv-768]=96;    NL[arxiv-768]=8192;  NT[arxiv-768]=319488;  BLK[arxiv-768]=512
MM[openai3-1536]=192; NL[openai3-1536]=8192; NT[openai3-1536]=319488; BLK[openai3-1536]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
MM[bge-m3]=128;      NL[bge-m3]=32768;    NT[bge-m3]=1277952;    BLK[bge-m3]=512
MM[stella]=128;      NL[stella]=32768;    NT[stella]=1277952;    BLK[stella]=512

# dataset np alphas -- alphas: the sampled-rule median of Table budget_current
CONFIGS="vogue-768:128:64 vogue-768:512:64 arxiv-768:128:100,200 arxiv-768:512:200
openai3-1536:128:8 openai3-1536:512:8 openai3-3072:128:4 openai3-3072:512:4
bge-m3:128:32 bge-m3:512:64 stella:128:16 stella:512:16"

# h <dataset> <np> <alphas> <out prefix> <rounds> <reps> <nq or ""> [command prefix...]
# One harness process; the command prefix (nsys, compute-sanitizer) wraps it.
h() {
  local ds=$1 np=$2 al=$3 pre=$4 ro=$5 re=$6 nqv=$7; shift 7
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_AB_ALPHAS="$al" JHQ_AB_ROUNDS=$ro JHQ_AB_REPS=$re \
      ${nqv:+JHQ_AB_NQ=$nqv} \
      timeout 7200 "$@" build/demo_jhq_v61_ab ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} $np 8 1024 "$pre" 3
}

echo "### A: results and timing $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds np al <<< "$c"
  [ -d "$D/arms6_cache/$ds" ] || echo "WARNING: no arms6 cache for $ds -- retrained index, not the paper's"
  tag=${ds}_np${np}
  h $ds $np $al $R/ab/$tag 5 3 "" > $R/ab/$tag.txt 2>&1
  echo "A $tag alphas=$al rc=$? $(grep -c 'AB_OK' $R/ab/$tag.txt) ok"
done

echo "### B: kernel time (nsys) $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds np al <<< "$c"
  for a in ${al//,/ }; do
    tag=${ds}_np${np}_a${a}
    h $ds $np $a $NS/$tag 1 5 "" nsys profile -t cuda --cuda-graph-trace=node --sample=none \
        --cpuctxsw=none --force-overwrite true -o $NS/$tag > $R/nsys/$tag.txt 2>&1
    rc=$?
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o $R/nsys/$tag $NS/$tag.nsys-rep > /dev/null 2>&1
    echo "B $tag rc=$rc $(ls $R/nsys/${tag}_cuda_gpu_kern_sum.csv 2>/dev/null | wc -l) csv"
  done
done

echo "### C: memory safety (compute-sanitizer, 64 queries) $(date -u +%T)"
san() {  # tool ds np alpha
  local tool=$1 ds=$2 np=$3 a=$4 tag=${2}_np${3}_${1}
  h $ds $np $a $NS/san_$tag 1 1 64 compute-sanitizer --tool $tool --error-exitcode 99 \
      --print-limit 50 --kernel-name kns=residual_refine --kernel-name kns=batched_topk_final \
      > $R/sanitizer/$tag.txt 2>&1
  echo "C $tag rc=$? $(grep -m1 'ERROR SUMMARY' $R/sanitizer/$tag.txt)"
}
for c in vogue-768:128:64 arxiv-768:128:200 openai3-1536:128:8 openai3-3072:128:4 bge-m3:128:32 stella:128:16; do
  IFS=: read ds np a <<< "$c"; san memcheck $ds $np $a
done
for c in vogue-768:128:64 openai3-3072:128:4; do
  IFS=: read ds np a <<< "$c"; san racecheck $ds $np $a; san synccheck $ds $np $a
done

echo "### D: summary $(date -u +%T)"
python3 experiments/warp_refine/summarize_warp.py $R
echo "=== WARP_REFINE""_DONE $(date -u +%FT%TZ) ==="
