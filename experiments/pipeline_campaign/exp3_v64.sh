#!/bin/bash
# Campaign Experiment 3 (v64, routing by Pi^T c), pilot on pipeline_v62's
# workloads (arxiv/128 at alpha 100 and 200, vogue/128 at 64, openai3-3072/128
# at 4; the arms6 index), batch sizes 8, 32, 256, 1000. Three routing modes in
# one process (demo_jhq_v64_batch): 0 = v61 (rotate, then score against c),
# 1 = score the unrotated query against Pi^T c, serial, 2 = as 1 with the
# routing branch on a second stream. 1 against 0 isolates the change of
# coordinates; 2 against 1 the concurrency.
#
#   A  per-query checks of 1 and 2 against 0: probe ids, centroid dot products
#      (the routing distance error is twice theirs), primary top-ck, final ids
#      and distances, recall; and the rotation's orthogonality.
#   B  compute-sanitizer memcheck of the whole pipeline, all three modes (64 queries).
#   gate  a NON-TIE primary difference on a query whose probes agree stops
#      here (that would be an ordering bug, not routing arithmetic).
#   C  5 plain rounds (paired, order rotated) and 1 nsys round per workload;
#      summarize_v64.py: e2e speedups, and per search the GPU span from the
#      first kernel to the scan against the sum of those kernels' times.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v64_batch -j8
#   nohup bash experiments/pipeline_campaign/exp3_v64.sh > results/pipeline_campaign_exp3.log 2>&1 & tail -f results/pipeline_campaign_exp3.log
set -u
exec 9>/root/.lock_campaign_exp3; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_campaign/exp3
NS=/root/autodl-tmp/campaign_exp3
mkdir -p $R/check $R/perf $R/sanitizer $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v64_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== EXP3""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/jhq_v64_exp.dir/flags.make; } > $R/build_env.txt 2>&1

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
# h <dataset> <alphas> <prefix> <rounds> <check> <nq or ""> <batches> [command prefix...]
h() {
  local ds=$1 al=$2 pre=$3 ro=$4 ck=$5 nqv=$6 bs=$7; shift 7
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$ro JHQ_BP_CHECK=$ck \
      JHQ_BP_LANES=16 JHQ_BP_ROUTES=0,1,2 JHQ_BP_BATCHES=$bs ${nqv:+JHQ_BP_NQ=$nqv} \
      timeout 3600 "$@" build/demo_jhq_v64_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 "$pre" 3
}

echo "### A: per-query checks $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  h $ds $al $R/check/$ds 0 1 "" 8,32,256,1000 > $R/check/$ds.txt 2>&1
  echo "check $ds rc=$? $(grep -c BP_OK $R/check/$ds.txt) ok; $(grep -m1 'rotation:' $R/check/$ds.txt)"
done
grep -h "route [12] vs 0" $R/check/*.txt > $R/check/all_routes.txt
grep -h "vs G=0" $R/check/*.txt > $R/check/all_comparisons.txt
BAD=$(awk '{ v = $NF; s += v } END { print s + 0 }' $R/check/all_routes.txt)
echo "comparisons: $(wc -l < $R/check/all_routes.txt); NON-TIE on queries with identical probes: $BAD"

echo "### B: memcheck, every kernel, modes 0, 1, 2 $(date -u +%T)"
h vogue-768 64 $NS/san_vogue 0 1 64 8,64 compute-sanitizer --tool memcheck --error-exitcode 99 \
    --print-limit 50 > $R/sanitizer/vogue-768_memcheck.txt 2>&1
echo "san vogue memcheck rc=$? $(grep -m1 'ERROR SUMMARY' $R/sanitizer/vogue-768_memcheck.txt)"

if [ "$BAD" -ne 0 ]; then
  echo "=== EXP3""_DONE correctness gate FAILED: no performance run ==="; exit 1
fi

echo "### C: performance $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  h $ds $al $R/perf/$ds 5 0 "" 8,32,256,1000 > $R/perf/$ds.txt 2>&1
  echo "plain $ds rc=$? $(grep -c BP_OK $R/perf/$ds.txt) ok"
  h $ds $al $NS/$ds 1 0 "" 8,32,256,1000 nsys profile -t cuda,nvtx --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none --force-overwrite true -o $NS/$ds > $R/perf/${ds}_nsys.txt 2>&1
  rc=$?
  nsys stats -q --force-export=true -r cuda_gpu_trace:nvtx-name -f csv -o $R/perf/${ds}_trace \
      $NS/$ds.nsys-rep > /dev/null 2> $R/perf/${ds}_stats_err.txt
  echo "nsys $ds rc=$rc $(ls $R/perf/${ds}_trace*.csv 2>/dev/null | wc -l) csv"
done
python3 experiments/pipeline_campaign/summarize_v64.py $R
gzip -f $R/perf/*_trace*.csv
echo "=== EXP3""_DONE $(date -u +%FT%TZ) ==="
