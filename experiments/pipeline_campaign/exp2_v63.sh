#!/bin/bash
# Campaign Experiment 2, the one optimisation the pilot left standing: v63's
# shared bound (exact cross-group pruning, no quota, no certificate, no
# fallback), measured against v62's independent groups on one index.
#
# Variants (demo_jhq_v63_batch): 0 = v61's scan (the reference), G = 2, 4, 8
# = v62's independent groups, 100 + G = the same G with the shared bound.
#
#   A  per-query checks: every variant against 0 (probe ids, primary top-ck
#      sets -- tie-only or NON-TIE --, final ids and distances, recall), and
#      each 100 + G against G: identical top-ck and final results expected.
#   B  compute-sanitizer on scan_ivf_group / merge_group_topk: memcheck and
#      synccheck at 64 queries, racecheck at 8 queries, batch 8 (vogue).
#   gate  any NON-TIE difference, or any shared-bound result that differs from
#      its independent-groups twin, stops here.
#   C  5 plain rounds (paired, order rotated) and 1 nsys round per workload;
#      stage_times.py and summarize_v63.py.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v63_batch -j8
#   nohup bash experiments/pipeline_campaign/exp2_v63.sh > results/pipeline_campaign_exp2b.log 2>&1 & tail -f results/pipeline_campaign_exp2b.log
set -u
exec 9>/root/.lock_campaign_exp2b; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_campaign/exp2_v63
NS=/root/autodl-tmp/campaign_exp2_v63
mkdir -p $R/check $R/perf $R/sanitizer $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v63_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== EXP2B""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/jhq_v63_exp.dir/flags.make; } > $R/build_env.txt 2>&1

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
VARIANTS=0,2,102,4,104,8,108
# h <dataset> <alphas> <prefix> <rounds> <check> <nq or ""> <batches> [command prefix...]
h() {
  local ds=$1 al=$2 pre=$3 ro=$4 ck=$5 nqv=$6 bs=$7; shift 7
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$ro JHQ_BP_CHECK=$ck \
      JHQ_BP_LANES=16 JHQ_BP_GROUPS=$VARIANTS JHQ_BP_BATCHES=$bs ${nqv:+JHQ_BP_NQ=$nqv} \
      timeout 3600 "$@" build/demo_jhq_v63_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 "$pre" 3
}

echo "### A: per-query checks $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  h $ds $al $R/check/$ds 0 1 "" 8,32,256,1000 > $R/check/$ds.txt 2>&1
  echo "check $ds rc=$? $(grep -c BP_OK $R/check/$ds.txt) ok"
done
grep -h "vs G=0" $R/check/*.txt > $R/check/all_comparisons.txt
grep -h "shared bound vs independent" $R/check/*.txt > $R/check/all_shared.txt
NONTIE=$(awk '{for (i = 1; i <= NF; ++i) if ($i == "NON-TIE") { v = $(i + 1); sub(",", "", v); s += v } } END { print s + 0 }' $R/check/all_comparisons.txt)
DIFF=$(awk '{ for (i = 1; i <= NF; ++i) if ($i ~ /^[0-9]+\/[0-9]+,?$/) { v = $i; sub(",", "", v); split(v, a, "/"); if (a[1] != a[2]) bad++ } } END { print bad + 0 }' $R/check/all_shared.txt)
echo "comparisons: $(wc -l < $R/check/all_comparisons.txt) vs G=0, NON-TIE $NONTIE; $(wc -l < $R/check/all_shared.txt) shared-vs-independent, differing: $DIFF"

echo "### B: compute-sanitizer $(date -u +%T)"
san() {  # tool nq batches
  local tool=$1 nq=$2 bs=$3
  h vogue-768 64 $NS/san_$tool 0 1 $nq $bs compute-sanitizer --tool $tool --error-exitcode 99 \
      --print-limit 50 --kernel-name kns=scan_ivf_group --kernel-name kns=merge_group_topk \
      > $R/sanitizer/vogue-768_$tool.txt 2>&1
  echo "san vogue $tool rc=$? $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' $R/sanitizer/vogue-768_$tool.txt)"
}
san memcheck 64 8,64; san racecheck 8 8; san synccheck 64 8,64

if [ "$NONTIE" -ne 0 ] || [ "$DIFF" -ne 0 ]; then
  echo "=== EXP2B""_DONE correctness gate FAILED: no performance run ==="; exit 1
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
python3 experiments/pipeline_v62/stage_times.py $R/perf --md $R/perf/stages.md > /dev/null
python3 experiments/pipeline_campaign/summarize_v63.py $R
gzip -f $R/perf/*_trace*.csv
echo "=== EXP2B""_DONE $(date -u +%FT%TZ) ==="
