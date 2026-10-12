#!/bin/bash
# pipeline_v62 Phases 2 and 3: correctness of list-group primary screening,
# then its performance against v61's query-centric scan, on Phase 0's
# configurations (arxiv/128 at alpha 100 and 200, vogue/128 at 64,
# openai3-3072/128 at 4; the paper's index), batch sizes 8, 32, 256, 1000,
# G = 0 (v61's scan, in the same binary and process) and 1, 2, 4, 8.
#
# Phase 2a  every query, every batch size, every G against G = 0: probe ids,
#           primary top-ck sets (a difference is "tie-only" when every differing
#           candidate sits exactly at its side's ck-th distance and the two
#           ck-th distances are equal, "NON-TIE" otherwise), primary distance
#           error, final ids and distances, recall.
# Phase 2b  compute-sanitizer memcheck (three datasets), racecheck and
#           synccheck (two) on the group-scan and merge kernels, 64 queries.
#   gate    any NON-TIE difference or probe mismatch stops here: no performance
#           numbers are produced for a pipeline that loses candidates.
# Phase 3   5 plain rounds (paired, variant order rotated) and 1 nsys round per
#           dataset; stage_times.py and summarize_v62.py.
#
# Runs from the repository it lives in (a worktree is fine). Nothing here reads
# or writes credentials.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v62_batch -j8
#   nohup bash experiments/pipeline_v62/run_v62.sh > results/pipeline_v62_run.log 2>&1 & tail -f /root/JHQ_GPU/results/pipeline_v62_run.log
set -u
exec 9>/root/.lock_pipeline_v62; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_v62
NS=/root/autodl-tmp/pipeline_v62_nsys
mkdir -p $R/check $R/sanitizer $R/perf $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/GPU: /'
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v62_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== V62""_DONE build failed, nothing run ==="; exit 1; }
{ grep -E "^CMAKE_(CUDA_ARCHITECTURES|BUILD_TYPE|CUDA_FLAGS)" build/CMakeCache.txt
  cat build/CMakeFiles/jhq_v62_exp.dir/flags.make; cat build/CMakeFiles/demo_jhq_v62_batch.dir/flags.make
  nvcc --version | tail -2; nsys --version; compute-sanitizer --version | tail -1
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; } > $R/build_env.txt 2>&1

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
      JHQ_BP_LANES=16 JHQ_BP_GROUPS=0,1,2,4,8 JHQ_BP_BATCHES=$bs ${nqv:+JHQ_BP_NQ=$nqv} \
      timeout 7200 "$@" build/demo_jhq_v62_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 "$pre" 3
}

echo "### Phase 2a: per-query checks $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  [ -d "$D/arms6_cache/$ds" ] || echo "WARNING: no arms6 cache for $ds -- retrained index, not the paper's"
  h $ds $al $R/check/$ds 0 1 "" 8,32,256,1000 > $R/check/$ds.txt 2>&1
  echo "check $ds rc=$? $(grep -c BP_OK $R/check/$ds.txt) ok"
done
grep -h "vs G=0" $R/check/*.txt > $R/check/all_comparisons.txt
NONTIE=$(awk '{for (i = 1; i <= NF; ++i) if ($i == "NON-TIE") { v = $(i + 1); sub(",", "", v); s += v } } END { print s + 0 }' $R/check/all_comparisons.txt)
PROBEBAD=$(awk '{for (i = 1; i <= NF; ++i) if ($i == "probes") { split($(i + 1), a, "/"); if (a[1] != a[2]) bad++ } } END { print bad + 0 }' $R/check/all_comparisons.txt)
NCMP=$(wc -l < $R/check/all_comparisons.txt)
echo "comparisons: $NCMP lines; non-tie primary top-ck differences: $NONTIE; probe mismatches: $PROBEBAD"

echo "### Phase 2b: compute-sanitizer $(date -u +%T)"
san() {  # tool ds alpha
  local tool=$1 ds=$2 a=$3 tag=${2}_${1}
  h $ds $a $NS/san_$tag 0 1 64 8,64 compute-sanitizer --tool $tool --error-exitcode 99 \
      --print-limit 50 --kernel-name kns=scan_ivf_group --kernel-name kns=merge_group_topk \
      > $R/sanitizer/$tag.txt 2>&1
  echo "san $tag rc=$? $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' $R/sanitizer/$tag.txt)"
}
san memcheck arxiv-768 200; san memcheck vogue-768 64; san memcheck openai3-3072 4
san racecheck vogue-768 64; san synccheck vogue-768 64
san racecheck openai3-3072 4; san synccheck openai3-3072 4

if [ "$NONTIE" -ne 0 ] || [ "$PROBEBAD" -ne 0 ] || [ "$NCMP" -eq 0 ]; then
  echo "=== V62""_DONE correctness gate FAILED: no performance run ==="; exit 1
fi

echo "### Phase 3: performance $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al <<< "$c"
  h $ds $al $R/perf/$ds 5 0 "" 8,32,256,1000 > $R/perf/$ds.txt 2>&1
  echo "plain $ds rc=$? $(grep -c BP_OK $R/perf/$ds.txt) ok"
  h $ds $al $NS/$ds 1 0 "" 8,32,256,1000 nsys profile -t cuda,nvtx --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none --force-overwrite true -o $NS/$ds > $R/perf/${ds}_nsys.txt 2>&1
  rc=$?
  nsys stats -q -r cuda_gpu_trace:nvtx-name -f csv -o $R/perf/${ds}_trace $NS/$ds.nsys-rep > /dev/null 2>&1
  echo "nsys $ds rc=$rc $(ls $R/perf/${ds}_trace*.csv 2>/dev/null | wc -l) csv"
done
python3 experiments/pipeline_v62/stage_times.py $R/perf --md $R/perf/stages.md > /dev/null
python3 experiments/pipeline_v62/summarize_v62.py $R
echo "=== V62""_DONE $(date -u +%FT%TZ) ==="
