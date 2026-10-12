#!/bin/bash
# Campaign Experiment 2 (v63, budget-aware local quotas), pilot before any
# implementation. Two questions, both answerable with the v62 binary:
#
#   A  How much could a smaller local quota save at all? A group that keeps
#      only its top-b runs the v62 group kernel at ck = b. So v62 at a reduced
#      alpha (ck = b) times the first pass of any quota scheme with zero
#      certification and zero fallback cost -- and its merge, which outputs b
#      rather than ck, is cheaper than the real one. Scan + merge time at
#      alpha/2, /4, /8 against alpha is therefore an upper bound on the gain.
#      nsys, one round, every G and batch size.
#   B  How often would a quota certify? JHQ_BP_DUMP=1 writes each query's exact
#      top-ck, probe ids and the list sizes; exp2_quota_study.py places every
#      top-ck member in its v62 group and scores the quota policies.
#
# Workloads: pipeline_v62's. Dumps (tens of MB) stay in
# /root/autodl-tmp/campaign_exp2; the study's tables go to results/.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake -S . -B build && cmake --build build --target demo_jhq_v62_batch -j8
#   nohup bash experiments/pipeline_campaign/exp2_v63_pilot.sh > results/pipeline_campaign_exp2.log 2>&1 & tail -f results/pipeline_campaign_exp2.log
set -u
exec 9>/root/.lock_campaign_exp2; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
echo "repo: $(pwd)"
R=results/pipeline_campaign/exp2
NS=/root/autodl-tmp/campaign_exp2
mkdir -p $R/nsys $NS
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v62_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== EXP2""_DONE build failed, nothing run ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
# dataset : the workload alphas : the reduced alphas of part A (alpha/2, /4, /8)
CONFIGS="arxiv-768:100,200:200,100,50,25,12.5 vogue-768:64:64,32,16,8 openai3-3072:4:4,2,1"
# h <dataset> <alphas> <prefix> <rounds> <check> <groups> <batches> [env...] -- [command prefix...]
h() {
  local ds=$1 al=$2 pre=$3 ro=$4 ck=$5 gr=$6 bs=$7; shift 7
  local extra=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$ro JHQ_BP_CHECK=$ck \
      JHQ_BP_LANES=16 JHQ_BP_GROUPS=$gr JHQ_BP_BATCHES=$bs ${extra[@]+"${extra[@]}"} \
      timeout 3600 "$@" build/demo_jhq_v62_batch ${P[$ds]} ${MM[$ds]} 8 8 ${al%%,*} 10 ${NL[$ds]} 128 8 1024 "$pre" 3
}

echo "### B: dumps for the quota study $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al red <<< "$c"
  h $ds $al $NS/$ds 0 1 0,1 1000 JHQ_BP_DUMP=1 -- > $R/dump_$ds.txt 2>&1
  echo "dump $ds rc=$? $(grep -c 'dumped' $R/dump_$ds.txt) dumps"
done
python3 experiments/pipeline_campaign/exp2_quota_study.py $R/quota_study.md $NS/*.dump > /dev/null
echo "quota study rc=$? -> $R/quota_study.md"

echo "### A: reduced-alpha kernel times (nsys) $(date -u +%T)"
for c in $CONFIGS; do
  IFS=: read ds al red <<< "$c"
  h $ds $red $NS/a_$ds 1 0 0,1,2,4,8 8,32,256,1000 -- nsys profile -t cuda,nvtx --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none --force-overwrite true -o $NS/a_$ds > $R/nsys/$ds.txt 2>&1
  rc=$?
  cp $NS/a_$ds.json $R/nsys/$ds.json
  nsys stats -q --force-export=true -r cuda_gpu_trace:nvtx-name -f csv -o $R/nsys/${ds}_trace \
      $NS/a_$ds.nsys-rep > /dev/null 2> $R/nsys/${ds}_stats_err.txt
  echo "nsys $ds rc=$rc $(ls $R/nsys/${ds}_trace*.csv 2>/dev/null | wc -l) csv"
done
python3 experiments/pipeline_v62/stage_times.py $R/nsys --md $R/nsys/stages.md > /dev/null
echo "stage_times rc=$?"
gzip -f $R/nsys/*_trace*.csv
echo "=== EXP2""_DONE $(date -u +%FT%TZ) ==="
