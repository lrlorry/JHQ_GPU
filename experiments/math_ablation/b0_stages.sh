#!/bin/bash
# B0 stage breakdown: v65 (variant -1) at the paper's configuration (np = 128; arxiv alpha 100, vogue 64,
# openai3 4), batches 8, 32, 256, 1000: one plain run (3 rounds) and one nsys round per workload, then
# experiments/pipeline_v62/stage_times.py, as in experiments/pipeline_campaign/exp2.
#   bash b0_stages.sh <out dir> [binary] [extra env...]     (run from the repository root)
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
OUT=$1; BIN=${2:-build/demo_jhq_v65_batch}; mkdir -p $OUT
shift; shift || true
EXTRA="$*"
exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024
NS=/root/autodl-tmp/math_ablation/nsys; mkdir -p $NS
for c in arxiv-768:100 vogue-768:64 openai3-3072:4; do
  IFS=: read ds al <<< "$c"
  run() {
    env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
        JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
        JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS="$al" JHQ_BP_ROUNDS=$1 JHQ_BP_CHECK=0 \
        JHQ_BP_LANES=16 JHQ_BP_GROUPS=-1 JHQ_BP_BATCHES=8,32,256,1000 $EXTRA \
        timeout 3600 "${@:3}" $BIN ${P[$ds]} ${MM[$ds]} 8 8 $al 10 ${NL[$ds]} 128 8 1024 $2 3
  }
  run 3 $OUT/$ds > $OUT/$ds.txt 2>&1
  echo "plain $ds rc=$? $(grep -c BP_OK $OUT/$ds.txt) ok"
  run 1 $NS/$ds nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite true -o $NS/$ds > $OUT/${ds}_nsys.txt 2>&1
  rc=$?
  nsys stats -q --force-export=true -r cuda_gpu_trace:nvtx-name -f csv -o $OUT/${ds}_trace $NS/$ds.nsys-rep > /dev/null 2> $OUT/${ds}_stats_err.txt
  echo "nsys $ds rc=$rc $(ls $OUT/${ds}_trace* 2>/dev/null | wc -l) trace"
done
flock -u 8
python3 experiments/pipeline_v62/stage_times.py $OUT --md $OUT/stages.md > $OUT/stage_times.txt 2>&1
echo "stage_times rc=$?"
gzip -f $OUT/*_trace*.csv
echo "STAGES_DONE"
