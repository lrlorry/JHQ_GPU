#!/bin/bash
# COMP8800 poster, experiment E1: JHQ-GPU v69 against cuVS CAGRA (fp32, int8), IVF-RaBitQ and
# IVF-PQ on one RTX 5090, one query set, one Recall@10, one host-to-host timed region.
#
#   bash experiments/comp8800_poster_e1/run_e1.sh <stage> <dataset> [<dataset> ...]
#   stages: env build jhq rabitq cagra cagra-int8 ivfpq ivfpq-refine bruteforce
#           e1a (= jhq rabitq cagra cagra-int8 ivfpq bruteforce, per dataset, in that order)
#
# Runs in the worktree it lives in and never fetches, resets or checks anything out: the box
# must already be at the commit under test (shipped by bundle). Every GPU stage takes the box's
# GPU lock. Results: $OUT (default experiments/comp8800_poster_e1/RAW_RESULTS), one file per
# (stage, dataset) plus PROGRESS.log; nothing is overwritten (jsonl files are appended to and
# every record carries its UTC time).
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO" || exit 1
E=experiments/comp8800_poster_e1
OUT=${OUT:-$REPO/$E/RAW_RESULTS}; mkdir -p "$OUT"
BE=$REPO/build_e1; mkdir -p "$BE"
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
PROG=$OUT/PROGRESS.log
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$PROG"; }

D=/root/autodl-tmp; V=/root/data
declare -A P NL MM NT BLK AL PQ
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-1536]="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
# JHQ: the paper configuration of the arms6 / pipeline-campaign runs (same trained-state cache);
# AL is the calibrated alpha at nprobe 128. IVF-PQ and IVF-RaBitQ use the same nlist.
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512;  AL[vogue-768]=64
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512;  AL[arxiv-768]=100
MM[openai3-1536]=192; NL[openai3-1536]=8192; NT[openai3-1536]=319488; BLK[openai3-1536]=512; AL[openai3-1536]=8
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024; AL[openai3-3072]=4
PQ[vogue-768]="192,384,768"; PQ[arxiv-768]="192,384,768"; PQ[openai3-1536]="384,768,1536"; PQ[openai3-3072]="384,768,1536"

gpu_lock(){
  # E1_LOCK_HELD=1: the caller already holds /root/.gpu_lock (taking it again here would wait
  # on our own parent forever)
  [ -z "${E1_LOCK_HELD:-}" ] && { exec 8>/root/.gpu_lock; flock 8; }
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  [ -n "$apps" ] && say "WARNING: compute apps on the GPU after taking the lock: $apps"
  say "gpu lock held; loadavg $(cut -d' ' -f1-3 /proc/loadavg); gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader)"
}

stage_env(){
  local f=$OUT/env_$(date -u +%Y%m%dT%H%M%SZ).txt
  { echo "## git"; git log --oneline -1; git status --short -- $E | head; git rev-parse HEAD
    echo "## gpu"; nvidia-smi; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER | grep -v '^$' | head -60
    echo "## toolchain"; nvcc --version | tail -2; g++-12 --version | head -1
    python3 -c "import cuvs,cupy,numpy;print('cuvs',cuvs.__version__,'cupy',cupy.__version__,'numpy',numpy.__version__)"
    echo "## build"; grep -E "CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE" build/CMakeCache.txt
    echo "## host"; uptime; nproc; free -g | head -2; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null
    echo "## processes"; ps aux --sort=-%cpu | head -12
    echo "## data"; for ds in "${!P[@]}"; do for x in ${P[$ds]}; do ls -l "$x"; done; done
  } > "$f" 2>&1
  say "env -> $f"
}

stage_build(){
  timeout 2400 cmake --build build -j 16 --target demo_jhq_v69_batch > "$OUT/build_jhq.log" 2>&1
  local rc=$?; say "build demo_jhq_v69_batch rc=$rc"
  [ $rc -ne 0 ] && { say "=== E1 build failed, nothing run ==="; exit 1; }
  local INC="-I$SP/libraft/include/rapids" LD=""
  for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
  for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
  timeout 1800 nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda -arch=sm_120 \
      $INC $E/bench_rabitq_sweep.cu -o "$BE/bench_rabitq_sweep" $LD -lcuvs -lrmm -lcudart -lcublas \
      > "$OUT/build_rabitq.log" 2>&1
  rc=$?; say "build bench_rabitq_sweep rc=$rc"
  [ $rc -ne 0 ] && { grep -m 20 error "$OUT/build_rabitq.log"; say "=== E1 build failed, nothing run ==="; exit 1; }
  ldd "$BE/bench_rabitq_sweep" | grep -i "not found" && { say "=== unresolved libraries ==="; exit 1; }
  return 0
}

stage_jhq(){  # $1 dataset; NPROBES and JHQ_ALPHA_MULT override the grid
  local ds=$1 al=${AL[$1]}
  local alphas; alphas=$(python3 -c "import sys;print(','.join('%g'%(float(m)*$al) for m in '${JHQ_ALPHA_MULT:-0.5,1,2}'.split(',')))")
  for np in ${NPROBES:-64 128 256}; do
    local pre=$OUT/jhq_${ds}_np${np}_$(date -u +%H%M%S)
    env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
        JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
        JHQ_RES_TRAIN_N=100000 JHQ_BP_LANES=16 JHQ_BP_ALPHAS=$alphas JHQ_BP_BATCHES=1000 \
        JHQ_BP_GROUPS=0,-1 JHQ_BP_ROUNDS=5 JHQ_BP_MIN_BATCHES=24 JHQ_BP_CHECK=1 \
        timeout 3600 build/demo_jhq_v69_batch ${P[$ds]} ${MM[$ds]} 8 8 $al 10 ${NL[$ds]} $np 8 1024 "$pre" 3 \
        > "$pre.txt" 2>&1
    say "jhq $ds np=$np alphas=$alphas rc=$? $(grep -c 'BP_OK' "$pre.txt") checks ok, $(grep -ci 'mismatch\|NON-TIE [1-9]\|error' "$pre.txt") flagged"
  done
}

stage_rabitq(){  # $1 dataset; RQ_BITS, RQ_NLISTS override
  local ds=$1
  for nl in ${RQ_NLISTS:-${NL[$ds]}}; do
    for bits in ${RQ_BITS:-9 8 4}; do
      RQ_IDX=$D/rq_e1_${ds}.idx timeout 7200 "$BE/bench_rabitq_sweep" ${P[$ds]} $nl $bits \
          "$OUT/rabitq_${ds}.jsonl" > "$OUT/rabitq_${ds}_nl${nl}_b${bits}.log" 2>&1
      say "rabitq $ds nlist=$nl bits=$bits rc=$? $(grep -c '^  bits' "$OUT/rabitq_${ds}_nl${nl}_b${bits}.log") points, $(grep -c FAILED "$OUT/rabitq_${ds}_nl${nl}_b${bits}.log") failed"
    done
  done
}

stage_cuvs(){  # $1 method $2 dataset
  local m=$1 ds=$2 extra=""
  case $m in
    ivfpq|ivfpq-refine) extra="--nlist ${NL[$ds]} --pq-dims ${PQ[$ds]}";;
  esac
  [ $m = ivfpq-refine ] && extra="$extra --lut f16"
  timeout 10800 python3 $E/bench_cuvs.py --dataset $ds --method $m --out "$OUT/cuvs_${ds}_${m}.jsonl" $extra \
      > "$OUT/cuvs_${ds}_${m}.log" 2>&1
  say "cuvs $m $ds rc=$? $(grep -c '^  ' "$OUT/cuvs_${ds}_${m}.log") lines, $(grep -c 'FAILED' "$OUT/cuvs_${ds}_${m}.log") failed"
}

STAGE=${1:?stage}; shift
case $STAGE in
  env) stage_env;;
  build) stage_build;;
  *)
    gpu_lock
    for ds in "$@"; do
      [ -z "${P[$ds]:-}" ] && { say "unknown dataset $ds"; continue; }
      case $STAGE in
        jhq) stage_jhq $ds;;
        rabitq) stage_rabitq $ds;;
        cagra|cagra-int8|ivfpq|ivfpq-refine|bruteforce) stage_cuvs $STAGE $ds;;
        e1a) stage_jhq $ds; stage_rabitq $ds; stage_cuvs cagra $ds; stage_cuvs cagra-int8 $ds
             stage_cuvs ivfpq $ds; stage_cuvs bruteforce $ds;;
        *) say "unknown stage $STAGE"; exit 1;;
      esac
    done;;
esac
say "=== E1_STAGE_DONE $STAGE $* ==="
