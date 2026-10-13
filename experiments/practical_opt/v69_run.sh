#!/bin/bash
# Practical optimisation, item 2: v69's probe selection (radix select) against v65's (repeated bitonic
# compaction), at the paper's configuration (np = 128; arxiv alpha 100, vogue 64, openai3 4).
#   1. build demo_jhq_v65_batch and demo_jhq_v69_batch; abort on a failed build
#   2. probe_select_test: both kernels cut verbatim from the sources, compared word for word on adversarial
#      inputs (ties, both zeros, infinities, nlist < nprobe, nprobe > BLOCK), then timed alone; stop on a mismatch
#   3. end to end, same index: G = 0, 1, -1 with the harness's checks and the G = 1 dump (probes + primary
#      top-ck of all 1000 queries), b = 8 and 1000; the two binaries' dumps must be byte-identical
#   4. timing: v65 v69 v69 v65 per workload, 5 rounds each, b = 8, 32, 256, 1000, variants 0 and -1
#   5. nsys stage breakdown of both binaries (one round each), experiments/pipeline_v62/stage_times.py
# Dumps and nsys reports stay on the data disk (/root/autodl-tmp/practical_opt), not in results/.
#   bash experiments/practical_opt/v69_run.sh        (from the repository root)
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
R=results/practical_opt/v69; mkdir -p $R
X=/root/autodl-tmp/practical_opt; mkdir -p $X/v69_dumps $X/nsys
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) v69 probe selection ==="
timeout 600 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 >/dev/null 2>&1
timeout 2400 cmake --build build -j 16 --target demo_jhq_v65_batch demo_jhq_v69_batch > $R/build.log 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { grep -E "error" $R/build.log | head -20; echo "=== DONE build failed, nothing run ==="; exit 1; }
python3 experiments/practical_opt/probe_select_test.py build/probe_select_test.cu
timeout 900 nvcc -O3 --expt-relaxed-constexpr --use_fast_math "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
    -o build/probe_select_test build/probe_select_test.cu > $R/probe_test_build.log 2>&1
rc=$?; echo "probe_test_build_rc=$rc"
[ $rc -ne 0 ] && { head -30 $R/probe_test_build.log; echo "=== DONE test build failed, nothing run ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv > $R/gpu.txt 2>&1
timeout 1800 build/probe_select_test > $R/probe_test.txt 2>&1
echo "probe_test rc=$? $(grep -E 'exactness' $R/probe_test.txt) $(tail -1 $R/probe_test.txt)"
grep -q PROBE_TEST_OK $R/probe_test.txt || { grep FAIL $R/probe_test.txt | head; flock -u 8; echo "=== DONE exactness failed, nothing else run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
declare -A P MM NL NT BLK AL
P[vogue-768]="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
P[arxiv-768]="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
P[openai3-3072]="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
MM[vogue-768]=96;     NL[vogue-768]=4096;    NT[vogue-768]=159744;    BLK[vogue-768]=512;  AL[vogue-768]=64
MM[arxiv-768]=96;     NL[arxiv-768]=8192;    NT[arxiv-768]=319488;    BLK[arxiv-768]=512;  AL[arxiv-768]=100
MM[openai3-3072]=384; NL[openai3-3072]=8192; NT[openai3-3072]=319488; BLK[openai3-3072]=1024; AL[openai3-3072]=4
DS="arxiv-768 vogue-768 openai3-3072"
# bench <binary> <out prefix> [JHQ_*=... ...] [-- <command prefix ...>]: the B0 invocation, paper alpha, np 128
bench() {
  local bin=$1 out=$2; shift 2
  local envs=() pre=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && { shift; pre=("$@"); }
  env JHQ_INDEX_CACHE="$D/arms6_cache/$ds" JHQ_BLOCK=${BLK[$ds]} JHQ_TILE_M_RT=${MM[$ds]} \
      JHQ_N_TRAIN=${NT[$ds]} JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 JHQ_BP_ALPHAS=${AL[$ds]} JHQ_BP_LANES=16 "${envs[@]}" \
      timeout 3600 "${pre[@]}" $bin ${P[$ds]} ${MM[$ds]} 8 8 ${AL[$ds]} 10 ${NL[$ds]} 128 8 1024 $out 3
}

# 3. end-to-end identity
for ds in $DS; do
  for v in 65 69; do
    bench build/demo_jhq_v${v}_batch $R/id_${ds}_v$v JHQ_BP_ROUNDS=0 JHQ_BP_CHECK=1 JHQ_BP_DUMP=1 \
        JHQ_BP_GROUPS=0,1,-1 JHQ_BP_BATCHES=8,1000 > $R/id_${ds}_v$v.txt 2>&1
    echo "identity $ds v$v rc=$? $(grep -c BP_OK $R/id_${ds}_v$v.txt) ok  $(date -u +%T)"
  done
  a65=$R/id_${ds}_v65_a${AL[$ds]}.dump; a69=$R/id_${ds}_v69_a${AL[$ds]}.dump
  if cmp -s $a65 $a69; then echo "dump $ds IDENTICAL"; else echo "dump $ds DIFFERS"; fi
  sha256sum $a65 $a69 >> $R/dumps.sha256
  python3 experiments/practical_opt/compare_v69.py $R/id_${ds}_v65 $R/id_${ds}_v69 $a65 $a69 > $R/id_${ds}_compare.txt 2>&1
  tail -3 $R/id_${ds}_compare.txt
  mv $a65 $a69 $X/v69_dumps/
done

# 4. timing, interleaved
for ds in $DS; do
  n=0
  for v in 65 69 69 65; do
    n=$((n+1))
    bench build/demo_jhq_v${v}_batch $R/t_${ds}_${n}_v$v JHQ_BP_ROUNDS=5 JHQ_BP_CHECK=0 \
        JHQ_BP_GROUPS=0,-1 JHQ_BP_BATCHES=8,32,256,1000 > $R/t_${ds}_${n}_v$v.txt 2>&1
    echo "timing $ds #$n v$v rc=$? $(grep -c BP_OK $R/t_${ds}_${n}_v$v.txt) ok  $(date -u +%T)"
  done
done

# 5. nsys stage breakdown
for v in 65 69; do
  S=$R/stages_v$v; mkdir -p $S
  for ds in $DS; do
    bench build/demo_jhq_v${v}_batch $S/$ds JHQ_BP_ROUNDS=3 JHQ_BP_CHECK=0 JHQ_BP_GROUPS=0,-1 \
        JHQ_BP_BATCHES=8,32,256,1000 > $S/$ds.txt 2>&1
    echo "plain v$v $ds rc=$? $(grep -c BP_OK $S/$ds.txt) ok"
    bench build/demo_jhq_v${v}_batch $X/nsys/v${v}_$ds JHQ_BP_ROUNDS=1 JHQ_BP_CHECK=0 JHQ_BP_GROUPS=0,-1 \
        JHQ_BP_BATCHES=8,32,256,1000 -- nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none \
        --cpuctxsw=none --force-overwrite true -o $X/nsys/v${v}_$ds > $S/${ds}_nsys.txt 2>&1
    rc=$?
    nsys stats -q --force-export=true -r cuda_gpu_trace:nvtx-name -f csv -o $S/${ds}_trace $X/nsys/v${v}_$ds.nsys-rep \
        > /dev/null 2> $S/${ds}_stats_err.txt
    echo "nsys v$v $ds rc=$rc $(ls $S/${ds}_trace* 2>/dev/null | wc -l) trace"
  done
  python3 experiments/pipeline_v62/stage_times.py $S --md $S/stages.md > $S/stage_times.txt 2>&1
  echo "stage_times v$v rc=$?"
  gzip -f $S/*_trace*.csv
done
flock -u 8
python3 experiments/practical_opt/analyze_v69.py $R > $R/summary.txt 2>&1
echo "analyze rc=$?"; cat $R/summary.txt
echo "=== PO_V69_DONE $(date -u +%FT%TZ) ==="
