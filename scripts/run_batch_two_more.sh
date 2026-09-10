#!/bin/bash
# The two datasets the matched-recall batch sweep is missing.
#
# Table 4 has vogue-768 and openai3-3072 and the paper never said why. Two of
# the remaining four can never be added: IVF-RaBitQ's build path runs out of
# memory on bge-m3 and stella-trec24, which Section 6.1 already reports. That
# leaves arxiv-768 and openai3-1536, which were simply never run.
#
# Same binaries, same nprobe grid, same timed region and the same matched-recall
# construction as bt3, the run that produced batch_matched.log, so the new rows
# interpolate into the same table.
#
# Serial with the CPU baseline, not parallel: run_cpu_four.sh is measuring host
# throughput, and a GPU benchmark loading its own data would perturb exactly the
# thing being measured. This blocks on that job's lock and starts when it ends.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_batch_two_more.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_batch2more; flock -n 9 || { echo "already running"; exit 0; }
L=/root/batch2more.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "queued behind the CPU baseline; head=$(git log --oneline -1)"

# Wait for run_cpu_four.sh to release its lock, however long that takes.
exec 7>/root/.lock_cpu_four; flock 7
say "CPU baseline finished or was never running; taking the GPU"

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
CACHE=/root/autodl-tmp/paper_cache; D=/root/autodl-tmp
SP=/root/miniconda3/lib/python3.12/site-packages
# bench_rq_batch2 dies on librapids_logger.so without this, and awk then turns
# the silence into empty fields that read as data.
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export JHQ_RQ_IDX=/root/autodl-tmp/rq_batch2more.idx

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>"$L" 2>&1
cmake --build build -j 16 --target demo_jhq_v57 >>"$L" 2>&1; rc1=$?
INC="-I$SP/libraft/include/rapids"; for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""; for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC examples/bench_ivf_rabitq.cu -o build/bench_rq_batch2 \
     $LD -lcuvs -lrmm -lcudart -lcublas >>"$L" 2>&1; rc2=$?
say "build_rc=$rc1/$rc2"
[ $rc1 -ne 0 -o $rc2 -ne 0 ] && { say "=== BATCH2MORE""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"

jhq(){ # tag paths M nlist n_train blk np alpha batch
  local C=$CACHE/${1}; mkdir -p "$C"
  env $E JHQ_INDEX_CACHE="$C" JHQ_BLOCK=$6 JHQ_TILE_M_RT=$3 JHQ_N_TRAIN=$5 \
      timeout 9000 build/demo_jhq_v57 $2 $3 8 8 $8 10 $4 $7 8 $9 "" 3 \
      >/tmp/b2.$$ 2>/tmp/b2e.$$
  local r q
  r=$(awk '/^Recall@10/{print $3;exit}' /tmp/b2.$$); q=$(awk '/^QPS/{print $3;exit}' /tmp/b2.$$)
  printf "  JHQ     %-14s np=%-5s a=%-5s batch=%-6s recall=%-8s qps=%-9s%s\n" \
    "$1" "$7" "$8" "$9" "${r:-MISSING}" "${q:-MISSING}" \
    "$([ -z "${r:-}" ] && echo '  <-- produced nothing')" >> "$L"
  rm -f /tmp/b2.$$ /tmp/b2e.$$
}
rq(){ # tag paths nlist np batch
  env JHQ_RQ_BATCH=$5 timeout 9000 build/bench_rq_batch2 $2 $3 8 $4 10 2 3 \
      >/tmp/r2.$$ 2>/tmp/r2e.$$
  local r q
  r=$(awk '/^Recall@10/{print $3;exit}' /tmp/r2.$$); q=$(awk '/^QPS/{print $3;exit}' /tmp/r2.$$)
  printf "  RaBitQ  %-14s np=%-5s %-11s batch=%-6s recall=%-8s qps=%-9s%s\n" \
    "$1" "$4" "" "$5" "${r:-MISSING}" "${q:-MISSING}" \
    "$([ -z "${r:-}" ] && echo '  <-- produced nothing')" >> "$L"
  rm -f /tmp/r2.$$ /tmp/r2e.$$
}

AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
NP="32 64 128 256 512"

# nlist, n_train, BLOCK and alpha follow the frontier's own settings for each
# dataset, as bt3 did for the two it covered.
say "########## arxiv-768: batch x nprobe, both systems ##########"
for b in 32 128 512 1024; do
  for np in $NP; do
    jhq arxiv-768 "$AX" 96 8192 319488 512 $np 64.0 $b
    rq  arxiv-768 "$AX" 8192 $np $b
  done
done
say "########## openai3-1536: batch x nprobe, both systems ##########"
for b in 32 128 512 1024; do
  for np in $NP; do
    jhq openai3-1536 "$O15" 192 4096 159744 512 $np 8.0 $b
    rq  openai3-1536 "$O15" 4096 $np $b
  done
done
rm -f "$JHQ_RQ_IDX"
say "=== BATCH2MORE""_DONE ==="
