#!/bin/bash
# The four datasets the CPU baseline is missing.
#
# Why this run exists: the CPU comparison covers vogue-768 and openai3-3072 and
# nothing else, and the paper does not say why.  There *is* six-dataset CPU data
# in results/jhq_cpu_ivf_*.csv, but it is the superseded set --
# data/cpu_baseline_artifact_note.md records that it came from ~/JHQ_repro,
# "not the paper's code, at settings that were never recorded", and its recall
# column reaches 1.0 because that build compared k results against the whole
# 100-wide ground-truth row.  cpu5.log replaced it with the authors' own
# artifact and only got through two datasets.  This finishes the other four on
# that same binary.
#
# Same binary, same alpha grid, same nprobe cap as cpu5.sh, so the new rows sit
# in one envelope with the old ones.  Mixing binaries across a frontier is the
# error CLAUDE.md opens with.
#
# nlist and M follow run_front6.sh's second block -- the parameters the GPU
# frontier reports -- so both sides of the comparison are the same index.
#
# Order is smallest first: arxiv and openai3-1536 finish inside half an hour,
# so the run is useful long before stella is done.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_cpu_four.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_cpu_four; flock -n 9 || { echo "already running"; exit 0; }
L=/root/cpu_four.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

R=/root/JHQ_paper2
B=$R/build/examples/bench_vogue768
if [ ! -x "$B" ]; then
    echo "=== CPUFOUR_DONE $B missing, nothing run ==="; exit 1
fi
echo "binary=$B  mtime=$(stat -c %y "$B")"

D=/root/autodl-tmp
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"

A="2.0,4.0,8.0,16.0,32.0,100.0"

run(){  # tag paths M nlist threads
    local tag=$1 paths=$2 M=$3 nl=$4 th=$5
    echo "### $tag M=$M nlist=$nl threads=$th alphas=$A $(date -u +%H:%M:%S)"
    OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores \
    JHQ_NLIST=$nl JHQ_M=$M JHQ_BR=8 JHQ_ALPHA=$A JHQ_NPROBE_MAX=1024 \
    timeout 28800 "$B" $paths
    echo "run_rc=$? tag=$tag threads=$th $(date -u +%H:%M:%S)"
}

# The envelope takes the best of 16 and 32 threads on the two datasets it
# already has.  Giving the new four only one thread count would hold their CPU
# side down and inflate the speedup, so both are run here too.
for th in 32 16; do
    run arxiv-768    "$AX"  96  8192  "$th"
    run openai3-1536 "$O15" 192 4096  "$th"
done
echo "--- the two large ones, 32 threads first $(date -u +%H:%M:%S) ---"
for th in 32 16; do
    run bge-m3 "$BG" 128 16384 "$th"
    run stella "$ST" 128 32768 "$th"
done
echo "=== CPUFOUR""_DONE $(date -u +%FT%TZ) ==="
