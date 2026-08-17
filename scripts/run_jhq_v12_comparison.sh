#!/bin/bash
# Run GPU (jhq_v12_transposed) + CPU (JHQ_repro's demo_jhq_ivf) sweeps on
# every dataset whose GPU side is actually ready, and write CSVs under
# results/ that plot_v12_vs_cpu_all.py turns into per-dataset comparison
# figures matching the style of the existing results/jhq_v12_vs_cpu.png.
#
# Only 3 of the 6 JHQ paper datasets are included here:
#   arxiv-abstracts-768, openai3-1536, openai3-3072
# bge-m3 (41GB) and stella-trec24 (73GB) are NOT included -- their base
# set alone exceeds this GPU's 32GB VRAM in JHQGpuIndex::rotate_on_gpu()
# (two full n*d float buffers), which is a separate, larger add()
# batching rewrite, not something this script's mmap host-loading fix
# solves. Vogue-768 already has results (results/jhq_v12_vogue.csv).
#
# Resumable: both sweep_gpu.py (per nprobe point) and sweep_cpu_ivf.py
# (whole run, since demo_jhq_ivf sweeps nprobe internally in one process)
# skip work that's already recorded in their output CSV -- rerunning this
# whole script after a kill/disconnect picks up where it left off instead
# of redoing completed points.
set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# name : dim (M = dim/8, same rule on both GPU and CPU sides)
DATASETS=(
  "arxiv-abstracts-768:768"
  "openai3-1536:1536"
  "openai3-3072:3072"
)

for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    dim="${entry##*:}"
    M=$((dim / 8))

    base="$DATA_ROOT/$name/base.fvecs"
    query="$DATA_ROOT/$name/query.fvecs"
    gt="$DATA_ROOT/$name/groundtruth.ivecs"

    if [[ ! -f "$base" || ! -f "$query" || ! -f "$gt" ]]; then
        echo "=== [$name] SKIPPED -- missing $base / $query / $gt ==="
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "[$name] GPU sweep (M=$M)"
    echo "======================================================================"
    python3 scripts/sweep_gpu.py \
        --version jhq_v12_transposed \
        --base "$base" --query "$query" --gt "$gt" \
        --M "$M" \
        --output "$RESULTS_DIR/jhq_v12_${name}.csv"

    echo ""
    echo "======================================================================"
    echo "[$name] CPU sweep (M=$M, auto)"
    echo "======================================================================"
    python3 scripts/sweep_cpu_ivf.py \
        --base "$base" --query "$query" --gt "$gt" \
        --output "$RESULTS_DIR/jhq_cpu_ivf_${name}.csv"
done

echo ""
echo "All done. Generating comparison figures..."
python3 scripts/plot_v12_vs_cpu_all.py
