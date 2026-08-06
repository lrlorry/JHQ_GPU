#!/bin/bash
# Build and run hblock_v36_1 (v38 + GPU-accelerated balanced-kmeans) on a
# SPACEV .i8bin subset. Defaults to the FULL 100M corpus and MAX_EF=256 --
# the whole point of this version is making a full-scale run practical,
# where the CPU-only balanced-kmeans in v38/v39/v42/v43 took an unknown,
# possibly 30+ minute amount of time (single-threaded, purely CPU-bound,
# confirmed via ps/nvidia-smi while it ran -- see git log for that
# investigation). The official 100M-track ground truth
# (ids.100M.i32bin + groundtruth.30K.i32bin) is already correctly
# restricted for the full 100M corpus -- do not pass a smaller N_BASE here
# expecting a "real" recall number, it will just re-restrict the ground
# truth down further (see results/hblock_v39_findings.md).
#
# stdout is line-buffered now (setvbuf in demo_hblock_v36_1.cu's main),
# so nohup/tee/redirected runs show live progress -- no more silent gaps.

set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

cmake --build build --target demo_hblock_v36_1 -j"${BUILD_JOBS:-8}"
BIN=./build/demo_hblock_v36_1

SPACEV_DIR="${SPACEV_DIR:-/root/autodl-tmp/spacev100m}"
BASE="${SPACEV_DIR}/base.100M.i8bin"
QUERY="${SPACEV_DIR}/query.30K.i8bin"
IDS="${SPACEV_DIR}/ids.100M.i32bin"
GT="${SPACEV_DIR}/groundtruth.30K.i32bin"

N_BASE="${N_BASE:-100000000}"
N_QUERY="${N_QUERY:-30000}"
MAX_EF="${MAX_EF:-256}"
BATCH="${BATCH:-1024}"
K1=16; K2=16; K3=16
DEGREE=32; ENTRY=4
D_PROJ=64; PER_BLOCK_R=16

mkdir -p results
CSV="results/hblock_v36_1.csv"
$BIN "$BASE" "$QUERY" "$IDS" "$GT" \
    "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
    "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
    "$CSV" 2>&1 | tee results/hblock_v36_1.log

echo ""
echo "Done. results/hblock_v36_1.{csv,log}."
