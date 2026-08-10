#!/bin/bash
# Build and run hblock_v36_2 -- a straight fork of v36_1, no algorithm
# change. Exists purely to run diagnose_missed_gt() (routing/graph miss
# classification: A=routing miss, B=graph unreachable, C=depth miss) on a
# clean, dedicated version, to get a real A/B/C breakdown on full-scale
# SPACEV-100M before deciding which recall lever to pull next.
# Defaults to the FULL 100M corpus and MAX_EF=256. The official 100M-track
# ground truth (ids.100M.i32bin + groundtruth.30K.i32bin) is already
# correctly restricted for the full 100M corpus -- do not pass a smaller
# N_BASE here expecting a "real" recall number, it will just re-restrict
# the ground truth down further (see results/hblock_v39_findings.md).
#
# stdout is line-buffered (setvbuf in demo_hblock_v36_2.cu's main), so
# nohup/tee/redirected runs show live progress -- no silent gaps.

set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

# New targets (jhq_hblock_v36_2/demo_hblock_v36_2) were added to
# CMakeLists.txt after build/ was originally configured -- reconfigure
# explicitly (reuses the existing cache: generator, CUDA arch, build type)
# so `cmake --build` doesn't fail with "No rule to make target" against a
# stale Makefile.
cmake -S . -B build
cmake --build build --target demo_hblock_v36_2 -j"${BUILD_JOBS:-8}"
BIN=./build/demo_hblock_v36_2

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
CSV="results/hblock_v36_2.csv"
$BIN "$BASE" "$QUERY" "$IDS" "$GT" \
    "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
    "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
    "$CSV" 2>&1 | tee results/hblock_v36_2.log

echo ""
echo "Done. results/hblock_v36_2.{csv,log}."
