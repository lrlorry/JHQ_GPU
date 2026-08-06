#!/bin/bash
# Build and run hblock_v42 end-to-end on a SPACEV .i8bin subset: v39's
# region pool, backed by real mmap'd files on disk (see
# hblock_v42/jhq_gpu_index.cuh). One straightforward run.
#
# Pool cap is 0/0 (auto-size to fit the whole store) -- no need to know
# region counts ahead of time. Adjust SPACEV_DIR / REGION_DIR to your
# actual paths. N_BASE defaults to a 5M-vector subset (fast); export
# N_BASE=100000000 for the full corpus once this works.

set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

cmake --build build --target demo_hblock_v42 -j"${BUILD_JOBS:-8}"
BIN=./build/demo_hblock_v42

SPACEV_DIR="${SPACEV_DIR:-/root/autodl-tmp/spacev100m}"
BASE="${SPACEV_DIR}/base.100M.i8bin"
QUERY="${SPACEV_DIR}/query.30K.i8bin"
IDS="${SPACEV_DIR}/ids.100M.i32bin"
GT="${SPACEV_DIR}/groundtruth.30K.i32bin"

# Where the two real region files get written -- fast local disk, not a
# network mount, or region fetch latency will dominate the numbers.
REGION_DIR="${REGION_DIR:-/root/autodl-tmp/hblock_v42_regions}"
mkdir -p "$REGION_DIR"

N_BASE="${N_BASE:-5000000}"
N_QUERY="${N_QUERY:-1000}"
MAX_EF="${MAX_EF:-64}"
BATCH="${BATCH:-64}"
REGION_MIB=1
CODE_CAP=0  # auto-size to fit the whole store
RAW_CAP=0
K1=16; K2=16; K3=16
DEGREE=32; ENTRY=4
D_PROJ=64; PER_BLOCK_R=16

mkdir -p results
CSV="results/hblock_v42.csv"
$BIN "$BASE" "$QUERY" "$IDS" "$GT" \
    "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
    "$REGION_MIB" "$CODE_CAP" "$RAW_CAP" \
    "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
    "$REGION_DIR" \
    "$CSV" 2>&1 | tee results/hblock_v42.log

echo ""
echo "Done. results/hblock_v42.{csv,log}. Region files under ${REGION_DIR}:"
ls -la "$REGION_DIR"
