#!/bin/bash
# Build and run hblock_v43 end-to-end on a SPACEV .i8bin subset: v42 +
# graph-aware region packing (see hblock_v43/jhq_gpu_index.cuh).
#
# Same N_BASE/batch defaults as run_hblock_v42.sh on purpose: this is the
# exact configuration results/hblock_v39_findings.md measured naive packing
# at (5M vectors, batch=64 -> 57-91% of all regions touched per batch).
# Compare this run's "uniq_this_call" / per-batch region-touch numbers
# against that baseline to see whether graph-aware packing actually helped.
#
# Pool cap is 0/0 (auto-size to fit the whole store) -- no need to know
# region counts ahead of time. Adjust SPACEV_DIR / REGION_DIR to your
# actual paths.

set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

cmake --build build --target demo_hblock_v43 -j"${BUILD_JOBS:-8}"
BIN=./build/demo_hblock_v43

SPACEV_DIR="${SPACEV_DIR:-/root/autodl-tmp/spacev100m}"
BASE="${SPACEV_DIR}/base.100M.i8bin"
QUERY="${SPACEV_DIR}/query.30K.i8bin"
IDS="${SPACEV_DIR}/ids.100M.i32bin"
GT="${SPACEV_DIR}/groundtruth.30K.i32bin"

# Where the two real region files get written -- fast local disk, not a
# network mount, or region fetch latency will dominate the numbers.
REGION_DIR="${REGION_DIR:-/root/autodl-tmp/hblock_v43_regions}"
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
CSV="results/hblock_v43.csv"
$BIN "$BASE" "$QUERY" "$IDS" "$GT" \
    "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
    "$REGION_MIB" "$CODE_CAP" "$RAW_CAP" \
    "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
    "$REGION_DIR" \
    "$CSV" 2>&1 | tee results/hblock_v43.log

echo ""
echo "Done. results/hblock_v43.{csv,log}. Region files under ${REGION_DIR}:"
ls -la "$REGION_DIR"
echo ""
echo "Compare 'uniq_this_call' and per-batch region-touch % against"
echo "results/hblock_v39_findings.md's batch=64 numbers (155-248 / 273"
echo "code regions, 57-91%) to see whether graph-aware packing helped."
