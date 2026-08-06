#!/bin/bash
# Build and run hblock_v38 end-to-end on a SPACEV .i8bin subset: the plain,
# fully GPU-resident baseline (no region partitioning). Same N_BASE/batch/ef
# and same dataset as run_hblock_v39.sh / v42 / v43 on purpose -- this is
# the controlled comparison point. Any recall/QPS difference between this
# and v39/v42/v43 on identical settings is attributable to region
# partitioning, not to the dataset or encoding changing at the same time.

set -e
ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
cd "$ROOT_DIR"

cmake --build build --target demo_hblock_v38 -j"${BUILD_JOBS:-8}"
BIN=./build/demo_hblock_v38

SPACEV_DIR="${SPACEV_DIR:-/root/autodl-tmp/spacev100m}"
BASE="${SPACEV_DIR}/base.100M.i8bin"
QUERY="${SPACEV_DIR}/query.30K.i8bin"
IDS="${SPACEV_DIR}/ids.100M.i32bin"
GT="${SPACEV_DIR}/groundtruth.30K.i32bin"

N_BASE="${N_BASE:-5000000}"
N_QUERY="${N_QUERY:-1000}"
MAX_EF="${MAX_EF:-64}"
BATCH="${BATCH:-64}"
K1=16; K2=16; K3=16
DEGREE=32; ENTRY=4
D_PROJ=64; PER_BLOCK_R=16

mkdir -p results
CSV="results/hblock_v38.csv"
$BIN "$BASE" "$QUERY" "$IDS" "$GT" \
    "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
    "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
    "$CSV" 2>&1 | tee results/hblock_v38.log

echo ""
echo "Done. results/hblock_v38.{csv,log}."
echo "Compare recall@10/QPS directly against results/hblock_v39_findings.md"
echo "and results/hblock_v42.csv / hblock_v43.csv (same N_BASE/batch/ef)."
