#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
SPACEV_DIR="${SPACEV_DIR:-/root/autodl-tmp/spacev100m}"
BASE="${BASE:-${SPACEV_DIR}/base.100M.i8bin}"
QUERY="${QUERY:-${SPACEV_DIR}/query.30K.i8bin}"
IDS="${IDS:-${SPACEV_DIR}/ids.100M.i32bin}"
GT="${GT:-${SPACEV_DIR}/groundtruth.30K.i32bin}"
NBASE="${NBASE:--1}"
NQUERY="${NQUERY:--1}"
MAX_EF="${MAX_EF:-256}"
REPS="${REPS:-3}"
REGION_MIB="${REGION_MIB:-1}"
CODE_REGION_CAP="${CODE_REGION_CAP:-256}"
RAW_REGION_CAP="${RAW_REGION_CAP:-256}"
BATCH="${BATCH:-1024}"

cd "$ROOT_DIR"
cmake --build build --target demo_hblock_v41_spacev -j"${BUILD_JOBS:-8}"

mkdir -p results
TS="$(date +%Y%m%d_%H%M%S)"
TAG="r${REGION_MIB}mib_c${CODE_REGION_CAP}_raw${RAW_REGION_CAP}"
OUT="results/hblock_v41_spacev_${TAG}_${TS}.txt"
CSV="results/hblock_v41_spacev_${TAG}_${TS}.csv"

{
    echo "=== HBlock v41: wave-streamed region pools, SPACEV ==="
    echo "date: $(date)"
    echo "base: $BASE"
    echo "query: $QUERY"
    echo "ids: $IDS"
    echo "gt: $GT"
    echo "nbase: $NBASE  nquery: $NQUERY"
    echo "region_bytes_mib=$REGION_MIB code_region_cap=$CODE_REGION_CAP raw_region_cap=$RAW_REGION_CAP"
    echo
    ./build/demo_hblock_v41_spacev \
        "$BASE" "$QUERY" "$IDS" "$GT" \
        "$NBASE" "$NQUERY" "$MAX_EF" \
        16 16 16 32 4 64 16 "$BATCH" "$REPS" \
        "$REGION_MIB" "$CODE_REGION_CAP" "$RAW_REGION_CAP" \
        "$CSV"
} 2>&1 | tee "$OUT"

echo "Done: $OUT  CSV: $CSV"
