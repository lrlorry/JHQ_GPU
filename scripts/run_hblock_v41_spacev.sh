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
REPS="${REPS:-1}"
REGION_MIB="${REGION_MIB:-1}"
REGION_CAP="${REGION_CAP:-512}"
# SPACEV has 30K queries. One large query batch lets all queries sharing a
# logical region reuse the same H2D transfer before the next wave.
BATCH="${BATCH:-32768}"

cd "$ROOT_DIR"
cmake -S . -B build
cmake --build build --target demo_hblock_v41_spacev -j"${BUILD_JOBS:-8}"

mkdir -p results
TS="$(date +%Y%m%d_%H%M%S)"
TAG="r${REGION_MIB}mib_cap${REGION_CAP}_b${BATCH}"
OUT="results/hblock_v41_spacev_${TAG}_${TS}.txt"
CSV="results/hblock_v41_spacev_${TAG}_${TS}.csv"

{
    echo "=== HBlock v41: fused wave-streamed regions, SPACEV ==="
    echo "date: $(date)"
    echo "base: $BASE"
    echo "query: $QUERY"
    echo "ids: $IDS"
    echo "gt: $GT"
    echo "nbase: $NBASE  nquery: $NQUERY"
    echo "region_bytes_mib=$REGION_MIB region_cap=$REGION_CAP batch=$BATCH"
    echo
    ./build/demo_hblock_v41_spacev \
        "$BASE" "$QUERY" "$IDS" "$GT" \
        "$NBASE" "$NQUERY" "$MAX_EF" \
        16 16 16 32 4 64 16 "$BATCH" "$REPS" \
        "$REGION_MIB" "$REGION_CAP" \
        "$CSV"
} 2>&1 | tee "$OUT"

echo "Done: $OUT  CSV: $CSV"
