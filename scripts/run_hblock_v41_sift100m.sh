#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
DATA_DIR="${SIFT100M_DIR:-/root/data/sift100m}"
BASE="${BASE:-${DATA_DIR}/bigann_base.bvecs}"
QUERY="${QUERY:-${DATA_DIR}/bigann_query.bvecs}"
GT="${GT:-${DATA_DIR}/gnd/idx_100M.ivecs}"
NBASE="${NBASE:-100000000}"
MAX_EF="${MAX_EF:-256}"
REPS="${REPS:-1}"
REGION_MIB="${REGION_MIB:-1}"
REGION_CAP="${REGION_CAP:-512}"
BATCH="${BATCH:-10000}"

cd "$ROOT_DIR"
cmake --build build --target demo_hblock_v41_sift100m -j"${BUILD_JOBS:-8}"

mkdir -p results
TS="$(date +%Y%m%d_%H%M%S)"
TAG="r${REGION_MIB}mib_cap${REGION_CAP}_b${BATCH}"
OUT="results/hblock_v41_sift100m_${TAG}_${TS}.txt"
CSV="results/hblock_v41_sift100m_${TAG}_${TS}.csv"

{
    echo "=== HBlock v41: fused wave-streamed regions, SIFT100M ==="
    echo "date: $(date)"
    echo "base: $BASE"
    echo "query: $QUERY"
    echo "gt: $GT"
    echo "nbase: $NBASE"
    echo "region_bytes_mib=$REGION_MIB region_cap=$REGION_CAP batch=$BATCH"
    echo
    ./build/demo_hblock_v41_sift100m \
        "$BASE" "$QUERY" "$GT" \
        "$NBASE" "$MAX_EF" \
        16 16 16 32 4 64 16 "$BATCH" "$REPS" \
        "$REGION_MIB" "$REGION_CAP" \
        "$CSV"
} 2>&1 | tee "$OUT"

echo "Done: $OUT  CSV: $CSV"
