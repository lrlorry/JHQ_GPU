#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
DATA_DIR="${SIFT100M_DIR:-/root/data/sift100m}"
BASE="${BASE:-${DATA_DIR}/bigann_base.bvecs}"
QUERY="${QUERY:-${DATA_DIR}/bigann_query.bvecs}"
GT="${GT:-${DATA_DIR}/gnd/idx_100M.ivecs}"
NBASE="${NBASE:-100000000}"
MAX_EF="${MAX_EF:-256}"
REPS="${REPS:-3}"

cd "$ROOT_DIR"
cmake --build build --target demo_hblock_v40 -j"${BUILD_JOBS:-8}"

mkdir -p results
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/hblock_v40_sift100m_${TS}.txt"
CSV="results/hblock_v40_sift100m_${TS}.csv"

{
    echo "=== HBlock v40: split PQ/exact resident SIFT100M baseline ==="
    echo "date: $(date)"
    echo "base: $BASE"
    echo "query: $QUERY"
    echo "gt: $GT"
    echo "nbase: $NBASE"
    echo
    ./build/demo_hblock_v40 \
        "$BASE" "$QUERY" "$GT" \
        "$NBASE" "$MAX_EF" \
        16 16 16 32 4 64 16 1024 "$REPS" "$CSV"
} 2>&1 | tee "$OUT"

echo "Done: $OUT  CSV: $CSV"
