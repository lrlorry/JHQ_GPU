#!/usr/bin/env bash
# Mirrors run_hblock_v40_sift100m.sh exactly (same data, same paths, same
# NBASE/MAX_EF/REPS knobs) so the two results are directly comparable --
# v40 is the full-resident split-kernel baseline; this is the same task
# with v39's logical region partitioning + bounded GPU region pool on top.
set -euo pipefail

ROOT_DIR="${JHQ_GPU_ROOT:-/root/JHQ_GPU}"
DATA_DIR="${SIFT100M_DIR:-/root/data/sift100m}"
BASE="${BASE:-${DATA_DIR}/bigann_base.bvecs}"
QUERY="${QUERY:-${DATA_DIR}/bigann_query.bvecs}"
GT="${GT:-${DATA_DIR}/gnd/idx_100M.ivecs}"
NBASE="${NBASE:-100000000}"
MAX_EF="${MAX_EF:-256}"
REPS="${REPS:-3}"

# Region-pool knobs, specific to v39. REGION_MIB_CAP below is not a single
# number -- see the CONFIGS sweep: each entry is
# "region_bytes_mib gpu_code_region_cap gpu_raw_region_cap". First entry
# should comfortably exceed the whole store (looks like v40's full-resident
# behavior); later entries shrink the pool so eviction actually happens.
CONFIGS=(
    "4 100000 100000"
    "1 512    512"
)

cd "$ROOT_DIR"
cmake --build build --target demo_hblock_v39_sift100m -j"${BUILD_JOBS:-8}"

mkdir -p results
TS="$(date +%Y%m%d_%H%M%S)"

for CFG in "${CONFIGS[@]}"; do
    read -r REGION_MIB CODE_CAP RAW_CAP <<< "$CFG"
    TAG="r${REGION_MIB}mib_c${CODE_CAP}"
    OUT="results/hblock_v39_sift100m_${TAG}_${TS}.txt"
    CSV="results/hblock_v39_sift100m_${TAG}_${TS}.csv"
    {
        echo "=== HBlock v39: region-partitioned GPU pool, SIFT100M ==="
        echo "date: $(date)"
        echo "base: $BASE"
        echo "query: $QUERY"
        echo "gt: $GT"
        echo "nbase: $NBASE"
        echo "region_bytes_mib=$REGION_MIB gpu_code_region_cap=$CODE_CAP gpu_raw_region_cap=$RAW_CAP"
        echo
        ./build/demo_hblock_v39_sift100m \
            "$BASE" "$QUERY" "$GT" \
            "$NBASE" "$MAX_EF" \
            16 16 16 32 4 64 16 1024 "$REPS" \
            "$REGION_MIB" "$CODE_CAP" "$RAW_CAP" \
            "$CSV"
    } 2>&1 | tee "$OUT"
    echo "Done: $OUT  CSV: $CSV"
    echo ""
done
