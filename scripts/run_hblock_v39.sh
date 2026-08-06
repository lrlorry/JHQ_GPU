#!/bin/bash
# Run hblock_v39 on a SPACEV .i8bin subset: streamed int8 add() + logical
# region partitioning + bounded GPU region pool. Sweeps region-pool capacity
# so the same recall/QPS numbers can be compared under "pool holds
# everything" (looks like v38's full-resident baseline) vs "pool is
# genuinely capacity-bounded" (out-of-core fetch/evict path is exercised).
#
# Adjust SPACEV_DIR and the *_FILE names below to your actual SPACEV subset
# layout (see common/spacev_io.cuh for the expected file formats).

set -e
BIN=./build/demo_hblock_v39

SPACEV_DIR="${SPACEV_DIR:-/root/data/spacev-100m}"
BASE="${SPACEV_DIR}/base.i8bin"
QUERY="${SPACEV_DIR}/query.i8bin"
IDS="${SPACEV_DIR}/ids.100M.i32bin"
GT="${SPACEV_DIR}/groundtruth.100K.i32bin"

MAX_EF=128
BATCH=256
K1=16; K2=16; K3=16
DEGREE=32; ENTRY=4
D_PROJ=64; PER_BLOCK_R=16

# region_bytes_mib gpu_code_region_cap gpu_raw_region_cap
# "full" first: pool comfortably exceeds the whole store, so region_reuse
# from search()'s own stdout diagnostics reflects "no real paging happens".
# "bounded" second: small enough that eviction genuinely happens each batch
# -- this is the configuration that actually tests out-of-core behavior.
CONFIGS=(
    "4 100000 100000"
    "1 512    512"
    "1 128    128"
)

mkdir -p results
for CFG in "${CONFIGS[@]}"; do
    read -r REGION_MIB CODE_CAP RAW_CAP <<< "$CFG"
    TAG="r${REGION_MIB}mib_c${CODE_CAP}"
    CSV="results/hblock_v39_${TAG}.csv"
    echo "========================================"
    echo "region_bytes=${REGION_MIB}MiB  gpu_code_region_cap=${CODE_CAP}  gpu_raw_region_cap=${RAW_CAP}"
    echo "-> ${CSV}"
    echo "========================================"
    $BIN "$BASE" "$QUERY" "$IDS" "$GT" \
        -1 -1 "$MAX_EF" "$BATCH" \
        "$REGION_MIB" "$CODE_CAP" "$RAW_CAP" \
        "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
        "$CSV" 2>&1 | tee "results/hblock_v39_${TAG}.log"
    echo ""
done

echo "All done. Per-config CSV + full stdout log (incl. region fetch-planning"
echo "diagnostics printed by search() each call) under results/hblock_v39_*.{csv,log}"
