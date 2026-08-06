#!/bin/bash
# Run hblock_v42 on a SPACEV .i8bin subset: v39's region pool, now backed by
# real mmap'd files on disk (see hblock_v42/jhq_gpu_index.cuh).
#
# Defaults below are informed by hblock_v39's actual findings on this same
# dataset (see results/hblock_v39_findings.md):
#   - n_base=5000000 is the smallest size where region count (273 code
#     regions) is large enough for pool-capacity sweeps to mean anything;
#     500K only has 39 regions total, not enough room to test eviction.
#   - batch_size matters a lot more than pool size: a batch of 64 queries
#     touches 57-91% of all regions with this naive (physical block order)
#     layout, a batch of 8 touches only 11-33%. Both are swept below.
#
# Adjust SPACEV_DIR / REGION_DIR to your actual paths.

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
MAX_EF=64
K1=16; K2=16; K3=16
DEGREE=32; ENTRY=4
D_PROJ=64; PER_BLOCK_R=16

# batch  region_bytes_mib  gpu_code_region_cap  gpu_raw_region_cap
#
# The two "100/200" configs are deliberately undersized relative to what
# hblock_v39_findings.md measured a batch=64 call needs (up to ~248/273
# code regions at ef=64) -- they are expected to hit the batch-exceeds-pool
# guard and abort with a clear error, on purpose, to show that boundary.
# No `set -e`: each config runs independently and a thrown/aborted config
# does not stop the rest of the sweep.
CONFIGS=(
    "64 1 300 550"
    "64 1 100 200"
    "8  1 300 550"
    "8  1 40  80"
)

mkdir -p results
declare -a RESULT_LINES
for CFG in "${CONFIGS[@]}"; do
    read -r BATCH REGION_MIB CODE_CAP RAW_CAP <<< "$CFG"
    TAG="b${BATCH}_r${REGION_MIB}mib_c${CODE_CAP}"
    CSV="results/hblock_v42_${TAG}.csv"
    echo "========================================"
    echo "batch=${BATCH}  region_bytes=${REGION_MIB}MiB  gpu_code_region_cap=${CODE_CAP}  gpu_raw_region_cap=${RAW_CAP}"
    echo "-> ${CSV}"
    echo "========================================"
    set +e
    $BIN "$BASE" "$QUERY" "$IDS" "$GT" \
        "$N_BASE" "$N_QUERY" "$MAX_EF" "$BATCH" \
        "$REGION_MIB" "$CODE_CAP" "$RAW_CAP" \
        "$K1" "$K2" "$K3" "$DEGREE" "$ENTRY" "$D_PROJ" "$PER_BLOCK_R" \
        "$REGION_DIR" \
        "$CSV" 2>&1 | tee "results/hblock_v42_${TAG}.log"
    STATUS="${PIPESTATUS[0]}"
    set -e
    if [ "$STATUS" -eq 0 ]; then
        RESULT_LINES+=("OK   $TAG")
    else
        RESULT_LINES+=("FAIL $TAG (exit $STATUS -- check results/hblock_v42_${TAG}.log)")
    fi
    echo ""
done

echo "========================================"
echo "Summary:"
for line in "${RESULT_LINES[@]}"; do echo "  $line"; done
echo "========================================"
echo "Per-config CSV + full stdout log (incl. region fetch-planning"
echo "diagnostics printed by search() each call) under results/hblock_v42_*.{csv,log}"
echo "Region files themselves are under ${REGION_DIR} -- ls -la it to confirm"
echo "they're real files, not an in-process buffer."
