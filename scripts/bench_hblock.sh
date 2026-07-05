#!/usr/bin/env bash
# bench_hblock.sh — build and benchmark hblock v2/v3/v4, save results to file
# Usage: bash scripts/bench_hblock.sh [output_dir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$ROOT/results}"
mkdir -p "$OUT_DIR"

B=/root/data/vogue-768_base.fvecs
Q=/root/data/vogue-768_query.fvecs
G=/root/data/vogue-768_groundtruth.ivecs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT="$OUT_DIR/hblock_${TIMESTAMP}.txt"

log() { echo "$*" | tee -a "$RESULT"; }

# ── Build ─────────────────────────────────────────────────────────────────────
log "=== BUILD $(date) ==="
cd "$ROOT"
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 | tee -a "$RESULT"
cmake --build build --target demo_hblock_v2 demo_hblock_v3 demo_hblock_v4 -j"$(nproc)" 2>&1 | tail -5 | tee -a "$RESULT"
log ""

# ── Common params ─────────────────────────────────────────────────────────────
K1=64; K2=128; CK1=8; CK2=32; CK3=256; K=10; BATCH=1024

run_bench() {
    local label="$1"; shift
    log "=== $label ==="
    log "CMD: $*"
    log "--- output ---"
    "$@" 2>&1 | tee -a "$RESULT"
    log ""
}

# ── v2: JL + analytical product code ─────────────────────────────────────────
run_bench "hblock_v2 (JL + analytical, ck3=$CK3)" \
    ./build/demo_hblock_v2 "$B" "$Q" "$G" \
    $K1 $K2 $CK1 $CK2 $CK3 $K $BATCH

# ── v3: PCA cascade ───────────────────────────────────────────────────────────
run_bench "hblock_v3 (PCA cascade, k1=k2=32, ck3=$CK3)" \
    ./build/demo_hblock_v3 "$B" "$Q" "$G" \
    $K1 $K2 32 32 $CK1 $CK2 $CK3 $K $BATCH

# ── v4: Discriminative S_B projection ────────────────────────────────────────
run_bench "hblock_v4 (S_B discriminative, k1=k2=32, ck3=$CK3)" \
    ./build/demo_hblock_v4 "$B" "$Q" "$G" \
    $K1 $K2 32 32 $CK1 $CK2 $CK3 $K $BATCH

# ── v3 ck3 sweep: does better routing allow smaller ck3? ─────────────────────
log "=== hblock_v3 ck3 sweep (recall vs speed tradeoff) ==="
for ck3 in 64 128 256; do
    log "--- v3 ck3=$ck3 ---"
    ./build/demo_hblock_v3 "$B" "$Q" "$G" \
        $K1 $K2 32 32 $CK1 $CK2 $ck3 $K $BATCH 2>&1 | \
        grep -E "Recall|QPS|Latency|LeafFine|TOTAL" | tee -a "$RESULT"
    log ""
done

log "=== DONE: results saved to $RESULT ==="
echo ""
echo "Results: $RESULT"
