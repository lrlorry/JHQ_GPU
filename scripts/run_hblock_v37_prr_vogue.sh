#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768_base.fvecs
QUERY=/root/data/vogue-768_query.fvecs
GT=/root/data/vogue-768_groundtruth.ivecs
TS=$(date +%Y%m%d_%H%M%S)
OUT=results/hblock_v37_prr_${TS}.txt
CSV=results/hblock_v37_prr_${TS}.csv
mkdir -p results

# search_mode: 0=FIXED_PER_BLOCK 1=CORRECTED_FIXED 2=CERTIFIED_PRR
# eps_mode:    0=BLOCK_128 1=SUBBLOCK_32 2=VECTOR_LEVEL
SEARCH_MODE=${1:-0}
EPS_MODE=${2:-0}

{
echo "=== HBlock v37_prr: certified PQ racing (PRR) ==="
echo "date: $(date)"
echo "search_mode=${SEARCH_MODE}  eps_mode=${EPS_MODE}"
echo ""
./build/demo_hblock_v37_prr $BASE $QUERY $GT 256 16 16 16 32 4 64 16 1024 $SEARCH_MODE $EPS_MODE $CSV
} 2>&1 | tee $OUT

echo "Done: $OUT  CSV: $CSV"
