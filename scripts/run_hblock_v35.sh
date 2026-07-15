#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768_base.fvecs
QUERY=/root/data/vogue-768_query.fvecs
GT=/root/data/vogue-768_groundtruth.ivecs
TS=$(date +%Y%m%d_%H%M%S)
OUT=results/hblock_v35_${TS}.txt
CSV=results/hblock_v35_${TS}.csv
mkdir -p results

{
echo "=== HBlock v35: W=4 batched expansion + HNSW early termination, beam=ef (no cap), no early termination, ceiling test ==="
echo "date: $(date)"
echo ""
./build/demo_hblock_v35 $BASE $QUERY $GT 256 16 16 16 32 4 64 16 1024 $CSV
} 2>&1 | tee $OUT

echo "Done: $OUT  CSV: $CSV"
