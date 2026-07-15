#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768_base.fvecs
QUERY=/root/data/vogue-768_query.fvecs
GT=/root/data/vogue-768_groundtruth.ivecs
TS=$(date +%Y%m%d_%H%M%S)
OUT=results/hblock_v33_${TS}.txt
CSV=results/hblock_v33_${TS}.csv
mkdir -p results

{
echo "=== HBlock v33: W=4 batched expansion + HNSW early termination, beam=ef (no cap) ==="
echo "date: $(date)"
echo ""
./build/demo_hblock_v33 $BASE $QUERY $GT 256 16 16 16 32 4 64 16 1024 $CSV
} 2>&1 | tee $OUT

echo "Done: $OUT  CSV: $CSV"
