#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768_base.fvecs
QUERY=/root/data/vogue-768_query.fvecs
GT=/root/data/vogue-768_groundtruth.ivecs
TS=$(date +%Y%m%d_%H%M%S)
OUT=results/hblock_v32_${TS}.txt
CSV=results/hblock_v32_${TS}.csv
mkdir -p results

{
echo "=== HBlock v32: v30 semantics, single ef parameter (depth=ef, beam=min(ef,128)) ==="
echo "date: $(date)"
echo ""
./build/demo_hblock_v32 $BASE $QUERY $GT 256 16 16 16 32 4 64 16 1024 $CSV
} 2>&1 | tee $OUT

echo "Done: $OUT  CSV: $CSV"
