#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768/base.fvecs
QUERY=/root/data/vogue-768/query.fvecs
GT=/root/data/vogue-768/gt.ivecs
OUT=results/hblock_v32_$(date +%Y%m%d_%H%M%S).txt
mkdir -p results

{
echo "=== HBlock v32: single beam parameter, build once + sweep ef ==="
echo "date: $(date)"
echo ""
# Build with max_ef=128; demo sweeps ef=8,16,32,64,128 at search time
./build/demo_hblock_v32 $BASE $QUERY $GT 128
} 2>&1 | tee $OUT

echo "Done: $OUT"
