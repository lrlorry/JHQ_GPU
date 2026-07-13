#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768/base.fvecs
QUERY=/root/data/vogue-768/query.fvecs
GT=/root/data/vogue-768/gt.ivecs
OUT=results/hblock_v32_$(date +%Y%m%d_%H%M%S).txt
mkdir -p results

BEAMS="8 16 32 64 128"

{
echo "=== HBlock v32: single beam parameter ==="
echo "date: $(date)"
echo ""
for beam in $BEAMS; do
    echo "--- beam=$beam ---"
    ./build/demo_hblock_v32 $BASE $QUERY $GT $beam
    echo ""
done
} 2>&1 | tee $OUT

echo "Done: $OUT"
