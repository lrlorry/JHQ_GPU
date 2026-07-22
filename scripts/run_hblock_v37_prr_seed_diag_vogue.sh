#!/bin/bash
set -e
cd /root/JHQ_GPU
BASE=/root/data/vogue-768_base.fvecs
QUERY=/root/data/vogue-768_query.fvecs
GT=/root/data/vogue-768_groundtruth.ivecs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results
OUT_PREFIX=results/hblock_v37_prr_seed_diag_${TS}

{
echo "=== HBlock v37_prr: exact-seed threshold diagnostic (seed tau2 go/no-go) ==="
echo "date: $(date)"
echo ""
./build/demo_hblock_v37_prr_seed_diag $BASE $QUERY $GT $OUT_PREFIX
} 2>&1 | tee ${OUT_PREFIX}.log

echo "Done: ${OUT_PREFIX}.txt  CSV: ${OUT_PREFIX}.csv  console log: ${OUT_PREFIX}.log"
