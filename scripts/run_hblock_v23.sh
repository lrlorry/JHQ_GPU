#!/bin/bash
# Run hblock_v23 on both vogue-768 and arxiv-768, sweep graph_budget
# Output: results/hblock_v23_vogue.csv  results/hblock_v23_arxiv.csv

set -e
BIN=./build/demo_hblock_v23

VOGUE_BASE=/root/data/vogue-768_base.fvecs
VOGUE_QRY=/root/data/vogue-768_query.fvecs
VOGUE_GT=/root/data/vogue-768_groundtruth.ivecs

ARXIV_BASE=/root/autodl-tmp/arxiv-abstracts-768/base.fvecs
ARXIV_QRY=/root/autodl-tmp/arxiv-abstracts-768/query.fvecs
ARXIV_GT=/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs

# Fixed params
K1=16; K2=16; K3=16
CK1=2; CK2=2; CK3=4
K=10; BATCH=1024; D_PROJ=64; RERANK_R=128; KM_ITERS=30
DEGREE=32; ENTRY=4; C2N=4; C1N=2

BUDGETS="8 16 32 64"

run_dataset() {
    local NAME=$1; local BASE=$2; local QRY=$3; local GT=$4
    local CSV=results/hblock_v23_${NAME}.csv
    echo "========================================"
    echo "Dataset: $NAME  ->  $CSV"
    echo "========================================"
    echo "budget,recall@10,latency_ms,qps" > $CSV
    for BUD in $BUDGETS; do
        echo "--- budget=$BUD ---"
        OUT=$($BIN $BASE $QRY $GT \
            $K1 $K2 $K3 $CK1 $CK2 \
            $K $BATCH $D_PROJ $RERANK_R $KM_ITERS \
            $DEGREE $BUD $ENTRY $C2N $C1N 2>&1)
        echo "$OUT"
        RECALL=$(echo "$OUT" | grep "Recall@10" | awk '{print $NF}')
        LATENCY=$(echo "$OUT" | grep "Latency" | awk '{print $3}')
        QPS=$(echo "$OUT" | grep "QPS" | awk '{print $NF}')
        echo "$BUD,$RECALL,$LATENCY,$QPS" | tee -a $CSV
        echo ""
    done
    echo "Done: $NAME  csv=$CSV"
}

run_dataset vogue "$VOGUE_BASE" "$VOGUE_QRY" "$VOGUE_GT"
run_dataset arxiv "$ARXIV_BASE" "$ARXIV_QRY" "$ARXIV_GT"

echo ""
echo "All done. Results in results/hblock_v23_vogue.csv and results/hblock_v23_arxiv.csv"
