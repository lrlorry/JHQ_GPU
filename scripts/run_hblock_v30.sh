#!/bin/bash
# Run hblock_v30 on both vogue-768 and arxiv-768
# v29: per-block exact rerank — PQ prefilter (top per_block_r) → exact L2 → merge.
# Eliminates cross-block PQ ranking error; oracle recall ≈ actual recall.

set -e
BIN=./build/demo_hblock_v30

VOGUE_BASE=/root/data/vogue-768_base.fvecs
VOGUE_QRY=/root/data/vogue-768_query.fvecs
VOGUE_GT=/root/data/vogue-768_groundtruth.ivecs

ARXIV_BASE=/root/autodl-tmp/arxiv-abstracts-768/base.fvecs
ARXIV_QRY=/root/autodl-tmp/arxiv-abstracts-768/query.fvecs
ARXIV_GT=/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs

K1=16; K2=16; K3=16
CK1=2; CK2=2
K=10; BATCH=1024; D_PROJ=64; KM_ITERS=30
DEGREE=32; ENTRY=4; C2N=4; C1N=2; MINI_KM=5

# per_block_r: PQ candidates per block before exact L2
# klocal: per-block exact top-k (must be >= k=10)
PER_BLOCK_R=16
KLOCAL=10

DEPTHS="32 64 128 256"
BEAM_SIZES="32 64 128"

run_dataset() {
    local NAME=$1; local BASE=$2; local QRY=$3; local GT=$4
    local CSV=results/hblock_v30_${NAME}.csv
    echo "========================================"
    echo "Dataset: $NAME  ->  $CSV"
    echo "========================================"
    echo "depth,beam_size,oracle_recall@10,recall@10,latency_ms,qps" > $CSV
    for DEP in $DEPTHS; do
        for BS in $BEAM_SIZES; do
            echo "--- depth=$DEP  beam_size=$BS  per_block_r=$PER_BLOCK_R  klocal=$KLOCAL ---"
            OUT=$($BIN $BASE $QRY $GT \
                $K1 $K2 $K3 $CK1 $CK2 \
                $K $BATCH $D_PROJ $PER_BLOCK_R $KM_ITERS \
                $DEGREE $DEP $ENTRY $C2N $C1N $BS $KLOCAL $MINI_KM 2>&1)
            echo "$OUT"
            ORACLE=$(echo "$OUT" | grep "Oracle Recall@" | tail -1 | awk '{print $NF}')
            RECALL=$(echo "$OUT" | grep "PQ    Recall@"  | awk '{print $NF}')
            LATENCY=$(echo "$OUT" | grep "Latency" | awk '{print $3}')
            QPS=$(echo "$OUT"    | grep "QPS"     | awk '{print $NF}')
            echo "$DEP,$BS,$ORACLE,$RECALL,$LATENCY,$QPS" | tee -a $CSV
            echo ""
        done
    done
    echo "Done: $NAME  csv=$CSV"
}

run_dataset vogue "$VOGUE_BASE" "$VOGUE_QRY" "$VOGUE_GT"
run_dataset arxiv "$ARXIV_BASE" "$ARXIV_QRY" "$ARXIV_GT"

echo ""
echo "All done. Results in results/hblock_v30_vogue.csv and results/hblock_v30_arxiv.csv"
