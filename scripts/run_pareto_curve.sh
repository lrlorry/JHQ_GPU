#!/usr/bin/env bash
# QPS-vs-recall front for JHQ v21 against cuVS IVF-PQ, matched batch.
#
# A single operating point is not a comparison: both systems trade recall for
# throughput along nprobe, and the claim has to hold along the curve. cuVS
# searches the whole query set in one call, so batch is pinned to 1024 here.
#
# The v16 point this replaces was run at batch 256 against a cuVS number taken
# at 1000, which is where most of the apparent 2x gap came from.
set -u
DATA=${DATA:-/root/data/vogue-768_}
BIN=${BIN:-./build}
BATCH=${BATCH:-1024}
ALPHA=${ALPHA:-100.0}
KEEP=${KEEP:-8}
BLK=${BLK:-1024}
PFXN=${PFXN:-1}; PFXD=${PFXD:-4}
D="${DATA}base.fvecs ${DATA}query.fvecs ${DATA}groundtruth.ivecs"

printf "%-10s %7s %9s %10s %9s\n" "variant" "nprobe" "recall" "scan_ms" "QPS"
for np in 8 16 32 64 128 256; do
  o=$($BIN/demo_jhq_v16_pq_primary $D 96 8 4 $ALPHA 10 1024 $np 8 $BATCH "" 5 2>&1)
  printf "%-10s %7s %9s %10s %9s\n" "v16" "$np" \
    "$(echo "$o" | grep '^Recall@' | awk '{print $3}')" "-" \
    "$(echo "$o" | grep '^QPS' | awk '{print $3}')"
  o=$(JHQ_BLOCK=$BLK JHQ_PFX_NUM=$PFXN JHQ_PFX_DEN=$PFXD JHQ_TILE_M_RT=96 \
      $BIN/demo_jhq_v21_k${KEEP} $D 96 8 4 $ALPHA 10 1024 $np 8 $BATCH "" 5 2>&1)
  printf "%-10s %7s %9s %10s %9s\n" "v21" "$np" \
    "$(echo "$o" | grep '^Recall@' | awk '{print $3}')" \
    "$(echo "$o" | grep scan_ivf | head -1 | awk '{print $2}')" \
    "$(echo "$o" | grep '^QPS' | awk '{print $3}')"
done
echo
echo "cuVS IVF-PQ 384B, batch 1000: see results/cuvs_ivfpq_vogue.csv"
