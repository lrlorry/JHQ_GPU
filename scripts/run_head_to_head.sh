#!/usr/bin/env bash
# JHQ vs cuVS at matched recall, matched batch.
#
# The standing comparison was not like-for-like. cuVS's 0.9441 @ 20907 QPS comes
# from ivf_pq.search(sp, idx, xq_d, k) with the whole 1000-query set in one
# call; the JHQ figure it was set against, 0.9448 @ 10491, was run at
# batch_size=256. The same log has Vogue at nprobe=8 going 16856 -> 40613 QPS
# between those two batch sizes, so the handicap is worth about 2.4x on its own.
#
# Matched point: M=96, alpha=100, nlist=1024, nprobe=128, k=10.
set -u
DATA=${DATA:-/root/data/vogue-768_}
BIN=${BIN:-./build}
D="${DATA}base.fvecs ${DATA}query.fvecs ${DATA}groundtruth.ivecs"

run() {  # name binary batch extra-env
    local name=$1 bin=$2 b=$3; shift 3
    local out
    out=$(env "$@" $BIN/$bin $D 96 8 4 100.0 10 1024 128 8 $b "" 5 2>&1)
    printf "%-26s %6s %9s %10s %9s\n" "$name" "$b" \
        "$(echo "$out" | grep '^Recall@'  | awk '{print $3}')" \
        "$(echo "$out" | grep scan_ivf | head -1 | awk '{print $2}')" \
        "$(echo "$out" | grep '^QPS'     | awk '{print $3}')"
}

printf "%-26s %6s %9s %10s %9s\n" "variant" "batch" "recall" "scan_ms" "QPS"
for b in 256 1024; do
    run "v16 (published point)" demo_jhq_v16_pq_primary $b JHQ_UNUSED=1
    run "v19 tiled"             demo_jhq_v19_timed      $b JHQ_UNUSED=1
    for blk in 256 1024; do
      run "v21 casc 1/4 k8 blk$blk" demo_jhq_v21_k8 $b \
          JHQ_BLOCK=$blk JHQ_PFX_NUM=1 JHQ_PFX_DEN=4
    done
done
echo
echo "cuVS-IVFPQ-384B nprobe=128: recall 0.9441, QPS 20907, build 4.52 s (batch 1000)"
