#!/usr/bin/env bash
# v21 cascade sweep: occupancy (JHQ_BLOCK) x prefix fraction x survivor count.
#
# The scan block carries a 48 KB table, so at 256 threads one block fits per SM
# and 256 of 1536 thread slots are live -- 17% occupancy, which is the standing
# explanation for the measured 235 GB/s marginal bandwidth (13% of peak).
# 1024 threads costs 89 KB, still one block, but four times the threads.
#
# The cascade is the other axis: pass 1 ranks candidates on a prefix of the
# subspaces, pass 2 completes only the best KEEP per thread. PFX 1/1 disables
# it, which is the v19 control.
set -u
DATA=${DATA:-/root/data/vogue-768_}
BIN=${BIN:-./build}
NP=${NP:-128}
BATCH=${BATCH:-1024}
ALPHA=${ALPHA:-100.0}

printf "%6s %5s %7s %9s %10s %9s\n" "BLOCK" "KEEP" "prefix" "recall" "scan_ms" "QPS"
for blk in 256 512 1024; do
  for keep in 4 8 16 32; do
    for pfx in "1 1" "1 2" "1 3" "1 4" "1 8"; do
      set -- $pfx
      out=$(JHQ_BLOCK=$blk JHQ_PFX_NUM=$1 JHQ_PFX_DEN=$2 \
            $BIN/demo_jhq_v21_k${keep} \
            ${DATA}base.fvecs ${DATA}query.fvecs ${DATA}groundtruth.ivecs \
            96 8 4 $ALPHA 10 1024 $NP 8 $BATCH "" 5 2>&1)
      printf "%6s %5s %7s %9s %10s %9s\n" "$blk" "$keep" "$1/$2" \
        "$(echo "$out" | grep '^Recall@'  | awk '{print $3}')" \
        "$(echo "$out" | grep scan_ivf | head -1 | awk '{print $2}')" \
        "$(echo "$out" | grep '^QPS'     | awk '{print $3}')"
    done
  done
done
