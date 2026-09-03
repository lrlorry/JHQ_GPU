#!/bin/bash
# Re-measure the fairness rows that were taken while another session's job
# shared the card. Recall was unaffected; qps_mean and vram_mib were not --
# one row recorded a negative memory delta and two had QPS spreads above 11%
# against the 0.02-0.64% an idle card gives.
#
# Waits for asg14 and for the card to be clear of compute processes, so this
# does not create the same problem in the other direction.
exec 9>/root/.lock_fair_rerun
flock -n 9 || { echo "another rerun holds the lock"; exit 0; }
exec >/root/p0_fair_rerun.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."

wait_for_card() {
  while pgrep -f "bash /root/asg14.sh" >/dev/null \
     || pgrep -f "[r]un_p0_fair.sh" >/dev/null; do sleep 20; done
  # and until nvidia-smi reports no compute process at all
  for _ in $(seq 1 360); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -eq 0 ] && return 0
    sleep 20
  done
}
wait_for_card
echo "card clear at $(date -u +%FT%TZ); compute apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"

export JHQ_INDEX_CACHE=/root/jhq_cache
R=results/final
python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 \
  --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
  --trainset-fraction 0.05 --out $R/fair_stella_ivfpq_f0.05.csv
echo "### P0_FAIR_RERUN_DONE"
