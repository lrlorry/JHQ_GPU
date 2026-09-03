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
  # Gate on memory in use, not on the compute-process list. Inside a container
  # that list can come back empty while the card is fully busy, because the
  # PIDs belong to another namespace -- so an empty list is not an all-clear.
  # Used memory is a direct reading and needs no PID visibility. asg14 uses the
  # same test, so the two scripts cannot talk past each other.
  while pgrep -f "bash /root/asg14.sh" >/dev/null \
     || pgrep -f "[r]un_p0_fair.sh" >/dev/null; do sleep 20; done
  for _ in $(seq 1 540); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    case "$used" in (''|*[!0-9]*) used=999999 ;; esac
    [ "$used" -lt 800 ] && { echo "card idle at ${used} MiB"; return 0; }
    sleep 20
  done
  echo "card never fell below 800 MiB; not starting"
  return 1
}
wait_for_card || exit 0
echo "card clear at $(date -u +%FT%TZ), memory.used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"

export JHQ_INDEX_CACHE=/root/jhq_cache
R=results/final
python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 \
  --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
  --trainset-fraction 0.05 --out $R/fair_stella_ivfpq_f0.05.csv
echo "### P0_FAIR_RERUN_DONE"
