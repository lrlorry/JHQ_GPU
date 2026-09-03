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

# Whole-card mutual exclusion, held for the duration of the measurements.
#
# The memory gate stops a job starting while the card is already busy, but it
# does not stop two waiters passing it in the same instant: both poll, both see
# an idle card, both start. asg15 and this script share the same gates and
# neither waits on the other, so that race is live. A lock removes it without
# either side having to wait on the other's process name -- which would
# deadlock if both did it.
exec 8>/root/.gpu_lock
echo "waiting for the shared GPU lock at $(date -u +%FT%TZ)"
flock 8
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
# Four rows across two files carry a contention signature, and one more is
# doubtful on memory alone (stella f0.05 pq128 np32 read 15,951 MiB against
# 3,486 for the same configuration at f0.02). Re-measuring whole files rather
# than single points keeps each file internally consistent.
python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 \
  --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
  --trainset-fraction 0.05 --out $R/fair_stella_ivfpq_f0.05.csv
python3 scripts/bench_all.py --dataset bge-m3 --reps 3 --nlist 8192 \
  --method ivfpq --pq-dims 128,256 --nprobe 32,128,256 \
  --trainset-fraction 0.10 --out $R/fair_bge_ivfpq_f0.10.csv
echo "### P0_FAIR_RERUN_DONE"
