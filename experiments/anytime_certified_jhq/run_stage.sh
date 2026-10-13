#!/bin/bash
# AC-JHQ offline stages on the box (CPU only; reuses the rank-optimal / progressive exports, no GPU).
#   /root/autodl-tmp/launch_b.sh exp/anytime-certified-jhq "experiments/anytime_certified_jhq/run_stage.sh synth pilot=openai3-3072_np128_a4" ac_stage1.log
set -u
exec 9>/root/.lock_ac; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.."
E=experiments/anytime_certified_jhq; R=results/anytime_certified_jhq
mkdir -p $R/raw
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) stages: $* ==="
for st in "$@"; do
  t0=$(date +%s)
  case $st in
    synth) python3 $E/synth_check.py $R > $R/raw/synth.txt 2>&1; echo "synth rc=$? $(tail -1 $R/raw/synth.txt)" ;;
    pilot=*) tag=${st#pilot=}
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 $E/acs_pilot.py $R $tag calib > $R/raw/pilot_$tag.txt 2>&1
      echo "pilot $tag rc=$?"; head -60 $R/raw/pilot_$tag.txt | cut -c1-260 ;;
    *) echo "unknown stage $st" ;;
  esac
  echo "stage $st took $(( $(date +%s) - t0 )) s"
done
echo "=== AC_STAGE_DONE $(date -u +%FT%TZ) ==="
