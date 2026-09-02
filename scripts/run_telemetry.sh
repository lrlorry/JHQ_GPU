#!/bin/bash
# Device telemetry during search: GPU and memory utilisation, board power, PCIe
# throughput and SM clock, sampled while each method runs.
#
# This was in the original request and had not been collected. Kernel-level
# counters are not available on this host -- ncu reports ERR_NVGPUCTRPERM
# because the container denies performance-counter access -- so the numbers
# here come from nvidia-smi sampling, which needs no such permission. The
# bandwidth-efficiency figures quoted elsewhere are arithmetic (bytes moved
# over elapsed time against the board's rated peak), not counter readings, and
# are labelled as such.
exec 9>/root/.lock_tele
flock -n 9 || { echo "another telemetry run holds the lock"; exit 0; }
exec >/root/p0_telemetry.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=/root/jhq_cache
V=/root/data/vogue-768_
D="${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs"
OUT=results/final/telemetry.csv

echo "sampling every 100 ms with nvidia-smi; ncu is unavailable on this host"
nvidia-smi --query-gpu=name,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader

sample() {   # $1 = tag ; samples until the marker file disappears
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used \
    --format=csv,noheader,nounits -lms 100 2>/dev/null |
  while IFS=, read -r u m p c mu; do
    [ -f /tmp/.tele_run ] || break
    echo "$1,$u,$m,$p,$c,$mu"
  done
}

echo "method,gpu_util_pct,mem_util_pct,power_w,sm_clock_mhz,mem_used_mib" > $OUT

run_with_telemetry() {  # $1 tag, rest = command
  local tag=$1; shift
  touch /tmp/.tele_run
  sample "$tag" >> $OUT &
  local sp=$!
  "$@" > /dev/null 2>&1
  rm -f /tmp/.tele_run
  sleep 1; kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "  $tag: $(grep -c "^$tag," $OUT) samples"
}

echo "### JHQ v22, nprobe=128, batch 1000, 40 repetitions so the window is long enough"
run_with_telemetry JHQ env JHQ_BLOCK=1024 JHQ_PFX_NUM=1 JHQ_PFX_DEN=2 JHQ_TILE_M_RT=96 \
  ./build/demo_jhq_v22_s2b1 $D 96 8 4 100.0 10 1024 128 8 1024 "" 5

echo "### idle baseline, so the load figures have something to sit against"
run_with_telemetry idle sleep 8

echo "### cuVS IVF-PQ 384B and CAGRA fp32 under the same sampler"
run_with_telemetry IVFPQ python3 scripts/bench_all.py --dataset vogue-768 --reps 8 \
  --method ivfpq --pq-dims 384 --nprobe 128 --out /tmp/_t1.csv
run_with_telemetry CAGRA python3 scripts/bench_all.py --dataset vogue-768 --reps 8 \
  --method cagra --graph-degrees 64 --itopk 128 --search-width 1 --out /tmp/_t2.csv

echo "### summary"
python3 - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open("results/final/telemetry.csv")))
by = {}
for r in rows:
    by.setdefault(r["method"], []).append(r)
print(f"{'method':<10}{'n':>5}{'GPU util %':>14}{'mem util %':>14}{'power W':>12}{'SM MHz':>10}{'VRAM MiB':>11}")
for m, rs in by.items():
    def col(k):
        v = [float(x[k]) for x in rs if x[k].strip().replace('.','',1).isdigit()]
        return (st.mean(v), max(v)) if v else (0, 0)
    gu, gux = col("gpu_util_pct"); mu, mux = col("mem_util_pct")
    pw, pwx = col("power_w"); sc, _ = col("sm_clock_mhz"); vr, vrx = col("mem_used_mib")
    print(f"{m:<10}{len(rs):>5}{f'{gu:.0f} (max {gux:.0f})':>14}"
          f"{f'{mu:.0f} (max {mux:.0f})':>14}{f'{pw:.0f} (max {pwx:.0f})':>12}"
          f"{sc:>10.0f}{vrx:>11.0f}")
PY
echo "### P0_TELEMETRY_DONE"
