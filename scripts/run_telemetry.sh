#!/bin/bash
# Device telemetry during search: SM and memory-controller utilisation, board
# power, and PCIe throughput in both directions.
#
# These were asked for at the start of this work and never collected. Kernel
# counters are unavailable on this host -- ncu reports ERR_NVGPUCTRPERM because
# the container denies performance-counter access -- so the numbers come from
# nvidia-smi dmon, which needs no such permission and does report PCIe rx/tx.
# The bandwidth-efficiency percentages quoted elsewhere are arithmetic (bytes
# moved over elapsed time against the board's rated peak), not counter reads.
exec 9>/root/.lock_tele
flock -n 9 || { echo "another telemetry run holds the lock"; exit 0; }
exec >/root/p0_telemetry.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_p0_gap.sh|[r]un_p0_stella.sh" >/dev/null; do sleep 30; done
export JHQ_INDEX_CACHE=/root/jhq_cache
V=/root/data/vogue-768_
D="${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs"
RAW=/tmp/dmon.txt
MARKS=/tmp/marks.txt

nvidia-smi --query-gpu=name,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max \
  --format=csv,noheader
: > $MARKS
nvidia-smi dmon -o T -s put -d 1 > $RAW 2>/dev/null &
DMON=$!
sleep 3

mark() { echo "$1 $(date +%s)" >> $MARKS; }

mark idle_start;  sleep 12;  mark idle_end

mark jhq_start
for i in 1 2 3 4 5 6; do
  JHQ_BLOCK=1024 JHQ_PFX_NUM=1 JHQ_PFX_DEN=2 JHQ_TILE_M_RT=96 \
    ./build/demo_jhq_v22_s2b1 $D 96 8 4 100.0 10 1024 128 8 1024 "" 5 >/dev/null 2>&1
done
mark jhq_end

mark ivfpq_start
python3 scripts/bench_all.py --dataset vogue-768 --reps 12 --method ivfpq \
  --pq-dims 384 --nprobe 128 --out /tmp/_t1.csv >/dev/null 2>&1
mark ivfpq_end

mark cagra_start
python3 scripts/bench_all.py --dataset vogue-768 --reps 12 --method cagra \
  --graph-degrees 64 --itopk 128 --search-width 1 --out /tmp/_t2.csv >/dev/null 2>&1
mark cagra_end

sleep 2; kill $DMON 2>/dev/null; wait $DMON 2>/dev/null
echo "dmon samples: $(grep -vc '^#' $RAW)"

python3 - "$RAW" "$MARKS" <<'PY'
import sys, statistics as st
raw, marks = sys.argv[1], sys.argv[2]
# dmon -o T prefixes each row with seconds since boot, not epoch, so segment on
# sample order between the marks instead: read wall-clock marks and the sample
# count, then split proportionally. Simpler and adequate at 1 Hz.
spans = {}
for line in open(marks):
    tag, t = line.split()
    key, edge = tag.rsplit("_", 1)
    spans.setdefault(key, {})[edge] = int(t)
rows = [l.split() for l in open(raw) if not l.startswith("#") and l.strip()]
if not rows:
    print("no dmon samples"); raise SystemExit
# columns: time gpu pwr gtemp mtemp sm mem enc dec jpg ofa rxpci txpci
idx = {"pwr": 2, "sm": 5, "mem": 6, "rx": 11, "tx": 12}
t0 = min(v["start"] for v in spans.values())
total = max(v["end"] for v in spans.values()) - t0
print(f"\n{'phase':<10}{'n':>5}{'SM %':>16}{'MEM ctrl %':>16}{'power W':>14}"
      f"{'PCIe rx MB/s':>15}{'PCIe tx MB/s':>15}")
for key, v in sorted(spans.items(), key=lambda kv: kv[1]["start"]):
    lo = int(len(rows) * (v["start"] - t0) / total)
    hi = int(len(rows) * (v["end"] - t0) / total)
    seg = rows[lo:hi]
    if not seg:
        continue
    def col(name):
        vals = []
        for r in seg:
            try: vals.append(float(r[idx[name]]))
            except (ValueError, IndexError): pass
        return (st.mean(vals), max(vals)) if vals else (0.0, 0.0)
    out = [col(c) for c in ("sm", "mem", "pwr", "rx", "tx")]
    print(f"{key:<10}{len(seg):>5}" + "".join(
        f"{f'{m:.0f} (max {x:.0f})':>16}" if c < 2 else
        (f"{f'{m:.0f} (max {x:.0f})':>14}" if c == 2 else f"{f'{m:.0f} (max {x:.0f})':>15}")
        for c, (m, x) in enumerate(out)))
PY
cp $RAW results/final/telemetry_dmon.txt 2>/dev/null
echo "### P0_TELEMETRY_DONE"
