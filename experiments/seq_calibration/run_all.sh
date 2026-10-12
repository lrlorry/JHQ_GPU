#!/bin/bash
# Reproduce every number in SEQUENTIAL_CALIBRATION_REPORT.md. CPU only, ~1 min.
# Needs python3 + numpy. Reads paper_adc2026/data/ and writes only to ./out/.
#
#   bash experiments/seq_calibration/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
python3 reproduce_baseline.py | tee out/baseline_repro.txt   # step 2: R1 exact + R2 protocol
python3 replay.py                                            # steps 3-6: all policies, 2000 draws x 2 splits
python3 summarize.py                                         # tables -> out/summary.md
