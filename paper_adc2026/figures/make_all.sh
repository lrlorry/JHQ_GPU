#!/bin/bash
# Redraw every figure from the frozen logs in ../data/.
set -e
cd "$(dirname "$0")"
for f in fig_pipeline.py fig_lut.py fig_rule.py fig_frontier.py fig_alpha.py fig_calibration.py fig_batch.py \
         fig_ablation.py fig_negatives.py fig_hierarchy.py fig_cost.py; do
  echo "== $f"
  python3 "$f"
done
echo "PDF and PNG in out/"
