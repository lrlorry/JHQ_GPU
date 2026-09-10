#!/bin/bash
# Redraw every figure from the frozen logs in ../data/.
set -e
cd "$(dirname "$0")"
for f in fig_pipeline.py fig_lut.py fig_layout.py fig_rule.py fig_frontier.py fig_alpha.py fig_calibration.py fig_batch.py fig_economics.py \
         fig_lutgroups.py fig_ablation.py fig_negatives.py fig_hierarchy.py fig_cost.py fig_build.py fig_memory.py; do
  echo "== $f"
  python3 "$f"
done
# Nothing ships until every number in the body traces to a source. Six errors
# of that kind reached a draft; this is the check that would have caught them.
echo "== audit_numbers.py"
python3 audit_numbers.py || { echo "AUDIT FAILED -- unsourced numbers above"; exit 1; }
echo "PDF and PNG in out/"
