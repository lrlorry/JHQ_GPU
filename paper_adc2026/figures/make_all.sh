#!/bin/bash
# Redraw every figure from the frozen logs in ../data/.
set -e
cd "$(dirname "$0")"
for f in fig_frontier.py fig_alpha.py fig_batch.py fig_ablation.py fig_hierarchy.py; do
  echo "== $f"
  python3 "$f"
done
echo "PDF and PNG in out/"
