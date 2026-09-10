#!/bin/bash
# Everything still outstanding, in one unattended pass.
#
#   1. remeasure_anchors.sh -- the two timing points the paper quotes from a
#      single measurement, five interleaved passes each, now with the RaBitQ
#      arm's library path set so it stops failing silently.
#   2. run_budget_arms.sh   -- rerun with the ENUM arm, which walks the whole
#      grid on the same sample bisection used, so the two can be compared draw
#      by draw rather than pool against sample.
#
# Sequential, not parallel: they share the card, and the CPU arm of (1) shares
# the host with everything. A previous comparison was thrown out for exactly
# that reason.
set -u
L=/root/finalise.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="
bash scripts/remeasure_anchors.sh
echo "--- anchors exit=$? $(date -u +%T)"
bash scripts/run_budget_arms.sh
echo "--- budget exit=$? $(date -u +%T)"
echo "=== FINALISE_DONE $(date -u +%FT%TZ) ==="
