#!/bin/bash
# Copy exactly the figures the manuscript includes, discovered from the tex.
# A hand-written list went stale the moment a figure was added: fig_build and
# fig_cpu were added after it and the paper kept using hour-old PDFs.
set -u
cd "$(dirname "$0")" 2>/dev/null || true
R=/Users/apple/github/JHQ_GPU/paper_adc2026
n=0
for f in $(grep -roh 'figs/[a-z_0-9]*\.pdf' $R/tex/ADC/*.tex | sed 's|figs/||;s|\.pdf||' | sort -u); do
  if [ -f "$R/figures/out/$f.pdf" ]; then
    cp "$R/figures/out/$f.pdf" "$R/tex/figs/$f.pdf"; n=$((n+1))
  else
    echo "  MISSING generator output: $f"
  fi
done
echo "  synced $n figures the manuscript includes"
