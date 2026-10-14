#!/bin/bash
# Right after E1-B: JHQ v69's high-recall end on openai3-3072, which E1-B's grid (alpha <= 32,
# nprobe <= 512) left at 0.972 while every baseline reaches 0.99. Holds the GPU lock.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
while ! grep -q E1B_ALL_DONE /root/autodl-tmp/e1b.log 2>/dev/null; do sleep 20; done
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
echo "=== E1E start $(date -u +%FT%TZ)"
NPROBES="128 256 512 1024" JHQ_ALPHA_MULT="16,32,64" bash $R jhq openai3-3072
NPROBES="1024" JHQ_ALPHA_MULT="2,4,8" bash $R jhq openai3-3072
# the low-recall end on the 768-d sets (E1-A's grid started at nprobe 64)
NPROBES="16 32" JHQ_ALPHA_MULT="0.25,0.5,1" bash $R jhq vogue-768 arxiv-768
NPROBES="32" JHQ_ALPHA_MULT="0.5,1,2,4" bash $R jhq openai3-3072
echo "=== E1E_ALL_DONE $(date -u +%FT%TZ)"
