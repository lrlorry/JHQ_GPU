#!/bin/bash
# JHQ v69's high-recall end on openai3-1536 (E1-C's grid, alpha <= 64, stopped at 0.975).
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
echo "=== E1F start $(date -u +%FT%TZ)"
NPROBES="256 512 1024" JHQ_ALPHA_MULT="16,32,64" bash $R jhq openai3-1536
NPROBES="1024" JHQ_ALPHA_MULT="1,2,4,8" bash $R jhq openai3-1536
echo "=== E1F_ALL_DONE $(date -u +%FT%TZ)"
