#!/bin/bash
# After E1-A: (1) JHQ v69 at more nprobe on vogue and arxiv, to fill its frontier and find its
# high-recall end; (2) E1-B on openai3-3072, every method. Holds the GPU lock throughout.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 RQ_POOL=2048 CUVS_POOL=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
echo "=== E1B start $(date -u +%FT%TZ) head=$(git log --oneline -1)"
NPROBES="96 192 384 512" bash $R jhq vogue-768 arxiv-768
NPROBES="1024" JHQ_ALPHA_MULT="0.5,1" bash $R jhq arxiv-768
# openai3-3072: recall moves with alpha here (paper alpha 4 at nprobe 128)
NPROBES="64 128 256 512" JHQ_ALPHA_MULT="0.5,1,2,4,8" bash $R jhq openai3-3072
RQ_NLISTS="8192 4096" RQ_MODES="1,2,3" bash $R rabitq openai3-3072
for st in cagra cagra-int8 ivfpq bruteforce; do bash $R $st openai3-3072; done
echo "=== E1B_ALL_DONE $(date -u +%FT%TZ) throttled=$(grep nr_throttled /sys/fs/cgroup/cpu.stat | head -1)"
