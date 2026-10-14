#!/bin/bash
# After E1-E: the optional fourth dataset, openai3-1536, every method. Holds the GPU lock.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
while ! grep -q E1E_ALL_DONE /root/autodl-tmp/e1e.log 2>/dev/null; do sleep 30; done
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 RQ_POOL=2048 CUVS_POOL=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
echo "=== E1C start $(date -u +%FT%TZ) head=$(git log --oneline -1)"
NPROBES="64 128 256 512" JHQ_ALPHA_MULT="0.5,1,2,4,8" bash $R jhq openai3-1536
RQ_NLISTS="8192 4096" RQ_MODES="0,1,2,3" bash $R rabitq openai3-1536
for st in cagra cagra-int8 ivfpq bruteforce; do bash $R $st openai3-1536; done
echo "=== E1C_ALL_DONE $(date -u +%FT%TZ)"
