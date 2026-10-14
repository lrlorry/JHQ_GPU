#!/bin/bash
# E1-A: vogue-768 and arxiv-768, every method, holding the GPU lock for the whole run.
# Pools for the baselines' search-time allocations (cuVS's own harness runs behind one): RaBitQ's
# RAFT workspace, and an RMM pool for the Python methods, both set only after each index is built.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 RQ_POOL=2048 CUVS_POOL=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
echo "=== E1A start $(date -u +%FT%TZ) head=$(git log --oneline -1) throttled=$(grep nr_throttled /sys/fs/cgroup/cpu.stat | head -1)"
bash $R env
for st in jhq rabitq cagra cagra-int8 ivfpq bruteforce; do
  bash $R $st vogue-768 arxiv-768
done
echo "=== E1A_ALL_DONE $(date -u +%FT%TZ) throttled=$(grep nr_throttled /sys/fs/cgroup/cpu.stat | head -1)"
