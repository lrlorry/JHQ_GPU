#!/bin/bash
# After E1-C: JHQ v69's low-recall end (nprobe 16 and 32) on the two 768-d sets, which E1-A's grid
# (nprobe >= 64) left unmeasured; then one cold build per dataset (empty trained-state cache), so
# JHQ's build time is measured rather than read from the cache. Holds the GPU lock.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
while ! grep -q E1C_ALL_DONE /root/autodl-tmp/e1c.log 2>/dev/null; do sleep 30; done
exec 8>/root/.gpu_lock; flock 8
export E1_LOCK_HELD=1 OMP_NUM_THREADS=16
R=experiments/comp8800_poster_e1/run_e1.sh
OUT=experiments/comp8800_poster_e1/RAW_RESULTS
echo "=== E1D start $(date -u +%FT%TZ)"
NPROBES="32" JHQ_ALPHA_MULT="0.5,1,2,4" bash $R jhq openai3-1536
D=/root/autodl-tmp; V=/root/data
cold(){  # ds base query gt M nlist ntrain block alpha
  local c=$D/e1_coldcache/$1; rm -rf "$c"; mkdir -p "$c"
  env JHQ_INDEX_CACHE="$c" JHQ_BLOCK=$8 JHQ_TILE_M_RT=$5 JHQ_N_TRAIN=$7 JHQ_GPU_CODEBOOK=1 \
      JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000 JHQ_BP_LANES=16 \
      JHQ_BP_BATCHES=1000 JHQ_BP_GROUPS=0,-1 JHQ_BP_ROUNDS=1 JHQ_BP_CHECK=0 \
      timeout 1800 build/demo_jhq_v69_batch $2 $3 $4 $5 8 8 $9 10 $6 128 8 1024 $OUT/jhqcold_$1 3 > $OUT/jhqcold_$1.txt 2>&1
  echo "$(date -u +%FT%TZ) jhq cold build $1 rc=$? $(grep -E '^train|^add' $OUT/jhqcold_$1.txt | tr '\n' ' ')" | tee -a $OUT/PROGRESS.log
  rm -rf "$c"
}
cold vogue-768 $V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs 96 4096 159744 512 64
cold arxiv-768 $D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs 96 8192 319488 512 100
cold openai3-3072 $D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs 384 8192 319488 1024 4
cold openai3-1536 $D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs 192 8192 319488 512 8
echo "=== E1D_ALL_DONE $(date -u +%FT%TZ)"
