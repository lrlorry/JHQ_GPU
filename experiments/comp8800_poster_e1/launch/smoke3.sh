#!/bin/bash
# E1 smoke 3: RaBitQ phase diagnostic with and without a workspace pool; every cuVS method on a
# tiny vogue grid, then CAGRA and IVF-PQ again behind an RMM pool. Results in SMOKE3/.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH OMP_NUM_THREADS=16
E=experiments/comp8800_poster_e1; O=$E/SMOKE3; mkdir -p $O
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')"
OUT=$PWD/$O bash $E/run_e1.sh build 2>&1 | tail -2
exec 8>/root/.gpu_lock; flock 8
thr(){ grep -E 'nr_throttled' /sys/fs/cgroup/cpu.stat | tr '\n' ' '; }
echo "lock $(date -u +%T) apps=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)] $(thr)"
V=/root/data; VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
for pool in 0 2048; do
  echo "--- rabitq pool=$pool"
  RQ_POOL=$pool RQ_DIAG=300 RQ_NPROBES=128 RQ_MODES=1 RQ_ROUNDS=5 RQ_CALLS=24 RQ_IDX=/root/autodl-tmp/rq_smoke.idx \
    build_e1/bench_rabitq_sweep $VG 4096 8 $O/rq_pool$pool.jsonl 2>&1 | grep -E "DIAG|search|total|bits=" 
done
echo "after rq $(thr)"
C="--dataset vogue-768 --warmup 3 --rounds 5 --calls 24"
for pool in 0 1; do
  export CUVS_POOL=$pool; echo "--- cuvs pool=$pool"
  python3 $E/bench_cuvs.py $C --method cagra --out $O/cagra_p$pool.jsonl --graph-degrees 32 --itopk 64,128 --search-width 1 2>&1 | grep -E "R=|FAIL|Error|Fault" | head -4
  python3 $E/bench_cuvs.py $C --method ivfpq --out $O/ivfpq_p$pool.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f32 2>&1 | grep -E "R=|FAIL|Error|Fault" | head -3
  [ $pool = 1 ] && continue
  python3 $E/bench_cuvs.py $C --method cagra-int8 --out $O/cagra8_p$pool.jsonl --graph-degrees 32 --itopk 64,128 --search-width 1 2>&1 | grep -E "R=|FAIL|Error|Fault" | head -4
  python3 $E/bench_cuvs.py $C --method ivfpq-refine --out $O/ivfpqr_p$pool.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f16 --refine-ratios 2 2>&1 | grep -E "R=|FAIL|Error|Fault" | head -3
  python3 $E/bench_cuvs.py $C --method bruteforce --out $O/bf_p$pool.jsonl 2>&1 | grep -E "R=|FAIL|Error|Fault" | head -3
done
echo "end $(thr)"
echo "=== SMOKE3_DONE $(date -u +%T)"
