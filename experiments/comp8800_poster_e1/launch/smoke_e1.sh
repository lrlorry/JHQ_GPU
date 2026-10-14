#!/bin/bash
# E1 smoke: every method once on vogue with a tiny grid; results go to SMOKE/, not RAW_RESULTS/.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
E=experiments/comp8800_poster_e1; O=$E/SMOKE; mkdir -p $O
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')"
exec 8>/root/.gpu_lock; flock 8
echo "lock $(date -u +%T) apps=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)]"
E1_LOCK_HELD=1 OUT=$PWD/$O NPROBES=128 JHQ_ALPHA_MULT=1 bash $E/run_e1.sh jhq vogue-768 2>&1 | tail -2
V=/root/data
C="--dataset vogue-768 --warmup 2 --rounds 2 --calls 5"
python3 $E/bench_cuvs.py $C --method cagra --out $O/cagra.jsonl --graph-degrees 32 --itopk 64 --search-width 1 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method cagra-int8 --out $O/cagra8.jsonl --graph-degrees 32 --itopk 64 --search-width 1 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method ivfpq --out $O/ivfpq.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f32 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method ivfpq-refine --out $O/ivfpqr.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f16 --refine-ratios 2 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method bruteforce --out $O/bf.jsonl 2>&1 | tail -3
RQ_NPROBES=128 RQ_MODES=1 RQ_WARMUP=2 RQ_ROUNDS=2 RQ_CALLS=5 RQ_IDX=/root/autodl-tmp/rq_smoke.idx \
  build_e1/bench_rabitq_sweep $V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs 4096 8 $O/rq.jsonl 2>&1 | tail -3
echo "=== SMOKE_DONE $(date -u +%T)"
