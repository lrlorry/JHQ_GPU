#!/bin/bash
# E1 smoke 2: rebuild RaBitQ, its phase diagnostic, and every cuVS method once on vogue.
cd /root/autodl-tmp/jhq_pipeline_wt || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
E=experiments/comp8800_poster_e1; O=$E/SMOKE2; mkdir -p $O
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')"
OUT=$PWD/$O bash $E/run_e1.sh build 2>&1 | tail -3
exec 8>/root/.gpu_lock; flock 8
echo "lock $(date -u +%T) apps=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)] cpu.max=$(cat /sys/fs/cgroup/cpu.max) throttled=$(grep -E 'nr_throttled|throttled_usec' /sys/fs/cgroup/cpu.stat | tr '\n' ' ')"
V=/root/data
RQ_DIAG=200 RQ_NPROBES=128 RQ_MODES=1 RQ_ROUNDS=3 RQ_CALLS=24 RQ_IDX=/root/autodl-tmp/rq_smoke.idx \
  build_e1/bench_rabitq_sweep $V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs 4096 8 $O/rq.jsonl 2>&1 | tail -9
echo "after rq: throttled=$(grep -E 'nr_throttled|throttled_usec' /sys/fs/cgroup/cpu.stat | tr '\n' ' ')"
C="--dataset vogue-768 --warmup 3 --rounds 3 --calls 24"
for m in cagra cagra-int8; do
  python3 $E/bench_cuvs.py $C --method $m --out $O/$m.jsonl --graph-degrees 32 --itopk 64,128 --search-width 1 2>&1 | grep -v "^ *$" | tail -4
done
python3 $E/bench_cuvs.py $C --method ivfpq --out $O/ivfpq.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f32 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method ivfpq-refine --out $O/ivfpqr.jsonl --nlist 4096 --pq-dims 192 --nprobe 64 --lut f16 --refine-ratios 2 2>&1 | tail -3
python3 $E/bench_cuvs.py $C --method bruteforce --out $O/bf.jsonl 2>&1 | tail -3
echo "end: throttled=$(grep -E 'nr_throttled|throttled_usec' /sys/fs/cgroup/cpu.stat | tr '\n' ' ')"
echo "=== SMOKE2_DONE $(date -u +%T)"
