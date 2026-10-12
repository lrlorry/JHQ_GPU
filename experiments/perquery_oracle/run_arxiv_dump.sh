#!/bin/bash
# Per-query ids for arxiv-768 at every alpha, on the index the paper's budget
# table measured -- the one input the per-query oracle gap is missing.
#
# ORACLE_GAP_NOTES.md: on the configurations with per-query dumps the gap is
# x1.00-1.33, and 8 of the paper's 12 configurations cannot gain more than 7%
# from any per-query scheme. arxiv is the exception (ceiling x3.81 at np=128,
# x2.01 at np=512), and whether its queries differ in difficulty or all need a
# large budget cannot be read off aggregate recall. This run answers that and
# nothing else: no kernel change, no new version directory.
#
# Same library and settings as run_budget_arms6.sh's arxiv row (jhq_v57_exp,
# M=96, nlist=8192, n_train=319488, BLOCK=512, JHQ_RES_TRAIN_N=100000,
# kmeans_iters=3) and the same JHQ_INDEX_CACHE directory. The cache key covers
# the build parameters and the training sample, not alpha or nprobe, and the
# two demos set identical parameters, so this loads the trained state arms6
# measured instead of training a new one. oracle_gap.py checks that rather
# than assuming it: held-out recall must equal budget_arms.log's FIXED arms at
# every alpha. A mismatch means a retrained index (cache gone), results that
# depend on batch composition (arms6 searched the 500 odd queries, this
# searches all 1000 at once), or -- on any GPU other than the 5090 -- the
# other card's arithmetic: cuBLAS and fast-math differ in the last bits, which
# can move a few coarse assignments even with the cache present. None of
# these matters to the question this run answers; all of them matter to
# lining its numbers up against the paper's.
#
# alpha=400 is the largest power-of-two-ish budget this build takes at
# BLOCK=512: the final top-k kernel asks (2*ck + 3*BLOCK)*4 B of dynamic shared
# memory without opting in, so ck <= 5376; the exact scan's buffer is
# (pow2 >= ck + BLOCK)*8 B against the 101,376 B opt-in, so ck <= 7680.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake --build build --target demo_jhq_v57_exp -j8
#   nohup bash experiments/perquery_oracle/run_arxiv_dump.sh > results/arxiv_perquery.log 2>&1 & tail -f /root/JHQ_GPU/results/arxiv_perquery.log
set -u
exec 9>/root/.lock_arxiv_pq; flock -n 9 || { echo "already running"; exit 0; }
echo "=== $(date -u +%FT%TZ) head=$(git -C /root/JHQ_GPU log --oneline -1) ==="
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/GPU: /'
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v57_exp >/dev/null 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { echo "=== ARXIV_PQ""_DONE build failed, nothing run ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp/arxiv-abstracts-768
OUT=/root/autodl-tmp/adump_arxiv
CACHE=/root/autodl-tmp/arms6_cache/arxiv-768
mkdir -p "$OUT"
[ -d "$CACHE" ] && echo "cache dir present: $CACHE ($(ls "$CACHE" | wc -l) files)" \
                || echo "WARNING: $CACHE missing -- the index will be retrained and differ from the paper's"
cp "$D/groundtruth.ivecs" "$OUT/gt_arxiv-768.ivecs"

for np in 128 512; do
  for a in 2 4 8 16 32 64 100 200 400; do
    pre="$OUT/arxiv-768_np${np}_a${a}"
    env JHQ_INDEX_CACHE="$CACHE" JHQ_BLOCK=512 JHQ_TILE_M_RT=96 JHQ_N_TRAIN=319488 \
        JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
        JHQ_RES_TRAIN_N=100000 \
        timeout 3600 build/demo_jhq_v57_exp "$D/base.fvecs" "$D/query.fvecs" "$D/groundtruth.ivecs" \
        96 8 8 $a 10 8192 $np 8 1024 "$pre" 3 > "$pre.txt" 2>&1
    rc=$?
    echo "np=$np a=$a rc=$rc $(grep -E '^Recall@|^QPS' "$pre.txt" | tr -s ' ' | tr '\n' ' ')"
    [ $rc -ne 0 ] && tail -3 "$pre.txt"
  done
done

python3 experiments/perquery_oracle/oracle_gap.py --pq-dir "$OUT" \
    --config arxiv-768:128 --config arxiv-768:512 \
    --arms-log paper_adc2026/data/budget_arms.log --tag arxiv
echo "py_rc=$?"
tar czf /root/autodl-tmp/adump_arxiv.tgz -C /root/autodl-tmp adump_arxiv && echo "dumps: /root/autodl-tmp/adump_arxiv.tgz"
echo "=== ARXIV_PQ""_DONE $(date -u +%FT%TZ) ==="
