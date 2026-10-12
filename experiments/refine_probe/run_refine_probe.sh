#!/bin/bash
# Two feasibility probes for refinement-side work, before either becomes a
# kernel: can the residual stage be made cheaper by ruling candidates out
# early (bounds), or by mapping it onto the GPU differently (lanes)?
#
# Both are capped by one number that has to be measured first: the share of
# search time refinement takes at the operating point the paper's own alpha
# rule selects. k_and_stages.log suggests 4-33% there (against 24-53% at
# alpha=100), so this run measures it directly, on the same index, by
# compiling refinement out.
#
#   lanes   demo_jhq_v60_exp, JHQ_REFINE_LANES = 0 (the v58 kernel), 1, 8, 16,
#           32: end-to-end QPS and recall, nothing else differing between
#           runs. L=0 runs first and last so drift shows.
#   share   demo_jhq_v58_jq (JHQ_NO_RESIDUAL; same sources, same cache).
#   bounds  JHQ_DUMP_REFINE (first 100 queries, written during the untimed
#           warm-up) -> analyze_refine_dump.py with the measured share.
#
# Index: run_budget_arms6.sh's settings and cache directories, so the trained
# state is the one the paper's budget table measured; v60 does not touch
# training. np=128, the adaptive alpha (vogue 64, openai3-3072 4, arxiv 200)
# and alpha=100. Decisions: summarize_probe.py, thresholds fixed in its header.
#
# On the box:
#   cd /root/JHQ_GPU && git pull
#   cmake --build build --target demo_jhq_v60_exp demo_jhq_v58_jq -j8
#   nohup bash experiments/refine_probe/run_refine_probe.sh > results/refine_probe.log 2>&1 & tail -f /root/JHQ_GPU/results/refine_probe.log
set -u
exec 9>/root/.lock_refine_probe; flock -n 9 || { echo "already running"; exit 0; }
echo "=== $(date -u +%FT%TZ) head=$(git -C /root/JHQ_GPU log --oneline -1) ==="
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
# Every number below belongs to this card. The paper's are all RTX 5090; on any
# other GPU the lanes result is a direction, not a figure to set beside them.
nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/GPU: /'
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v60_exp demo_jhq_v58_jq >/dev/null 2>&1
rc=$?; echo "build_rc=$rc"
[ $rc -ne 0 ] && { echo "=== REFINE_PROBE""_DONE build failed, nothing run ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; echo "got the GPU $(date -u +%T)"

D=/root/autodl-tmp; V=/root/data
OUT=/root/autodl-tmp/refine_probe; mkdir -p "$OUT"
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

# run <tag> <binary> <paths> <M> <nlist> <n_train> <BLOCK> <alpha> [VAR=value ...]
# <tag> starts with the dataset name, which is also its arms6 cache directory.
run() {
  local tag=$1 bin=$2 paths=$3 M=$4 nl=$5 nt=$6 blk=$7 a=$8; shift 8
  local pre="$OUT/$tag"
  env JHQ_INDEX_CACHE="$D/arms6_cache/${tag%%_*}" JHQ_BLOCK=$blk JHQ_TILE_M_RT=$M \
      JHQ_N_TRAIN=$nt JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 \
      JHQ_RES_TRAIN_N=100000 "$@" \
      timeout 3600 build/$bin $paths $M 8 8 $a 10 $nl 128 8 1024 "$pre" 3 > "$pre.txt" 2>&1
  local rc=$?
  echo "PROBE $tag rc=$rc $(grep -E '^Recall@|^QPS' "$pre.txt" | tr -s ' ' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$pre.txt"
}

# probe <ds> <paths> <M> <nlist> <n_train> <BLOCK> <adaptive alpha>
probe() {
  local ds=$1 paths=$2 M=$3 nl=$4 nt=$5 blk=$6 aa=$7
  echo "### $ds np=128 adaptive alpha=$aa $(date -u +%T)"
  [ -d "$D/arms6_cache/$ds" ] || echo "WARNING: no arms6 cache for $ds -- this index is retrained, not the paper's"
  run ${ds}_a${aa}_L0  demo_jhq_v60_exp "$paths" $M $nl $nt $blk $aa JHQ_DUMP_REFINE="$OUT/${ds}_a${aa}.refine"
  for L in 1 8 16 32; do
    run ${ds}_a${aa}_L$L demo_jhq_v60_exp "$paths" $M $nl $nt $blk $aa JHQ_REFINE_LANES=$L
  done
  run ${ds}_a${aa}_L0b demo_jhq_v60_exp "$paths" $M $nl $nt $blk $aa
  run ${ds}_a${aa}_jq  demo_jhq_v58_jq  "$paths" $M $nl $nt $blk $aa
  run ${ds}_a100_L0    demo_jhq_v60_exp "$paths" $M $nl $nt $blk 100
  run ${ds}_a100_L32   demo_jhq_v60_exp "$paths" $M $nl $nt $blk 100 JHQ_REFINE_LANES=32
  run ${ds}_a100_jq    demo_jhq_v58_jq  "$paths" $M $nl $nt $blk 100
}

# the frontier's settings per dataset, as run_budget_arms6.sh has them
probe vogue-768    "$VG"  96  4096 159744 512  64
probe openai3-3072 "$O30" 384 8192 319488 1024 4
probe arxiv-768    "$AX"  96  8192 319488 512  200

# refinement share at the adaptive alpha, from the clean L=0 run and jq
share() {
  local q0=$(grep -E '^QPS' "$OUT/$1_a$2_L0b.txt" | awk '{print $3}')
  local qj=$(grep -E '^QPS' "$OUT/$1_a$2_jq.txt"  | awk '{print $3}')
  awk -v a="$q0" -v b="$qj" 'BEGIN { if (a > 0 && b > 0) printf "%.4f", 1 - a / b }'
}
for spec in "vogue-768 64" "openai3-3072 4" "arxiv-768 200"; do
  set -- $spec
  s=$(share $1 $2)
  echo "### bounds $1 a=$2 refine share=${s:-NA}"
  python3 experiments/refine_probe/analyze_refine_dump.py "$OUT/$1_a$2.refine" \
      ${s:+--share $s} --json "$OUT/$1_a$2.bounds.json"
done
python3 experiments/refine_probe/summarize_probe.py "$OUT"
echo "=== REFINE_PROBE""_DONE $(date -u +%FT%TZ) ==="
