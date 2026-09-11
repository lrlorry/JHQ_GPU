#!/bin/bash
# The frontier with calibration and reporting on disjoint queries.
#
# The published frontier calibrates the budget rule on a 32-query sample
# strided out of the 1,000 queries it then scores -- Section 6.1 says so and
# Figure 2's caption calls the resulting gap optimistic.  It is a small leak,
# 32 of 1,000, but it is the paper's main end-to-end figure and the paper also
# says its held-out experiment does not validate this protocol, which leaves
# the claim with no clean support anywhere.
#
# demo_jhq_alpha_holdout calibrates on even-indexed queries and reports on odd
# -- the split Section 6.3 already uses -- so the two experiments share one
# protocol and the caveat goes.
#
# Both arms are re-run, not just the rule.  The reported set drops from 1,000
# queries to 500, and QPS at batch 1,000 is not comparable with QPS at batch
# 500, so a fixed-alpha curve carried over from the old run would be measured
# against a different batch size than the curve beside it.
#
# JHQ_AS_HOLDOUT=0 on the same binary reproduces the pooled numbers, so the
# log carries both and the difference is measured rather than asserted.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_frontier_holdout.sh /root/fh.sh \
#     && setsid nohup bash /root/fh.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_fh; flock -n 9 || { echo running; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
CACHE=/root/autodl-tmp/paper_cache; mkdir -p $CACHE
D=/root/autodl-tmp; V=/root/data
L=/root/fhold.log; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_alpha_holdout >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L | head -12 >>$L; say "=== FH""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"

# tag paths M nlist n_train BLOCK nprobe holdout
one(){
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 blk=$6 np=$7 ho=$8
  local C=$CACHE/${tag}; mkdir -p $C
  local out
  out=$(env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$blk JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt \
            JHQ_AS_SAMPLE=32 JHQ_AS_HOLDOUT=$ho \
            timeout 9000 build/demo_jhq_alpha_holdout \
            $paths $M 8 8 100.0 10 $nl $np 8 1024 "" 3 2>&1)
  local rc=$? res
  res=$(printf '%s\n' "$out" | grep -o 'AF_RESULT.*' | tail -1)
  if [ -z "${res:-}" ]; then
      printf "  HOLD%s %-14s M=%-4s nlist=%-6s np=%-5s MISSING rc=%s\n" \
        "$ho" "$tag" "$M" "$nl" "$np" "$rc" >> $L
      printf '%s\n' "$out" | tail -5 | sed 's/^/      | /' >> $L
  else
      printf "  HOLD%s %-14s M=%-4s nlist=%-6s np=%-5s %s rc=%s\n" \
        "$ho" "$tag" "$M" "$nl" "$np" "$res" "$rc" >> $L
  fi
}

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

# nlist, n_train and BLOCK follow the frontier; openai3-3072 is at the 8192 its
# own sweep chose, which the frontier now reports.
#
# Both arms of a cell run back to back.  Sweeping all of holdout=1 before any
# of holdout=0 -- which is how this was first written -- produces no comparable
# pair until run 37 of 72, and the deep probes make 72 runs a three-hour job,
# so an interrupted sweep yielded nothing.  Pairing here means every depth that
# finishes is complete.
pair(){ one "$@" 1; one "$@" 0; }
for np in 8 32 128 256 512 1024; do
  say "########## nprobe=$np ##########"
  pair vogue-768    "$VG"  96  4096  159744  512  $np
  pair arxiv-768    "$AX"  96  8192  319488  512  $np
  pair openai3-1536 "$O15" 192 8192  319488  512  $np
  pair openai3-3072 "$O30" 384 8192  319488  1024 $np
  pair bge-m3       "$BG"  128 32768 1277952 512  $np
  pair stella       "$ST"  128 32768 1277952 512  $np
  say "---------- nprobe=$np complete ----------"
done
say "=== FH""_DONE ==="
