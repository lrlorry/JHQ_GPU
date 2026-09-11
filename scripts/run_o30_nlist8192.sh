#!/bin/bash
# openai3-3072's frontier at the nlist its own sweep chose.
#
# results/front6/nlist6.log swept four nlist values per dataset at ~39 training
# points a centroid.  Five of the six datasets run on the frontier at the value
# that sweep picked.  openai3-3072 does not: the sweep picks 8192 at every
# recall target it resolves -- 48,690 against 44,821 QPS at R=0.90, 30,879
# against 29,524 at 0.95, 20,009 against 18,742 at 0.97 -- and the frontier
# runs at 4096.
#
# How it happened: run_front6.sh's second block quadrupled every dataset's
# published nlist, taking both OpenAI sets from 1024 to 4096.  openai3-1536 was
# later moved to the sweep's 8192 and openai3-3072 was not.  So the paper
# measures its highest-dimensional dataset at a partition its own experiment
# rejected, losing 4.6% to 8.6% of throughput.
#
# It also confounds the argument Section 6.2 now makes about dimension.  At
# nlist=4096 openai3-3072 holds 244 vectors a list against openai3-1536's 122,
# so the same probe depth scans twice the candidates on the higher-dimensional
# set, and the smaller margin at 3072 cannot be attributed to d.  At 8192 both
# hold 122 and the comparison is about dimension alone.
#
# Protocol is /root/_pa.sh's, unchanged except nlist: same binaries, same
# BLOCK=1024 for this dataset, same nprobe ladder, both arms.  n_train follows
# the 39-points-a-centroid rule every other dataset uses, so 39*8192=319,488
# rather than 4096's 159,744.
#
# The 4096 rows are not discarded.  They stay in paper_fronts.log and this
# writes its own file, so the pair is a measurement of what the partition is
# worth rather than a silent replacement.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_o30_nlist8192.sh /root/o30.sh \
#     && setsid nohup bash /root/o30.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_o30; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
CACHE=/root/autodl-tmp/o30_cache; rm -rf $CACHE; mkdir -p $CACHE
D=/root/autodl-tmp
L=/root/o30_nlist.log; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build -j 16 --target demo_jhq_v57 demo_jhq_alpha_v57 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L | head -20 >>$L; say "=== O30""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"

one(){  # tag paths M nlist n_train BLOCK nprobe mode
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 blk=$6 np=$7 mode=$8
  local C=$CACHE/${tag}_${nl}; mkdir -p $C
  local bin=demo_jhq_v57 alpha=100.0 env_extra=""
  if [ "$mode" = rule ]; then bin=demo_jhq_alpha_v57; env_extra="JHQ_AS_SAMPLE=32"; fi
  env $E JHQ_INDEX_CACHE=$C JHQ_BLOCK=$blk JHQ_TILE_M_RT=$M JHQ_N_TRAIN=$nt $env_extra \
      timeout 9000 build/$bin $paths $M 8 8 $alpha 10 $nl $np 8 1024 "" 3 \
      >/tmp/o30.$$ 2>/tmp/o30e.$$
  local rc=$?
  if [ "$mode" = rule ]; then
    printf "  RULE  %-14s M=%-4s nlist=%-6s np=%-5s %s rc=%s\n" "$tag" "$M" "$nl" "$np" \
      "$(grep -o 'AF_RESULT.*' /tmp/o30.$$)" "$rc" >> $L
  else
    printf "  FIX   %-14s M=%-4s nlist=%-6s np=%-5s a=100 recall=%-8s qps=%-9s train=%-9s add=%-10s vram=%-9s rc=%s\n" \
      "$tag" "$M" "$nl" "$np" \
      "$(awk '/^Recall@10/{print $3;exit}' /tmp/o30.$$)" \
      "$(awk '/^QPS/{print $3;exit}' /tmp/o30.$$)" \
      "$(awk '/^  train:/{print $2;exit}' /tmp/o30.$$)" \
      "$(awk '/^  add:/{print $2;exit}' /tmp/o30.$$)" \
      "$(awk '/^VRAM used/{print $4;exit}' /tmp/o30.$$)" "$rc" >> $L
  fi
  [ $rc -ne 0 ] && head -3 /tmp/o30e.$$ | sed 's/^/      ! /' >> $L
  rm -f /tmp/o30.$$ /tmp/o30e.$$
}

O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
for np in 8 32 128 256 512 1024; do
  for m in fix rule; do
    one openai3-3072 "$O30" 384 8192 319488 1024 $np $m
  done
done
say "=== O30""_DONE ==="
