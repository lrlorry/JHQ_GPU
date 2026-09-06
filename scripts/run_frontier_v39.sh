#!/bin/bash
# The JHQ side of the frontier, re-measured on the frozen path.
#
# The published fronts (results/pre_freeze_v22_s2b1/) came from
# demo_jhq_v22_s2b1 with the prefix cascade at 1/2 and a residual codebook
# trained for 25 Lloyd iterations on a sample. Since then: the fidelity freeze,
# 2000 iterations (worth 6-8e-3 of recall), equation 5 over all of Y by default,
# and the refine kernel's carveout (worth 2%). The cuVS baselines do not move
# and are not re-run.
#
# Two of the old grid's points cannot come back. openai3-1536 at M=96 gives
# Ds=16 and openai3-3072 at M=96 gives Ds=32, and equation 4 needs Ds to divide
# B=8. The pre-freeze binary accepted them; the frozen one rejects them before
# allocating. Only the admissible M are swept here.
set -u
exec 9>/root/.lock_front; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}; mkdir -p $JHQ_INDEX_CACHE
R=results/frontier_v39; mkdir -p $R
L=${LOG:-/root/frontier.log}; : > $L
BIN=demo_jhq_v39_exp
echo "=== $(date -u +%FT%TZ)  $(git log --oneline -1) ===" >> $L

# The old runs used JHQ_BLOCK=1024; kept so the only differences are the ones
# named above.
run(){ # ds nlist M Br nprobes tag
  echo "###### $1  M=$3 Br=$4 nlist=$2  nprobe=$5" >> $L
  python3 scripts/bench_all.py --dataset $1 --reps 3 --nlist $2 \
      --method jhq --binary $BIN --M $3 --Br $4 --nprobe $5 --block 1024 \
      --out $R/f_$1_M$3Br$4.csv >> $L 2>&1
  echo "  rc=$?" >> $L; }

run vogue-768     1024  96  4 1,4,8,16,32,64,128,256,512
run vogue-768     1024  96  8 32,64,128,256,512,1024
run vogue-768     1024  192 4 32,64,128,256
run vogue-768     1024  192 8 32,64,128,256,512,1024
run arxiv-768     2048  96  4 1,8,32,64,128,256
run arxiv-768     2048  96  8 32,128,256,512,1024
run bge-m3        8192  128 4 8,32,128,256
run bge-m3        8192  128 8 32,128,256,512,1024
run stella-trec24 16384 128 4 8,32,128,256
run stella-trec24 16384 128 8 32,64,128,256,512,1024
run openai3-1536  1024  192 4 8,32,128,256
run openai3-3072  1024  384 4 8,32,128,256
echo "=== FRONT""_DONE $(date -u +%FT%TZ) ===" >> $L
