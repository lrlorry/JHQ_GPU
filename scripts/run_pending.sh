#!/bin/bash
# Every pending experiment, in the order their answers change what else is worth
# doing. One process, one lock, one log per stage; each stage writes CSVs the
# harness stamps with the commit and parameters that produced them.
#
#   bash scripts/run_pending.sh          all of it, ~5 h
#   STAGES="1 2" bash scripts/run_pending.sh   just those
set -u
exec 9>/root/.lock_pending; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
export JHQ_INDEX_CACHE=${JHQ_INDEX_CACHE:-/root/jhq_cache}; mkdir -p $JHQ_INDEX_CACHE
D=${DATA_ROOT:-/root/autodl-tmp}; V=${VOGUE_DIR:-/root/data}
R=results/pending; mkdir -p $R
L=${LOG:-/root/pending.log}; : > $L
STAGES=${STAGES:-"1 2 3 4 5 6 7 8 9"}
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }

source /etc/network_turbo 2>/dev/null
git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1 && \
  git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>$L 2>&1
cmake --build build --target demo_jhq_v39_exp demo_jhq_v39_lut16 -j 16 >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== PENDING""_DONE build failed ==="; exit 1; }
exec 8>/root/.gpu_lock; flock 8

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
raw(){ # bin paths M B Br alpha nlist nprobe extra_env tag
  env $E JHQ_TILE_M_RT=$3 ${9} timeout 3000 build/$1 $2 $3 $4 $5 $6 10 $7 $8 8 1024 "" 5 \
      >/tmp/p.$$ 2>/dev/null
  printf "  %-34s M=%-4s B=%s Br=%s a=%-6s nlist=%-6s np=%-5s  recall=%-8s qps=%s\n" \
    "${10}" "$3" "$4" "$5" "$6" "$7" "$8" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/p.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/p.$$)" >> $L; rm -f /tmp/p.$$; }

has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ── 1. fp16 CAGRA. Decides whether the BGE-M3 half of the argument stands.
if has 1; then say "### 1 fp16 CAGRA"
  python3 scripts/bench_all.py --dataset bge-m3 --reps 3 --nlist 8192 --method cagra-fp16 \
    --graph-degrees 32 --itopk 64,128,256,512 --search-width 1,2 --out $R/fp16_bge-m3.csv >>$L 2>&1
  say "  rc=$?"
  python3 scripts/bench_all.py --dataset stella-trec24 --reps 3 --nlist 16384 --method cagra-fp16 \
    --graph-degrees 32 --itopk 64,128 --search-width 1 --out $R/fp16_stella.csv >>$L 2>&1
  say "  rc=$? (stella expected to OOM at 36 GiB -- that is the result)"
fi

# ── 2. JHQ_BLOCK. Decides whether the two existing measurement groups merge.
if has 2; then say "### 2 JHQ_BLOCK sweep"
  for b in 128 256 512 1024; do for r in 1 2; do
    raw demo_jhq_v39_exp "$VG" 96 8 8 100.0 1024 128 "JHQ_BLOCK=$b" "block/vogue B=$b r$r"
    raw demo_jhq_v39_exp "$ST" 128 8 8 100.0 16384 128 "JHQ_BLOCK=$b" "block/stella B=$b r$r"
  done; done
fi

# ── 3. lut16 at BLOCK=1024. The shared-memory verdict was taken at 256 only.
if has 3; then say "### 3 lut16 at BLOCK 256 and 1024"
  for b in 256 1024; do for r in 1 2; do
    raw demo_jhq_v39_exp   "$ST" 128 8 8 100.0 16384 128 "JHQ_BLOCK=$b" "lut32/stella B=$b r$r"
    raw demo_jhq_v39_lut16 "$ST" 128 8 8 100.0 16384 128 "JHQ_BLOCK=$b" "lut16/stella B=$b r$r"
  done; done
fi

# ── 4. alpha at the paper's range. ck drives several results here.
if has 4; then say "### 4 alpha sweep"
  for a in 2.0 4.0 8.0 100.0; do for np in 8 32 128; do
    raw demo_jhq_v39_exp "$VG" 96 8 8 $a 1024 $np "JHQ_BLOCK=1024" "alpha/vogue"
    raw demo_jhq_v39_exp "$ST" 128 8 8 $a 16384 $np "JHQ_BLOCK=1024" "alpha/stella"
  done; done
fi

# ── 5. M across the admissible set. Tests whether eq.4's price is an L=2 artefact.
if has 5; then say "### 5 M sweep (Ds | B)"
  for m in 96 192 384; do for np in 32 128; do
    raw demo_jhq_v39_exp "$VG" $m 8 8 100.0 1024 $np "JHQ_BLOCK=1024" "M/vogue"
  done; done
  for m in 128 256 512; do for np in 32 128; do
    raw demo_jhq_v39_exp "$ST" $m 8 8 100.0 16384 $np "JHQ_BLOCK=1024" "M/stella"
  done; done
fi

# ── 6. equation 4 against the Lloyd-refined product, same binary, one flag.
if has 6; then say "### 6 JHQ_PAPER_CODEBOOK A/B"
  for pc in 1 0; do for np in 8 32 128; do
    raw demo_jhq_v39_exp "$VG" 96 8 8 100.0 1024 $np "JHQ_BLOCK=1024 JHQ_PAPER_CODEBOOK=$pc" "eq4=$pc/vogue"
    raw demo_jhq_v39_exp "$ST" 128 8 8 100.0 16384 $np "JHQ_BLOCK=1024 JHQ_PAPER_CODEBOOK=$pc" "eq4=$pc/stella"
  done; done
fi

# ── 7. nlist. Aims at the 3-10x QPS gap: more, smaller lists scan fewer candidates.
if has 7; then say "### 7 nlist sweep"
  for nl in 8192 16384 65536; do for np in 32 128 256; do
    raw demo_jhq_v39_exp "$ST" 128 8 8 100.0 $nl $np "JHQ_BLOCK=1024" "nlist/stella"
  done; done
fi

# ── 8. add_batch and the assignment batch. add() is 95% of the build.
if has 8; then say "### 8 add_batch / JHQ_ASSIGN_BATCH"
  for ab in 16384 65536 262144; do
    raw demo_jhq_v39_exp "$ST" 128 8 8 100.0 16384 128 "JHQ_BLOCK=1024 JHQ_ADD_BATCH=$ab" "addb=$ab/stella"
  done
  for sb in 8192 32768 131072; do
    raw demo_jhq_v39_exp "$ST" 128 8 8 100.0 16384 128 "JHQ_BLOCK=1024 JHQ_ASSIGN_BATCH=$sb" "asgb=$sb/stella"
  done
fi

# ── 9. B=4, the paper's [4,8]. Ds | 4 so M >= d/4.
if has 9; then say "### 9 B=4"
  for np in 32 128; do
    raw demo_jhq_v39_exp "$VG" 192 4 8 100.0 1024 $np "JHQ_BLOCK=1024" "B4/vogue M=192"
    raw demo_jhq_v39_exp "$ST" 256 4 8 100.0 16384 $np "JHQ_BLOCK=1024" "B4/stella M=256"
  done
fi

say "=== PENDING""_DONE ==="
