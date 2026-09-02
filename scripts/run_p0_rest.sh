#!/bin/bash
# One process runs every remaining stage in order. The marker-and-waiter chain
# it replaces accumulated duplicate waiters across a flaky link, and duplicates
# fire together onto one GPU, which corrupts exactly the timings being measured.
exec 9>/root/.lock_rest
flock -n 9 || { echo "another p0_rest already holds the lock"; exit 0; }
exec >/root/p0_rest.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
for t in $(seq 1 12); do source /etc/network_turbo 2>/dev/null; git pull -q origin fix/recall-eval-v15 && break; sleep 10; done
echo "HEAD: $(git log --oneline -1)"
export JHQ_INDEX_CACHE=/root/jhq_cache; mkdir -p $JHQ_INDEX_CACHE
R=results/final
A=/root/autodl-tmp

echo "############ 1. cuVS re-measured with the dataset left on the host"
echo "   The first pass uploaded the fp32 set before building, so its memory"
echo "   figures counted the raw vectors and its bge-m3 runs failed on the"
echo "   upload rather than on the index. Also adds pq_dim=768, which only"
echo "   Vogue had, so the matched-budget comparison exists everywhere."
for spec in "vogue-768 1024" "arxiv-768 2048" "openai3-1536 1024" "openai3-3072 1024"; do
  set -- $spec; DS=$1; NL=$2; SHORT=${DS%-768}; [ "$DS" = arxiv-768 ] && SHORT=arxiv-768
  B="python3 scripts/bench_all.py --dataset $DS --reps 3 --nlist $NL"
  echo "###### $DS"
  $B --method ivfpq --pq-dims 96,192,384,768 --nprobe 8,32,128,256 --out $R/p0_${SHORT}_ivfpq.csv
  $B --method cagra --graph-degrees 32,64 --itopk 32,64,128,256 --search-width 1,2 --out $R/p0_${SHORT}_cagra.csv
  $B --method cagra-int8 --graph-degrees 32,64 --itopk 32,64,128,256 --search-width 1,2 --out $R/p0_${SHORT}_cagra_int8.csv
done

echo "############ 2. saturation: every ceiling claim needs the sweep to have"
echo "   stopped climbing. int8 CAGRA was still gaining +0.0069 per itopk step"
echo "   on openai3-3072 at the largest value tried; IVF-PQ at pq_dim=768 was"
echo "   gaining +0.0074 per nprobe step on Vogue."
for spec in "vogue-768 1024" "arxiv-768 2048" "openai3-1536 1024" "openai3-3072 1024"; do
  set -- $spec; DS=$1; NL=$2; SHORT=${DS%-768}; [ "$DS" = arxiv-768 ] && SHORT=arxiv-768
  B="python3 scripts/bench_all.py --dataset $DS --reps 3 --nlist $NL"
  echo "###### $DS saturation"
  $B --method cagra-int8 --graph-degrees 64 --itopk 512,1024,2048 --search-width 1,2 --out $R/sat_${SHORT}_cagra_int8.csv
  $B --method ivfpq --pq-dims 768 --nprobe 512,1024 --out $R/sat_${SHORT}_ivfpq768.csv
  $B --method jhq --binary demo_jhq_v22_s2b1 --M 96 --Br 8 --nprobe 512,1024 --out $R/sat_${SHORT}_jhq_Br8.csv
done

echo "############ 3. CPU JHQ on the openai3 sets, at the smallest M the CPU allows"
echo "   The reference packs per-dimension indices into the code byte and needs"
echo "   Ds = d/M <= B, so M >= d/B: 192 at d=1536, 384 at d=3072. The GPU port"
echo "   has no such constraint and runs both at M=96."
D=/root/JHQ_repro/build/examples/demo_jhq_ivf
for spec in "openai3-1536 192" "openai3-3072 384"; do
  set -- $spec; DS=$1; M=$2
  for th in 1 all; do
    echo "###### $DS  threads=$th  M=$M nlist=1024"
    if [ "$th" = 1 ]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 7200 \
        $D $A/$DS/base.fvecs $A/$DS/query.fvecs $A/$DS/groundtruth.ivecs $M 8 4 1024 100.0 10 2>&1 | tail -25
    else
      unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
      timeout 7200 $D $A/$DS/base.fvecs $A/$DS/query.fvecs $A/$DS/groundtruth.ivecs $M 8 4 1024 100.0 10 2>&1 | tail -25
    fi
  done
  python3 scripts/bench_all.py --dataset $DS --reps 3 --nlist 1024 --method jhq \
    --binary demo_jhq_v22_s2b1 --M $M --Br 4 --nprobe 8,32,128,256 \
    --out $R/p0_${DS}_jhq_M${M}Br4.csv
done
echo "### P0_REST_DONE"
