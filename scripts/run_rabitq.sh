#!/bin/bash
# IVF-RaBitQ against JHQ, one machine, one protocol, one dataset.
#
# OpenAI-3072-1M is the only dataset the IVF-RaBitQ paper and this project
# share: they list it at 1,000,000 x 3072 and we hold 999,000 x 3072 of the
# same source. Everything else about their setup differs -- L40S against
# RTX 5090, batch 10,000 against 1,000, GPU-resident timing against ours --
# so their published multiples cannot be carried over. This runs both here.
#
# bits_per_dim counts the 1-bit code, so the matched budgets are
#     JHQ Br=4  ->  5      JHQ Br=8  ->  9
# and the four search modes (LUT16/LUT32/QUANT4/QUANT8) are the two inner
# product families the paper ablates.
set -u
exec 9>/root/.lock_rabitq; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
D=${DATA_ROOT:-/root/autodl-tmp}
L=${LOG:-/root/rabitq.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null
if ! git fetch https://github.com/lrlorry/JHQ_GPU.git fix/recall-eval-v15 >>$L 2>&1; then
  say "=== fetch failed; refusing to run against a stale tree ==="; exit 1
fi
git reset --hard FETCH_HEAD >>$L 2>&1
say "HEAD $(git log --oneline -1)"

SP=/root/miniconda3/lib/python3.12/site-packages
# Four things this needs that are not obvious:
#   - every wheel that ships headers, including rapids_logger, which raft's
#     logger_macros.hpp includes and which lives in its own package;
#   - the CCCL the RAPIDS wheels vendor under <pkg>/include/rapids (3.4.3),
#     ahead of the toolkit's, since RMM refuses anything below 3.3 and CUDA
#     13.0 here ships 3.0.1;
#   - g++-12 as the host compiler, because GCC 11's <atomic> rejects a type
#     CCCL passes to std::atomic;
#   - librmm.so as well as libcuvs.so; the RMM symbols raft pulls in are not
#     in libcuvs.
INC="-I$SP/libraft/include/rapids"
for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""
for p in $SP/*/lib64 $SP/*/lib; do
  [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"
done
say "compiling"
export DEBIAN_FRONTEND=noninteractive
command -v g++-12 >/dev/null || apt-get install -y g++-12 >>$L 2>&1
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC \
     examples/bench_ivf_rabitq.cu -o build/bench_ivf_rabitq \
     $LD -lcuvs -lrmm -lcudart -lcublas >>$L 2>&1
rc=$?; say "compile_rc=$rc"
[ $rc -ne 0 ] && { say "=== RABITQ""_DONE compile failed ==="; exit 1; }

# librapids_logger.so lives in the rapids_logger wheel's own lib dir, which
# the rpath list above covers only if that dir is named lib64 or lib. Set the
# search path explicitly rather than guessing at wheel layouts.
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH"
say "LD_LIBRARY_PATH set"
ldd build/bench_ivf_rabitq 2>&1 | grep -i "not found" | sed 's/^/      ! /' >> $L

exec 8>/root/.gpu_lock; flock 8
say "got the GPU"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"
run(){ # nlist bits nprobe mode
  timeout 6000 build/bench_ivf_rabitq $O30 $1 $2 $3 10 $4 3 >/tmp/rq.$$ 2>/tmp/rqe.$$
  local rc=$?
  printf "  ivf_rabitq  nlist=%-7s bits=%-3s np=%-5s mode=%-3s recall=%-8s qps=%-9s train=%-10s vram=%-9s rc=%s\n" \
    "$1" "$2" "$3" "$4" \
    "$(awk '/^Recall@10/{print $3;exit}' /tmp/rq.$$)" \
    "$(awk '/^QPS/{print $3;exit}' /tmp/rq.$$)" \
    "$(awk '/^  train:/{print $2;exit}' /tmp/rq.$$)" \
    "$(awk '/^VRAM used/{print $4;exit}' /tmp/rq.$$)" "$rc" >> $L
  [ $rc -ne 0 ] && head -4 /tmp/rqe.$$|sed 's/^/      ! /' >>$L
  rm -f /tmp/rq.$$ /tmp/rqe.$$
}
say "########## bits=9, matching JHQ Br=8 ##########"
for np in 8 32 128 512; do for m in 2 1; do run 4096 9 $np $m; done; done
say "########## bits=5, matching JHQ Br=4 ##########"
for np in 8 32 128 512; do for m in 2 1; do run 4096 5 $np $m; done; done
say "########## their own default, bits=3 ##########"
for np in 32 128 512; do run 4096 3 $np 2; done
say "=== RABITQ""_DONE ==="
