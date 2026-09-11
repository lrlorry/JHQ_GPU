#!/bin/bash
# IVF-RaBitQ as its own paper runs it: quantised scan, then exact re-ranking.
#
# cuvs::neighbors::ivf_rabitq::search_params has n_probes and mode and nothing
# else -- the library ships RaBitQ's quantiser without the re-ranking stage its
# published results rely on.  Measured consequence: at bits_per_dim=1, the
# configuration RaBitQ is known for, recall tops out at 0.78 on vogue-768 and
# 0.79 on arxiv-768 at every probe depth (rq_bits_partial.log).  Reaching the
# recalls this paper compares at therefore forces bits_per_dim=8 and a scan
# eight times more expensive, which is why every RaBitQ number here sits 2.3x
# to 4.9x behind cuVS CAGRA-int8 at R=0.90-0.95 -- the reverse of RaBitQ's own
# published ordering, and the sign that the baseline is being run without half
# of itself.
#
# cuvs::neighbors::refine is shipped separately and refine.hpp's example is
# this exact pipeline.  So: bits_per_dim=1, search alpha*k candidates, refine
# to k against the fp32 dataset, and sweep alpha the way the paper sweeps
# JHQ's.  Both systems then do the same thing -- scan cheaply, spend exact
# work on a survivor set -- which is the comparison worth reporting whichever
# way it lands.
#
# bge-m3 and stella-trec24 are absent: RaBitQ's build path already OOMs there,
# and refine would need 38.8 GiB of fp32 dataset resident on a 32 GB card.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_rabitq_refine.sh /root/rqref.sh \
#     && setsid nohup bash /root/rqref.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_rqrefine; flock -n 9 || { echo "already running"; exit 0; }
L=/root/rq_refine.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "start; head=$(git log --oneline -1)"

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
D=/root/autodl-tmp; V=/root/data
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export JHQ_RQ_IDX=$D/rq_refine.idx

INC="-I$SP/libraft/include/rapids"; for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""; for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC examples/bench_ivf_rabitq_refine.cu -o build/bench_rq_refine \
     $LD -lcuvs -lrmm -lcudart -lcublas >>"$L" 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== RQREF""_DONE build failed ==="; tail -30 "$L" | sed 's/^/  | /' >> "$L"; exit 1; }

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths nlist bits alpha nprobe mode
    local tag=$1 paths=$2 nl=$3 bt=$4 al=$5 np=$6 md=$7 out r q
    out=$(env JHQ_RQ_TRAIN_PER_LIST=256 JHQ_RQ_ALPHA="$al" timeout 9000 \
          build/bench_rq_refine $paths "$nl" "$bt" "$np" 10 "$md" 3 2>&1)
    r=$(printf '%s\n' "$out" | awk '/^Recall@10/{print $3;exit}')
    q=$(printf '%s\n' "$out" | awk '/^QPS/{print $3;exit}')
    if [ -z "${r:-}" ]; then
        echo "  RQR $tag nlist=$nl bits=$bt a=$al np=$np MISSING" >> "$L"
        printf '%s\n' "$out" | tail -4 | sed 's/^/      | /' >> "$L"
    else
        printf "  RQR %-14s nlist=%-6s bits=%-3s a=%-4s mode=%-2s np=%-5s recall=%-8s qps=%s\n" \
            "$tag" "$nl" "$bt" "$al" "$md" "$np" "$r" "$q" >> "$L"
    fi
}

NP="8 32 128 256 512 1024"
# alpha is the refine multiplier, the same quantity JHQ calibrates.  bits=1 is
# RaBitQ's own configuration; 2 is carried as the fallback if 1 plus refine
# still cannot reach the high-recall end.
for bt in 1 2; do
  for al in 4 8 16; do
    say "########## bits=$bt alpha=$al ##########"
    for np in $NP; do one vogue-768    "$VG"  4096 "$bt" "$al" "$np" 1; done
    for np in $NP; do one arxiv-768    "$AX"  8192 "$bt" "$al" "$np" 1; done
    for np in $NP; do one openai3-1536 "$O15" 4096 "$bt" "$al" "$np" 0; done
    for np in $NP; do one openai3-3072 "$O30" 4096 "$bt" "$al" "$np" 2; done
  done
done
rm -f "$JHQ_RQ_IDX"
say "=== RQREF""_DONE ==="
