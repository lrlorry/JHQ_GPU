#!/bin/bash
# IVF-RaBitQ with the coarse-training budget as a controlled variable.
#
# Why this run exists.  Figure 5's caption says every index trains the coarse
# quantiser at 39 points a centroid.  That is true of JHQ and false of
# IVF-RaBitQ: cuVS defaults max_train_points_per_cluster to 256, so the
# archived runs trained on 932,328 points at vogue-768 against JHQ's 159,744 --
# essentially the whole dataset, and 5.8x to 6.6x JHQ's budget across the four
# datasets it builds on.  A build-time comparison across that gap measures the
# training budget, not the index.
#
# The fix is not to hold RaBitQ down to 39.  Coarse training cuts both ways:
# results/front6/README.md measures better-trained centroids as worth +5% to
# +96% of search throughput, because an even partition touches fewer vectors at
# a fixed nprobe.  Forcing 39 would buy our build number with its search
# number, which is the same error pointed the other way.
#
# So the budget becomes a variable: each dataset at 39 -- matching JHQ -- and at
# cuVS's own 256, build and search both.  Then the paper can say what the build
# gap is at equal training and what the library's default costs and buys,
# instead of quoting one number that silently contains both.
#
# openai3-1536 is also corrected here.  The archived RaBitQ runs used
# nlist=8192 on it while the JHQ frontier uses 4096, so that dataset was not a
# controlled comparison in either direction.  nlist now follows the frontier's
# own setting on every dataset.
#
# bge-m3 and stella-trec24 are absent because RaBitQ cannot build there at all:
# the k-means sampler materialises the dataset as float and asks for 41.3 GB
# and 72.8 GB on a 32 GiB card.
#
# Serial behind the queue already on the box: run_cpu_four.sh is measuring host
# throughput, run_batch_two_more.sh is waiting on it, and this waits on that.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_rabitq_train.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_rqtrain; flock -n 9 || { echo "already running"; exit 0; }
L=/root/rq_train.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "queued; head=$(git log --oneline -1)"

# Wait out the CPU baseline and then the batch sweep, in that order.
exec 7>/root/.lock_cpu_four;   flock 7; say "CPU baseline clear"
exec 6>/root/.lock_batch2more; flock 6; say "batch sweep clear"

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
D=/root/autodl-tmp; V=/root/data
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export JHQ_RQ_IDX=$D/rq_train.idx

INC="-I$SP/libraft/include/rapids"; for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""; for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC examples/bench_ivf_rabitq.cu -o build/bench_rq_train \
     $LD -lcuvs -lrmm -lcudart -lcublas >>"$L" 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== RQTRAIN""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

# mode selects cuVS's search kernel: 0=LUT16 1=LUT32 2=QUANT4 3=QUANT8.  The
# archived runs used 2, but paper_rabitq.log never recorded it -- the wrapper
# kept only its own parsed summary and swallowed the line the bench prints --
# so which kernel the published RaBitQ curves used cannot be read back from the
# archive.  It is recorded here, and swept once per dataset below, because a
# baseline compared at a kernel that is not its best is not a baseline.
one(){  # tag paths nlist train_per_list nprobe mode
    local tag=$1 paths=$2 nl=$3 tpl=$4 np=$5 md=${6:-2}
    local out
    out=$(env JHQ_RQ_TRAIN_PER_LIST="$tpl" timeout 9000 \
          build/bench_rq_train $paths "$nl" 8 "$np" 10 "$md" 3 2>&1)
    local r q b actual
    r=$(printf '%s\n' "$out" | awk '/^Recall@10/{print $3;exit}')
    q=$(printf '%s\n' "$out" | awk '/^QPS/{print $3;exit}')
    b=$(printf '%s\n' "$out" | grep -oE 'build_ms=[0-9.]+' | head -1 | cut -d= -f2)
    # The library reports what it actually used, which is the number that goes
    # in the paper -- 39 is a request, and min(n_rows, tpl*n_lists) is the cap.
    actual=$(printf '%s\n' "$out" | grep -oE 'train_per_list=[0-9]+' | head -1 | cut -d= -f2)
    if [ -z "${r:-}" ]; then
        echo "  RQ $tag nlist=$nl tpl=$tpl np=$np MISSING <-- produced nothing" >> "$L"
        printf '%s\n' "$out" | tail -4 | sed 's/^/      | /' >> "$L"
    else
        printf "  RQ %-14s nlist=%-6s tpl=%-4s used=%-6s mode=%-2s np=%-5s recall=%-8s qps=%-9s build_ms=%s\n" \
            "$tag" "$nl" "$tpl" "${actual:-?}" "$md" "$np" "$r" "$q" "${b:-?}" >> "$L"
    fi
}

# nlist follows run_front6.sh's second block, the settings the JHQ frontier
# reports, so both sides partition the same way.
NP="8 32 128 256 512 1024"

# Which kernel is RaBitQ's best here?  One probe depth per dataset, all four
# modes, at the library's own training budget.  Whatever wins is what the
# sweep below should have been run at all along.
say "########## search mode sweep, nprobe=128, tpl=256 ##########"
for md in 0 1 2 3; do
    one vogue-768    "$VG"  4096 256 128 "$md"
    one arxiv-768    "$AX"  8192 256 128 "$md"
    one openai3-1536 "$O15" 4096 256 128 "$md"
    one openai3-3072 "$O30" 4096 256 128 "$md"
done

for tpl in 39 256; do
    say "########## train_per_list=$tpl ##########"
    for np in $NP; do one vogue-768    "$VG"  4096 "$tpl" "$np"; done
    for np in $NP; do one arxiv-768    "$AX"  8192 "$tpl" "$np"; done
    for np in $NP; do one openai3-1536 "$O15" 4096 "$tpl" "$np"; done
    for np in $NP; do one openai3-3072 "$O30" 4096 "$tpl" "$np"; done
done
rm -f "$JHQ_RQ_IDX"
say "=== RQTRAIN""_DONE ==="
