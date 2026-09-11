#!/bin/bash
# IVF-RaBitQ cold build time, at JHQ's training budget and at cuVS's default.
#
# This number was measured three times already and thrown away three times.
# bench_ivf_rabitq.cu times the whole build -- k-means, add, and the
# serialize/deserialize round-trip cuVS's own test requires before search
# returns sensible ids -- and prints it as "  train: NNNN ms".  Every wrapper
# written after run_rabitq.sh greps for "build_ms=[0-9.]+" instead, which the
# binary never prints, so rq_train.log and rq_best.log carry "build_ms=?" on
# all 88 rows.  The pattern here is run_rabitq.sh's, which is the one that
# works.
#
# There is no index cache to defeat: JHQ_RQ_IDX names a scratch file the bench
# writes and reads back inside the timed region, not a cache it skips a build
# for.  So one run per cell is one cold build.
#
# Why both budgets.  Figure 5 currently plots IVF-RaBitQ's build from the
# archived sweep, which ran at cuVS's default of 256 training points a
# centroid against JHQ's 39 -- 5.8x to 6.6x the budget -- and the caption says
# so instead of measuring it.  With both numbers the caption can say what the
# build gap is at equal training and what the library's default costs, which
# is what the disclosure is standing in for.
#
# nprobe=8 throughout: search is not what this measures, and the shallowest
# probe is the cheapest way to reach the build print.  Kernel follows each
# dataset's own winner from rq_train.log's mode sweep, so a row here is
# comparable with rq_best.log.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_rabitq_build.sh /root/rqbuild.sh \
#     && setsid nohup bash /root/rqbuild.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_rqbuild; flock -n 9 || { echo "already running"; exit 0; }
L=/root/rq_build.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "start; head=$(git log --oneline -1)"

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
D=/root/autodl-tmp; V=/root/data
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export JHQ_RQ_IDX=$D/rq_build.idx

INC="-I$SP/libraft/include/rapids"; for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""; for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC examples/bench_ivf_rabitq.cu -o build/bench_rq_build \
     $LD -lcuvs -lrmm -lcudart -lcublas >>"$L" 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== RQBUILD""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths nlist tpl mode rep
    local tag=$1 paths=$2 nl=$3 tpl=$4 md=$5 rep=$6 out tr_ms used r
    out=$(env JHQ_RQ_TRAIN_PER_LIST="$tpl" timeout 9000 \
          build/bench_rq_build $paths "$nl" 8 8 10 "$md" 1 2>&1)
    tr_ms=$(printf '%s\n' "$out" | awk '/^  train:/{print $2;exit}')
    used=$(printf '%s\n' "$out"  | grep -oE 'train_per_list=[0-9]+' | head -1 | cut -d= -f2)
    r=$(printf '%s\n' "$out"     | awk '/^Recall@10/{print $3;exit}')
    if [ -z "${tr_ms:-}" ]; then
        echo "  RQB $tag nlist=$nl tpl=$tpl rep=$rep MISSING <-- no train line" >> "$L"
        printf '%s\n' "$out" | tail -4 | sed 's/^/      | /' >> "$L"
    else
        printf "  RQB %-14s nlist=%-6s tpl=%-4s used=%-6s mode=%-2s rep=%-2s recall=%-8s train_ms=%s\n" \
            "$tag" "$nl" "$tpl" "${used:-?}" "$md" "$rep" "${r:-?}" "$tr_ms" >> "$L"
    fi
}

# Three cold builds a cell: Figure 5's whiskers are a range over repeats, and
# one timing cannot produce one.
for rep in 1 2 3; do
  for tpl in 39 256; do
    say "########## rep=$rep train_per_list=$tpl ##########"
    one vogue-768    "$VG"  4096 "$tpl" 1 "$rep"
    one arxiv-768    "$AX"  8192 "$tpl" 1 "$rep"
    one openai3-1536 "$O15" 4096 "$tpl" 0 "$rep"
    one openai3-3072 "$O30" 4096 "$tpl" 2 "$rep"
  done
done
rm -f "$JHQ_RQ_IDX"
say "=== RQBUILD""_DONE ==="
