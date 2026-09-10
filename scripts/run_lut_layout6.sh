#!/bin/bash
# The table x layout square, on all six datasets.
#
# Why this run exists: Table 2's "packed 32-bit loads gain +30 to +48%" and the
# abstract's copy of it are typed as literals in figures/fig_ablation.py, and no
# log in paper_adc2026/data/ reproduces them.  The only measurement of that gain
# we hold is lut_groups.log's phase 3, which covers two datasets and gives +0.3%
# to +40.7% -- a different answer.  Either the published range came from a run
# whose log is lost, or it is wrong; six datasets settles which.
#
# The square is the same one v59's CMake comment lays out:
#
#            full table      factorised
#   byte     v59_b_g1        v59_b_g2
#   word     v59_g1          v59_g2
#
# all four from one source tree at one commit, so the layout gain is measured at
# fixed G rather than inferred across version directories.
#
# Lives in the repo rather than being copied to the box: the box reaches GitHub
# through /etc/network_turbo, so a fetch puts both the source and this driver at
# one commit.  CLAUDE.md has the longer version of why.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_lut_layout6.sh >/dev/null 2>&1 &
set -u
L=/root/layout6.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

# A script that runs after a failed build reports the old binary's output.
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 3600 cmake --build build -j 16 \
    --target demo_jhq_v59_g1 demo_jhq_v59_g2 \
             demo_jhq_v59_b_g1 demo_jhq_v59_b_g2
rc=$?; echo "build_rc=$rc"
[ "$rc" -ne 0 ] && { echo "=== LAYOUT6_DONE build failed, nothing run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
# The frontier driver's shared prefix.  Dropping it trains the residual codebook
# on every base vector instead of a 100k sample -- host-side, with the GPU idle.
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
# Its cache too: identical index parameters, so there is nothing to retrain.
# Per CLAUDE.md the cache is keyed on data and parameters, not on code, and all
# four binaries here train identically -- G and the layout change only the scan.
CACHE=$D/paper_cache

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

cell(){  # tag paths M nlist n_train nprobe layout G
    local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 np=$6 lay=$7 G=$8
    local bin=demo_jhq_v59_g$G
    [ "$lay" = byte ] && bin=demo_jhq_v59_b_g$G
    local C=$CACHE/$tag; mkdir -p "$C"
    local out
    out=$(env $E JHQ_INDEX_CACHE="$C" JHQ_BLOCK=512 JHQ_TILE_M_RT="$M" \
              JHQ_N_TRAIN="$nt" \
              timeout 3600 "build/$bin" $paths "$M" 8 8 100.0 10 "$nl" "$np" 8 1024 "" 3 2>&1)
    local r=$?
    # The recall and QPS the run reports, or a marker.  An arm that dies before
    # printing must not reach awk as empty fields and look like data -- that is
    # how three passes of the RaBitQ arm once went by looking like measurements.
    local rec qps
    rec=$(printf '%s\n' "$out" | sed -n 's/^Recall@10 *: *\([0-9.]*\).*/\1/p' | tail -1)
    qps=$(printf '%s\n' "$out" | sed -n 's/^QPS  *: *\([0-9]*\).*/\1/p' | tail -1)
    if [ -z "${rec:-}" ] || [ -z "${qps:-}" ]; then
        echo "  $tag M=$M layout=$lay G=$G np=$np MISSING <-- produced nothing, rc=$r"
        printf '%s\n' "$out" | tail -5 | sed 's/^/      | /'
    else
        echo "  $tag M=$M layout=$lay G=$G np=$np recall=$rec qps=$qps rc=$r"
    fi
}

square(){  # tag paths M nlist n_train nprobe...
    local tag=$1 paths=$2 M=$3 nl=$4 nt=$5; shift 5
    for np in "$@"; do
        for lay in byte word; do
            for G in 1 2; do cell "$tag" "$paths" "$M" "$nl" "$nt" "$np" "$lay" "$G"; done
        done
    done
}

# nlist and n_train follow run_front6.sh's second block, the one the paper's
# frontier reports, so the layout gain is measured at the paper's operating
# points rather than at a set chosen for this run.
echo "$(date -u +%T) ########## the 2x2, six datasets ##########"
square vogue-768    "$VG"  96  4096   159744  8 32 128 512
square arxiv-768    "$AX"  96  8192   319488  8 32 128 512
square openai3-1536 "$O15" 192 4096   159744  8 32 128 512
square openai3-3072 "$O30" 384 4096   159744  8 32 128 512
square bge-m3       "$BG"  128 16384  638976  8 32 128 512
square stella       "$ST"  128 32768  1277952 8 32 128 512
echo "=== LAYOUT6""_DONE $(date -u +%FT%TZ) ==="
