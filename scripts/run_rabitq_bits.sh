#!/bin/bash
# IVF-RaBitQ across its code-size knob, which no run in this project ever swept.
#
# Every archived and re-run RaBitQ number was taken at bits_per_dim=8.  That
# was a deliberate memory match -- bench_ivf_rabitq.cu's header notes JHQ's
# Br=8 corresponds to 9 bits a dimension, so 8 hands RaBitQ 11% less memory
# than parity -- but matching memory is not the same as finding the baseline's
# best recall-QPS frontier, and RaBitQ's own design point is 1 bit a dimension
# with re-ranking, not 8.
#
# Why it matters more than the kernel sweep did.  Section 6.1's explanation of
# JHQ's lead is that the primary scan reads d/8 bytes a candidate where
# IVF-RaBitQ at 8 bits a dimension reads d, and that the gap therefore widens
# with dimension -- which is what 768 parity and 1536/3072 leads look like.  At
# bits_per_dim=1 RaBitQ's scan reads d/8 too.  If its frontier is better there,
# the mechanism we report is an artefact of the operating point we pinned it
# at, and the comparison has to move.  That is worth 30 minutes of an idle card
# before the paper commits to the claim.
#
# Each dataset runs at the search kernel the mode sweep already found best for
# it (rq_train.log): LUT32 on vogue-768 and arxiv-768, LUT16 on openai3-1536,
# QUANT4 on openai3-3072.  Training stays at cuVS's own 256 points a centroid,
# and nlist follows the JHQ frontier's setting, so bits_per_dim is the only
# variable against rq_best.log.
#
# No chained flock.  run_cpu_big.sh took .lock_rqtrain to *wait* on it and so
# held the mutex the RaBitQ job needed, which silently blocked every launch.
# This takes only the GPU lock, which is the one it actually needs.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_rabitq_bits.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_rqbits; flock -n 9 || { echo "already running"; exit 0; }
L=/root/rq_bits.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "start; head=$(git log --oneline -1)"

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
D=/root/autodl-tmp; V=/root/data
SP=/root/miniconda3/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib64 $SP/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export JHQ_RQ_IDX=$D/rq_bits.idx

INC="-I$SP/libraft/include/rapids"; for p in $SP/*/include; do [ -d "$p" ] && INC="$INC -I$p"; done
LD=""; for p in $SP/*/lib64 $SP/*/lib; do [ -d "$p" ] && LD="$LD -L$p -Xlinker -rpath -Xlinker $p"; done
nvcc -O3 -std=c++20 -ccbin g++-12 --expt-relaxed-constexpr --extended-lambda \
     -arch=sm_120 $INC examples/bench_ivf_rabitq.cu -o build/bench_rq_bits \
     $LD -lcuvs -lrmm -lcudart -lcublas >>"$L" 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { say "=== RQBITS""_DONE build failed ==="; exit 1; }

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths nlist bits nprobe mode
    local tag=$1 paths=$2 nl=$3 bt=$4 np=$5 md=$6 out r q b
    out=$(env JHQ_RQ_TRAIN_PER_LIST=256 timeout 9000 \
          build/bench_rq_bits $paths "$nl" "$bt" "$np" 10 "$md" 3 2>&1)
    r=$(printf '%s\n' "$out" | awk '/^Recall@10/{print $3;exit}')
    q=$(printf '%s\n' "$out" | awk '/^QPS/{print $3;exit}')
    b=$(printf '%s\n' "$out" | grep -oE 'build_ms=[0-9.]+' | head -1 | cut -d= -f2)
    if [ -z "${r:-}" ]; then
        echo "  RQ $tag nlist=$nl bits=$bt np=$np mode=$md MISSING <-- produced nothing" >> "$L"
        printf '%s\n' "$out" | tail -4 | sed 's/^/      | /' >> "$L"
    else
        printf "  RQ %-14s nlist=%-6s bits=%-3s mode=%-2s np=%-5s recall=%-8s qps=%-9s build_ms=%s\n" \
            "$tag" "$nl" "$bt" "$md" "$np" "$r" "$q" "${b:-?}" >> "$L"
    fi
}

NP="8 32 128 256 512 1024"
# dataset|paths|nlist|best mode from rq_train.log's kernel sweep
SET="vogue-768|$VG|4096|1 arxiv-768|$AX|8192|1 openai3-1536|$O15|4096|0 openai3-3072|$O30|4096|2"

for bt in 1 2 4 8; do
    say "########## bits_per_dim=$bt ##########"
    for s in $SET; do
        IFS='|' read -r tag paths nl md <<< "$s"
        for np in $NP; do one "$tag" "$paths" "$nl" "$bt" "$np" "$md"; done
    done
done
rm -f "$JHQ_RQ_IDX"
say "=== RQBITS""_DONE ==="
