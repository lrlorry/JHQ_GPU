#!/bin/bash
# The rule against alpha=100, six datasets at S=32.
#
# Why this run exists: Section 6.4 quotes "over a broader 32-configuration sweep
# against alpha=100 -- six datasets at S=32 -- gain is 1.044--2.653x for at most
# 0.0048 recall, median payback 1.75 batches".  Nothing in paper_adc2026/data/
# produces 1.044 or 2.653 as a gain.  The strings exist in paper_fronts.log and
# in batch_matched_CONTAMINATED.log, which is how the old substring audit passed
# them, but neither file is this experiment.
#
# What we do hold is alpha_sample.log and alpha_fast.log: 32 rows between them,
# gains 0.997x to 2.014x -- and three of those configurations are below 1.0,
# that is, the rule loses to alpha=100 there.  The published range contains no
# such case.  Those two logs are also a different pairing from what the text
# describes (AS and AF are two methods, over four datasets, not six), so they
# cannot simply be relabelled as the source.  This run measures what the
# sentence actually claims.
#
# Lives in the repo rather than being copied to the box: the box reaches GitHub
# through /etc/network_turbo, so a fetch puts both the source and this driver at
# one commit.  CLAUDE.md has the longer version of why.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_alpha_sweep6.sh >/dev/null 2>&1 &
set -u
L=/root/alpha6cfg.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

# A script that runs after a failed build reports the old binary's output.
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_alpha_sample
rc=$?; echo "build_rc=$rc"
[ "$rc" -ne 0 ] && { echo "=== ALPHA6CFG_DONE build failed, nothing run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
CACHE=$D/paper_cache

VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
AX="$D/arxiv-abstracts-768/base.fvecs $D/arxiv-abstracts-768/query.fvecs $D/arxiv-abstracts-768/groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
O15="$D/openai3-1536/base.fvecs $D/openai3-1536/query.fvecs $D/openai3-1536/groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths M nlist n_train nprobe
    local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 np=$6
    local C=$CACHE/$tag; mkdir -p "$C"
    echo "### $tag M=$M nlist=$nl np=$np S=32 $(date -u +%T)"
    local out
    out=$(env $E JHQ_INDEX_CACHE="$C" JHQ_BLOCK=512 JHQ_TILE_M_RT="$M" \
              JHQ_N_TRAIN="$nt" \
              JHQ_AS_SAMPLE=32 JHQ_AS_EPS=0.001 \
              JHQ_AS_GRID=100,64,32,16,8,4,2 \
              timeout 5400 build/demo_jhq_alpha_sample \
              $paths "$M" 8 8 100.0 10 "$nl" "$np" 8 1024 "" 3 2>&1)
    local r=$?
    # A configuration that dies before printing must say so rather than reach
    # the parser as an empty field and be read as a measurement.
    local res
    res=$(printf '%s\n' "$out" | grep '^AS_RESULT' | tail -1)
    if [ -z "${res:-}" ]; then
        echo "  MISSING <-- produced no AS_RESULT, rc=$r"
        printf '%s\n' "$out" | tail -6 | sed 's/^/      | /'
    else
        echo "  $res  tag=$tag np=$np rc=$r"
    fi
}

# Six datasets x nprobe {8,32,128,512} = 24 configurations at S=32.  The text
# says 32; whether that count came from a different nprobe set is one of the
# things this settles, so the count is reported rather than assumed.
# Parameters follow run_front6.sh's second block, the one the frontier reports.
for np in 8 32 128 512; do
    one vogue-768    "$VG"  96  4096   159744  "$np"
    one arxiv-768    "$AX"  96  8192   319488  "$np"
    one openai3-1536 "$O15" 192 4096   159744  "$np"
    one openai3-3072 "$O30" 384 4096   159744  "$np"
    one bge-m3       "$BG"  128 16384  638976  "$np"
    one stella       "$ST"  128 32768  1277952 "$np"
done
echo "=== ALPHA6CFG""_DONE $(date -u +%FT%TZ) ==="
