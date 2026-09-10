#!/bin/bash
# Every budget arm on one index, for Section 6.4's fair comparison.
#
# Lives in the repo rather than being copied to the box: the box reaches GitHub
# through /etc/network_turbo, so a fetch puts both the source and this driver at
# one commit. Shipping the driver by hand while the driver git-fetches the
# source is how the two drift apart -- CLAUDE.md has the longer version of why.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_budget_arms.sh >/dev/null 2>&1 &
set -u
L=/root/arms.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

# The build tree predates this target, so gmake has no rule for it until cmake
# regenerates. A script that runs after a failed build reports the old binary.
timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_budget_arms
rc=$?; echo "build_rc=$rc"
[ "$rc" -ne 0 ] && { echo "=== ARMS_DONE build failed, nothing run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
CACHE=$D/arms_cache
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
O30="$D/openai3-3072/base.fvecs $D/openai3-3072/query.fvecs $D/openai3-3072/groundtruth.ivecs"

one(){  # tag paths M nlist n_train BLOCK nprobe
  local tag=$1 paths=$2 M=$3 nl=$4 nt=$5 blk=$6 np=$7
  local C=$CACHE/$tag; mkdir -p "$C"
  echo "### $tag M=$M nlist=$nl np=$np $(date -u +%T)"
  env JHQ_INDEX_CACHE="$C" JHQ_BLOCK="$blk" JHQ_TILE_M_RT="$M" JHQ_N_TRAIN="$nt" \
      JHQ_AS_GRID=200,100,64,32,16,8,4,2 JHQ_AS_SLOTS=1 \
      JHQ_ARM_S=32,64,128 JHQ_ARM_REPS=64 JHQ_ORACLE_TAU=0.001 \
      timeout 9000 build/demo_jhq_budget_arms $paths "$M" 8 8 100.0 10 "$nl" "$np" 8 1024 "" 3
  echo "run_rc=$? tag=$tag np=$np"
}
for np in 128 512; do one vogue-768    "$VG"  96 4096 159744 512 "$np"; done
for np in 128 512; do one openai3-3072 "$O30" 384 4096 159744 512 "$np"; done
echo "=== ARMS_DONE $(date -u +%FT%TZ) ==="
