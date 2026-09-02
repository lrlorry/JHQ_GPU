#!/bin/bash
# Does the device codebook agree with the host one, and how much time does it save?
#
# Bit-identity is not the claim -- the atomics run in whatever order the
# scheduler picks. The claim that has to hold is that the index built either way
# retrieves the same neighbours, so this runs both and compares recall at four
# decimal places as well as the build time.
exec 9>/root/.lock_cbk
flock -n 9 || { echo "another run holds the lock"; exit 0; }
exec >/root/gpu_codebook.log 2>&1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
make demo_jhq_v22_s2b1 -j24 2>&1 | grep -E "error|Error" | head -12
cd ..
V=/root/data/vogue-768_
D="${V}base.fvecs ${V}query.fvecs ${V}groundtruth.ivecs"
unset JHQ_INDEX_CACHE          # a cached codebook would defeat the whole test
for mode in host device; do
  echo "###### primary codebook trained on the $mode"
  if [ "$mode" = device ]; then export JHQ_GPU_CODEBOOK=1; else unset JHQ_GPU_CODEBOOK; fi
  JHQ_TRAIN_PHASES=1 JHQ_BLOCK=1024 JHQ_PFX_NUM=1 JHQ_PFX_DEN=2 JHQ_TILE_M_RT=96 \
    ./build/demo_jhq_v22_s2b1 $D 96 8 4 100.0 10 1024 128 8 1024 "" 5 2>&1 \
    | grep -E "\[train\]|train:|add:|Recall@|^QPS"
done
echo "### GPU_CODEBOOK_CHECK_DONE"
