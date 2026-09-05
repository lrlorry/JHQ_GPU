#!/bin/bash
# v37 vs v36: does moving the residuals off disk actually pay, and did the
# subspace-major store break anything?
#
# The layout check is the point of the first two blocks. v37 default keeps the
# residuals in host memory; JHQ_RES_HOST_GB=0 forces every subspace through the
# spill file instead. Both run the same kernel and the same estimator over the
# same values, so their recall must agree exactly. If it does not, the store's
# index arithmetic is wrong and the timing is meaningless.
#
#   bash scripts/run_v37.sh            # DATA_ROOT and SCRATCH below
set -u
D=${DATA_ROOT:-/root/autodl-tmp}
V=${VOGUE_DIR:-/root/data}
S=${SCRATCH:-/root/autodl-tmp}
L=${LOG:-/root/v37.log}
cd "$(dirname "$0")/.."
exec 9>/root/.lock_v37; flock -n 9 || { echo "already running"; exit 0; }
exec 8>/root/.gpu_lock; flock 8
: > $L
echo "=== $(date -u +%FT%TZ) ===" >> $L

echo "--- 0. 磁盘带宽:1.27 GB/s 到底是盘还是我们的单线程 fread ---" >> $L
dd if=/dev/zero of=$S/.bwtest bs=1M count=8192 oflag=direct 2>&1 | tail -1 | sed 's/^/    write /' >> $L
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
dd if=$S/.bwtest of=/dev/null bs=1M iflag=direct 2>&1 | tail -1 | sed 's/^/    read  /' >> $L
rm -f $S/.bwtest
echo "    cgroup memory.max = $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)" >> $L

E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1"
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"

run(){ # bin tag paths M nlist extra_env
  local t0=$(date +%s)
  env $E JHQ_TILE_M_RT=$4 JHQ_RES_SPILL=$S $6 \
      timeout 6000 build/$1 $3 $4 8 8 100.0 10 $5 128 8 1024 "" 5 >/tmp/o.$$ 2>/tmp/e.$$
  local rc=$?
  printf "%-18s %-8s M=%-4s %-24s rc=%s wall=%ss recall=%s train=%s add=%s\n" \
    "$1" "$2" "$4" "$6" "$rc" "$(( $(date +%s)-t0 ))" \
    "$(awk '/^Recall@10/{print $3; exit}' /tmp/o.$$)" \
    "$(awk '/^  train:/{print $2; exit}' /tmp/o.$$)" \
    "$(awk '/^  add:/{print $2; exit}' /tmp/o.$$)" >> $L
  grep -E "trains on|host memory|per chunk|one pass" /tmp/o.$$ /tmp/e.$$ \
    | sed 's/^[^ ]*://;s/^/      /' | head -4 >> $L
  grep -iE "bad_alloc|out of memory|what\(\):|CUDA error" /tmp/e.$$ | head -2 \
    | sed 's/^/      FAIL /' >> $L
  rm -f /tmp/o.$$ /tmp/e.$$; }

echo "--- 1. v37 默认:残差留主机内存,全量 ---" >> $L
run demo_jhq_v37_exp vogue  "$VG" 96  1024   ""
run demo_jhq_v37_exp bge    "$BG" 128 8192   ""
run demo_jhq_v37_exp stella "$ST" 128 16384  ""

echo "--- 2. 同一个 v37,强制全部落盘。recall 必须与上面逐位一致 ---" >> $L
run demo_jhq_v37_exp vogue  "$VG" 96  1024   "JHQ_RES_HOST_GB=0"
run demo_jhq_v37_exp bge    "$BG" 128 8192   "JHQ_RES_HOST_GB=0"
run demo_jhq_v37_exp stella "$ST" 128 16384  "JHQ_RES_HOST_GB=0"

echo "--- 3. v36,同样全部落盘:分离出 GPU 转置本身值多少 ---" >> $L
run demo_jhq_v36_exp stella "$ST" 128 16384  ""

echo "--- 4. stella M=64:C=8 时要 42.4 GiB,推导的 C 应当放行 ---" >> $L
run demo_jhq_v37_exp stella "$ST" 64 16384   ""
echo "--- 5. 对照:C=8 应当失败 ---" >> $L
run demo_jhq_v37_exp stella "$ST" 64 16384   "JHQ_RES_CHUNK=8"

echo "--- 6. 对照:采样 100K 应复现 v35 的 0.9848 / 0.9589 / 0.9914 ---" >> $L
run demo_jhq_v37_exp vogue  "$VG" 96  1024   "JHQ_RES_TRAIN_N=100000"
run demo_jhq_v37_exp bge    "$BG" 128 8192   "JHQ_RES_TRAIN_N=100000"
run demo_jhq_v37_exp stella "$ST" 128 16384  "JHQ_RES_TRAIN_N=100000"

echo "=== V37""_DONE $(date -u +%FT%TZ) ===" >> $L
