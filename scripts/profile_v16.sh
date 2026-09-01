#!/bin/bash
# Where does the time actually go? The static reading says the scan kernel is
# launched with one block per query -- 256 blocks against 170 SMs -- and that
# the byte LUT it reads is 25 MB per batch, larger than the 22 MB of candidate
# codes scanned at nprobe=1. Both are guesses until measured.
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/v16; mkdir -p $R
V=/root/data/vogue-768_
NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)

for np in 8 128; do
  echo ""
  echo "########## nprobe=$np ##########"
  $NCU --target-processes all --launch-count 24 \
       --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
       ./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs \
       ${V}groundtruth.ivecs 96 8 4 4.0 10 1024 $np 8 256 "" 5 \
    2>&1 | tee $R/ncu_np$np.log | grep -E "^ *(void|jhq)|Section|sm__throughput|dram_throughput|warps_active|grid_size|block_size|sectors_per_request|bank_conflicts" | head -60
done

echo ""
echo "########## kernel 时间占比 (nsys) ##########"
NSYS=$(command -v nsys || echo /usr/local/cuda/bin/nsys)
$NSYS profile -o $R/v16_trace --force-overwrite true --stats=true \
     ./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs \
     ${V}groundtruth.ivecs 96 8 4 4.0 10 1024 8 8 256 "" 5 \
  2>&1 | grep -A 25 -iE "cuda_gpu_kern_sum|CUDA Kernel Statistics" | head -30
echo "=========== PROFILE DONE $(date +%H:%M:%S) ==========="
