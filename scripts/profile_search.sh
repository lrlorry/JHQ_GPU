#!/bin/bash
# Profile the search path only. The earlier run pointed ncu at the whole
# program and the trace came back dominated by index construction -- GPU
# kernels totalled ~700 ms against a 27 s build that is almost entirely CPU
# work -- so the scan never appeared.
#
# --kernel-name selects the search kernels by name, and --launch-skip drops the
# encode launches that run during add(). The metrics are chosen to separate the
# three candidate bottlenecks rather than to confirm any one of them:
#
#   stall_long_scoreboard   waiting on a global load -- the LUT gather
#   stall_barrier           waiting at __syncthreads -- the top-k reduction
#   l1tex/lts hit rates     whether the LUT is being served from cache at all
#   sectors_per_request     coalescing of the [M,N] code reads
#   dram throughput         whether HBM is the limit at all (9.4% says not)
set -u
cd /root/JHQ_GPU
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=results/prof; mkdir -p $R
V=/root/data/vogue-768_
NCU=/usr/local/cuda/bin/ncu

for np in 8 128; do
  echo ""
  echo "########## nprobe=$np ##########"
  $NCU --kernel-name "regex:scan_ivf|batched_topk_final|residual_refine|build_byte_lut" \
       --launch-skip 40 --launch-count 12 \
       --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
gpu__time_duration.sum \
       ./build/demo_jhq_v16_pq_primary ${V}base.fvecs ${V}query.fvecs \
       ${V}groundtruth.ivecs 96 8 4 4.0 10 1024 $np 8 256 "" 5 \
    2>&1 | tee $R/search_np$np.log | grep -E "Kernel Name|scan_ivf|topk|residual_refine|build_byte|stalled|hit_rate|throughput|warps_active|sectors_per_request|duration" | head -70
done
echo "=========== SEARCH PROFILE DONE $(date +%H:%M:%S) ==========="
