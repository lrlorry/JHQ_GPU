#!/bin/bash
# The last two CPU datasets: bge-m3 and stella-trec24.
#
# run_cpu_four.sh finished arxiv-768 and openai3-1536 and was stopped before
# these two so the GPU queue could start. They are the datasets on which no
# compared baseline builds at all, so the CPU reference is the only comparison
# available there.
#
# Cost, calibrated rather than extrapolated. The August run
# (results/jhq_cpu_ivf_*.csv) measured every dataset under a narrower protocol,
# and its wall time reconstructs from build_time plus 1000/QPS a row: 202 s for
# bge-m3 and 345 s for stella. The same reconstruction on arxiv-768 and
# openai3-1536 gives 54 s and 65 s against 1380 s and 519 s actually measured
# under this protocol, so this protocol costs about 17x the August one -- six
# alphas instead of one, and nprobe to 1024 instead of 128. That puts bge-m3
# near an hour and stella near two.
#
# An earlier estimate here said 22 hours. It came from fitting build time to
# N*d*nlist on two points and extrapolating forty-fold past them, and the
# August build times -- 144 s and 239 s, not hours -- show how far wrong that
# was. Interpolating between measured protocols is the version to trust.
#
# 32 threads only. cpu_provenance.md records 16 and 32 as inside the noise of a
# single timing, and dropping 16 halves the run. The envelope on these two
# therefore has one thread configuration where the other four have two, which
# holds their CPU side slightly down and so does not inflate our ratio in the
# direction that flatters us -- but it is a difference and Section 6.8 says so.
#
# Queued last: the GPU jobs are measuring throughput on the same host, and a
# 32-thread memory-bound scan beside them would perturb the numbers the paper
# leads with.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_cpu_big.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_cpu_big; flock -n 9 || { echo "already running"; exit 0; }
L=/root/cpu_big.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

# Behind the two GPU jobs, in the order they were queued.
exec 7>/root/.lock_batch2more; flock 7; echo "batch sweep clear $(date -u +%H:%M:%S)"
exec 6>/root/.lock_rqtrain;    flock 6; echo "rabitq rerun clear $(date -u +%H:%M:%S)"

R=/root/JHQ_paper2
B=$R/build/examples/bench_vogue768
[ -x "$B" ] || { echo "=== CPUBIG_DONE $B missing ==="; exit 1; }
echo "binary=$B"

D=/root/autodl-tmp
BG="$D/bge-m3/base.fvecs $D/bge-m3/query.fvecs $D/bge-m3/groundtruth.ivecs"
ST="$D/stella-trec24/base.fvecs $D/stella-trec24/query.fvecs $D/stella-trec24/groundtruth.ivecs"
A="2.0,4.0,8.0,16.0,32.0,100.0"

run(){  # tag paths M nlist
    echo "### $1 M=$3 nlist=$4 threads=32 alphas=$A $(date -u +%H:%M:%S)"
    OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores \
    JHQ_NLIST=$4 JHQ_M=$3 JHQ_BR=8 JHQ_ALPHA=$A JHQ_NPROBE_MAX=1024 \
    timeout 28800 "$B" $2
    echo "run_rc=$? tag=$1 $(date -u +%H:%M:%S)"
}
# Smaller first, so its build time confirms or refutes the projection before
# the larger one commits two hours to it.
run bge-m3 "$BG" 128 16384
run stella "$ST" 128 32768
echo "=== CPUBIG""_DONE $(date -u +%FT%TZ) ==="
