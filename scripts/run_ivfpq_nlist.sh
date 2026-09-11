#!/bin/bash
# IVF-PQ over nlist, which it was never swept on.
#
# The archived cuVS runs sweep pq_dim over four code sizes and n_probes over
# six depths, and pin n_lists at one value a dataset: 1024 on vogue-768 and
# both OpenAI sets, 2048 on arxiv-768.  JHQ's own nlist was swept over four
# values per dataset (results/front6/nlist6.log) and the frontier reports the
# winner.  So one side of the comparison is tuned on a parameter the other
# side is not, which is the same error as comparing IVF-RaBitQ at a search
# kernel that is not its best.
#
# The size of it: at openai3-1536, IVF-PQ's n_lists=1024 puts 976 vectors in a
# list against JHQ's 122 at 8192, so the same probe depth scans eight times as
# many vectors.  JHQ's own sweep measures nlist as worth 1.4x to 2.1x over the
# middle of a front, so this is not a rounding difference.
#
# The grid is JHQ's grid for each dataset, so afterwards both systems have been
# offered the same partition choices and each is reported at its own best.
# CAGRA needs no equivalent: it already sweeps graph_degree, itopk_size and
# search_width, 24 configurations a dataset.
#
# The four smaller datasets first.  They carry the worst mismatch -- IVF-PQ at
# 1024 or 2048 against JHQ at 4096 or 8192 -- and they are the panels where the
# two systems are close enough for the partition to matter.  bge-m3 and stella
# follow if the card is free; there IVF-PQ is already at 8192 and 16384.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD && cp scripts/run_ivfpq_nlist.sh /root/pqnl.sh \
#     && setsid nohup bash /root/pqnl.sh >/dev/null 2>&1 &
set -u
exec 9>/root/.lock_pqnl; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
cd /root/JHQ_GPU
L=/root/pq_nlist.log; : > "$L"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$L"; }
say "start; head=$(git log --oneline -1)"
R=results/pq_nlist; mkdir -p "$R"

exec 8>/root/.gpu_lock; flock 8; say "got the GPU"

# dataset:nlist values JHQ swept for it, minus the one IVF-PQ already has
run(){  # dataset nprobe-grid nlist...
    local ds=$1 np=$2; shift 2
    for nl in "$@"; do
        say "### $ds nlist=$nl"
        timeout 7200 python3 scripts/bench_all.py --dataset "$ds" --method ivfpq \
            --nlist "$nl" --pq-dims 96,192,384,768 --nprobe "$np" --reps 3 \
            --out "$R/${ds}_ivfpq_nl${nl}.csv" >>"$L" 2>&1
        say "  rc=$? rows=$(grep -vc '^#' "$R/${ds}_ivfpq_nl${nl}.csv" 2>/dev/null)"
    done
}
NP="8,32,128,256,512,1024"
run vogue-768    "$NP" 2048 4096 8192
run arxiv-768    "$NP" 4096 8192 16384
run openai3-1536 "$NP" 2048 4096 8192
run openai3-3072 "$NP" 2048 4096 8192
say "=== PQNL""_DONE small four ==="

# The large two, where IVF-PQ's default k-means fraction does not fit the card
# and the archived runs already reduced it; keep that reduction so the only
# variable here is nlist.
runf(){  # dataset fraction nprobe-grid nlist...
    local ds=$1 fr=$2 np=$3; shift 3
    for nl in "$@"; do
        say "### $ds nlist=$nl frac=$fr"
        timeout 7200 python3 scripts/bench_all.py --dataset "$ds" --method ivfpq \
            --nlist "$nl" --pq-dims 128,256 --nprobe "$np" --reps 3 \
            --trainset-fraction "$fr" \
            --out "$R/${ds}_ivfpq_nl${nl}_f${fr}.csv" >>"$L" 2>&1
        say "  rc=$? rows=$(grep -vc '^#' "$R/${ds}_ivfpq_nl${nl}_f${fr}.csv" 2>/dev/null)"
    done
}
runf bge-m3        0.05 "$NP" 4096 16384 32768
runf stella-trec24 0.02 "$NP" 8192 32768 65536
say "=== PQNL""_DONE ==="
