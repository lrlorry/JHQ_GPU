#!/bin/bash
# The bounded selector's deviation, as returned ids.
#
# Why this run exists: Section 4's caveat says that with the index pinned, on
# vogue-768 at nprobe=128 and BLOCK=512, raising K_LOCAL from 4 to 8 changes 11
# of 10,000 returned id positions, touches 4 of 1,000 queries and gives 1 query
# a different result set.  Those three numbers appear in the paper and in a
# comment in jhq_v53_cap/search.cu, and nowhere else.  The K_LOCAL ablation that
# is archived (results/pre_freeze_v22_s2b1/abl_klocal.csv) is at nprobe=256 and
# records only recall and QPS -- it cannot speak to which ids came back.  A
# claim about ids needs the ids.
#
# Both arms must see the same index or the diff measures training, not the
# selector: JHQ_INDEX_CACHE points both at one directory, and per CLAUDE.md the
# cache is keyed on data and parameters, which are identical here -- K_LOCAL is
# a compile-time macro in the scan and changes nothing that is trained.
#
# demo_jhq_v36 writes <prefix>.ivecs (the returned ids) when given an
# out_prefix, so no new demo is needed; only the K_LOCAL=8 twin target is new,
# and jhq_v53_cap/ itself is untouched.
#
# On the box:
#   source /etc/network_turbo && cd /root/JHQ_GPU \
#     && git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> \
#     && git reset --hard FETCH_HEAD \
#     && setsid nohup bash scripts/run_klocal_iddiff.sh >/dev/null 2>&1 &
set -u
L=/root/klocal.log; : > "$L"
exec >> "$L" 2>&1
echo "=== $(date -u +%FT%TZ) head=$(git log --oneline -1) ==="

timeout 600 cmake -S . -B build >/dev/null 2>&1; echo "configure_rc=$?"
timeout 2400 cmake --build build -j 16 --target demo_jhq_v53_x1 demo_jhq_v53_kl8
rc=$?; echo "build_rc=$rc"
[ "$rc" -ne 0 ] && { echo "=== KLOCAL_DONE build failed, nothing run ==="; exit 1; }

D=/root/autodl-tmp; V=/root/data
E="JHQ_GPU_CODEBOOK=1 JHQ_ENCODE_GROUPED_OFF=1 JHQ_Y_TRANSPOSED=1 JHQ_RES_TRAIN_N=100000"
CACHE=$D/paper_cache/vogue-768          # the frontier's own cache: same index
OUT=/root/klocal_ids; mkdir -p "$OUT"
VG="$V/vogue-768_base.fvecs $V/vogue-768_query.fvecs $V/vogue-768_groundtruth.ivecs"

run(){  # tag binary
    local tag=$1 bin=$2
    echo "### $tag $(date -u +%T)"
    local out
    out=$(env $E JHQ_INDEX_CACHE="$CACHE" JHQ_BLOCK=512 JHQ_TILE_M_RT=96 \
              JHQ_N_TRAIN=159744 \
              timeout 3600 "build/$bin" $VG 96 8 8 100.0 10 4096 128 8 1024 \
              "$OUT/$tag" 3 2>&1)
    local r=$?
    local rec qps
    rec=$(printf '%s\n' "$out" | sed -n 's/^Recall@10 *: *\([0-9.]*\).*/\1/p' | tail -1)
    qps=$(printf '%s\n' "$out" | sed -n 's/^QPS  *: *\([0-9]*\).*/\1/p' | tail -1)
    if [ -z "${rec:-}" ]; then
        echo "  MISSING <-- $tag produced nothing, rc=$r"
        printf '%s\n' "$out" | tail -6 | sed 's/^/      | /'
        return 1
    fi
    echo "  $tag recall=$rec qps=$qps rc=$r  ids=$(stat -c%s "$OUT/$tag.ivecs" 2>/dev/null) bytes"
}

run kl4 demo_jhq_v53_x1 || exit 1
run kl8 demo_jhq_v53_kl8 || exit 1

# The diff, in the three units the paper quotes.  ivecs is [d][d ints] per row;
# here d = k = 10 and there are 1,000 rows.
python3 - "$OUT/kl4.ivecs" "$OUT/kl8.ivecs" <<'PY'
import struct, sys
def read(p):
    out = []
    with open(p, 'rb') as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack('<i', h)[0]
            out.append(list(struct.unpack('<%di' % d, f.read(4 * d))))
    return out
a, b = read(sys.argv[1]), read(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))
pos = sum(1 for x, y in zip(a, b) for i, j in zip(x, y) if i != j)
qs  = [n for n, (x, y) in enumerate(zip(a, b)) if x != y]
sets = [n for n in qs if set(a[n]) != set(b[n])]
print("KLOCAL_DIFF queries=%d k=%d positions=%d of %d  queries_touched=%d  "
      "different_set=%d" % (len(a), len(a[0]), pos, len(a) * len(a[0]),
                            len(qs), len(sets)))
for n in qs:
    print("   q%-4d kl4=%s" % (n, a[n]))
    print("        kl8=%s" % (b[n],))
PY
echo "=== KLOCAL""_DONE $(date -u +%FT%TZ) ==="
