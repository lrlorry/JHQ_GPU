#!/usr/bin/env python3
"""What raising K_LOCAL from 4 to 8 does to the returned ids.

Section 4's caveat used to say: with the index pinned, vogue-768 at nprobe=128
and BLOCK=512, raising K_LOCAL from 4 to 8 changes 11 of 10,000 returned id
positions, touches 4 of 1,000 queries, and gives 1 query a different result
set.  Those numbers lived in the paper and in a comment in
jhq_v53_cap/search.cu, and nowhere else.  The archived K_LOCAL ablation,
results/pre_freeze_v22_s2b1/abl_klocal.csv, is at nprobe=256 and records only
recall and QPS -- it cannot say which ids came back.

data/klocal_ids/ is the diff, run at the cell the sentence names: v53_x1
(K_LOCAL=4) against v53_kl8, one pinned index cache, ids dumped by
demo_jhq_v36's out_prefix.  Both arms return Recall@10 of 0.9647.

It does not reproduce, and it fails in the direction that makes the caveat
weaker than published: **no query gets a different result set**.  Every
difference is two adjacent ranks swapping inside an otherwise identical top-10.
The bounded selector discarded nothing on this cell; it reordered ties.

The published 11/4/1 was measured on a different index instance.  Training is
not reproducible on this box -- three cold starts of one binary give
0.9852/0.9842/0.9849 -- so which items sit at equal distance moves with the
codebook, and a count of ten positions out of ten thousand moves with it.  The
right statement is the one this run supports, with the sensitivity named.
"""
import glob
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style


def read_ivecs(p):
    out = []
    with open(p, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack("<i", h)[0]
            out.append(list(struct.unpack("<%di" % d, f.read(4 * d))))
    return out


def main():
    d = os.path.join(style.DATA, "klocal_ids")
    a_p, b_p = os.path.join(d, "kl4.ivecs"), os.path.join(d, "kl8.ivecs")
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        print("  klocal_ids/ not present"); return
    a, b = read_ivecs(a_p), read_ivecs(b_p)
    k = len(a[0])
    pos = sum(1 for x, y in zip(a, b) for i, j in zip(x, y) if i != j)
    qs = [n for n, (x, y) in enumerate(zip(a, b)) if x != y]
    sets = [n for n in qs if set(a[n]) != set(b[n])]
    rec = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        rec.append((os.path.basename(f)[:-5], j["eval"]["recall_at_k"],
                    j["perf"]["qps"], j["params"]["nprobe"]))
    for nm, r, q, np_ in rec:
        print("  %-4s recall=%.4f qps=%.0f nprobe=%d" % (nm, r, q, np_))
    print("  K_LOCAL 4 against 8, %d queries at k=%d:" % (len(a), k))
    print("     %d of %d id positions change" % (pos, len(a) * k))
    print("     %d queries touched" % len(qs))
    print("     %d queries get a different result set" % len(sets))
    if qs and not sets:
        print("     every difference is a swap of adjacent ranks inside an "
              "identical top-%d" % k)


if __name__ == "__main__":
    main()
