#!/usr/bin/env python3
"""Differential test for the one thing the faithful claim rests on.

Algorithm 1 line 4 selects the top-alpha*k by complete primary distance. This
reads the dump the search writes -- every candidate's complete primary distance
for one query, and the ck the selector returned -- and checks the second is the
exact top-ck of the first.

Both sides come from the same run and the same distances, so a mismatch is the
selection and nothing else. Recall is not consulted: it is an aggregate and a
changed result set can leave it untouched.

Ties are broken by ascending database position, in the reference and in the
report, and any tie at the ck-th distance is listed rather than counted as a
mismatch.
"""
import sys, struct, numpy as np

def read_dump(path):
    with open(path, "rb") as f:
        total, ck = struct.unpack("ii", f.read(8))
        ad = np.frombuffer(f.read(4 * total), dtype=np.float32)
        ap = np.frombuffer(f.read(4 * total), dtype=np.int32)
        sp = np.frombuffer(f.read(4 * ck), dtype=np.float32)
        sq = np.frombuffer(f.read(4 * ck), dtype=np.int32)
        tail = f.read()
    k = cd = fd = fi = None
    if len(tail) >= 4:
        k = struct.unpack("i", tail[:4])[0]
        off = 4
        cd = np.frombuffer(tail[off:off + 4 * ck], dtype=np.float32); off += 4 * ck
        fd = np.frombuffer(tail[off:off + 4 * k],  dtype=np.float32); off += 4 * k
        fi = np.frombuffer(tail[off:off + 4 * k],  dtype=np.int32)
    return total, ck, ad, ap, sp, sq, k, cd, fd, fi

def main(path):
    total, ck, ad, ap, sp, sq, k, cd, fd, fi = read_dump(path)
    print(f"  candidates evaluated : {total}")
    print(f"  ck (= ceil(alpha*k)) : {ck}")

    # Reference: exact top-ck by (distance, position), position breaking ties.
    order = np.lexsort((ap, ad))[:ck]
    ref_pos = ap[order]
    ref_val = ad[order]

    got = sq[sq >= 0]
    if len(got) != min(ck, total):
        print(f"  [FAIL] selector returned {len(got)} of an expected {min(ck,total)}")
        return 1

    ref_set, got_set = set(ref_pos.tolist()), set(got.tolist())
    missing = ref_set - got_set
    extra   = got_set - ref_set

    # A position missing only because it ties the ck-th distance is not an error:
    # the reference had to break that tie somehow too.
    cut = ref_val[-1]
    tied = {p for p in (missing | extra) if abs(float(ad[ap == p][0]) - cut) <= 0}
    hard_missing = missing - tied
    hard_extra   = extra - tied

    print(f"  reference cut distance: {cut:.6f}")
    print(f"  in reference, not returned : {len(missing)}  (of which tie at cut: {len(missing & tied)})")
    print(f"  returned, not in reference : {len(extra)}  (of which tie at cut: {len(extra & tied)})")

    if hard_missing or hard_extra:
        print(f"  [FAIL] top-alpha*k mismatch: {len(hard_missing)} missing, {len(hard_extra)} extra")
        for p in sorted(hard_missing)[:5]:
            print(f"         missing pos={p} dist={float(ad[ap==p][0]):.6f} < cut")
        return 1

    # The returned distances must also match what the dump says for those ids.
    idx = {int(p): i for i, p in enumerate(ap)}
    err = max(abs(float(sp[i]) - float(ad[idx[int(q)]]))
              for i, q in enumerate(got)) if len(got) else 0.0
    print(f"  max |returned - recomputed| primary distance: {err:.3e}")
    if err > 1e-3:
        print("  [FAIL] returned primary distances disagree with the dump")
        return 1

    print("  [PASS] top-alpha*k IDs        mismatched = 0")
    print("  [PASS] primary distances      max_abs_err = %.3e" % err)

    if k is None:
        print("  (no composite/final stages in this dump)")
        return 0

    # Stage G: the final top-k must be the exact top-k of the composite
    # distances the refinement produced, with ties broken by position as the
    # reference does.
    valid = sq >= 0
    order = np.lexsort((sq[valid], cd[valid]))[:k]
    ref_dist = cd[valid][order]
    got_dist = np.sort(fd)
    dmax = float(np.max(np.abs(np.sort(ref_dist) - got_dist))) if k else 0.0
    print(f"  final top-k: max |reference - returned| composite = {dmax:.3e}")
    if dmax > 1e-4:
        print("  [FAIL] final top-k is not the top-k of the composite distances")
        for i in range(min(k, 5)):
            print(f"         rank {i}: reference {np.sort(ref_dist)[i]:.6f} vs returned {got_dist[i]:.6f}")
        return 1
    print("  [PASS] final top-k            selected exactly from composite")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/root/topck.bin"))
