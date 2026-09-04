#!/usr/bin/env python3
"""Differential test for the two build stages the search test cannot reach.

Reads what the encoder produced -- rotated vectors, primary codes, residual
codes -- together with the two codebooks it used, and recomputes both codes by
brute force from those same codebooks. Nothing is retrained, so a mismatch is
the encoder and not a difference of training.

Primary (paper §3.2): yhat^(m) = argmin over the K codewords of C^m. The GPU
takes a separable shortcut, valid because Eq. 4 makes C^m a Cartesian product,
so the argmin factorises per dimension. The reference does not take it: it
enumerates all K codewords. That is the point of the check.

Residual (paper §4.2, Eq. 5): r = y - yhat, quantised to the nearest codeword
of the shared scalar codebook C_R^m.

Exact equality is required. A mismatch is only excused where two codewords are
equidistant from the value, which is reported explicitly rather than folded
into a tolerance.
"""
import sys, struct, numpy as np


def read_dump(path):
    with open(path, "rb") as f:
        nc, d, M, Ds, K, Kr, Br, bpv = struct.unpack("8i", f.read(32))
        y    = np.frombuffer(f.read(4 * nc * d), dtype=np.float32).reshape(nc, d)
        pc   = np.frombuffer(f.read(nc * M), dtype=np.uint8).reshape(nc, M)
        rc   = np.frombuffer(f.read(nc * bpv), dtype=np.uint8).reshape(nc, bpv)
        co   = np.frombuffer(f.read(4 * nc), dtype=np.float32)
        cent = np.frombuffer(f.read(4 * M * K * Ds), dtype=np.float32).reshape(M, K, Ds)
        rcb  = np.frombuffer(f.read(4 * M * Kr), dtype=np.float32).reshape(M, Kr)
    return dict(nc=nc, d=d, M=M, Ds=Ds, K=K, Kr=Kr, Br=Br, bpv=bpv,
                y=y, pc=pc, rc=rc, co=co, cent=cent, rcb=rcb)


def check_primary(D):
    """Brute force over all K codewords per subspace, in float64."""
    nc, M, Ds, K = D["nc"], D["M"], D["Ds"], D["K"]
    y, cent, pc = D["y"].astype(np.float64), D["cent"].astype(np.float64), D["pc"]
    bad, ties, first = 0, 0, None
    for m in range(M):
        ym = y[:, m * Ds:(m + 1) * Ds]            # [nc, Ds]
        # [nc, K] squared distance to every codeword of this subspace
        dist = ((ym[:, None, :] - cent[m][None, :, :]) ** 2).sum(axis=2)
        ref = dist.argmin(axis=1)
        got = pc[:, m].astype(np.int64)
        diff = np.nonzero(ref != got)[0]
        for i in diff:
            # Equidistant codewords are a genuine tie, not an encoder error.
            if abs(dist[i, ref[i]] - dist[i, got[i]]) <= 0.0:
                ties += 1
                continue
            bad += 1
            if first is None:
                first = (int(i), m, int(got[i]), int(ref[i]),
                         float(dist[i, got[i]]), float(dist[i, ref[i]]))
    return bad, ties, first, nc * M


def check_residual(D):
    """yhat from the primary code, then nearest scalar codeword per dimension."""
    nc, M, Ds, Br, Kr, bpv = D["nc"], D["M"], D["Ds"], D["Br"], D["Kr"], D["bpv"]
    y, cent, pc, rc, rcb = (D["y"].astype(np.float64), D["cent"].astype(np.float64),
                            D["pc"], D["rc"], D["rcb"].astype(np.float64))
    bad, ties, first = 0, 0, None
    for m in range(M):
        ym    = y[:, m * Ds:(m + 1) * Ds]
        yhat  = cent[m][pc[:, m].astype(np.int64)]        # [nc, Ds]
        resid = ym - yhat
        dist  = (resid[:, :, None] - rcb[m][None, None, :]) ** 2   # [nc, Ds, Kr]
        ref   = dist.argmin(axis=2)
        for k in range(Ds):
            j = m * Ds + k
            if Br == 4:
                packed = rc[:, j // 2].astype(np.int64)
                got = np.where(j % 2 == 0, packed & 0x0F, packed >> 4)
            else:
                got = rc[:, j].astype(np.int64)
            diff = np.nonzero(ref[:, k] != got)[0]
            for i in diff:
                if abs(dist[i, k, ref[i, k]] - dist[i, k, got[i]]) <= 0.0:
                    ties += 1
                    continue
                bad += 1
                if first is None:
                    first = (int(i), j, int(got[i]), int(ref[i, k]))
    return bad, ties, first, nc * M * Ds


def main(path):
    D = read_dump(path)
    print(f"  shape: {D['nc']} vectors, d={D['d']} M={D['M']} Ds={D['Ds']} "
          f"K={D['K']} Kr={D['Kr']} Br={D['Br']}")

    bad, ties, first, total = check_primary(D)
    print(f"  primary codes checked   : {total}  ({D['nc']} vectors x {D['M']} subspaces)")
    print(f"  primary code mismatches : {bad}   (exact ties, not counted: {ties})")
    if first:
        i, m, g, r, dg, dr = first
        print(f"    first: vector {i} subspace {m}: encoder {g} (d={dg:.9f}) "
              f"vs reference {r} (d={dr:.9f})")
    rc_bad, rc_ties, rc_first, rc_total = check_residual(D)
    print(f"  residual codes checked  : {rc_total}")
    print(f"  residual code mismatches: {rc_bad}   (exact ties, not counted: {rc_ties})")
    if rc_first:
        i, j, g, r = rc_first
        print(f"    first: vector {i} dim {j}: encoder {g} vs reference {r}")

    ok = (bad == 0 and rc_bad == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] primary codes   mismatch = {bad}")
    print(f"  [{'PASS' if ok else 'FAIL'}] residual codes  mismatch = {rc_bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/root/enc.bin"))
