#!/usr/bin/env python3
"""Residual-codebook convergence, and the isolation check that licenses it.

Reads the encode dumps a training-size sweep produced. Each carries the
rotated vectors, both levels' codes, the primary codebook, the residual
codebook and the IVF centroids.

Two questions, in this order:

1. Did anything except the residual codebook move? The primary codebook and
   the IVF centroids must be bit-identical across the sweep, or the sweep
   varied more than one thing and no recall difference can be attributed.

2. How far apart are the residual codebooks, and how many residual code
   assignments actually change on a fixed sample? Recall is a coarse
   instrument; a converged codebook is the direct evidence.
"""
import sys, struct, glob, os, numpy as np


def read(path):
    with open(path, "rb") as f:
        nc, d, M, Ds, K, Kr, Br, bpv = struct.unpack("8i", f.read(32))
        y    = np.frombuffer(f.read(4*nc*d), dtype=np.float32).reshape(nc, d)
        pc   = np.frombuffer(f.read(nc*M), dtype=np.uint8).reshape(nc, M)
        rc   = np.frombuffer(f.read(nc*bpv), dtype=np.uint8).reshape(nc, bpv)
        co   = np.frombuffer(f.read(4*nc), dtype=np.float32)
        cent = np.frombuffer(f.read(4*M*K*Ds), dtype=np.float32).reshape(M, K, Ds)
        rcb  = np.frombuffer(f.read(4*M*Kr), dtype=np.float32).reshape(M, Kr)
        tail = f.read()
    ivf = None
    if len(tail) >= 4:
        nlist = struct.unpack("i", tail[:4])[0]
        ivf = np.frombuffer(tail[4:4+4*nlist*d], dtype=np.float32).reshape(nlist, d)
    return dict(nc=nc, d=d, M=M, Ds=Ds, K=K, Kr=Kr, Br=Br, bpv=bpv,
                y=y, pc=pc, rc=rc, cent=cent, rcb=rcb, ivf=ivf)


def codes_from(rcb, y, cent, pc, M, Ds, Br, Kr):
    """Residual codes this codebook would assign to a fixed sample."""
    out = []
    yd, cd, rd = y.astype(np.float64), cent.astype(np.float64), rcb.astype(np.float64)
    for m in range(M):
        ym    = yd[:, m*Ds:(m+1)*Ds]
        resid = ym - cd[m][pc[:, m].astype(np.int64)]
        out.append(np.abs(resid[:, :, None] - rd[m][None, None, :]).argmin(axis=2))
    return np.concatenate(out, axis=1)     # [nc, M*Ds]


def main(ref_path, paths):
    R = read(ref_path)
    print(f"reference: {os.path.basename(ref_path)}  "
          f"d={R['d']} M={R['M']} Ds={R['Ds']} Br={R['Br']} Kr={R['Kr']}")
    ref_codes = codes_from(R["rcb"], R["y"], R["cent"], R["pc"],
                           R["M"], R["Ds"], R["Br"], R["Kr"])
    scale = float(np.abs(R["rcb"]).max())
    print()
    print(f"  {'variant':<22}{'prim cb':>10}{'IVF':>10}"
          f"{'mean |dc|':>12}{'max |dc|':>12}{'norm':>10}{'codes moved':>13}")
    for p in paths:
        A = read(p)
        same_cent = "same" if np.array_equal(A["cent"], R["cent"]) else "DIFFERS"
        if A["ivf"] is None or R["ivf"] is None:
            same_ivf = "n/a"
        else:
            same_ivf = "same" if np.array_equal(A["ivf"], R["ivf"]) else "DIFFERS"
        dc = np.abs(A["rcb"].astype(np.float64) - R["rcb"].astype(np.float64))
        a_codes = codes_from(A["rcb"], R["y"], R["cent"], R["pc"],
                             R["M"], R["Ds"], R["Br"], R["Kr"])
        moved = float((a_codes != ref_codes).mean())
        print(f"  {os.path.basename(p)[:22]:<22}{same_cent:>10}{same_ivf:>10}"
              f"{dc.mean():>12.3e}{dc.max():>12.3e}{dc.max()/scale:>10.3e}"
              f"{moved*100:>12.4f}%")
    print()
    print("  prim cb / IVF must read 'same': otherwise the sweep moved more than")
    print("  the residual training set and no difference is attributable to it.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
