#!/usr/bin/env python3
"""Raw files of a quantize.py variant for the GPU prototype (demo_jhq_v67_repr_bench), next to its codes:
lo.f32, hi.f32 [d], res_cb.f32 [M][Kr], corr.f32 [N] = 2<y_hat, r_hat> (float32, as the paper index stores it).
bits.u8 and res.u8 are already in the layout import_state takes ([N][M] and [N][d], position order).

  python3 export_variant.py <tag> <variant>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402


def main():
    tag, v = sys.argv[1], sys.argv[2]
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    vd = os.path.join(C.RQ, tag, "var_" + v)
    pz = np.load(os.path.join(vd, "params.npz"))
    if bool(pz["per_list"]):
        raise SystemExit("per-list variants need per-list tables; the v67 import takes one codebook")
    lo, hi, cb = pz["lo"].astype(np.float32), pz["hi"].astype(np.float32), pz["res_cb"].astype(np.float32)
    lo.tofile(os.path.join(vd, "lo.f32")); hi.tofile(os.path.join(vd, "hi.f32")); cb.tofile(os.path.join(vd, "res_cb.f32"))
    bits = np.memmap(os.path.join(vd, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M))
    res = np.memmap(os.path.join(vd, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d))
    corr = np.empty(N, np.float32)
    for a in range(0, N, 131072):
        b = min(N, a + 131072)
        yh = np.where(np.unpackbits(np.asarray(bits[a:b]), axis=1).astype(bool), hi[None, :], lo[None, :])
        rh = C.decode_residual(np.asarray(res[a:b]), cb)
        corr[a:b] = 2.0 * (yh * rh).sum(1, dtype=np.float32)
    corr.tofile(os.path.join(vd, "corr.f32"))
    print("exported", vd, "corr mean %.4g sd %.4g" % (corr.mean(), corr.std()))


if __name__ == "__main__":
    main()
