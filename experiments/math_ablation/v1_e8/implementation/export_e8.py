#!/usr/bin/env python3
"""Raw files of an E8 representation for demo_jhq_v68_e8_bench (jhq_v68_e8::import_state), next to the codes.

  python3 export_e8.py <tag> [grouped]

Writes into MA/<tag>/v1_e8[_grouped]/: cent.f32 [M][256][8] (the scaled E8 codewords of every subspace),
res_cb.f32 [M][256] (sorted residual codebooks), corr.f32 [N] (2 <y_hat, r_hat> per vector, position order, as
JHQ's encoder stores it) and list_ids.i32 [N] (the offline position order, checked by the benchmark).
codes.u8 [N][M] and res.u8 [N][d] are already there (e8.py build).
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import ma_common as MC  # noqa: E402
from e8 import CB  # noqa: E402

C = MC.C


def main():
    tag = sys.argv[1]
    grouped = len(sys.argv) > 2 and sys.argv[2] == "grouped"
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    vd = os.path.join(MC.MA, tag, "v1_e8" + ("_grouped" if grouped else ""))
    pz = np.load(os.path.join(vd, "params.npz"))
    scales, cb = pz["scales"].astype(np.float64), pz["res_cb"].astype(np.float32)
    assert np.all(np.diff(cb, axis=1) >= 0), "residual codebooks must be sorted"
    cent = (scales[:, None, None] * CB[None, :, :]).astype(np.float32)          # [M][256][8]
    cent.tofile(os.path.join(vd, "cent.f32"))
    cb.tofile(os.path.join(vd, "res_cb.f32"))
    shutil.copyfile(os.path.join(C.RQ, tag, "list_ids.i32"), os.path.join(vd, "list_ids.i32"))
    codes = np.memmap(os.path.join(vd, "codes.u8"), dtype=np.uint8, mode="r", shape=(N, M))
    res = np.memmap(os.path.join(vd, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d))
    corr = np.empty(N, np.float32)
    off = (np.arange(M) * 256).repeat(8)[None, :]
    cbf = cb.reshape(-1)
    for a in range(0, N, 16384):
        cd = np.asarray(codes[a:a + 16384]).astype(np.int64)
        yh = cent[np.arange(M)[None, :], cd].reshape(len(cd), d)
        rh = cbf[np.asarray(res[a:a + 16384]).astype(np.int64) + off]
        corr[a:a + 16384] = 2.0 * np.einsum("ij,ij->i", yh.astype(np.float64), rh.astype(np.float64))
    corr.tofile(os.path.join(vd, "corr.f32"))
    info = dict(tag=tag, grouped=grouped, vardir=vd, corr_mean=float(corr.mean()), corr_abs_max=float(np.abs(corr).max()))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
