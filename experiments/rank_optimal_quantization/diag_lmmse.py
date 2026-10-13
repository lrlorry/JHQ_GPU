#!/usr/bin/env python3
"""Derivation check for a candidate dropped before any experiment: the correlation-aware (linear MMSE,
Bussgang/arcsine-law) decoder of the sign bits.

  python3 diag_lmmse.py <results dir> [variant ...]

For a code b (the paper index's own bits, or a variant's), the linear MMSE estimate of the rotated vector is
y_hat(b) = m + B (b - p), B = Cov(y, b) Cov(b)^-1 (a d x d matrix, global). Its inner product with the query is
linear in the bits, so it would keep the table structure (weights B^T q'). The derivation (THEORY.md 7) says that for
centred Gaussian coordinates Cov(b) and Cov(y, b) share their correlation structure to first order (arcsine law),
so B is close to diagonal and the decoder adds nothing over the per-coordinate conditional means. Measured here on a
database sample (fit on 100K vectors, scored on 50K others): the off-diagonal share of B, and the reconstruction
MSE and inner-product estimation error of the LMMSE decoder against the per-coordinate decoder. Output
lmmse_check.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402


def main():
    R = sys.argv[1]
    variants = sys.argv[2:]
    out = {}
    for tag in C.TAGS:
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        xp = C.xpos(tag)
        rng = np.random.default_rng(C.SEED + 7)
        idx = rng.choice(N, 150000, replace=False)
        fit, ev = np.sort(idx[:100000]), np.sort(idx[100000:])
        codes = {"JHQ": np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M))}
        for v in variants:
            codes[v] = np.memmap(os.path.join(C.RQ, tag, "var_" + v, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M))
        yf = np.asarray(xp[fit], dtype=np.float64); ye = np.asarray(xp[ev], dtype=np.float64)
        mu = yf.mean(0)
        out[tag] = {}
        for name, mm in codes.items():
            bf = np.unpackbits(np.asarray(mm[fit]), axis=1).astype(np.float64)
            be = np.unpackbits(np.asarray(mm[ev]), axis=1).astype(np.float64)
            p = bf.mean(0)
            Zf = bf - p
            Cbb = Zf.T @ Zf / len(Zf)
            Cyb = (yf - mu).T @ Zf / len(Zf)
            lam = 1e-6 * np.trace(Cbb) / d
            B = np.linalg.solve(Cbb + lam * np.eye(d), Cyb.T).T              # Cyb Cbb^-1
            diag = np.diag(np.diag(B))
            off = float(np.linalg.norm(B - diag) ** 2 / np.linalg.norm(B) ** 2)
            # per-coordinate decoder: conditional means of y given each bit (fit sample)
            hi = np.where(bf > 0, yf, 0).sum(0) / np.maximum(bf.sum(0), 1)
            lo = np.where(bf == 0, yf, 0).sum(0) / np.maximum((1 - bf).sum(0), 1)
            y_pc = np.where(be > 0, hi, lo)
            y_lm = mu + (be - p) @ B.T
            mse_pc = float(((ye - y_pc) ** 2).sum(1).mean()); mse_lm = float(((ye - y_lm) ** 2).sum(1).mean())
            # inner-product estimation against 200 held-out "queries" (other database vectors)
            qv = ye[:200]
            ip_true = qv @ ye[200:20200].T
            e_pc = float(np.mean((qv @ y_pc[200:20200].T - ip_true) ** 2))
            e_lm = float(np.mean((qv @ y_lm[200:20200].T - ip_true) ** 2))
            out[tag][name] = dict(offdiag_share_of_B=off, recon_mse_per_coordinate_decoder=mse_pc, recon_mse_lmmse=mse_lm,
                                  recon_mse_gain=1 - mse_lm / mse_pc, ip_mse_per_coordinate=e_pc, ip_mse_lmmse=e_lm,
                                  ip_mse_gain=1 - e_lm / e_pc)
            print(tag, name, json.dumps(out[tag][name]), flush=True)
    json.dump(out, open(os.path.join(R, "lmmse_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
