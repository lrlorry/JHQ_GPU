#!/usr/bin/env python3
"""Build a primary-quantizer variant over the whole base, with a residual trained and encoded the paper
index's way (roq_common.train_residual / encode_residual: per-subspace 256-level scalar Lloyd on the same
100K-vector strided sample), so primary and residual are always consistent.

  python3 quantize.py <tag> <variant> [<variant> ...]

Variants (all 1 bit per coordinate; per-vector storage identical to JHQ's: d bits + 8 d bits + nothing else):
  jhq_re     t_j = 0, levels +-c shared (c = the paper index's level): the re-implementation check
  lloydmax   per coordinate, the reconstruction-optimal 2-level Lloyd-Max quantizer (threshold = midpoint
             of the two conditional means, iterated)
  median     t_j = median of y_j, levels = conditional means on each side (rank-optimal threshold under the
             symmetric location model, THEORY.md 5)
  mean       t_j = mean of y_j, levels = conditional means
  list_mean  per IVF list l: t_lj = mean of y_j over the list's members, levels = conditional means within
             the list (THEORY.md 5.6); metadata nlist x d x 3
Statistics come from the database only: the strided training sample for per-coordinate rules, each list's
own members for per-list rules. Output RQ/<tag>/var_<variant>/: bits.u8 [N][M], res.u8 [N][d] (position
order), params.npz.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402

STEP = 131072


def cond_means(y, t):
    """per column: mean of the values above t and at or below t (t broadcasts)."""
    above = y > t
    na = np.maximum(above.sum(0), 1)
    nb = np.maximum((~above).sum(0), 1)
    hi = np.where(above, y, 0).sum(0) / na
    lo = np.where(~above, y, 0).sum(0) / nb
    return lo, hi


def fit_global(kind, ytr, c):
    d = ytr.shape[1]
    if kind == "jhq_re":
        return np.zeros(d), np.full(d, -c), np.full(d, c)
    if kind == "median":
        t = np.median(ytr, axis=0)
    elif kind == "mean":
        t = ytr.mean(0)
    elif kind == "lloydmax":
        t = np.median(ytr, axis=0)
        for _ in range(100):
            lo, hi = cond_means(ytr, t)
            tn = 0.5 * (lo + hi)
            if np.abs(tn - t).max() < 1e-9:
                break
            t = tn
    lo, hi = cond_means(ytr, t)
    return t, lo, hi


def fit_list(kind, xp, off):
    nl = len(off) - 1
    d = xp.shape[1]
    T = np.zeros((nl, d), np.float32); LO = np.zeros_like(T); HI = np.zeros_like(T)
    for l in range(nl):
        a, b = int(off[l]), int(off[l + 1])
        if b == a:
            continue
        y = np.asarray(xp[a:b], dtype=np.float64)
        t = y.mean(0) if kind == "list_mean" else np.median(y, axis=0)
        lo, hi = cond_means(y, t)
        # a coordinate whose list members all fall on one side keeps a usable second level
        lo = np.where((y <= t).sum(0) == 0, t - (hi - t), lo)
        hi = np.where((y > t).sum(0) == 0, t + (t - lo), hi)
        T[l], LO[l], HI[l] = t, lo, hi
    return T, LO, HI


def build(tag, kind):
    t0 = time.time()
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    c = float(np.float32(m["levels"][1]))
    xp = C.xpos(tag)
    ids = np.fromfile(os.path.join(C.RQ, tag, "list_ids.i32"), dtype=np.int32)
    inv = np.empty(N, np.int64); inv[ids] = np.arange(N)
    ptr = np.sort(inv[C.res_sample_rows(N)])                      # the residual training sample, by position
    ytr = np.asarray(xp[ptr], dtype=np.float64)
    out = os.path.join(C.RQ, tag, "var_" + kind)
    os.makedirs(out, exist_ok=True)
    per_list = kind.startswith("list_")
    if per_list:
        off = C.list_offsets(tag)
        T, LO, HI = fit_list(kind, xp, off)
        lpos = C.list_of_pos(tag)
        thr = lambda a, b: T[lpos[a:b]]
        lev = lambda a, b, bits: np.where(bits, HI[lpos[a:b]], LO[lpos[a:b]])
        lt = lpos[ptr]
        ytr_hat = np.where(ytr > T[lt], HI[lt], LO[lt])
    else:
        t, lo, hi = fit_global(kind, ytr, c)
        T, LO, HI = t.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)
        thr = lambda a, b: T[None, :]
        lev = lambda a, b, bits: np.where(bits, HI[None, :], LO[None, :])
        ytr_hat = np.where(ytr > T[None, :], HI[None, :], LO[None, :])
    cb, iters = C.train_residual((ytr - ytr_hat).astype(np.float32), M)
    bits_mm = np.memmap(os.path.join(out, "bits.u8"), dtype=np.uint8, mode="w+", shape=(N, M))
    res_mm = np.memmap(os.path.join(out, "res.u8"), dtype=np.uint8, mode="w+", shape=(N, d))
    ones = 0.0; mse_p = 0.0; mse_h = 0.0
    for a in range(0, N, STEP):
        b = min(N, a + STEP)
        y = np.asarray(xp[a:b])
        bits = y > thr(a, b)
        yh = lev(a, b, bits).astype(np.float32)
        r = y - yh
        rc = C.encode_residual(r, cb)
        bits_mm[a:b] = np.packbits(bits, axis=1)
        res_mm[a:b] = rc
        ones += bits.sum()
        mse_p += float(((y - yh).astype(np.float64) ** 2).sum())
        mse_h += float(((r - C.decode_residual(rc, cb)).astype(np.float64) ** 2).sum())
    bits_mm.flush(); res_mm.flush()
    np.savez(os.path.join(out, "params.npz"), kind=kind, per_list=per_list, thr=T, lo=LO, hi=HI, res_cb=cb)
    info = dict(tag=tag, kind=kind, per_list=per_list, seconds=time.time() - t0, bit_one_fraction=ones / (N * d),
                primary_mse_per_vector=mse_p / N, hierarchical_mse_per_vector=mse_h / N,
                lloyd_iterations=[int(min(iters)), int(np.median(iters)), int(max(iters))],
                metadata_floats=int(T.size + LO.size + HI.size + cb.size))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    for kind in sys.argv[2:]:
        build(sys.argv[1], kind)
