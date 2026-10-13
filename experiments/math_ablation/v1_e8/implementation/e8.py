#!/usr/bin/env python3
"""V1 E8-JHQ: a 256-word E8 primary codebook per 8-D subspace (8 bits, the budget of JHQ's {-c, +c}^8).

  python3 e8.py test  <out dir>                   codebook checks, encoder consistency, Gaussian MSE (no data)
  python3 e8.py build <out dir> <tag> [grouped]   encode the base, scale(s) trained on the residual training sample

Codebook C_E8 (THEORY.md 1): the 240 roots of E8 in the even coordinate system,
  112 integer roots     (+-1, +-1, 0^6) over all positions and signs                       norm^2 2
  128 half-integer roots (+-1/2)^8 with an even number of minus signs                      norm^2 2
plus the 16 axis points +-2 e_i of the second shell                                         norm^2 4,
all in E8 = {x in Z^8 u (Z + 1/2)^8 : sum x_i = 0 mod 2}. Codewords: s * C_E8 with a scale s fitted on training
vectors only (global, or one per subspace with "grouped").

Encoders: brute force over the 256 words (reference), and the structured one: the nearest word of each norm
class is the class's maximum inner product (two largest |x_i| with their signs; sign(x) with the parity fixed on
the smallest |x_i|; the largest |x_i|), then the three candidates are compared by full distance, since the
norms differ (4 for the axis points) and nearest != largest inner product across classes.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import ma_common as MC  # noqa: E402

C = MC.C


def e8_codebook():
    words = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    w = np.zeros(8); w[i] = si; w[j] = sj
                    words.append(w)
    for m in range(256):
        signs = np.array([-1.0 if (m >> k) & 1 else 1.0 for k in range(8)])
        if (signs < 0).sum() % 2 == 0:
            words.append(0.5 * signs)
    for i in range(8):
        for s in (2.0, -2.0):
            w = np.zeros(8); w[i] = s
            words.append(w)
    return np.array(words)


def check_codebook(Cb):
    n = len(Cb)
    keys = {tuple(np.round(2 * w).astype(int)) for w in Cb}
    twice = np.round(2 * Cb)
    integer = np.all(twice % 2 == 0, axis=1)
    half = np.all(np.abs(twice % 2) == 1, axis=1)
    in_e8 = (integer | half) & (np.round(Cb.sum(1)) % 2 == 0) & np.allclose(2 * Cb, twice)
    nrm = (Cb ** 2).sum(1)
    d2 = ((Cb[:, None, :] - Cb[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return dict(words=n, unique=len(keys), all_in_E8=bool(in_e8.all()),
                norm2_counts={str(k): int(v) for k, v in zip(*np.unique(np.round(nrm, 6), return_counts=True))},
                integer_roots=int((integer & (np.abs(nrm - 2) < 1e-9)).sum()), half_integer_roots=int((half & (np.abs(nrm - 2) < 1e-9)).sum()),
                min_distance2=float(d2.min()), symmetric=bool(len({tuple(np.round(2 * w).astype(int)) for w in -Cb} & keys) == n))


CB = e8_codebook()
CB_NORM2 = (CB ** 2).sum(1)
_KEYS = np.round(2 * CB).astype(np.int64) + 4                      # entries in {0, 2, 3, 4, 5, 6, 8}
_KEYV = (_KEYS * (9 ** np.arange(8))[None, :]).sum(1)
_ORDER = np.argsort(_KEYV)
_SORTED = _KEYV[_ORDER]


def index_of(words):
    k = ((np.round(2 * words).astype(np.int64) + 4) * (9 ** np.arange(8))[None, :]).sum(1)
    i = np.searchsorted(_SORTED, k)
    assert np.all(_SORTED[np.minimum(i, 255)] == k), "not a codeword"
    return _ORDER[i].astype(np.uint8)


def encode_brute(X, s, chunk=65536):
    """reference: argmin_k ||x - s C_k||^2 (first index on ties), X [n, 8]."""
    out = np.empty(len(X), np.uint8)
    for a in range(0, len(X), chunk):
        x = np.asarray(X[a:a + chunk], dtype=np.float64)
        d = (x ** 2).sum(1)[:, None] - 2 * s * x @ CB.T + s * s * CB_NORM2[None, :]
        out[a:a + chunk] = d.argmin(1)
    return out


def encode_fast(X, s):
    """structured nearest codeword, X [n, 8] -> codes [n] (uint8)."""
    u = np.asarray(X, dtype=np.float64) / s
    n = len(u)
    a = np.abs(u)
    sg = np.where(u < 0, -1.0, 1.0)
    rows = np.arange(n)
    # class 1: (+-1, +-1, 0^6), inner product |u_i1| + |u_i2|
    top2 = np.argpartition(-a, 1, axis=1)[:, :2]
    w1 = np.zeros_like(u)
    w1[rows, top2[:, 0]] = sg[rows, top2[:, 0]]
    w1[rows, top2[:, 1]] = sg[rows, top2[:, 1]]
    ip1 = a[rows, top2[:, 0]] + a[rows, top2[:, 1]]
    # class 2: (+-1/2)^8, even number of minus signs: sign(u), parity fixed on the smallest |u_i|
    eps = sg.copy()
    odd = (eps < 0).sum(1) % 2 == 1
    jmin = a.argmin(1)
    eps[rows[odd], jmin[odd]] *= -1
    w2 = 0.5 * eps
    ip2 = (w2 * u).sum(1)
    # class 3: +-2 e_i
    jmax = a.argmax(1)
    w3 = np.zeros_like(u)
    w3[rows, jmax] = 2 * sg[rows, jmax]
    ip3 = 2 * a[rows, jmax]
    # compare by distance (norms 2, 2, 4): ||u||^2 - 2 ip + norm^2
    dist = np.stack([2 - 2 * ip1, 2 - 2 * ip2, 4 - 2 * ip3], 1)
    pick = dist.argmin(1)
    W = np.where((pick == 0)[:, None], w1, np.where((pick == 1)[:, None], w2, w3))
    return index_of(W)


def fit_scale(X, iters=30, chunk=1 << 20):
    """Lloyd for a scale-only codebook: codes for s, then s = sum <x, C_code> / sum ||C_code||^2 (MSE never rises).
    Chunked: X can hold tens of millions of 8-D rows."""
    n = len(X)
    x2 = sum(float((np.asarray(X[a:a + chunk], dtype=np.float64) ** 2).sum()) for a in range(0, n, chunk))
    s = np.sqrt(x2 / n / CB_NORM2.mean())
    hist = []
    for _ in range(iters):
        xw = ww = 0.0
        for a in range(0, n, chunk):
            x = np.asarray(X[a:a + chunk], dtype=np.float64)
            W = CB[encode_fast(x, s)]
            xw += float((x * W).sum()); ww += float((W ** 2).sum())
        s_new = xw / ww
        hist.append(float((x2 - 2 * s_new * xw + s_new * s_new * ww) / n))   # MSE of the codes for s, at s_new
        if abs(s_new - s) <= 1e-10 * s:
            s = s_new
            break
        s = s_new
    return s, hist


def cube_mse(X, c):
    X = np.asarray(X, dtype=np.float64)
    return float(((X - c * np.where(X > 0, 1.0, -1.0)) ** 2).sum(1).mean())


def test(out):
    os.makedirs(out, exist_ok=True)
    rep = dict(codebook=check_codebook(CB))
    rng = np.random.default_rng(20261013)
    # encoder consistency: fast vs brute on Gaussian, heavy-tailed and on-codeword inputs
    res = {}
    for name, X in (("gauss", rng.standard_normal((200000, 8))), ("student3", rng.standard_t(3, (200000, 8))),
                    ("codewords_plus_noise", CB[rng.integers(0, 256, 50000)] + 1e-3 * rng.standard_normal((50000, 8)))):
        for s in (0.5, 1.0, 1.7):
            fb, ff = encode_brute(X, s), encode_fast(X, s)
            db = ((X - s * CB[fb]) ** 2).sum(1); df = ((X - s * CB[ff]) ** 2).sum(1)
            res["%s_s%.1f" % (name, s)] = dict(n=len(X), index_agreement=float((fb == ff).mean()),
                                              max_distance_gap=float(np.abs(db - df).max()),
                                              distance_ties_where_indices_differ=int((fb != ff).sum()))
    rep["encoder_consistency"] = res
    # nearest != largest inner product across norm classes
    X = rng.standard_normal((200000, 8))
    s, _ = fit_scale(X)
    near = encode_fast(X, s)
    ipmax = (X @ CB.T).argmax(1)
    rep["nearest_vs_max_inner_product"] = dict(scale=s, disagree=float((near != ipmax).mean()))
    # Gaussian MSE per dimension: JHQ's cube at its level 0.6745 sigma, the cube at the 1-bit Lloyd-Max level
    # sqrt(2/pi) sigma, E8 with a fitted scale (independent samples, fit and test separate)
    Xf, Xt = rng.standard_normal((400000, 8)), rng.standard_normal((400000, 8))
    s, hist = fit_scale(Xf)
    e8 = float(((Xt - s * CB[encode_fast(Xt, s)]) ** 2).sum(1).mean()) / 8
    rep["gaussian_mse_per_dim"] = dict(cube_jhq_c06745=cube_mse(Xt, 0.6744897501960817) / 8,
                                       cube_lloydmax_c0798=cube_mse(Xt, np.sqrt(2 / np.pi)) / 8,
                                       e8_fitted=e8, e8_scale=s, e8_scale_fit_mse_history=hist,
                                       e8_vs_cube_lloydmax=e8 / (cube_mse(Xt, np.sqrt(2 / np.pi)) / 8),
                                       e8_codeword_use_share_axis=float((encode_fast(Xt, s) >= 240).mean()))
    json.dump(rep, open(os.path.join(out, "e8_test.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    ok = rep["codebook"]["unique"] == 256 and rep["codebook"]["all_in_E8"] and all(v["max_distance_gap"] < 1e-9 for v in res.values())
    print("E8_TEST", "OK" if ok else "FAILED")
    return ok


def build(out, tag, grouped=False):
    """codes [N][M] uint8 (position order) for the E8 primary; residual retrained with JHQ's procedure."""
    t0 = time.time()
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    xp = C.xpos(tag)
    ptr = MC.residual_sample_positions(tag)
    ytr = np.asarray(xp[ptr], dtype=np.float32)
    if grouped:
        scales = np.array([fit_scale(ytr[:, k * 8:(k + 1) * 8])[0] for k in range(M)])
    else:
        s, _ = fit_scale(ytr.reshape(-1, 8))
        scales = np.full(M, s)
    vd = os.path.join(MC.MA, tag, "v1_e8" + ("_grouped" if grouped else ""))
    os.makedirs(vd, exist_ok=True)
    # residual training sample with the E8 primary
    rtr = np.empty(ytr.shape, np.float32)
    for k in range(M):
        sl = slice(k * 8, (k + 1) * 8)
        rtr[:, sl] = ytr[:, sl] - scales[k] * CB[encode_fast(ytr[:, sl], scales[k])]
    cb, iters = C.train_residual(rtr, M)
    del rtr
    codes = np.memmap(os.path.join(vd, "codes.u8"), dtype=np.uint8, mode="w+", shape=(N, M))
    res = np.memmap(os.path.join(vd, "res.u8"), dtype=np.uint8, mode="w+", shape=(N, d))
    mse_p = mse_h = mse_cube = 0.0
    lev = float(np.float32(m["levels"][1]))
    STEP = max(4096, (1 << 24) // d)
    t_enc = 0.0
    for a in range(0, N, STEP):
        b = min(N, a + STEP)
        y = np.asarray(xp[a:b], dtype=np.float64)
        te = time.time()
        cd = np.stack([encode_fast(y[:, k * 8:(k + 1) * 8], scales[k]) for k in range(M)], 1)
        t_enc += time.time() - te
        yh = np.concatenate([scales[k] * CB[cd[:, k]] for k in range(M)], 1)
        r = (y - yh).astype(np.float32)
        rc = C.encode_residual(r, cb)
        codes[a:b] = cd; res[a:b] = rc
        mse_p += float(((y - yh) ** 2).sum())
        mse_h += float(((r - C.decode_residual(rc, cb)).astype(np.float64) ** 2).sum())
        mse_cube += float(((y - lev * np.where(y > 0, 1.0, -1.0)) ** 2).sum())
    codes.flush(); res.flush()
    np.savez(os.path.join(vd, "params.npz"), scales=scales, res_cb=cb, codebook=CB)
    use = np.bincount(np.asarray(codes[::97]).ravel(), minlength=256)
    info = dict(tag=tag, grouped=grouped, N=N, d=d, M=M, scales_min_med_max=[float(scales.min()), float(np.median(scales)), float(scales.max())],
                jhq_level=lev, primary_mse_per_vector=mse_p / N, jhq_primary_mse_per_vector=mse_cube / N,
                primary_mse_ratio_e8_over_jhq=mse_p / mse_cube, hierarchical_mse_per_vector=mse_h / N,
                encode_seconds_cpu=t_enc, codeword_use_entropy_bits=float(-(use / use.sum() * np.log2(np.maximum(use / use.sum(), 1e-300))).sum()),
                axis_word_share=float(use[240:].sum() / use.sum()), lloyd_iterations=[int(min(iters)), int(np.median(iters)), int(max(iters))],
                seconds=time.time() - t0)
    json.dump(info, open(os.path.join(vd, "info.json"), "w"), indent=1)
    json.dump(info, open(os.path.join(out, "build_%s%s.json" % (tag, "_grouped" if grouped else "")), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "test":
        sys.exit(0 if test(out) else 1)
    build(out, sys.argv[3], grouped=len(sys.argv) > 4 and sys.argv[4] == "grouped")
