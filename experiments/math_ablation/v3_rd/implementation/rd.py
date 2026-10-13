#!/usr/bin/env python3
"""V3 RD-JHQ: residual bits allocated across subspaces under sum_m 8 b_m = 8 d (the same total as JHQ's 8 bits per
dimension), with the primary unchanged (JHQ's 1-bit code), codebooks retrained with JHQ's 1-D Lloyd at 2^b_m
levels, and the codes packed at their real variable widths into the same d bytes per vector.

  python3 rd.py test  <out>                 pack / unpack round trip on random widths and codes (no data)
  python3 rd.py stats <out> <tag>           residual statistics, distortion-rate table D_m(b), allocations
  python3 rd.py build <out> <tag> <alloc>   encode + pack the base for one allocation (uniform, waterfill,
                                            empirical, queryweighted)
  python3 rd.py eval  <out> [--test]        recall against T on the probed candidates (JHQ's primary order, each
                                            allocation's D_H), against the paper index and the uniform rebuild

Allocation per subspace m (8 coordinates share one codebook, as in JHQ), b_m in {6, ..., 10}:
  uniform        b_m = 8
  waterfill      reverse water-filling on the residual variances: b_m = [1/2 log2(lambda_m / theta)], theta set so the
                 sum is 8 M, rounded under the constraint
  empirical      greedy on the measured distortion-rate table: start at 6 and give each next bit to the subspace with
                 the largest drop D_m(b) - D_m(b + 1), until sum b_m = 8 M (optimal for convex D_m)
  queryweighted  the same greedy on w_m D_m(b), w_m = mean over calibration queries of sum_{j in m} (q'_j - y_j)^2 over
                 the query's 100 nearest candidates by the paper index's D_H (no ground truth read)
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import ma_common as MC  # noqa: E402

C = MC.C
BMIN, BMAX = 6, 10


# ------------------------------------------------------------------ packing
def offsets(widths):
    return np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)


def pack(codes, widths):
    """codes [n, d] (each < 2^width_j) -> [n, sum(widths) / 8] bytes, MSB first, coordinates in order."""
    n, d = codes.shape
    tot = int(widths.sum())
    assert tot % 8 == 0
    o = offsets(widths)
    bits = np.zeros((n, tot), np.uint8)
    c = codes.astype(np.int64)
    for k in range(int(widths.max())):
        j = np.where(widths > k)[0]
        bits[:, o[j] + k] = (c[:, j] >> (widths[j] - 1 - k)) & 1
    return np.packbits(bits, axis=1)


def unpack(packed, widths):
    """[n, sum(widths) / 8] bytes -> codes [n, len(widths)]. A code at stream bit o with width w lies inside the 24-bit
    window of bytes o // 8 .. o // 8 + 2 (o % 8 <= 7 and w <= 10, so o % 8 + w <= 17): one shift and one mask."""
    o = offsets(widths)
    n = packed.shape[0]
    p = np.concatenate([packed, np.zeros((n, 2), np.uint8)], axis=1).astype(np.int32)
    by = o // 8
    win = (p[:, by] << 16) | (p[:, by + 1] << 8) | p[:, by + 2]
    return (win >> (24 - (o % 8) - widths)[None, :]) & ((1 << widths) - 1)[None, :]


def test(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261013)
    rep = {}
    for trial in range(20):
        M = int(rng.integers(4, 64))
        b = rng.integers(BMIN, BMAX + 1, M)
        # force the total to 8 M by moving bits between subspaces
        while b.sum() != 8 * M:
            i = rng.integers(0, M)
            if b.sum() > 8 * M and b[i] > BMIN:
                b[i] -= 1
            elif b.sum() < 8 * M and b[i] < BMAX:
                b[i] += 1
        w = np.repeat(b, 8)
        codes = rng.integers(0, 1 << w, (257, len(w)))
        p = pack(codes, w)
        ok = p.shape[1] == len(w) and np.array_equal(unpack(p, w), codes)
        rep["trial%d" % trial] = dict(M=M, bytes_per_vector=int(p.shape[1]), d=int(len(w)), roundtrip=bool(ok))
    allok = all(v["roundtrip"] and v["bytes_per_vector"] == v["d"] for v in rep.values())
    json.dump(rep, open(os.path.join(out, "rd_pack_test.json"), "w"), indent=1)
    print("RD_PACK_TEST", "OK" if allok else "FAILED")
    return allok


# ------------------------------------------------------------------ quantizer pieces
def nearest(cb, v):
    mids = 0.5 * (cb[:-1] + cb[1:])
    return np.searchsorted(mids, v, side="left").astype(np.uint16)


def jhq_primary(tag, pos):
    """JHQ's primary reconstruction c * s for the base rows at positions pos (paper index's codes)."""
    m = C.meta(tag)
    c = float(np.float32(m["levels"][1]))
    bits = np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(m["N"], m["M"]))
    return c * (2.0 * np.unpackbits(np.asarray(bits[pos]), axis=1) - 1.0)


def heldout_positions(tag):
    m = C.meta(tag)
    ids = np.fromfile(os.path.join(C.RQ, tag, "list_ids.i32"), dtype=np.int32)
    inv = np.empty(m["N"], np.int64); inv[ids] = np.arange(m["N"])
    stride = m["N"] // C.RES_TRAIN_N
    return np.sort(inv[np.arange(C.RES_TRAIN_N, dtype=np.int64) * stride + stride // 2])


def _dr_job(args):
    k, rtr, rho = args
    row, cbs = [], {}
    for b in range(BMIN, BMAX + 1):
        cb, it = C.lloyd_1d(rtr, 1 << b)
        e = rho - cb[nearest(cb, rho)]
        row.append(float((e.astype(np.float64) ** 2).mean()))
        cbs[b] = cb
    return k, row, cbs


def residual_samples(tag):
    xp = C.xpos(tag)
    ptr, pho = MC.residual_sample_positions(tag), heldout_positions(tag)
    ytr, yho = np.asarray(xp[ptr], dtype=np.float64), np.asarray(xp[pho], dtype=np.float64)
    return (ytr - jhq_primary(tag, ptr)).astype(np.float32), (yho - jhq_primary(tag, pho)).astype(np.float32)


def greedy(D, weights, M):
    b = np.full(M, BMIN)
    gain = lambda k: weights[k] * (D[k][b[k] - BMIN] - D[k][b[k] - BMIN + 1]) if b[k] < BMAX else -np.inf
    while b.sum() < 8 * M:
        g = np.array([gain(k) for k in range(M)])
        b[int(np.argmax(g))] += 1
    return b


def waterfill(lam, M):
    lo, hi = 1e-30, float(lam.max())
    for _ in range(200):
        th = np.sqrt(lo * hi)
        s = np.clip(0.5 * np.log2(lam / th), BMIN, BMAX).sum()
        lo, hi = (th, hi) if s > 8 * M else (lo, th)
    x = np.clip(0.5 * np.log2(lam / th), BMIN, BMAX)
    b = np.clip(np.round(x), BMIN, BMAX).astype(int)
    frac = x - b
    while b.sum() > 8 * M:
        i = int(np.argmin(np.where(b > BMIN, frac, np.inf))); b[i] -= 1; frac[i] += 1
    while b.sum() < 8 * M:
        i = int(np.argmax(np.where(b < BMAX, frac, -np.inf))); b[i] += 1; frac[i] -= 1
    return b


def stats(out, tag):
    t0 = time.time()
    m = C.meta(tag)
    d, M = m["d"], m["M"]
    rtr, rho = residual_samples(tag)
    var_dim = rho.astype(np.float64).var(0)
    var_sub = var_dim.reshape(M, 8).mean(1)
    kurt = ((rho - rho.mean(0)) ** 4).mean(0) / np.maximum(var_dim, 1e-300) ** 2 - 3
    am_gm = lambda v: float(v.mean() / np.exp(np.log(v).mean()))
    pool = get_context("fork").Pool(min(48, M))
    res = pool.map(_dr_job, [(k, rtr[:, k * 8:(k + 1) * 8], rho[:, k * 8:(k + 1) * 8]) for k in range(M)])
    pool.close(); pool.join()
    res.sort(key=lambda r: r[0])
    D = np.array([r[1] for r in res])                 # [M, 5] held-out MSE per coordinate
    # query weights from calibration queries (no labels): their 100 nearest candidates by the paper index's D_H
    files = C.bin_files(tag)
    qids = [int(os.path.basename(f)[4:-4]) for f in files]
    sp = C.splits(qids)
    qrot = C.qrot(tag)
    xp = C.xpos(tag)
    w = np.zeros(M); nq = 0
    for f, x in zip(files, qids):
        if sp[x] != "calib":
            continue
        q = C.load_bin(f)
        top = q["pos"][np.lexsort((q["pos"], q["DH"]))[:100]].astype(np.int64)
        diff = qrot[x][None, :] - np.asarray(xp[np.sort(top)], dtype=np.float64)
        w += (diff ** 2).reshape(len(top), M, 8).sum(2).mean(0); nq += 1
    w /= nq
    allocs = dict(uniform=np.full(M, 8), waterfill=waterfill(var_sub, M), empirical=greedy(D, np.ones(M), M),
                  queryweighted=greedy(D, w / w.mean(), M))
    pred = {k: float(D[np.arange(M), b - BMIN].mean()) for k, b in allocs.items()}
    predw = {k: float((w / w.mean() * D[np.arange(M), b - BMIN]).mean()) for k, b in allocs.items()}
    rep = dict(tag=tag, d=d, M=M,
               residual_var_dim=dict(mean=float(var_dim.mean()), cv=float(var_dim.std() / var_dim.mean()), min=float(var_dim.min()), max=float(var_dim.max())),
               residual_var_subspace=dict(cv=float(var_sub.std() / var_sub.mean()), min=float(var_sub.min()), max=float(var_sub.max())),
               excess_kurtosis_mean=float(kurt.mean()),
               gaussian_waterfilling_gain_bound=dict(per_dimension_AM_over_GM=am_gm(var_dim), per_subspace_AM_over_GM=am_gm(var_sub)),
               distortion_rate_mean_over_subspaces={str(b): float(D[:, b - BMIN].mean()) for b in range(BMIN, BMAX + 1)},
               distortion_rate_slope_bits=float(np.log2(D[:, 2] / D[:, 3]).mean()),
               query_weight_cv=float(w.std() / w.mean()),
               allocations={k: dict(histogram={str(b): int((v == b).sum()) for b in range(BMIN, BMAX + 1)}, total_bits_per_vector=int(8 * v.sum()))
                            for k, v in allocs.items()},
               predicted_heldout_mse_per_coordinate=pred, predicted_query_weighted_mse=predw,
               predicted_mse_ratio_vs_uniform={k: v / pred["uniform"] for k, v in pred.items()},
               seconds=time.time() - t0)
    od = os.path.join(MC.MA, tag, "v3_rd")
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "stats.npz"), D=D, w=w, var_dim=var_dim, **{"alloc_" + k: v for k, v in allocs.items()})
    json.dump(rep, open(os.path.join(out, "stats_%s.json" % tag), "w"), indent=1)
    print(json.dumps(rep), flush=True)


def build(out, tag, alloc):
    t0 = time.time()
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    od = os.path.join(MC.MA, tag, "v3_rd")
    b = np.load(os.path.join(od, "stats.npz"))["alloc_" + alloc].astype(np.int64)
    widths = np.repeat(b, 8)
    rtr, _ = residual_samples(tag)
    cbs = np.zeros((M, 1 << BMAX), np.float32)
    for k in range(M):
        cb, _ = C.lloyd_1d(rtr[:, k * 8:(k + 1) * 8], 1 << int(b[k]))
        cbs[k, :len(cb)] = cb
    vd = os.path.join(od, alloc)
    os.makedirs(vd, exist_ok=True)
    packed = np.memmap(os.path.join(vd, "res_packed.u8"), dtype=np.uint8, mode="w+", shape=(N, d))
    xp = C.xpos(tag)
    STEP = max(4096, (1 << 22) // d)
    mse = 0.0
    t_pack = t_unpack = 0.0
    for a in range(0, N, STEP):
        e = min(N, a + STEP)
        pos = np.arange(a, e)
        r = (np.asarray(xp[a:e], dtype=np.float64) - jhq_primary(tag, pos)).astype(np.float32)
        codes = np.empty((e - a, d), np.uint16)
        for k in range(M):
            codes[:, k * 8:(k + 1) * 8] = nearest(cbs[k, :1 << int(b[k])], r[:, k * 8:(k + 1) * 8])
        tp = time.time(); p = pack(codes, widths); t_pack += time.time() - tp
        packed[a:e] = p
        if a == 0 or a + STEP >= N:
            tu = time.time(); back = unpack(p, widths); t_unpack += time.time() - tu
            assert np.array_equal(back, codes.astype(np.int64)), "pack / unpack mismatch"
        rh = cbs[np.arange(M).repeat(8)[None, :], codes.astype(np.int64)]
        mse += float(((r - rh).astype(np.float64) ** 2).sum())
    packed.flush()
    np.savez(os.path.join(vd, "params.npz"), bits=b, res_cb=cbs)
    info = dict(tag=tag, alloc=alloc, bits_histogram={str(x): int((b == x).sum()) for x in range(BMIN, BMAX + 1)},
                bytes_per_vector=int(packed.shape[1]), hierarchical_mse_per_vector=mse / N,
                pack_seconds_cpu=t_pack, unpack_check_seconds_cpu=t_unpack, seconds=time.time() - t0)
    json.dump(info, open(os.path.join(vd, "info.json"), "w"), indent=1)
    json.dump(info, open(os.path.join(out, "build_%s_%s.json" % (tag, alloc)), "w"), indent=1)
    print(json.dumps(info), flush=True)


def rd_dist(tag, vd):
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    pz = np.load(os.path.join(vd, "params.npz"))
    b, cbs = pz["bits"].astype(np.int64), pz["res_cb"].astype(np.float64)
    widths = np.repeat(b, 8)
    packed = np.memmap(os.path.join(vd, "res_packed.u8"), dtype=np.uint8, mode="r", shape=(N, d))
    sub = np.arange(M).repeat(8)[None, :]

    def f(q, qr, pos):
        n = len(pos)
        dh = np.empty(n)
        for a in range(0, n, 4096):
            ps = pos[a:a + 4096]
            codes = unpack(np.asarray(packed[ps]), widths)
            e = qr[None, :] - jhq_primary(tag, ps) - cbs[sub, codes]
            dh[a:a + 4096] = np.einsum("ij,ij->i", e, e)
        return q["P"], dh                       # the primary order is the paper index's (unchanged primary)
    return f


def evaluate(out, test):
    rows, summ = [], {}
    label = "eval_rd" + ("_test" if test else "")
    for tag in C.TAGS:
        # only the distinct allocations: one with the same bits as an earlier one gives the same codes
        st = np.load(os.path.join(MC.MA, tag, "v3_rd", "stats.npz"))
        methods, same = {}, {}
        for alloc in ("uniform", "empirical", "waterfill", "queryweighted"):
            twin = next((a for a in same if np.array_equal(st["alloc_" + a], st["alloc_" + alloc])), None)
            if twin is not None:
                print(tag, alloc, "has the bits of", twin, "-- not evaluated again", flush=True)
                continue
            same[alloc] = True
            vd = os.path.join(MC.MA, tag, "v3_rd", alloc)
            if os.path.exists(os.path.join(vd, "params.npz")):
                methods["RD-" + alloc] = rd_dist(tag, vd)
        if list(methods) == ["RD-uniform"]:
            print(tag, "every allocation is the uniform one: its codes are JHQ's offline re-implementation's", flush=True)
            continue
        r, s = MC.evaluate(out, label, tag, methods, test=test)
        rows += r
        s["paired_calib_val"] = {a + " - RD-uniform": MC.paired(out, label, tag, a, "RD-uniform") for a in methods if a != "RD-uniform"}
        s["paired_calib_val"]["RD-uniform - JHQ"] = MC.paired(out, label, tag, "RD-uniform", "JHQ")
        summ[tag] = s
        print(tag, json.dumps(s["methods"]), flush=True)
        print("  paired", json.dumps(s["paired_calib_val"]), flush=True)
        MC.write_rows(os.path.join(out, label + ".csv"), rows)
        json.dump(summ, open(os.path.join(out, label + ".json"), "w"), indent=1)


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if cmd == "test":
        sys.exit(0 if test(out) else 1)
    elif cmd == "stats":
        stats(out, sys.argv[3])
    elif cmd == "build":
        build(out, sys.argv[3], sys.argv[4])
    elif cmd == "eval":
        evaluate(out, "--test" in sys.argv)
