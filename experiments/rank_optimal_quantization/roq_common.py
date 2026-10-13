"""Shared pieces of the rank-optimal quantization study (offline, on the GPU box).

Data (all read-only, produced by earlier steps):
  LF/<tag>/rb_q<qid>.bin   per query: v65's probed candidates (position, id, probe rank), FP64 D_prim
                           and D_H of the paper index, v65's top-10, ground truth top-10
  PP/<tag>/                primary.u8 [N][M] (position order), list_offsets.i32, qrot.f64 (q' = Pi q,
                           FP64), qids.i32, meta.json (levels)
  RQ/<tag>/                Pi.f32, res_c1d.f32, res_codes.u8 [N][d], corr.f32, list_ids.i32, meta.json
  RQ/<tag>/xpos.f32        [N][d] y = Pi x for every base vector, in POSITION order (prep.py)
  RQ/<tag>/var_<name>/     a quantizer variant: bits.u8 [N][M] and res.u8 [N][d] in position order,
                           params.npz (levels per dimension or per list, residual codebooks, ...)

Conventions (checked by prep.py against the exports): bit j of a vector is the MSB-first bit j%8 of
byte j//8; JHQ's bit is 1 iff y_j > 0; s = 2 b - 1. Primary distance D_P = ||q' - y_hat||^2,
hierarchical D_H = ||q' - y_hat - r_hat||^2. Selection by (score, position), final top-10 by
(D_H, position), recall@10 against the ground-truth top-10.
"""
import json
import os

import numpy as np

LF = "/root/autodl-tmp/lf_pilot"
PP = "/root/autodl-tmp/pp_pilot"
RQ = "/root/autodl-tmp/roq"
K = 10
SEED = 20261013
TAGS = ("arxiv-768_np128_a100", "vogue-768_np128_a64", "openai3-3072_np128_a4")
BASE = {"arxiv-768_np128_a100": "/root/autodl-tmp/arxiv-abstracts-768",
        "vogue-768_np128_a64": "/root/data/vogue-768",
        "openai3-3072_np128_a4": "/root/autodl-tmp/openai3-3072"}
RES_TRAIN_N = 100000
BITS = ((np.arange(256)[:, None] >> (7 - np.arange(8))[None, :]) & 1).astype(np.uint8)   # [256, 8], MSB first


def base_paths(tag):
    b = BASE[tag]
    if tag.startswith("vogue"):
        return b + "_base.fvecs.raw_f32", b + "_query.fvecs", b + "_groundtruth.ivecs"
    return b + "/base.fvecs.raw_f32", b + "/query.fvecs", b + "/groundtruth.ivecs"


def meta(tag):
    m = json.load(open(os.path.join(RQ, tag, "meta.json")))
    m.update({k: v for k, v in json.load(open(os.path.join(PP, tag, "meta.json"))).items() if k not in m})
    return m


def fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def rotation(tag):
    d = meta(tag)["d"]
    P = np.fromfile(os.path.join(RQ, tag, "Pi.f32"), dtype=np.float32)
    return np.ascontiguousarray(P.reshape(d, d).T)          # A with y = A x  (Pi stored column-major)


def xpos(tag):
    m = meta(tag)
    return np.memmap(os.path.join(RQ, tag, "xpos.f32"), dtype=np.float32, mode="r", shape=(m["N"], m["d"]))


def list_offsets(tag):
    return np.fromfile(os.path.join(PP, tag, "list_offsets.i32"), dtype=np.int32).astype(np.int64)


def list_of_pos(tag):
    off = list_offsets(tag)
    return np.repeat(np.arange(len(off) - 1), np.diff(off)).astype(np.int32)


def qrot(tag):
    m = meta(tag)
    qids = np.fromfile(os.path.join(PP, tag, "qids.i32"), dtype=np.int32)
    q = np.fromfile(os.path.join(PP, tag, "qrot.f64"), dtype=np.float64).reshape(len(qids), m["d"])
    return {int(x): q[i] for i, x in enumerate(qids)}


def res_sample_rows(N):
    """the residual training sample of the paper index: base rows i * (N // 100000), in id order."""
    stride = N // RES_TRAIN_N
    return np.arange(RES_TRAIN_N, dtype=np.int64) * stride


def load_bin(path):
    with open(path, "rb") as f:
        b = f.read()
    o = 0

    def take(n, dt):
        nonlocal o
        a = np.frombuffer(b, dtype=dt, count=n, offset=o)
        o += a.nbytes
        return a
    qid, n, nprobe, ck, k = (int(x) for x in take(5, np.int32))
    q = dict(qid=qid, n=n, nprobe=nprobe, ck=ck, k=k)
    q["probes"] = take(nprobe, np.int32)
    q["pos"], q["id"], q["rank"] = take(n, np.int32), take(n, np.int32), take(n, np.int32)
    q["P"], q["DH"], _lc, _lg = (take(n, np.float64) for _ in range(4))
    q["qn2"] = float(take(1, np.float64)[0])
    q["v65_ids"], q["v65_d"], q["v65_dh"] = take(k, np.int32), take(k, np.float32), take(k, np.float64)
    q["topck"], q["gt"] = take(ck, np.int32), take(k, np.int32)
    assert o == len(b), path
    return q


def bin_files(tag):
    d = os.path.join(LF, tag)
    fs = sorted(f for f in os.listdir(d) if f.startswith("rb_q") and f.endswith(".bin"))
    return [os.path.join(d, f) for f in fs]


def splits(qids):
    perm = np.random.default_rng(SEED).permutation(sorted(int(x) for x in qids))
    c1, c2 = int(0.4 * len(perm)), int(0.6 * len(perm))
    return {int(x): ("calib" if i < c1 else "val" if i < c2 else "test") for i, x in enumerate(perm)}


# ---------------------------------------------------------------- residual (JHQ's procedure)
def lloyd_1d(vals, Kr=256, max_iter=2000, tol=1e-6):
    """train_1d_kmeans_hold of cpu/codebook.cpp: quantile seeds, cell = [midpoint below, midpoint
    above), empty cells hold, stop when max |delta| <= tol * max |c|. Sorted values + prefix sums, so
    an iteration costs O(Kr log n)."""
    s = np.sort(np.asarray(vals, dtype=np.float64).ravel())
    n = len(s)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    c = s[np.minimum(((np.arange(Kr) + 0.5) / Kr * n).astype(np.int64), n - 1)].copy()
    for it in range(max_iter):
        mids = 0.5 * (c[:-1] + c[1:])
        cut = np.concatenate([[0], np.searchsorted(s, mids, side="left"), [n]])
        cnt = np.diff(cut)
        sm = cs[cut[1:]] - cs[cut[:-1]]
        new = np.where(cnt > 0, sm / np.maximum(cnt, 1), c)
        delta = np.abs(new - c).max()
        c = np.sort(new)
        mx = np.abs(c).max()
        if mx > 0 and delta <= tol * mx:
            break
    return c.astype(np.float32), it + 1


def nearest_sorted(cb, v):
    """index of the nearest entry of the sorted codebook cb for every value of v (ties to the lower)."""
    mids = 0.5 * (cb[:-1] + cb[1:])
    return np.searchsorted(mids, v, side="left").astype(np.uint8)


def train_residual(resid, M, Ds=8, Kr=256):
    """per subspace m: one scalar codebook over the residuals of its Ds coordinates."""
    n, d = resid.shape
    cb = np.zeros((M, Kr), np.float32)
    iters = []
    for m in range(M):
        cb[m], it = lloyd_1d(resid[:, m * Ds:(m + 1) * Ds], Kr)
        iters.append(it)
    return cb, iters


def encode_residual(resid, cb, Ds=8):
    n, d = resid.shape
    out = np.empty((n, d), np.uint8)
    for m in range(cb.shape[0]):
        out[:, m * Ds:(m + 1) * Ds] = nearest_sorted(cb[m], resid[:, m * Ds:(m + 1) * Ds])
    return out


def decode_residual(codes, cb, Ds=8):
    n, d = codes.shape
    M = cb.shape[0]
    idx = np.arange(M).repeat(Ds)[None, :]
    return cb[idx, codes.astype(np.int64)]


# ---------------------------------------------------------------- evaluation
def t_grid(ck, n_max):
    base = {10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640, 960, 1280, 1920, 2560}
    base |= {max(K, int(round(ck * r))) for r in (1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1.0, 1.25, 1.5, 2.0)}
    return sorted(t for t in base if t <= n_max)


def recall_curve(key, dh, pos, isgt, Ts):
    """recall@10 when the primary top-T by (key, position) is refined by (D_H, position)."""
    order = np.lexsort((pos, key))
    rH = np.empty(len(dh), np.int64)
    rH[np.lexsort((pos, dh))] = np.arange(len(dh))
    out = np.zeros(len(Ts), np.int16)
    for i, T in enumerate(Ts):
        sel = order[:T]
        if len(sel) > K:
            sel = sel[np.argpartition(rH[sel], K - 1)[:K]]
        out[i] = isgt[sel].sum()
    return out, order


def ranks_of(order, which):
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order))
    return r[which]


def misrank(key, pos_idx, neg_idx):
    """fraction of (positive, negative) pairs with key(pos) >= key(neg) (lower key = closer)."""
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan
    kp = np.sort(key[pos_idx])
    kn = np.sort(key[neg_idx])
    # for every positive, how many negatives have key <= its key
    return float(np.searchsorted(kn, kp, side="right").sum()) / (len(kp) * len(kn))


def boot_ci(x, rng, B=5000):
    x = np.asarray(x, np.float64)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
