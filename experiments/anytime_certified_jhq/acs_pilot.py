#!/usr/bin/env python3
"""AC-JHQ offline pilot: how much can certified sequential bounds eliminate before all primary words are read?

  python3 acs_pilot.py <out dir> <tag> [splits=calib] [max_queries]

Representation (checked against the exported FP64 D_prim): D_P = ||q'||^2 + d c^2 - 2c S, S = sum_w y_w,
y_w = <q'_w, s_w> over the 32 dimensions of word w, s = +-1 the primary bits. Lower D_P = higher S. The primary
top-T (T = ck) is the T largest S (ties by position). Per query, one uniformly random word permutation pi (shared
by all candidates, so reads stay coalesced); after t words a candidate x knows y_{pi_1..pi_t}(x).

Bounds on S(x) after t words (THEORY.md): population v_w = y_w - c_w over the W words, c_w fixed before sampling,
range [a, b] known from the query (and the list for C3):
  C0   deterministic: S <= sum_read y + sum_unread ||q'_w||_1                           (risk 0)
  C1   Hoeffding-Serfling, Bardenet-Maillard 2015 Corollary 2.5, c_w = 0                 (risk delta_c per check)
  C2   empirical Bernstein-Serfling, Bardenet-Maillard 2015 Theorem 4.3 (prob 1 - 5 delta'), c_w = 0
  C3   C2 with the bucket control variate c_w = <q'_w, mu_{l,w}>, mu_l = the list's sign means (int8, as stored)
Every certified bound is also clipped by C0. A candidate is eliminated at a checkpoint when its upper bound on S is
below S_tau, the T-th largest S of the query (the ORACLE threshold: the most favourable valid threshold, an upper
bound on any realisable saving). Risk: a primary top-T member is eliminated only if its bound fails, so
delta_c = delta_query / (T K) over the T members and the K checkpoints (union bound, any dependence).
Reported per checkpoint set and delta_query: eliminated share, words read per candidate, 32 B sector reads of the
[W][N] word rows (a sector = 8 consecutive positions of one row), warp-word issues (32 lanes), top-T retention.
Audit: empirical violation rate of each bound over random permutations at large delta (where violations must
be possible but rare), against the nominal rate.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rank_optimal_quantization"))
import roq_common as C  # noqa: E402

G = {}
KAPPA = 7.0 / 3.0 + 3.0 / np.sqrt(2.0)
DELTAS = (1e-2, 1e-3, 1e-4)
FRACS = (0.125, 0.25, 0.5, 0.75)


def rho(t, W):
    return np.where(t <= W / 2, 1.0 - (t - 1.0) / W, (1.0 - t / W) * (1.0 + 1.0 / t))


def word_sums(pos, qr, W):
    """y [n, W]: y_w = <q'_w, s_w> from the code bytes (s = 2 bit - 1, MSB first = the subspace's first dim)."""
    n = len(pos)
    y = np.empty((n, W))
    qw = qr.reshape(W, 32)
    for a in range(0, n, 4096):
        b = np.unpackbits(np.asarray(G["prim"][pos[a:a + 4096]]), axis=1).astype(np.float64)
        s = (2.0 * b - 1.0).reshape(-1, W, 32)
        y[a:a + 4096] = np.einsum("nwj,wj->nw", s, qw)
    return y


def bounds(yp, cw, h, t, delta_c, kind):
    """upper bounds on S for every candidate after reading the first t words of the permuted rows yp [n, W]
    (cw [n, W] the fixed offsets in the same permuted order, h [W] the half-ranges ||q'_w||_1 permuted)."""
    W = yp.shape[1]
    h2 = h if h.ndim == 2 else h[None, :]          # one permuted h for all rows, or one per row
    read = yp[:, :t]
    det = read.sum(1) + h2[:, t:].sum(1)
    if kind == "C0" or t >= W:
        return det if t < W else yp.sum(1)
    v = read - cw[:, :t]
    vbar = v.mean(1)
    csum = cw.sum(1)
    lo = (-h2 - cw).min(1)
    hi = (h2 - cw).max(1)
    R = hi - lo
    r = rho(float(t), float(W))
    if kind == "C1":
        eps = R * np.sqrt(r * np.log(1.0 / delta_c) / (2.0 * t))
    else:
        dp = delta_c / 5.0
        sig = np.sqrt(np.maximum(((v - vbar[:, None]) ** 2).mean(1), 0.0))
        eps = sig * np.sqrt(2.0 * r * np.log(1.0 / dp) / t) + KAPPA * R * np.log(1.0 / dp) / t
    cs = csum + W * (vbar + eps)
    return np.minimum(cs, det)


def per_query(path):
    q = C.load_bin(path)
    pos = q["pos"].astype(np.int64)
    n, ck = len(pos), q["ck"]
    qr = G["qrot"][q["qid"]]
    W = G["W"]
    y = word_sums(pos, qr, W)
    S = y.sum(1)
    c = G["c"]
    dp = float(qr @ qr) + G["d"] * c * c - 2.0 * c * S
    err = float(np.abs(dp - q["P"]).max() / np.abs(q["P"]).max())
    T = ck
    order = np.lexsort((pos, -S))
    topT = order[:T]
    S_tau = S[order[T - 1]]
    h = np.abs(qr).reshape(W, 32).sum(1)
    # bucket control variate: the list sign means (int8), per candidate's list
    lp = G["lpos"][pos]
    mu = G["mu"]
    rng = np.random.default_rng(q["qid"])
    out = dict(qid=q["qid"], n=n, T=T, reconstruction_rel_err=err, res={})
    # variance of the word population with and without the control variate
    ul, inv = np.unique(lp, return_inverse=True)
    bw = np.einsum("lwj,wj->lw", mu[ul].reshape(len(ul), W, 32), qr.reshape(W, 32))     # [lists, W]
    cv = bw[inv]                                                                          # [n, W]
    out["var_plain"] = float(y.var(1).mean())
    out["var_cv"] = float((y - cv).var(1).mean())
    out["range_plain"] = float((2 * h).max())
    out["range_cv_mean"] = float(((h[None, :] - cv).max(1) - (-h[None, :] - cv).min(1)).mean())
    # sectors and warps: 8 / 32 consecutive positions of one word row
    sec = pos // 8
    warp = pos // 32
    full_sectors = len(np.unique(sec)) * W
    full_warps = len(np.unique(warp)) * W
    out["full_sectors"] = full_sectors
    perm = rng.permutation(W)
    yp, cvp, hp = y[:, perm], cv[:, perm], h[perm]
    zero = np.zeros_like(yp)
    for fr_set in (FRACS, (0.5, 0.75), (0.75,)):
        ts = sorted({int(np.ceil(f * W)) for f in fr_set})
        K = len(ts)
        for delta in DELTAS:
            dc = delta / (T * K)
            for kind in ("C0", "C1", "C2", "C3"):
                if kind == "C0" and delta != DELTAS[0]:
                    continue
                active = np.ones(n, bool)
                words = np.zeros(n)
                sectors = warps = 0
                prev = 0
                elim_at = {}
                for t in ts + [W]:
                    # read words prev..t-1 for the active candidates
                    act_idx = np.where(active)[0]
                    sectors += len(np.unique(sec[act_idx])) * (t - prev)
                    warps += len(np.unique(warp[act_idx])) * (t - prev)
                    words[act_idx] += t - prev
                    prev = t
                    if t >= W:
                        break
                    ub = bounds(yp[act_idx], (cvp if kind == "C3" else zero)[act_idx], hp, t, dc, kind)
                    kill = act_idx[ub < S_tau]
                    active[kill] = False
                    elim_at[t] = float(len(kill) / n)
                lost = int((~active[topT]).sum())
                out["res"]["%s|%s|%g" % (kind, "-".join(map(str, ts)), delta)] = dict(
                    eliminated=float((~active).mean()), elim_at=elim_at, words_per_cand=float(words.mean()) / W,
                    sector_frac=sectors / full_sectors, warp_frac=warps / full_warps, topT_lost=lost)
    # audit: violation frequency of the per-candidate bounds at a large delta over random permutations
    aud = {}
    pick = np.concatenate([topT[:min(T, 20)], rng.choice(n, size=min(n, 40), replace=False)])
    for kind in ("C1", "C2", "C3"):
        for dc in (0.2, 0.05):
            viol = tot = 0
            for _ in range(50):
                pm = rng.permutation(W)
                for t in sorted({int(np.ceil(f * W)) for f in FRACS}):
                    ub = bounds(y[pick][:, pm], (cv[pick][:, pm] if kind == "C3" else np.zeros((len(pick), W))), h[pm], t, dc, kind)
                    viol += int((ub < S[pick] - 1e-12).sum()); tot += len(pick)
            aud["%s|%g" % (kind, dc)] = (viol, tot)
    out["audit"] = aud
    return out


def main():
    out_dir, tag = sys.argv[1], sys.argv[2]
    splits = (sys.argv[3] if len(sys.argv) > 3 else "calib").split(",")
    maxq = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    meta = json.load(open(os.path.join(C.PP, tag, "meta.json")))
    N, d, M = meta["N"], meta["d"], meta["M"]
    G.update(W=M // 4, d=d, c=float(np.float32(meta["levels"][1])),
             prim=np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
             qrot=C.qrot(tag), lpos=C.list_of_pos(tag))
    # list sign means mu_l (int8-quantised, as they would be stored)
    off = C.list_offsets(tag)
    nl = len(off) - 1
    mu = np.zeros((nl, d))
    for l0 in range(0, nl, 256):
        l1 = min(nl, l0 + 256)
        a, b = int(off[l0]), int(off[l1])
        bt = np.unpackbits(np.asarray(G["prim"][a:b]), axis=1).astype(np.int32)
        cs = np.zeros((b - a + 1, d), np.int64); np.cumsum(bt, axis=0, out=cs[1:])
        st = off[l0:l1 + 1] - a
        cnt = np.maximum(np.diff(st), 1)
        mu[l0:l1] = 2.0 * (cs[st[1:]] - cs[st[:-1]]) / cnt[:, None] - 1.0
    G["mu"] = np.round(mu * 127.0) / 127.0
    files = C.bin_files(tag)
    qids = [int(os.path.basename(f)[4:-4]) for f in files]
    sp = C.splits(qids)
    files = [f for f, x in zip(files, qids) if sp[x] in splits][:maxq]
    pool = mp.get_context("fork").Pool(24)
    res = pool.map(per_query, files, chunksize=2)
    pool.close(); pool.join()
    keys = list(res[0]["res"].keys())
    summ = dict(tag=tag, splits=splits, queries=len(res), W=G["W"], T=res[0]["T"],
                reconstruction_rel_err_max=max(o["reconstruction_rel_err"] for o in res),
                var_plain=float(np.mean([o["var_plain"] for o in res])), var_cv=float(np.mean([o["var_cv"] for o in res])),
                range_plain=float(np.mean([o["range_plain"] for o in res])), range_cv=float(np.mean([o["range_cv_mean"] for o in res])),
                methods={})
    for k in keys:
        rs = [o["res"][k] for o in res]
        summ["methods"][k] = dict(eliminated=float(np.mean([r["eliminated"] for r in rs])),
                                  words_per_cand=float(np.mean([r["words_per_cand"] for r in rs])),
                                  sector_frac=float(np.mean([r["sector_frac"] for r in rs])),
                                  sector_frac_p90=float(np.quantile([r["sector_frac"] for r in rs], 0.9)),
                                  warp_frac=float(np.mean([r["warp_frac"] for r in rs])),
                                  topT_lost_queries=int(sum(r["topT_lost"] > 0 for r in rs)),
                                  elim_at={str(t): float(np.mean([r["elim_at"].get(t, 0.0) for r in rs])) for t in rs[0]["elim_at"]})
    aud = {}
    for k in res[0]["audit"]:
        v = sum(o["audit"][k][0] for o in res); tt = sum(o["audit"][k][1] for o in res)
        aud[k] = dict(violations=v, checks=tt, rate=v / max(tt, 1))
    summ["audit"] = aud
    summ["seconds"] = time.time() - t0
    json.dump(summ, open(os.path.join(out_dir, "pilot_%s.json" % tag), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ("tag", "queries", "W", "T", "reconstruction_rel_err_max", "var_plain", "var_cv", "range_plain", "range_cv")}))
    for k, v in summ["methods"].items():
        print("%-22s elim %.3f  words/cand %.3f  sectors %.3f (p90 %.3f)  warps %.3f  topT-lost queries %d  elim_at %s" % (
            k, v["eliminated"], v["words_per_cand"], v["sector_frac"], v["sector_frac_p90"], v["warp_frac"], v["topT_lost_queries"],
            {t: round(x, 3) for t, x in v["elim_at"].items()}))
    print("audit", json.dumps(aud))


if __name__ == "__main__":
    main()
