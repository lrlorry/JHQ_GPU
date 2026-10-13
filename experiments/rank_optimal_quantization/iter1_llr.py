#!/usr/bin/env python3
"""Iteration 1: near-neighbour likelihood-ratio scores on the paper index's own primary codes.

  python3 iter1_llr.py <results dir> [workers]

Every score is a weighted sign inner product S = sum_j w_j(q) s_j (higher = closer; THEORY.md 4):
  JHQ          w_j = q'_j                                   (the primary distance's own ranking)
  LLR-iso      w_j = psi(lam q'_j / sigma) / 2              (H0 = N(0, sigma^2 I); lam -> 0 is JHQ)
  LLR-bias     w_j = [psi(mu1_j / s1_j) - psi(mu_j / sd_j)] / 2,  H0 = N(mu_j, sd_j^2) per coordinate (measured),
               H1 = H0 * exp(-beta ||y - q'||^2): s1_j^2 = sd_j^2 / (1 + g_j), mu1_j = s1_j^2 (mu_j / sd_j^2 + 2 beta q'_j),
               g_j = 2 beta sd_j^2; parametrised by g = 2 beta sigma^2
  LMP          w_j = q'_j phi(k q'_j / sigma) / (Phi(k q'_j/sigma) Phi(-k q'_j/sigma))   (locally most powerful for a
               correlation level k = rho / sqrt(1 - rho^2); k -> 0 is JHQ)
psi(t) = log Phi(t) - log Phi(-t). The codes, candidates and D_H are the paper index's (v65's): only the
order of the primary selection changes. Parameters are chosen on the CALIBRATION split by mean recall at
T in {ck/4, ck/2, ck}; the matched-recall budget is read on VALIDATION. The test split is not read here.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.special import log_ndtr, ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402

GRID = {"LLR-iso": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        "LLR-bias": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        "LMP": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)}
G = {}


def psi(t):
    return log_ndtr(t) - log_ndtr(-t)


def weights(qr):
    sig, mu, sd = G["sigma"], G["mu"], G["sd"]
    W, names = [qr], [("JHQ", 0.0)]
    for lam in GRID["LLR-iso"]:
        W.append(0.5 * psi(lam * qr / sig)); names.append(("LLR-iso", lam))
    for g in GRID["LLR-bias"]:
        beta = g / (2 * sig * sig)
        gj = 2 * beta * sd * sd
        s1 = sd / np.sqrt(1 + gj)
        mu1 = s1 * s1 * (mu / (sd * sd) + 2 * beta * qr)
        W.append(0.5 * (psi(mu1 / s1) - psi(mu / sd))); names.append(("LLR-bias", g))
    for k in GRID["LMP"]:
        t = k * qr / sig
        W.append(qr * np.exp(-0.5 * t * t) / np.maximum(ndtr(t) * ndtr(-t), 1e-300)); names.append(("LMP", k))
    return np.stack(W, 1), names


def per_query(path):
    q = C.load_bin(path)
    n, ck = q["n"], q["ck"]
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    s = np.unpackbits(np.asarray(G["prim"][pos]), axis=1).astype(np.float64) * 2 - 1      # [n, d]
    W, names = weights(qr)
    S = s @ W                                       # [n, V], FP64 so that JHQ's column orders as its D_P does
    isgt = np.isin(q["id"], q["gt"][:C.K])
    Ts = G["Ts"]
    dh = q["DH"]
    rH = np.empty(n, np.int64); rH[np.lexsort((pos, dh))] = np.arange(n)
    hard = np.where((rH >= C.K) & (rH < 100))[0]
    rng = np.random.default_rng(q["qid"])
    rnd = rng.choice(np.where(~isgt)[0], size=min(2000, int((~isgt).sum())), replace=False)
    gti = np.where(isgt)[0]
    R = np.zeros((len(names), len(Ts)), np.int16)
    MR = np.zeros((len(names), 2))
    RK = np.zeros((len(names), 3))
    for v in range(len(names)):
        key = -S[:, v]
        R[v], order = C.recall_curve(key, dh, pos, isgt, Ts)
        MR[v] = (C.misrank(key, gti, rnd), C.misrank(key, gti, hard[~isgt[hard]]))
        rk = C.ranks_of(order, gti) if len(gti) else np.array([n])
        RK[v] = (np.median(rk), np.max(rk), np.mean(rk < ck))
    # agreement of the weights with JHQ's
    cosw = (W * W[:, :1]).sum(0) / np.linalg.norm(W, axis=0) / np.linalg.norm(W[:, 0])
    full = int(isgt[np.lexsort((pos, dh))[:C.K]].sum())
    return dict(qid=q["qid"], n=n, ck=ck, R=R, MR=MR, RK=RK, cosw=cosw, full=full, names=names, gt_in=int(isgt.sum()))


def main():
    R = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    os.makedirs(R, exist_ok=True)
    rows, summ = [], {}
    for tag in C.TAGS:
        t0 = time.time()
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        xp = C.xpos(tag)
        smp = np.asarray(xp[np.arange(0, N, max(1, N // 200000))], dtype=np.float64)
        G.clear()
        G.update(prim=np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                 qrot=C.qrot(tag), sigma=float(m["levels"][1]) / 0.6744897501960817,
                 mu=smp.mean(0), sd=smp.std(0))
        files = C.bin_files(tag)
        q0 = C.load_bin(files[0])
        G["Ts"] = C.t_grid(q0["ck"], 4 * q0["ck"])
        pool = mp.get_context("fork").Pool(workers)       # close + join: see eval_variants.py
        res = pool.map(per_query, files, chunksize=4)
        pool.close(); pool.join()
        sp = C.splits([o["qid"] for o in res])
        names = res[0]["names"]
        Ts = G["Ts"]
        ck = res[0]["ck"]
        Rr = np.stack([o["R"] for o in res]).astype(np.float64) / C.K          # [nq, V, T]
        MR = np.stack([o["MR"] for o in res]); RK = np.stack([o["RK"] for o in res])
        cosw = np.stack([o["cosw"] for o in res])
        S = {s: np.array([sp[o["qid"]] == s for o in res]) for s in ("calib", "val", "test")}
        ti = [Ts.index(t) for t in sorted({max(C.K, ck // 4), max(C.K, ck // 2), ck}) if t in Ts]
        # choose each family's parameter on calibration
        chosen = {"JHQ": 0}
        for fam in GRID:
            idx = [i for i, nm in enumerate(names) if nm[0] == fam]
            sc = [Rr[S["calib"]][:, i, ti].mean() for i in idx]
            chosen[fam] = idx[int(np.argmax(sc))]
        target = {s: float(Rr[S[s]][:, 0, Ts.index(ck)].mean()) for s in ("calib", "val")}

        def tmin(vi, split):
            cur = Rr[S[split]][:, vi, :].mean(0)
            ok = [t for t, r in zip(Ts, cur) if r >= target[split] - 1e-12]
            return min(ok) if ok else None
        for vi, nm in enumerate(names):
            for split in ("calib", "val"):
                cur = Rr[S[split]][:, vi, :].mean(0)
                rows.append(dict(iteration=1, tag=tag, method=nm[0], param=nm[1], split=split, chosen_on_calib=(chosen.get(nm[0]) == vi),
                                 T_min_matched=tmin(vi, split), T_min_JHQ=tmin(0, split), target_recall=target[split],
                                 **{"recall_T%d" % t: float(r) for t, r in zip(Ts, cur)},
                                 misrank_random=float(np.nanmean(MR[S[split]][:, vi, 0])), misrank_hard=float(np.nanmean(MR[S[split]][:, vi, 1])),
                                 gt_rank_median=float(np.mean(RK[S[split]][:, vi, 0])), gt_rank_max_mean=float(np.mean(RK[S[split]][:, vi, 1])),
                                 gt_within_ck=float(np.mean(RK[S[split]][:, vi, 2])), cos_weights_vs_jhq=float(cosw[S[split]][:, vi].mean())))
        summ[tag] = dict(ck=ck, T_grid=Ts, target=target, full_ivf_recall={s: float(np.mean([o["full"] for o, f in zip(res, S[s]) if f]) / C.K) for s in ("calib", "val")},
                         chosen={fam: dict(param=names[vi][1], T_min_val=tmin(vi, "val"), T_min_val_JHQ=tmin(0, "val"),
                                           reduction_val=(None if tmin(vi, "val") is None or tmin(0, "val") is None else 1 - tmin(vi, "val") / tmin(0, "val")),
                                           recall_at_ck_val=float(Rr[S["val"]][:, vi, Ts.index(ck)].mean()))
                                 for fam, vi in chosen.items()},
                         seconds=time.time() - t0)
        np.savez_compressed(os.path.join(C.RQ, tag, "iter1_perquery.npz"), qid=np.array([o["qid"] for o in res]), R=Rr, MR=MR, RK=RK,
                            names=np.array(["%s:%g" % nm for nm in names]), Ts=np.array(Ts))
        print(tag, json.dumps(summ[tag]["chosen"]), "%.0fs" % summ[tag]["seconds"], flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith(("iteration", "tag", "method", "param", "split", "chosen", "T_min", "target")), k))
    with open(os.path.join(R, "iter1_llr.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(summ, open(os.path.join(R, "iter1_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
