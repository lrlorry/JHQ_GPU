#!/usr/bin/env python3
"""Pairwise misranking: the bounds of THEORY.md 4 against what the data does (calibration + validation).

  python3 rank_error.py <results dir> [variant ...]      (variants built by quantize.py; JHQ is always included)

Per query, for a code (JHQ's own bits, or a variant's), with positives = the ground-truth top-10 among the probed
candidates and negatives = all other probed candidates:
  empirical       fraction of (positive, negative) pairs the code's primary distance misorders (ties count)
  bhatt_model     prod_j rho_j^2 of THEORY.md 4.1, with p0_j = the bit frequency over the query's candidates
                  (measured) and p1_j from the independent Gaussian neighbour model
                  y_j ~ N(m_j + r (q'_j - m_j), (1 - r^2) sd_j^2), r = the measured mean centred correlation of
                  the ground-truth top-10 (prep.py), m_j and sd_j the per-coordinate mean and spread, threshold t_j
                  of the code; valid for the LLR score only and only under independent coordinates
  chernoff_score  THEORY.md 4.2 for the code's own score sum_j w_j s_j with the same model p0, p1
  cantelli        Var D / (Var D + (E D)^2) of the empirical pair differences D = S(neg) - S(pos) (S = primary
                  distance) when E D > 0: assumption-free, about the empirical pair distribution itself
"""
import csv
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402

G = {}


def per_query(path):
    q = C.load_bin(path)
    if G["split"][q["qid"]] not in ("calib", "val"):
        return None
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    isgt = np.isin(q["id"], q["gt"][:C.K])
    if isgt.sum() == 0:
        return None
    out = dict(qid=q["qid"], split=G["split"][q["qid"]], rows=[])
    r = G["r"]
    m, sd = G["mu"], G["sd"]
    for v, P in G["codes"].items():
        bits = np.unpackbits(np.asarray(P["bits"][pos]), axis=1).astype(bool)
        if P["lo"] is None:
            c = G["c"]
            dp = q["P"]
            t = np.zeros_like(qr); lo = np.full_like(qr, -c); hi = np.full_like(qr, c)
        else:
            t, lo, hi = P["thr"], P["lo"], P["hi"]
            yh = np.where(bits, hi[None, :], lo[None, :])
            dp = ((qr[None, :] - yh) ** 2).sum(1)
        # the primary distance is affine in the bits: dp = const + sum_j b_j u_j, u_j = (q-hi)^2 - (q-lo)^2
        u = (qr - hi) ** 2 - (qr - lo) ** 2
        p0 = np.clip(bits[~isgt].mean(0), 1e-6, 1 - 1e-6)
        p1 = np.clip(ndtr((m + r * (qr - m) - t) / (sd * np.sqrt(1 - r * r))), 1e-12, 1 - 1e-12)
        rho = np.sqrt(p1 * p0) + np.sqrt((1 - p1) * (1 - p0))
        log10_b = 2 * np.log10(rho).sum()
        # Chernoff for the code's own score: lower dp is closer; misrank = dp(pos) >= dp(neg),
        # i.e. sum_j u_j (b+_j - b-_j) >= 0; MGF of u_j (b+ - b-) at t >= 0
        def logmgf(tt):
            e1 = np.exp(np.clip(tt * u, -700, 700)); e2 = np.exp(np.clip(-tt * u, -700, 700))
            return np.log(p1 * e1 + 1 - p1).sum() + np.log(p0 * e2 + 1 - p0).sum()
        scale = 1.0 / max(np.abs(u).max(), 1e-30)
        res = minimize_scalar(lambda z: logmgf(z * scale), bounds=(0, 50), method="bounded")
        log10_ch = min(0.0, res.fun / np.log(10))
        gp, gn = dp[isgt], dp[~isgt]
        emp = float(np.searchsorted(np.sort(gn), gp, side="right").sum()) / (len(gp) * len(gn))
        D = (gn[None, :] - gp[:, None]).ravel() if len(gn) * len(gp) < 3_000_000 else \
            (gn[np.random.default_rng(q["qid"]).integers(0, len(gn), 300000)] - np.repeat(gp, 300000 // len(gp) + 1)[:300000])
        ED, VD = float(D.mean()), float(D.var())
        cant = VD / (VD + ED * ED) if ED > 0 else 1.0
        out["rows"].append((v, emp, log10_b, log10_ch, cant))
    return out


def main():
    R = sys.argv[1]
    variants = sys.argv[2:]
    rows = []
    for tag in C.TAGS:
        mt = C.meta(tag)
        N, d, M = mt["N"], mt["d"], mt["M"]
        prep = json.load(open(os.path.join(R, "prep_%s.json" % tag)))
        xp = C.xpos(tag)
        smp = np.asarray(xp[np.arange(0, N, max(1, N // 200000))], dtype=np.float64)
        G.clear()
        G.update(qrot=C.qrot(tag), r=prep["neighbours"]["centred_corr_top10_mean"], mu=smp.mean(0), sd=smp.std(0),
                 c=float(np.float32(mt["levels"][1])), codes={})
        G["codes"]["JHQ"] = dict(bits=np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M)), lo=None)
        for v in variants:
            pz = np.load(os.path.join(C.RQ, tag, "var_" + v, "params.npz"))
            if bool(pz["per_list"]):
                continue
            G["codes"][v] = dict(bits=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                                 thr=pz["thr"].astype(np.float64), lo=pz["lo"].astype(np.float64), hi=pz["hi"].astype(np.float64))
        files = C.bin_files(tag)
        G["split"] = C.splits([int(os.path.basename(f)[4:-4]) for f in files])
        pool = mp.get_context("fork").Pool(64)            # close + join: see eval_variants.py
        res = [o for o in pool.map(per_query, files, chunksize=4) if o]
        pool.close(); pool.join()
        for v in G["codes"]:
            for s in ("calib", "val"):
                a = np.array([[x[1], x[2], x[3], x[4]] for o in res if o["split"] == s for x in o["rows"] if x[0] == v])
                rows.append(dict(tag=tag, code=v, split=s, queries=len(a), neighbour_corr_model=G["r"],
                                 empirical_misrank_mean=float(a[:, 0].mean()), empirical_misrank_median=float(np.median(a[:, 0])),
                                 log10_bhattacharyya_model_median=float(np.median(a[:, 1])),
                                 log10_chernoff_own_score_median=float(np.median(a[:, 2])),
                                 cantelli_mean=float(a[:, 3].mean()),
                                 frac_queries_model_bound_below_empirical=float((10 ** a[:, 1] < a[:, 0]).mean()),
                                 frac_queries_chernoff_below_empirical=float((10 ** a[:, 2] < a[:, 0]).mean())))
        print(tag, [(r["code"], r["split"], round(r["empirical_misrank_mean"], 5), round(r["log10_bhattacharyya_model_median"], 1),
                     round(r["log10_chernoff_own_score_median"], 1), round(r["cantelli_mean"], 3)) for r in rows if r["tag"] == tag], flush=True)
    with open(os.path.join(R, "RANK_ERROR_ANALYSIS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
