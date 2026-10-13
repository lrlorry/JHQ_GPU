#!/usr/bin/env python3
"""V2 EVT-JHQ, offline: list-level pruning of the primary scan, simulated on the paper index's probed candidates.

  python3 evt_sim.py <out dir> [--test]

Per query the paper index's v65 probes nprobe lists; they are taken in increasing distance from the query to
the list centre (the list mean in the rotated space, a proxy for the routing centroid); every candidate's FP64 primary distance
D_P and D_H are in the candidate file. A pruning policy scans the first L1 lists (phase 1), takes tau = the ck-th
smallest D_P seen so far, and then, phase by phase (h lists per phase, tau updated after each), skips a list
whose estimated risk P(min_{x in list} D_P(x) < tau) is below eps. The risk uses only what is known before the
list is read: tau, the query-to-list-centre distance delta (computed by the routing anyway), the list size n and
the probe rank. The primary top-ck of the scanned candidates is refined by D_H (v65's T = ck).

Risk models (fitted on calibration queries only):
  global      P(y = 1 | u) binned on u = tau - delta (one fixed global statistic)
  iid_evt     the brief's formula 1 - (1 - F(tau - delta))^n with F the pooled per-candidate CDF of D_P - delta:
              exact only for i.i.d. candidates
  gev         a GEV fitted to the per-list minimum offsets (min D_P - delta), per list-size band; fit quality checked
  logistic    calibrated logistic regression on (u / s, log n, probe rank, delta - delta_1)
Baseline: the same index with a smaller nprobe (the first n' lists), and the per-query oracle that skips exactly
the lists holding none of the full scan's primary top-ck (the most any list-level rule could save).
Outputs: evt_sim.json, EVT_POLICIES.csv (recall@10 and candidates scanned for every policy and split),
evt_calibration.csv (reliability of each model on validation).
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import ma_common as MC  # noqa: E402

C = MC.C
G = {}
L1S = (8, 16, 32)
H = 16
EPS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
NPRIME = (8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128)


def load_query(path):
    """per query: lists in probe order with size, centre distance, min D_P; candidates grouped by list."""
    q = C.load_bin(path)
    qr = G["qrot"][q["qid"]]
    # lists in increasing query-to-list-centre distance (the exported probe array is not guaranteed sorted)
    d0 = ((G["lmean"][q["probes"]].astype(np.float64) - qr[None, :]) ** 2).sum(1)
    po = np.argsort(d0, kind="stable")
    inv = np.empty_like(po); inv[po] = np.arange(len(po))
    rk = inv[q["rank"].astype(np.int64)]
    order = np.argsort(rk, kind="stable")
    rk = rk[order]
    P, DH = q["P"][order], q["DH"][order]
    pos = q["pos"][order].astype(np.int64)
    isgt = np.isin(q["id"][order], q["gt"][:C.K])
    npb = q["nprobe"]
    start = np.searchsorted(rk, np.arange(npb))
    end = np.searchsorted(rk, np.arange(npb), side="right")
    size = end - start
    mins = np.array([P[a:b].min() if b > a else np.inf for a, b in zip(start, end)])
    delta = d0[po]
    return dict(qid=q["qid"], ck=q["ck"], P=P, DH=DH, pos=pos, isgt=isgt, start=start, end=end, size=size, mins=mins, delta=delta, n=len(P))


def features(z, tau, j):
    """features of the lists j of query z before they are read, given the current tau."""
    return np.stack([tau - z["delta"][j], np.log(np.maximum(z["size"][j], 1)), j / len(z["delta"]), z["delta"][j] - z["delta"][0]], 1)


def topck_recall(z, scanned_lists):
    idx = np.concatenate([np.arange(z["start"][j], z["end"][j]) for j in scanned_lists]) if len(scanned_lists) else np.array([], int)
    ck = z["ck"]
    if len(idx) == 0:
        return 0, 0
    sel = idx[np.lexsort((z["pos"][idx], z["P"][idx]))[:ck]]
    fin = sel[np.lexsort((z["pos"][sel], z["DH"][sel]))[:C.K]]
    return int(z["isgt"][fin].sum()), len(idx)


def run_policy(z, L1, risk, eps):
    """phase 1: the first L1 lists; then phases of H lists, each list skipped when risk < eps (tau from what was read)."""
    npb = len(z["delta"])
    scanned = list(range(min(L1, npb)))
    seen = np.sort(np.concatenate([z["P"][z["start"][j]:z["end"][j]] for j in scanned]))
    for a in range(L1, npb, H):
        js = np.arange(a, min(npb, a + H))
        tau = seen[z["ck"] - 1] if len(seen) >= z["ck"] else np.inf
        keep = js if not np.isfinite(tau) else js[risk(z, tau, js) >= eps]
        scanned += keep.tolist()
        if len(keep):
            seen = np.sort(np.concatenate([seen] + [z["P"][z["start"][j]:z["end"][j]] for j in keep]))
    hit, ncand = topck_recall(z, scanned)
    return hit, ncand, len(scanned)


def training_rows(z, L1):
    """(features, label) of the lists after phase 1, label = the list holds a candidate below tau (phase-1 tau)."""
    npb = len(z["delta"])
    seen = np.sort(np.concatenate([z["P"][z["start"][j]:z["end"][j]] for j in range(min(L1, npb))]))
    if len(seen) < z["ck"] or L1 >= npb:
        return None
    tau = seen[z["ck"] - 1]
    js = np.arange(L1, npb)
    return features(z, tau, js), (z["mins"][js] < tau).astype(np.float64), z["mins"][js] - z["delta"][js], z["size"][js]


def fit_logistic(X, y, iters=50, l2=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -40, 40)))
        g = Z.T @ (p - y) + l2 * w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda X2: 1 / (1 + np.exp(-np.clip(np.concatenate([(X2 - mu) / sd, np.ones((len(X2), 1))], 1) @ w, -40, 40)))


def per_query_sim(path):
    z = load_query(path)
    out = dict(qid=z["qid"], nprime={}, oracle=None, pol={})
    full_hit, full_n = topck_recall(z, list(range(len(z["delta"]))))
    out["full"] = (full_hit, full_n)
    for npr in NPRIME:
        out["nprime"][npr] = topck_recall(z, list(range(min(npr, len(z["delta"])))))
    # oracle: lists holding any of the full scan's primary top-ck
    idx = np.lexsort((z["pos"], z["P"]))[:z["ck"]]
    need = np.unique(np.searchsorted(z["end"], idx, side="right"))
    out["oracle"] = topck_recall(z, need.tolist()) + (len(need),)
    for name, models in G["models"].items():
        for L1 in L1S:
            risk = models[L1]
            for eps in EPS:
                out["pol"][(name, L1, eps)] = run_policy(z, L1, risk, eps)
    return out


def fit_models(zs):
    models = {k: {} for k in ("global", "iid_evt", "gev", "logistic")}
    fitinfo = {}
    # pooled per-candidate offsets D_P - delta (for the i.i.d. model), from a subsample of calibration queries
    offs = np.concatenate([np.concatenate([z["P"][z["start"][j]:z["end"][j]] - z["delta"][j] for j in range(len(z["delta"]))])[::7] for z in zs])
    offs.sort()
    F = lambda u: np.searchsorted(offs, u, side="right") / len(offs)
    for L1 in L1S:
        rows = [r for r in (training_rows(z, L1) for z in zs) if r is not None]
        X = np.concatenate([r[0] for r in rows]); y = np.concatenate([r[1] for r in rows])
        mo = np.concatenate([r[2] for r in rows]); sz = np.concatenate([r[3] for r in rows])
        # global: binned on u
        edges = np.quantile(X[:, 0], np.linspace(0, 1, 41)); edges[0], edges[-1] = -np.inf, np.inf
        bins = np.clip(np.searchsorted(edges, X[:, 0], side="right") - 1, 0, 39)
        rate = np.array([(y[bins == b].sum() + 0.5) / ((bins == b).sum() + 1.0) for b in range(40)])
        rate = np.maximum.accumulate(rate)                                     # monotone in u
        models["global"][L1] = (lambda e, r: lambda z, tau, js: r[np.clip(np.searchsorted(e, tau - z["delta"][js], side="right") - 1, 0, 39)])(edges, rate)
        models["iid_evt"][L1] = lambda z, tau, js: 1 - (1 - F(tau - z["delta"][js])) ** np.maximum(z["size"][js], 0)
        # GEV on minus the list-minimum offset (maxima form), per size band
        bands = np.quantile(sz, [0, 0.25, 0.5, 0.75, 1.0]); bands[0], bands[-1] = 0, np.inf
        gp, ks = [], []
        for b in range(4):
            sel = (sz >= bands[b]) & (sz < bands[b + 1]) & np.isfinite(mo)
            v = -mo[sel]
            sub = v[np.random.default_rng(L1 + b).choice(len(v), size=min(len(v), 20000), replace=False)]
            c, loc, sc = stats.genextreme.fit(sub)
            gp.append((c, loc, sc))
            ks.append(float(stats.kstest(sub, "genextreme", args=(c, loc, sc)).statistic))
        def gev_risk(z, tau, js, bands=bands, gp=gp):
            b = np.clip(np.searchsorted(bands, z["size"][js], side="right") - 1, 0, 3)
            out = np.empty(len(js))
            for k in range(4):
                s = b == k
                c, loc, sc = gp[k]
                out[s] = stats.genextreme.sf(-(tau - z["delta"][js[s]]), c, loc, sc)   # P(-min_off > -(tau - delta)) = P(min_off < tau - delta)
            return out
        models["gev"][L1] = gev_risk
        lg = fit_logistic(X, y)
        models["logistic"][L1] = (lambda f: lambda z, tau, js: f(features(z, tau, js)))(lg)
        fitinfo[L1] = dict(rows=int(len(y)), positive_rate=float(y.mean()), gev_ks_by_size_band=ks,
                           gev_params=[list(map(float, g)) for g in gp])
    return models, fitinfo


def calibration_table(zs, models):
    out = []
    for L1 in L1S:
        rows = [(z, r) for z in zs for r in [training_rows(z, L1)] if r is not None]
        for name, mm in models.items():
            pr, ys = [], []
            for z, r in rows:
                npb = len(z["delta"])
                seen = np.sort(np.concatenate([z["P"][z["start"][j]:z["end"][j]] for j in range(L1)]))
                tau = seen[z["ck"] - 1]
                js = np.arange(L1, npb)
                pr.append(mm[L1](z, tau, js)); ys.append(r[1])
            pr, ys = np.concatenate(pr), np.concatenate(ys)
            for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)):
                s = (pr >= lo) & (pr < hi)
                out.append(dict(L1=L1, model=name, bin="[%g,%g)" % (lo, hi), lists=int(s.sum()),
                                predicted_mean=float(pr[s].mean()) if s.any() else None, observed_rate=float(ys[s].mean()) if s.any() else None))
            out.append(dict(L1=L1, model=name, bin="brier", lists=int(len(ys)), predicted_mean=float(((pr - ys) ** 2).mean()), observed_rate=float(ys.mean())))
    return out


def main():
    R = sys.argv[1]
    test = "--test" in sys.argv
    os.makedirs(R, exist_ok=True)
    summ, prow, crow = {}, [], []
    for tag in C.TAGS:
        t0 = time.time()
        m = C.meta(tag)
        G.clear()
        G["qrot"] = C.qrot(tag)
        G["lmean"] = np.fromfile(os.path.join(C.RQ, tag, "lmean.f32"), dtype=np.float32).reshape(-1, m["d"])
        files = C.bin_files(tag)
        qids = [int(os.path.basename(f)[4:-4]) for f in files]
        sp = C.splits(qids)
        by = {s: [f for f, x in zip(files, qids) if sp[x] == s] for s in ("calib", "val", "test")}
        pool = mp.get_context("fork").Pool(24)
        zs = pool.map(load_query, by["calib"], chunksize=4)
        pool.close(); pool.join()
        models, fitinfo = fit_models(zs)
        G["models"] = models
        val = [load_query(f) for f in by["val"][:100]]
        crow += [dict(tag=tag, **r) for r in calibration_table(val, models)]
        del zs, val
        ent = dict(fit=fitinfo, splits={})
        for s in ("calib", "val") + (("test",) if test else ()):
            pool = mp.get_context("fork").Pool(24)
            res = pool.map(per_query_sim, by[s], chunksize=4)
            pool.close(); pool.join()
            nq = len(res)
            fh, fn = np.mean([o["full"][0] for o in res]) / C.K, np.mean([o["full"][1] for o in res])
            e = dict(queries=nq, full=dict(recall=fh, candidates=fn), nprime={}, oracle={}, policies={})
            for npr in NPRIME:
                h = np.mean([o["nprime"][npr][0] for o in res]) / C.K; n = np.mean([o["nprime"][npr][1] for o in res])
                e["nprime"][npr] = dict(recall=h, candidates=n, frac=n / fn)
                prow.append(dict(tag=tag, split=s, policy="nprobe", L1="", eps="", nprobe_eff=npr, recall=h, candidates=n, frac_candidates=n / fn, lists=npr))
            oh = np.mean([o["oracle"][0] for o in res]) / C.K; on = np.mean([o["oracle"][1] for o in res]); ol = np.mean([o["oracle"][2] for o in res])
            e["oracle"] = dict(recall=oh, candidates=on, frac=on / fn, lists=ol)
            prow.append(dict(tag=tag, split=s, policy="oracle", L1="", eps="", nprobe_eff="", recall=oh, candidates=on, frac_candidates=on / fn, lists=ol))
            for key in res[0]["pol"]:
                name, L1, eps = key
                h = np.mean([o["pol"][key][0] for o in res]) / C.K; n = np.mean([o["pol"][key][1] for o in res]); l = np.mean([o["pol"][key][2] for o in res])
                e["policies"]["%s|%d|%g" % key] = dict(recall=h, candidates=n, frac=n / fn, lists=l)
                prow.append(dict(tag=tag, split=s, policy=name, L1=L1, eps=eps, nprobe_eff="", recall=h, candidates=n, frac_candidates=n / fn, lists=l))
            # per-query recall arrays for paired comparisons (policy vs the nprobe cut at matched mean candidates)
            np.savez_compressed(os.path.join(R, "evt_%s_%s_perquery.npz" % (tag, s)), qid=np.array([o["qid"] for o in res]),
                                full=np.array([o["full"] for o in res]),
                                **{"nprime_%d" % k: np.array([o["nprime"][k] for o in res]) for k in NPRIME},
                                **{"pol_%s_%d_%g" % k: np.array([o["pol"][k] for o in res]) for k in res[0]["pol"]})
            ent["splits"][s] = e
        ent["seconds"] = time.time() - t0
        summ[tag] = ent
        print(tag, "full", ent["splits"]["val"]["full"], "oracle", ent["splits"]["val"]["oracle"], "%.0fs" % ent["seconds"], flush=True)
        json.dump(summ, open(os.path.join(R, "evt_sim%s.json" % ("_test" if test else "")), "w"), indent=1, default=float)
        for fn_, rows in (("EVT_POLICIES%s.csv" % ("_test" if test else ""), prow), ("evt_calibration.csv", crow)):
            with open(os.path.join(R, fn_), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
