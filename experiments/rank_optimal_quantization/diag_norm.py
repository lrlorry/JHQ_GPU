#!/usr/bin/env python3
"""Round-5 diagnosis, second step: is the within-list loss of the per-list codes the ||y_hat||^2 term of D_P?
Calibration and validation queries only; a diagnostic, not a sixth round (nothing here is carried to the test split).

  python3 diag_norm.py <results dir> [variant ...]        (default: median list_median)

D_P = ||q'||^2 - 2<q', y_hat> + ||y_hat||^2. Every base vector of the three workloads has norm 1, so the last term
is pure estimation noise for the ranking: it is constant for JHQ (+-c levels), varies with the bits for per-dimension
levels whose midpoints are not 0, and varies with the bits and the list for per-list levels. Per query and code:
  the spread of ||y_hat||^2 over the candidates, between lists and within lists
  recall@10 against T (refined by the code's own D_H) and pairwise misranking (same list / other lists) for two primary
  keys: D_P, and the inner product alone (IP: rank by -<q', y_hat>, i.e. D_P with ||y_hat||^2 replaced by the known 1)
Output: diag_norm.json, diag_norm_<tag>_perquery.npz (recall curves) and a printed table.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402
import eval_variants as EV  # noqa: E402

G = EV.G


def code_terms(P, qr, pos, chunk=8192):
    """<q', y_hat> and ||y_hat||^2 of the candidates' primary reconstructions."""
    n = len(pos)
    ip = np.empty(n); n2 = np.empty(n)
    for a in range(0, n, chunk):
        ps = pos[a:a + chunk]
        bits = np.unpackbits(np.asarray(P["bits"][ps]), axis=1).astype(bool)
        if P["per_list"]:
            l = G["lpos"][ps]
            yh = np.where(bits, P["hi32"][l], P["lo32"][l]).astype(np.float64)
        else:
            yh = np.where(bits, P["hi32"][None, :], P["lo32"][None, :]).astype(np.float64)
        ip[a:a + chunk] = yh @ qr
        n2[a:a + chunk] = np.einsum("ij,ij->i", yh, yh)
    return ip, n2


def split_var(x, inv, cnt):
    mu = np.bincount(inv, weights=x) / cnt
    return float(np.sum(cnt * (mu - x.mean()) ** 2) / len(x)), float(np.sum((x - mu[inv]) ** 2) / len(x))


def per_query(path):
    q = C.load_bin(path)
    pos = q["pos"].astype(np.int64)
    qr = G["qrot"][q["qid"]]
    lp = G["lpos"][pos]
    isgt = np.isin(q["id"], q["gt"][:C.K])
    neg = ~isgt
    _, inv, cnt = np.unique(lp, return_inverse=True, return_counts=True)
    keys = {("JHQ", "D_P"): (q["P"], q["DH"])}
    out = dict(qid=q["qid"], R={}, MR={}, NV={})
    for v in G["variants"]:
        dp, dh = EV.distances(G["var"][v], qr, pos)
        ip, n2 = code_terms(G["var"][v], qr, pos)
        keys[(v, "D_P")] = (dp, dh)
        keys[(v, "IP")] = (-ip, dh)
        out["NV"][v] = split_var(n2, inv, cnt)
    for k, (key, dh) in keys.items():
        out["R"][k], _ = C.recall_curve(key, dh, pos, isgt, G["Ts"])
        ms = ns = mc = nc = 0
        for g in np.where(isgt)[0]:
            same = neg & (lp == lp[g])
            ms += int((key[same] <= key[g]).sum()); ns += int(same.sum())
            mc += int((key[neg & ~same] <= key[g]).sum()); nc += int((neg & ~same).sum())
        out["MR"][k] = (ms / max(ns, 1), mc / max(nc, 1))
    return out


def boot(x, rng, B=10000):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    R = sys.argv[1]
    variants = sys.argv[2:] or ["median", "list_median"]
    rng = np.random.default_rng(C.SEED)
    summ = {}
    for tag in C.TAGS:
        t0 = time.time()
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        G.clear()
        G["qrot"] = C.qrot(tag)
        G["lpos"] = C.list_of_pos(tag)
        G["variants"] = variants
        G["var"] = {}
        for v in variants:
            pz = np.load(os.path.join(C.RQ, tag, "var_" + v, "params.npz"))
            G["var"][v] = dict(per_list=bool(pz["per_list"]), lo32=pz["lo"].astype(np.float32), hi32=pz["hi"].astype(np.float32),
                               res_cb=pz["res_cb"].astype(np.float32),
                               bits=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                               res=np.memmap(os.path.join(C.RQ, tag, "var_" + v, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d)))
        files = C.bin_files(tag)
        qids = [int(os.path.basename(f)[4:-4]) for f in files]
        sp = C.splits(qids)
        files = [f for f, x in zip(files, qids) if sp[x] in ("calib", "val")]       # the test split is not read here
        ck = C.load_bin(files[0])["ck"]
        G["Ts"] = Ts = C.t_grid(ck, 4 * ck)
        pool = mp.get_context("fork").Pool(24)
        res = pool.map(per_query, files, chunksize=4)
        pool.close(); pool.join()
        split = np.array([sp[o["qid"]] for o in res])
        assert set(split) <= {"calib", "val"}
        ks = list(res[0]["R"].keys())
        Rr = {k: np.stack([o["R"][k] for o in res]).astype(np.float64) / C.K for k in ks}
        ent = dict(ck=ck, T_grid=Ts, target={}, codes={}, norm_spread={})
        for s in ("calib", "val"):
            ent["target"][s] = float(Rr[("JHQ", "D_P")][split == s, Ts.index(ck)].mean())
        for v in variants:
            ent["norm_spread"][v] = dict(var_between=float(np.mean([o["NV"][v][0] for o in res])),
                                         var_within=float(np.mean([o["NV"][v][1] for o in res])))
        J = Rr[("JHQ", "D_P")]
        for k in ks:
            e = dict(misrank_same_list=float(np.mean([o["MR"][k][0] for o in res])),
                     misrank_other_list=float(np.mean([o["MR"][k][1] for o in res])), T_min={}, diff_vs_JHQ={})
            for s in ("calib", "val"):
                cur = Rr[k][split == s].mean(0)
                ok = [t for t, r in zip(Ts, cur) if r >= ent["target"][s] - 1e-12]
                e["T_min"][s] = min(ok) if ok else None
            for t in sorted({max(10, ck // 8), max(10, ck // 4), max(10, ck // 2), ck}):
                dd = Rr[k][:, Ts.index(t)] - J[:, Ts.index(t)]
                e["diff_vs_JHQ"][str(t)] = dict(mean=float(dd.mean()), ci95=list(boot(dd, rng)))
            ent["codes"]["%s|%s" % k] = e
        ent["seconds"] = time.time() - t0
        summ[tag] = ent
        np.savez_compressed(os.path.join(R, "diag_norm_%s_perquery.npz" % tag), qid=np.array([o["qid"] for o in res]), split=split,
                            Ts=np.array(Ts), **{"R_%s|%s" % k: Rr[k].astype(np.float32) for k in ks})
        print("==", tag, "targets", ent["target"], "  spread of ||y_hat||^2 (between / within lists):",
              {v: "%.2e / %.2e" % (x["var_between"], x["var_within"]) for v, x in ent["norm_spread"].items()}, flush=True)
        for k, e in ent["codes"].items():
            print("   %-18s T_min calib %-5s val %-5s | misrank same %.4f other %.2e | %s" % (
                k, e["T_min"]["calib"], e["T_min"]["val"], e["misrank_same_list"], e["misrank_other_list"],
                " ".join("T=%s %+.2f [%+.2f,%+.2f]" % (t, 100 * x["mean"], 100 * x["ci95"][0], 100 * x["ci95"][1])
                         for t, x in e["diff_vs_JHQ"].items())), flush=True)
        json.dump(summ, open(os.path.join(R, "diag_norm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
