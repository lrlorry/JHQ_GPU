#!/usr/bin/env python3
"""Synthetic check of analyze.py: a small clustered index with 1-bit codes, written in the two
export formats (the local-first per-query bins and demo_jhq_v66_export_primary's files), then

  1  per_query on one query against independent re-implementations: the primary distance
     rebuilt from the code bytes, A0's stage-1 selection, its stage-2 sectors counted as a set
     of 32-byte addresses, and its recall;
  2  analyze.py end to end: p = 1 must reproduce the baseline exactly, coverage must not fall
     as L grows, and the simulated v65 must equal the bins' v65 on every query.

  python3 selftest.py <scratch dir>
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402


def build(out):
    rng = np.random.default_rng(7)
    d, nlist, N, nq, nprobe, ck, k = 256, 64, 12803, 60, 8, 50, 10
    M = d // 8
    c = np.float32(0.0625)
    cent = rng.standard_normal((nlist, d))
    assign = np.sort(rng.integers(0, nlist, N))
    X = cent[assign] * 0.7 + rng.standard_normal((N, d))
    off = np.searchsorted(assign, np.arange(nlist + 1)).astype(np.int32)
    prim = np.packbits((X > 0).astype(np.uint8), axis=1)                      # [N, M], MSB = first dim
    Qs = cent[rng.integers(0, nlist, nq)] * 0.7 + rng.standard_normal((nq, d))
    Qs *= 0.05
    Xs = X * 0.05
    tag = "syn_np8_a5"
    pp, lf = os.path.join(out, "pp", tag), os.path.join(out, "lf", tag)
    os.makedirs(pp, exist_ok=True); os.makedirs(lf, exist_ok=True)
    prim.tofile(os.path.join(pp, "primary.u8"))
    off.tofile(os.path.join(pp, "list_offsets.i32"))
    qids = rng.permutation(nq).astype(np.int32)
    Qs[qids].astype(np.float64).tofile(os.path.join(pp, "qrot.f64"))
    qids.tofile(os.path.join(pp, "qids.i32"))
    json.dump(dict(N=N, d=d, M=M, Ds=8, K=256, nlist=nlist, nprobe=nprobe, alpha=5, seed=0, n_queries=nq,
                   levels=[-float(c), float(c)], levels_symmetric=True, codebook_level_mismatches=0,
                   words_per_vector=d // 32), open(os.path.join(pp, "meta.json"), "w"))
    yh = np.where(X > 0, float(c), -float(c))
    cd = ((cent[:, None, :] * 0.7 * 0.05 - Qs[None, :, :]) ** 2).sum(2)        # [nlist, nq]
    for qid in range(nq):
        q = Qs[qid]
        probes = np.argsort(cd[:, qid], kind="stable")[:nprobe].astype(np.int32)
        pos = np.concatenate([np.arange(off[l], off[l + 1]) for l in probes]).astype(np.int32)
        rank = np.concatenate([np.full(off[l + 1] - off[l], r) for r, l in enumerate(probes)]).astype(np.int32)
        P = ((q[None, :] - yh[pos]) ** 2).sum(1)
        DH = ((q[None, :] - Xs[pos]) ** 2).sum(1) - (q ** 2).sum() + rng.normal(0, 1e-4, len(pos))
        ids = (pos * 7 + 3).astype(np.int32)                                   # ids differ from positions
        allD = ((Xs - q[None, :]) ** 2).sum(1)
        gt = (np.argsort(allD, kind="stable")[:k] * 7 + 3).astype(np.int32)
        oP = np.lexsort((pos, P))
        top = oP[:ck]
        fin = top[np.lexsort((pos[top], DH[top]))[:k]]
        n = len(pos)
        with open(os.path.join(lf, "rb_q%d.bin" % qid), "wb") as f:
            np.array([qid, n, nprobe, ck, k], np.int32).tofile(f)
            probes.tofile(f); pos.tofile(f); ids.tofile(f); rank.tofile(f)
            for a in (P, DH, np.zeros(n), np.zeros(n)):
                a.astype(np.float64).tofile(f)
            np.array([(q ** 2).sum()], np.float64).tofile(f)
            ids[fin].tofile(f); np.zeros(k, np.float32).tofile(f); np.zeros(k, np.float64).tofile(f)
            pos[top].astype(np.int32).tofile(f); gt.tofile(f)
    return out, tag, dict(N=N, d=d, M=M, ck=ck, k=k, c=float(c))


def check_one(out, tag, cfg):
    analyze.setup(os.path.join(out, "lf"), os.path.join(out, "pp"), tag)
    path = os.path.join(out, "lf", tag, "rb_q5.bin")
    o = analyze.per_query((path, [cfg["ck"]]))
    assert o["prim_rel_dev"] < 1e-12, o["prim_rel_dev"]
    assert o["sim_equals_v65"] and o["topck_pos_match"]
    q = analyze.load(path)
    N, W, ck = cfg["N"], cfg["d"] // 32, cfg["ck"]
    prim = np.fromfile(os.path.join(out, "pp", tag, "primary.u8"), np.uint8).reshape(N, -1)
    qr = analyze.G["qrot"][analyze.G["qrow"][5]]
    s = np.unpackbits(prim[q["pos"]], axis=1).astype(np.float64) * 2 - 1      # [n, d]
    Sw = (s * cfg["c"] * qr[None, :]).reshape(len(q["pos"]), W, 32).sum(2)
    vi = analyze.VARIANTS.index(("A0", 0.0))
    pos = q["pos"].astype(np.int64)
    gt = set(q["gt"].tolist())
    for pi, p in enumerate(analyze.PS):
        nJ = max(1, int(round(p * W)))
        order1 = np.lexsort((pos, -Sw[:, :nJ].sum(1)))
        for li, m in enumerate(analyze.LMULT):
            L = min(m * ck, len(pos))
            sel = order1[:L]
            # per list, as analyze.py counts; the true distinct count can only be lower (a sector
            # shared at the boundary of two probed lists adjacent in storage)
            per_list = sum(len({((w * N + int(x)) * 4) // 32 for w in range(nJ, W) for x in pos[sel][q["rank"][sel] == r]})
                           for r in range(q["nprobe"]))
            secs = {((w * N + int(x)) * 4) // 32 for w in range(nJ, W) for x in pos[sel]}
            assert abs(o["S2"][vi, pi, li] - 32 * per_list) < 1e-9, (p, m, o["S2"][vi, pi, li], 32 * per_list)
            assert len(secs) <= per_list
            s1 = {((w * N + int(x)) * 4) // 32 for w in range(nJ) for x in pos}
            # stage 1 counts each probed list's segment on its own; lists adjacent in storage share
            # at most one sector a row, so the set count may be lower by at most that much
            assert 0 <= o["S1"][vi, pi] - 32 * len(s1) <= 32 * nJ * q["nprobe"], (o["S1"][vi, pi], 32 * len(s1))
            top = sel[np.lexsort((pos[sel], q["P"][sel]))[:ck]]
            fin = top[np.lexsort((pos[top], q["DH"][top]))[:10]]
            rec = len(gt & set(q["id"][fin].tolist())) / 10
            assert o["R"][vi, pi, li, 0] == round(rec * 10), (p, m, o["R"][vi, pi, li, 0], rec)
    print("per-query checks passed: reconstruction %.2e, A0 sectors/recall match the brute force" % o["prim_rel_dev"])


def check_all(out):
    res = os.path.join(out, "results")
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), os.path.join(out, "lf"),
                        os.path.join(out, "pp"), res, "4"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-4000:])
    assert r.returncode == 0
    rows = list(csv.DictReader(open(os.path.join(res, "RECALL_VS_BYTES.csv"))))
    for x in rows:
        if float(x["p"]) == 1.0:
            assert float(x["delta"]) == 0.0 and float(x["same_final_as_baseline"]) == 1.0, x
            assert abs(float(x["ideal_bytes_ratio"]) - 1.0) < 1e-12 and abs(float(x["sector_bytes_ratio"]) - 1.0) < 1e-12, x
    cov = list(csv.DictReader(open(os.path.join(res, "COVERAGE_VS_READ_RATIO.csv"))))
    by = {}
    for x in cov:
        by.setdefault((x["policy"], x["p"], x["lam"]), []).append((int(x["L_over_ck"]), float(x["coverage_full_primary_topck"])))
    for key, v in by.items():
        v.sort()
        assert all(b[1] >= a[1] - 1e-12 for a, b in zip(v[:-1], v[1:])), (key, v)
    s = json.load(open(os.path.join(res, "analysis_summary.json")))
    w = s["workloads"]["syn_np8_a5"]
    assert w["sim_equals_v65_final"] == 1.0 and w["prim_reconstruction_max_rel_dev"] < 1e-12, w
    print("end-to-end checks passed:", len(rows), "rows; T* =", w["T_star"])


if __name__ == "__main__":
    out, tag, cfg = build(sys.argv[1])
    check_one(out, tag, cfg)
    check_all(out)
