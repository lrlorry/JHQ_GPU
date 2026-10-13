#!/usr/bin/env python3
"""Step 0 of the rank-optimal quantization study, per workload:

  1  y = Pi x for every base vector (GPU, fp32, TF32 off), stored in position order: RQ/<tag>/xpos.f32
  2  checks of the offline re-implementation against the paper index's own export:
       primary bits  b = 1[y > 0] against primary.u8
       residual codes (nearest level of res_c1d to y - y_hat) and corr = 2<y_hat, r_hat>
       D_H = ||q' - y_hat - r_hat||^2 against the per-candidate FP64 D_H of experiments/local_first_refinement
       residual codebook retrained offline (lloyd_1d, the paper index's procedure) against res_c1d
  3  data diagnostics behind THEORY.md: norms, per-coordinate mean and spread after the rotation, sign-bit
     bias, query-vs-base offset, coordinate correlation, marginal shape, neighbour correlations
  4  per-list means of y (RQ/<tag>/lmean.f32) for the list-conditional variants

  python3 prep.py <tag> <results dir>
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402


def main():
    tag, R = sys.argv[1], sys.argv[2]
    os.makedirs(R, exist_ok=True)
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    c = float(np.float32(m["levels"][1]))
    out = dict(tag=tag, N=N, d=d, M=M, level=c)
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    A = torch.from_numpy(C.rotation(tag)).cuda()
    basep, queryp, gtp = C.base_paths(tag)
    base = np.memmap(basep, dtype=np.float32, mode="r", shape=(N, d))
    ids = np.fromfile(os.path.join(C.RQ, tag, "list_ids.i32"), dtype=np.int32)
    xp_path = os.path.join(C.RQ, tag, "xpos.f32")
    xp = np.memmap(xp_path, dtype=np.float32, mode="w+", shape=(N, d))
    step = 131072
    for a in range(0, N, step):
        b = min(N, a + step)
        x = torch.from_numpy(np.ascontiguousarray(base[ids[a:b]])).cuda()
        xp[a:b] = (x @ A.T).cpu().numpy()
    xp.flush()
    del xp
    out["rotate_seconds"] = time.time() - t0
    xp = C.xpos(tag)

    # ---- 2. checks
    prim = np.memmap(os.path.join(C.PP, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M))
    rcodes = np.memmap(os.path.join(C.RQ, tag, "res_codes.u8"), dtype=np.uint8, mode="r", shape=(N, d))
    corr = np.fromfile(os.path.join(C.RQ, tag, "corr.f32"), dtype=np.float32)
    cb = np.fromfile(os.path.join(C.RQ, tag, "res_c1d.f32"), dtype=np.float32).reshape(M, 256)
    bit_mis = 0; res_mis = 0; corr_dev = 0.0
    for a in range(0, N, step):
        b = min(N, a + step)
        y = np.asarray(xp[a:b])
        bits = np.packbits(y > 0, axis=1)
        bit_mis += int(np.unpackbits(bits ^ np.asarray(prim[a:b]), axis=1).sum())
        yhat = np.where(np.unpackbits(np.asarray(prim[a:b]), axis=1).astype(bool), c, -c).astype(np.float32)
        rc = C.encode_residual(y - yhat, cb)
        res_mis += int((rc != np.asarray(rcodes[a:b])).sum())
        rh = C.decode_residual(np.asarray(rcodes[a:b]), cb)
        cr = 2.0 * (yhat.astype(np.float64) * rh).sum(1)
        corr_dev = max(corr_dev, float(np.abs(cr - corr[a:b]).max()))
    out["check_primary_bit_mismatches"] = bit_mis
    out["check_primary_bits_total"] = N * d
    out["check_residual_code_mismatches"] = res_mis
    out["check_corr_max_abs_dev"] = corr_dev
    qr = C.qrot(tag)
    files = C.bin_files(tag)
    dh_dev = 0.0
    for f in files[:30]:
        q = C.load_bin(f)
        pos = q["pos"].astype(np.int64)
        yhat = np.where(np.unpackbits(np.asarray(prim[pos]), axis=1).astype(bool), c, -c)
        xh = yhat + C.decode_residual(np.asarray(rcodes[pos]), cb)
        dh = ((qr[q["qid"]][None, :] - xh) ** 2).sum(1)
        dh_dev = max(dh_dev, float(np.max(np.abs(dh - q["DH"]) / np.maximum(q["DH"], 1e-30))))
    out["check_DH_max_rel_dev_30_queries"] = dh_dev
    # residual codebook, retrained offline the paper index's way, on its training sample
    rows = C.res_sample_rows(N)
    inv = np.empty(N, np.int64); inv[ids] = np.arange(N)
    ps = np.sort(inv[rows])
    y = np.asarray(xp[ps])
    resid = y - np.where(y > 0, c, -c).astype(np.float32)
    t1 = time.time()
    cb2, iters = C.train_residual(resid, M)
    out["retrain_seconds"] = time.time() - t1
    spacing = np.median(np.diff(cb, axis=1), axis=1)
    out["retrain_codebook_max_abs_dev_over_median_spacing"] = float((np.abs(cb2 - cb).max(1) / spacing).max())
    out["retrain_lloyd_iterations"] = [int(min(iters)), int(np.median(iters)), int(max(iters))]
    sub = np.arange(0, N, max(1, N // 200000))
    yh = np.where(np.asarray(xp[sub]) > 0, c, -c).astype(np.float32)
    out["retrain_code_agreement_on_sample"] = float((C.encode_residual(np.asarray(xp[sub]) - yh, cb2) == np.asarray(rcodes[sub])).mean())

    # ---- 3. diagnostics
    smp = np.asarray(xp[np.arange(0, N, max(1, N // 200000))], dtype=np.float64)
    mu = smp.mean(0); sd = smp.std(0)
    nrm = np.linalg.norm(smp, axis=1)
    pb = (smp > 0).mean(0)
    Q = C.fvecs(queryp)
    qy = (Q.astype(np.float64) @ C.rotation(tag).T.astype(np.float64))
    cov = np.cov((smp[:50000] - mu).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    corr_m = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    off = np.abs(corr_m[np.triu_indices(d, 1)])
    z = (smp - mu) / sd
    out["diag"] = dict(
        norm_mean=float(nrm.mean()), norm_std=float(nrm.std()),
        rms_per_coordinate=float(np.sqrt((smp ** 2).mean())), sigma_paper=float(c / 0.6744897501960817),
        mean_vector_norm=float(np.linalg.norm(mu)), mean_vector_norm_over_rms_norm=float(np.linalg.norm(mu) / np.sqrt((nrm ** 2).mean())),
        abs_mean_over_sd_mean=float((np.abs(mu) / sd).mean()), abs_mean_over_sd_max=float((np.abs(mu) / sd).max()),
        sd_min_over_max=float(sd.min() / sd.max()), sd_cv=float(sd.std() / sd.mean()),
        bit_bias_abs_mean=float(np.abs(2 * pb - 1).mean()), bit_bias_abs_p90=float(np.quantile(np.abs(2 * pb - 1), 0.9)),
        bit_entropy_mean=float(np.mean(-(pb * np.log2(np.maximum(pb, 1e-12)) + (1 - pb) * np.log2(np.maximum(1 - pb, 1e-12))))),
        query_norm_mean=float(np.linalg.norm(qy, axis=1).mean()),
        query_mean_minus_base_mean_norm=float(np.linalg.norm(qy.mean(0) - mu)),
        query_mean_norm=float(np.linalg.norm(qy.mean(0))),
        query_bit_bias_abs_mean=float(np.abs(2 * (qy > 0).mean(0) - 1).mean()),
        cov_top1_share=float(ev[0] / ev.sum()), cov_top10_share=float(ev[:10].sum() / ev.sum()),
        cov_participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
        coord_abs_corr_mean=float(off.mean()), coord_abs_corr_p99=float(np.quantile(off, 0.99)),
        skew_abs_mean=float(np.abs((z ** 3).mean(0)).mean()), excess_kurtosis_mean=float(((z ** 4).mean(0) - 3).mean()))
    # neighbour correlations (raw cosine and centred correlation), 200 queries
    gt = C.ivecs(gtp)[:, :C.K]
    sel = np.arange(min(200, len(Q)))
    nb = np.asarray(base[np.unique(gt[sel].ravel())], dtype=np.float64)
    idmap = {int(v): i for i, v in enumerate(np.unique(gt[sel].ravel()))}
    cos1, cos10, ccor1, ccor10 = [], [], [], []
    mux = base[np.arange(0, N, max(1, N // 200000))].astype(np.float64).mean(0)
    for i in sel:
        qq = Q[i].astype(np.float64)
        v = nb[[idmap[int(t)] for t in gt[i]]]
        cs = v @ qq / (np.linalg.norm(v, axis=1) * np.linalg.norm(qq))
        cc = (v - mux) @ (qq - mux) / (np.linalg.norm(v - mux, axis=1) * np.linalg.norm(qq - mux))
        cos1.append(cs[0]); cos10.append(cs.mean()); ccor1.append(cc[0]); ccor10.append(cc.mean())
    cand_cos, cand_ccor = [], []
    for f in files[:50]:
        q = C.load_bin(f)
        qq = Q[q["qid"]].astype(np.float64)
        v = np.asarray(base[np.sort(q["id"][::max(1, q["n"] // 2000)])], dtype=np.float64)
        cand_cos.append(np.median(v @ qq / (np.linalg.norm(v, axis=1) * np.linalg.norm(qq))))
        cand_ccor.append(np.median((v - mux) @ (qq - mux) / (np.linalg.norm(v - mux, axis=1) * np.linalg.norm(qq - mux))))
    out["neighbours"] = dict(cos_top1=float(np.mean(cos1)), cos_top10_mean=float(np.mean(cos10)),
                             centred_corr_top1=float(np.mean(ccor1)), centred_corr_top10_mean=float(np.mean(ccor10)),
                             candidate_median_cos=float(np.mean(cand_cos)), candidate_median_centred_corr=float(np.mean(cand_ccor)))

    # ---- 4. per-list means
    offl = C.list_offsets(tag)
    lm = np.zeros((len(offl) - 1, d), np.float32)
    for l0 in range(0, len(offl) - 1, 256):
        l1 = min(len(offl) - 1, l0 + 256)
        a, b = int(offl[l0]), int(offl[l1])
        y = np.asarray(xp[a:b], dtype=np.float64)
        cs = np.vstack([np.zeros((1, d)), np.cumsum(y, axis=0)])
        st = offl[l0:l1 + 1] - a
        cnt = np.maximum(np.diff(st), 1)[:, None]
        lm[l0:l1] = ((cs[st[1:]] - cs[st[:-1]]) / cnt).astype(np.float32)
    lm.tofile(os.path.join(C.RQ, tag, "lmean.f32"))
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(R, "prep_%s.json" % tag), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
