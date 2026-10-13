#!/usr/bin/env python3
"""Query-adaptive progressive primary screening: the offline pilot (FP64).

  python3 analyze.py <lf dir> <pp dir> <results dir> [workers]

Inputs, per workload tag:
  <lf dir>/<tag>/rb_q<qid>.bin   experiments/local_first_refinement's export (the paper index,
      v65's own search): every probed candidate in v65's probe order with its position, id,
      probe rank, FP64 D_prim and D_H; ||q'||^2; v65's top-10; the ground truth.
  <pp dir>/<tag>/                demo_jhq_v66_export_primary: primary.u8 [N][M] code bytes,
      list_offsets.i32, qrot.f64 (q' = Pi q in FP64, the lf export's own product), qids.i32,
      meta.json (levels, the bitwise symmetry check, the word map).

Representation (checked by the exporter and again here): 1 bit a dimension, two levels -c, +c,
yhat_j = c s_j. D_prim = ||q'||^2 + d c^2 - 2 sum_w S_w with S_w = sum_{j in w} q'_j yhat_j over
the 32-dimension words w the scan reads (word w = dimensions 32w..32w+31). Every candidate's
S_w is rebuilt from its code bytes and the reconstruction is compared with the exported D_prim.

The method: stage 1 reads a fraction p of every candidate's words (|J| = round(p W)), estimates
the score, keeps the global top-L of the query's candidates by (estimate, position); stage 2 reads
the remaining words of those L, takes the exact top-T by (D_prim, position), then refines them as
v65 does and returns the 10 smallest D_H (ties by position). Word choice:
  A0   storage order (words 0, 1, ...), missing words counted as 0 -- the old v21 cascade's prefix
  A1   the query's order: largest I_w = sum_{j in w} (c q'_j)^2 first, one order for all lists
  A2   per probed list l: largest sum_{j in w} (c q'_j)^2 (1 - (lam mu_lj)^2) first,
       mu_lj = E[s_j | l] over the list's members (int8-quantised, as it would be stored)
  A3   A2's words, plus the missing words' list-conditional mean sum_{j not in J} c q'_j lam mu_lj
  A0c, A1c   A0's / A1's words plus the same mean correction (separates order from correction)
lam in {0, .25, .5, .75, 1} is chosen on the calibration queries only, per (policy, p, L), by mean
coverage of the full-primary top-ck; lam = 0 turns A0c/A1c/A2/A3 back into A0/A1/A1/A1.

Baselines: full(T) = the exact primary top-T, refined (full(ck) is v65; the simulation is checked
against v65's own results). T* = the smallest T of the grid whose validation recall reaches
full(ck)'s (the local-first pilot's rule). Progressive configurations end in T in {ck, T*}.

Pre-declared decision rules (fixed before any result was seen):
  non-inferiority  test-split paired bootstrap 95% CI of mean recall@10 difference vs the baseline
                   has lower end >= -0.001 (0.1 pp). A CI containing 0 is not called equality.
  operating point  per (policy, T_final): among (p, L) whose VALIDATION mean difference vs
                   full(T_final) is >= -0.0005, the one with the smallest validation ideal bytes;
                   reported on TEST.
  severe tail      test 1st percentile of the per-query difference < -0.1, or worst < -0.2.
  novelty          the best of A1/A1c/A2/A3 must beat A0 (the old fixed prefix) by >= 0.05 of
                   B_old in ideal bytes at its operating point; otherwise the result is the
                   v21/v39 cascade again.
Splits: seeded permutation of the query ids, 40% calibration / 20% validation / 40% test.

Bytes per query (primary codes only): B_old = n W 4. Ideal B_new = n |J| 4 + L (W-|J|) 4.
Sector level (32 B): stage 1 reads, per probed list, the contiguous segment of each read word row;
stage 2 reads, per unread word row, the distinct sectors holding the selected candidates (a sector
holds 8 consecutive positions of one row; row w starts at element w N, so its alignment is
(w N) mod 8). Lists adjacent in storage that are both probed share at most one sector per row at
their boundary, counted twice here, for B_old as well. Metadata: nprobe d bytes of int8 mu for the
policies that use mu. Scratch: the top-L (position, score) pairs written and read, 16 L bytes.
"""
import csv
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

K = 10
SEED = 20261013
PS = (0.125, 0.25, 0.5, 0.75, 1.0)
LMULT = (1, 2, 4, 8, 16, 32)
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
RATIOS = (1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2, 3 / 4, 1.0, 1.25, 1.5)
POLS = ("A0", "A0c", "A1", "A1c", "A2", "A3")
LAM_POLS = ("A0c", "A1c", "A2", "A3")
VARIANTS = [("A0", 0.0), ("A1", 0.0)] + [(p, l) for p in LAM_POLS for l in LAMS]
USES_MU = {"A0": False, "A1": False, "A0c": True, "A1c": True, "A2": True, "A3": True}
NI_MARGIN = 0.001
VAL_TOL = 0.0005
NOVELTY = 0.05
NBOOT = 5000

G = {}   # per-workload state, inherited by the workers


def load(path):
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


def t_grid(ck):
    return sorted(set(max(K, int(np.floor(ck * r))) for r in RATIOS))


def setup(lf_dir, pp_dir, tag):
    """the workload's codes, levels, list means; returns the summary of the checks."""
    meta = json.load(open(os.path.join(pp_dir, tag, "meta.json")))
    N, d, M = meta["N"], meta["d"], meta["M"]
    assert meta["Ds"] == 8 and meta["K"] == 256 and d == 8 * M and M % 4 == 0
    W = M // 4
    lev = np.array(meta["levels"], dtype=np.float32).astype(np.float64)   # the float32 levels, exactly
    assert meta["levels_symmetric"] and lev[0] == -lev[1] and meta["codebook_level_mismatches"] == 0, meta
    prim = np.memmap(os.path.join(pp_dir, tag, "primary.u8"), dtype=np.uint8, mode="r", shape=(N, M))
    off = np.fromfile(os.path.join(pp_dir, tag, "list_offsets.i32"), dtype=np.int32).astype(np.int64)
    qids = np.fromfile(os.path.join(pp_dir, tag, "qids.i32"), dtype=np.int32)
    qrot = np.fromfile(os.path.join(pp_dir, tag, "qrot.f64"), dtype=np.float64).reshape(len(qids), d)
    nlist = len(off) - 1
    # bits of a code byte, most significant first = the subspace's first dimension; yhat = lev[bit]
    bits = ((np.arange(256)[:, None] >> (7 - np.arange(8))[None, :]) & 1).astype(np.int64)
    levmat = lev[bits]                                                      # [256, 8]
    # mu_lj = E[s_j | l] from all members, then int8 (as stored): round(127 mu) / 127
    t0 = time.time()
    frac = np.zeros((nlist, d), dtype=np.float64)
    cnt = np.diff(off)
    step = 128
    for l0 in range(0, nlist, step):
        l1 = min(nlist, l0 + step)
        a, b = int(off[l0]), int(off[l1])
        if b == a:
            continue
        bt = np.unpackbits(np.asarray(prim[a:b]), axis=1)                      # [rows, d], MSB first
        cs = np.zeros((b - a + 1, d), dtype=np.int32)
        np.cumsum(bt, axis=0, dtype=np.int32, out=cs[1:])
        st = off[l0:l1 + 1] - a
        frac[l0:l1] = (cs[st[1:]] - cs[st[:-1]]) / np.maximum(cnt[l0:l1], 1)[:, None]
    mu = 2.0 * frac - 1.0
    mu[cnt == 0] = 0.0
    mu_q = np.round(mu * 127.0) / 127.0
    mu_stats = dict(seconds=time.time() - t0,
                    mean_mu2_weighted=float((mu ** 2 * cnt[:, None]).sum() / (cnt.sum() * d)),
                    mean_abs_mu_weighted=float((np.abs(mu) * cnt[:, None]).sum() / (cnt.sum() * d)),
                    global_mean_s_abs_mean=float(np.abs((mu * cnt[:, None]).sum(0) / cnt.sum()).mean()),
                    int8_max_abs_err=float(np.abs(mu_q - mu).max()),
                    list_size_mean=float(cnt.mean()), list_size_max=int(cnt.max()), empty_lists=int((cnt == 0).sum()))
    G.clear()
    G.update(N=N, d=d, M=M, W=W, c=float(lev[1]), y2=d * float(lev[1]) ** 2, prim=prim, off=off,
             qrow={int(x): i for i, x in enumerate(qids)}, qrot=qrot, levmat=levmat, mu_q=mu_q,
             shift=np.array([(w * N) % 8 for w in range(W)], dtype=np.int64))
    G["onehot"] = np.zeros((W, 8)); G["onehot"][np.arange(W), G["shift"]] = 1.0
    return meta, mu_stats


def per_query(args):
    path, Tfin = args
    q = load(path)
    n, ck, nprobe = q["n"], q["ck"], q["nprobe"]
    pos, rank, P, DH, ids = q["pos"].astype(np.int64), q["rank"].astype(np.int64), q["P"], q["DH"], q["id"]
    isgt = np.isin(ids, q["gt"][:K])
    orderP = np.lexsort((pos, P))
    rH = np.empty(n, dtype=np.int64); rH[np.lexsort((pos, DH))] = np.arange(n)
    orderH = np.argsort(rH)

    def final(members):
        if len(members) > K:
            members = members[np.argpartition(rH[members], K - 1)[:K]]
        return np.sort(members[np.argsort(rH[members])])

    out = dict(qid=q["qid"], n=n, ck=ck, nprobe=nprobe,
               recall_v65_actual=float(np.isin(q["v65_ids"], q["gt"][:K]).sum()) / K)
    Tg = t_grid(ck)
    full = {T: final(orderP[:min(T, n)]) for T in sorted(set(Tg) | set(Tfin or []))}
    out["full_recall"] = {T: float(isgt[f].sum()) / K for T, f in full.items()}
    f_ck = full[ck]
    out["sim_equals_v65"] = bool(set(ids[f_ck].tolist()) == set(q["v65_ids"].tolist()))
    if Tfin is None:
        return out

    W, M, d = G["W"], G["M"], G["d"]
    qr = G["qrot"][G["qrow"][q["qid"]]]
    Tb = qr.reshape(M, 8) @ G["levmat"].T                                   # [M, 256]
    codes = np.asarray(G["prim"][pos])                                      # [n, M]
    Sw = Tb[np.arange(M)[None, :], codes].reshape(n, W, 4).sum(2)           # [n, W]
    ip = Sw.sum(1)
    Prec = q["qn2"] + G["y2"] - 2.0 * ip
    out["prim_rel_dev"] = float(np.max(np.abs(Prec - P) / np.maximum(np.abs(P), 1e-30)))
    out["topck_pos_match"] = bool(np.array_equal(np.sort(pos[orderP[:min(ck, n)]]), np.sort(q["topck"][q["topck"] >= 0]))) \
        if (q["topck"] >= 0).sum() == min(ck, n) else False

    probes = q["probes"].astype(np.int64)
    c = G["c"]
    my = G["mu_q"][probes] * c                                              # [nprobe, d] E[yhat_j | l]
    U1 = (qr[None, :] * my).reshape(nprobe, W, 32).sum(2)                  # [nprobe, W] E[S_w | l]
    I1 = (c * c * qr * qr).reshape(W, 32).sum(1)                            # [W]
    Q = ((qr * qr)[None, :] * my * my).reshape(nprobe, W, 32).sum(2)       # sum (c q')^2 mu^2
    lsz = np.bincount(rank, minlength=nprobe).astype(np.float64)
    st = G["off"][probes]; en = G["off"][probes + 1]
    sh = np.arange(8)[None, :]
    nsec = np.where((en > st)[:, None], ((en[:, None] - 1 + sh) >> 3) - ((st[:, None] + sh) >> 3) + 1, 0).astype(np.float64)
    allcnt = G["onehot"].sum(0)
    out["B_old"] = float(n * W * 4)
    out["B_old_sec"] = float((nsec * allcnt[None, :]).sum() * 32)
    full_top_ck = orderP[:min(ck, n)]
    top10H = orderH[:K]
    Ls = [min(m * ck, n) for m in LMULT]
    out["L"] = Ls
    nV, nP, nL, nT = len(VARIANTS), len(PS), len(LMULT), len(Tfin)
    R = np.zeros((nV, nP, nL, nT), np.int16)          # hits of the ground-truth top-10
    SAME = np.zeros((nV, nP, nL, nT), bool)
    COV = np.zeros((nV, nP, nL, 3), np.float32)
    S2 = np.zeros((nV, nP, nL), np.float64)
    S1 = np.zeros((nV, nP), np.float64)
    IMP = np.zeros((nV, nP, 2), np.float32)
    LNEED = np.zeros((nV, nP, 2), np.int64)
    fullsets = {T: set(full[T].tolist()) for T in Tfin}
    for pi, p in enumerate(PS):
        nJ = max(1, int(round(p * W)))
        masks = {}
        m0 = np.zeros(W, bool); m0[:nJ] = True
        masks["A0"] = np.broadcast_to(m0, (nprobe, W))
        m1 = np.zeros(W, bool); m1[np.argsort(-I1, kind="stable")[:nJ]] = True
        masks["A1"] = np.broadcast_to(m1, (nprobe, W))
        for lam in LAMS:
            I2 = I1[None, :] - lam * lam * Q
            m2 = np.zeros((nprobe, W), bool)
            np.put_along_axis(m2, np.argsort(-I2, axis=1, kind="stable")[:, :nJ], True, axis=1)
            masks[("A2", lam)] = m2
        psum = {"A0": Sw[:, m0].sum(1), "A1": Sw[:, m1].sum(1)}
        for lam in LAMS:
            psum[("A2", lam)] = (Sw * masks[("A2", lam)][rank]).sum(1)
        for vi, (pol, lam) in enumerate(VARIANTS):
            key = "A0" if pol in ("A0", "A0c") else "A1" if pol in ("A1", "A1c") else ("A2", lam)
            mk = masks[key]
            score = psum[key]
            if pol in ("A0c", "A1c", "A3") and lam > 0:
                score = score + lam * (U1 * ~mk).sum(1)[rank]
            order1 = np.lexsort((pos, -score))
            r1 = np.empty(n, dtype=np.int64); r1[order1] = np.arange(n)
            LNEED[vi, pi] = (r1[full_top_ck].max() + 1, r1[f_ck].max() + 1)
            share = (mk * I1[None, :]).sum(1) / I1.sum()
            IMP[vi, pi] = ((share * lsz).sum() / lsz.sum(), np.corrcoef(score, ip)[0, 1] if p < 1 else 1.0)
            readc = mk.astype(np.float64) @ G["onehot"]
            unreadc = (~mk).astype(np.float64) @ G["onehot"]
            S1[vi, pi] = (nsec * readc).sum() * 32
            for li, L in enumerate(Ls):
                sel = order1[:L]
                selm = np.zeros(n, bool); selm[sel] = True
                COV[vi, pi, li] = (selm[full_top_ck].mean(), selm[top10H].mean(), 1.0 - selm[f_ck].mean())
                ss = np.sort(sel)
                ps, rs = pos[ss], rank[ss]
                brk = np.ones(len(ss), bool); brk[1:] = rs[1:] != rs[:-1]
                cnt = np.zeros((nprobe, 8))
                for s in range(8):
                    sec = (ps + s) >> 3
                    new = brk.copy(); new[1:] |= sec[1:] != sec[:-1]
                    cnt[:, s] = np.bincount(rs, weights=new, minlength=nprobe)
                S2[vi, pi, li] = (cnt * unreadc).sum() * 32
                inP = selm[orderP]
                cum = np.cumsum(inP)
                for ti, T in enumerate(Tfin):
                    f = final(orderP[inP & (cum <= T)])
                    R[vi, pi, li, ti] = isgt[f].sum()
                    SAME[vi, pi, li, ti] = set(f.tolist()) == fullsets[T]
    out.update(R=R, SAME=SAME, COV=COV, S1=S1, S2=S2, IMP=IMP, LNEED=LNEED)
    return out


def boot_ci(dl, idx):
    m = dl[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def qlow(x, a):
    s = np.sort(x)
    return float(s[int(np.floor(a * (len(s) - 1)))])


def main():
    lf_dir, pp_dir, R = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else min(64, os.cpu_count() or 8)
    os.makedirs(R, exist_ok=True)
    tags = sorted(t for t in os.listdir(pp_dir) if os.path.isfile(os.path.join(pp_dir, t, "meta.json")))
    summary = dict(policies=dict(A0="storage-order prefix (the v21 cascade's)", A1="query order", A2="IVF-conditioned per-list order",
                                 A3="A2 + mean correction", A0c="A0 + mean correction", A1c="A1 + mean correction"),
                   rules=dict(non_inferiority_margin=NI_MARGIN, validation_tolerance=VAL_TOL, novelty_bytes=NOVELTY,
                              severe_tail="test P1 < -0.1 or worst < -0.2", bootstrap=NBOOT, seed=SEED,
                              split="40/20/40 calibration/validation/test"), workloads={})
    rows_rb, rows_cov, rows_imp, rows_pq = [], [], [], []
    for tag in tags:
        t0 = time.time()
        meta, mu_stats = setup(lf_dir, pp_dir, tag)
        files = sorted(f for f in os.listdir(os.path.join(lf_dir, tag)) if f.startswith("rb_q") and f.endswith(".bin"))
        files = [os.path.join(lf_dir, tag, f) for f in files]
        qids = sorted(int(os.path.basename(f)[4:-4]) for f in files)
        perm = np.random.default_rng(SEED).permutation(qids)
        c1, c2 = int(0.4 * len(perm)), int(0.6 * len(perm))
        split = {int(x): ("calib" if i < c1 else "val" if i < c2 else "test") for i, x in enumerate(perm)}
        with mp.get_context("fork").Pool(workers) as pool:   # the workers inherit G
            first = pool.map(per_query, [(f, None) for f in files], chunksize=8)
        ck = first[0]["ck"]
        Tg = t_grid(ck)
        val = [o for o in first if split[o["qid"]] == "val"]
        target = np.mean([o["full_recall"][ck] for o in val])
        Tstar = next(T for T in Tg if np.mean([o["full_recall"][T] for o in val]) >= target - 1e-9)
        Tfin = [ck] if Tstar == ck else [ck, Tstar]
        with mp.get_context("fork").Pool(workers) as pool:   # the workers inherit G
            res = pool.map(per_query, [(f, Tfin) for f in files], chunksize=4)
        res.sort(key=lambda o: o["qid"])
        sp = np.array([split[o["qid"]] for o in res])
        S = {s: np.where(sp == s)[0] for s in ("calib", "val", "test")}
        Rr = np.stack([o["R"] for o in res]); SAME = np.stack([o["SAME"] for o in res])
        COV = np.stack([o["COV"] for o in res]); S1 = np.stack([o["S1"] for o in res]); S2 = np.stack([o["S2"] for o in res])
        IMP = np.stack([o["IMP"] for o in res]); LNEED = np.stack([o["LNEED"] for o in res])
        nvec = np.array([o["n"] for o in res], np.float64)
        Lq = np.array([o["L"] for o in res], np.float64)                    # [nq, nL]
        Bold = np.array([o["B_old"] for o in res]); Bsec = np.array([o["B_old_sec"] for o in res])
        nprobe = res[0]["nprobe"]; W, d = G["W"], G["d"]
        fullR = {T: np.array([o["full_recall"][T] for o in res]) for T in Tfin}
        # lam per (policy, p, L): calibration coverage of the full-primary top-ck
        vix = {v: i for i, v in enumerate(VARIANTS)}
        chosen = {}
        for pol in POLS:
            for pi in range(len(PS)):
                for li in range(len(LMULT)):
                    if pol in ("A0", "A1"):
                        chosen[(pol, pi, li)] = vix[(pol, 0.0)]
                        continue
                    best, bv = None, -1.0
                    for lam in LAMS:
                        v = COV[S["calib"], vix[(pol, lam)], pi, li, 0].mean()
                        if v > bv + 1e-12:
                            best, bv = vix[(pol, lam)], v
                    chosen[(pol, pi, li)] = best
        rng = np.random.default_rng(SEED)
        bidx = {s: rng.integers(0, len(S[s]), (NBOOT, len(S[s]))) for s in ("val", "test")}

        def ideal(pi, li):
            nJ = max(1, int(round(PS[pi] * W)))
            return (nvec * nJ * 4 + Lq[:, li] * (W - nJ) * 4) / Bold

        def stats(pol, pi, li, ti, s):
            vi = chosen[(pol, pi, li)]
            ix = S[s]
            T = Tfin[ti]
            rec = Rr[ix, vi, pi, li, ti].astype(np.float64) / K
            dl = rec - fullR[T][ix]
            dck = rec - fullR[ck][ix]
            meta_b = nprobe * d * 1.0 if (USES_MU[pol] and VARIANTS[vi][1] > 0) else 0.0
            scratch = 16.0 * Lq[ix, li]
            o = dict(recall=rec.mean(), delta=dl.mean(), delta_vs_ck=dck.mean(),
                     ideal=ideal(pi, li)[ix].mean(),
                     sec=((S1[ix, vi, pi] + S2[ix, vi, pi, li]) / Bsec[ix]).mean(),
                     sec_s1=(S1[ix, vi, pi] / Bsec[ix]).mean(), sec_s2=(S2[ix, vi, pi, li] / Bsec[ix]).mean(),
                     meta=(meta_b / Bsec[ix]).mean(), scratch=(scratch / Bsec[ix]).mean(),
                     total=((S1[ix, vi, pi] + S2[ix, vi, pi, li] + meta_b + scratch) / Bsec[ix]).mean(),
                     same=SAME[ix, vi, pi, li, ti].mean(), lam=VARIANTS[vi][1], dl=dl, dck=dck)
            return o

        ops = {}
        for pol in POLS:
            for ti, T in enumerate(Tfin):
                for pi, p in enumerate(PS):
                    for li, m in enumerate(LMULT):
                        a = stats(pol, pi, li, ti, "test")
                        v = stats(pol, pi, li, ti, "val")
                        lo, hi = boot_ci(a["dl"], bidx["test"])
                        lo2, hi2 = boot_ci(a["dck"], bidx["test"])
                        rows_rb.append(dict(tag=tag, policy=pol, lam=a["lam"], p=p, words_read=max(1, int(round(p * W))), W=W,
                                            L_over_ck=m, L_mean=Lq[S["test"], li].mean(), T_final=T, recall=a["recall"],
                                            baseline_recall=fullR[T][S["test"]].mean(), delta=a["delta"], ci_lo=lo, ci_hi=hi,
                                            non_inferior=lo >= -NI_MARGIN, delta_vs_full_ck=a["delta_vs_ck"], ci_lo_vs_ck=lo2, ci_hi_vs_ck=hi2,
                                            p50=qlow(a["dl"], 0.5), p5=qlow(a["dl"], 0.05), p1=qlow(a["dl"], 0.01), worst=float(a["dl"].min()),
                                            frac_worse=float((a["dl"] < 0).mean()), frac_better=float((a["dl"] > 0).mean()),
                                            same_final_as_baseline=a["same"], ideal_bytes_ratio=a["ideal"],
                                            sector_bytes_ratio=a["sec"], sector_stage1=a["sec_s1"], sector_stage2=a["sec_s2"],
                                            metadata_ratio=a["meta"], scratch_ratio=a["scratch"], total_bytes_ratio=a["total"],
                                            val_delta=v["delta"], val_ideal=v["ideal"]))
                # operating point, chosen on validation
                best = None
                for pi in range(len(PS)):
                    for li in range(len(LMULT)):
                        v = stats(pol, pi, li, ti, "val")
                        if v["delta"] >= -VAL_TOL - 1e-12 and (best is None or v["ideal"] < best[0] - 1e-12):
                            best = (v["ideal"], pi, li)
                if best is None:
                    ops["%s/T%d" % (pol, T)] = None
                    continue
                _, pi, li = best
                a = stats(pol, pi, li, ti, "test")
                lo, hi = boot_ci(a["dl"], bidx["test"])
                ops["%s/T%d" % (pol, T)] = dict(policy=pol, T_final=T, p=PS[pi], L_over_ck=LMULT[li], lam=a["lam"],
                                                 test_recall=a["recall"], baseline_recall=float(fullR[T][S["test"]].mean()),
                                                 delta=a["delta"], ci95=[lo, hi], non_inferior=lo >= -NI_MARGIN,
                                                 p1=qlow(a["dl"], 0.01), worst=float(a["dl"].min()),
                                                 severe_tail=qlow(a["dl"], 0.01) < -0.1 or float(a["dl"].min()) < -0.2,
                                                 frac_worse=float((a["dl"] < 0).mean()), ideal_bytes_ratio=a["ideal"],
                                                 sector_bytes_ratio=a["sec"], total_bytes_ratio=a["total"],
                                                 sector_stage1=a["sec_s1"], sector_stage2=a["sec_s2"], metadata_ratio=a["meta"],
                                                 scratch_ratio=a["scratch"], same_final=a["same"])
                for qi in S["test"]:
                    rows_pq.append(dict(tag=tag, op="%s/T%d" % (pol, T), qid=res[qi]["qid"],
                                        recall=float(Rr[qi, chosen[(pol, pi, li)], pi, li, ti]) / K, baseline=float(fullR[T][qi])))
        # coverage against read ratio (test), and importance (test)
        for pol in POLS:
            for pi, p in enumerate(PS):
                for li, m in enumerate(LMULT):
                    vi = chosen[(pol, pi, li)]
                    ix = S["test"]
                    rows_cov.append(dict(tag=tag, policy=pol, lam=VARIANTS[vi][1], p=p, L_over_ck=m, L_mean=Lq[ix, li].mean(),
                                         ideal_read_ratio=ideal(pi, li)[ix].mean(),
                                         coverage_full_primary_topck=COV[ix, vi, pi, li, 0].mean(),
                                         any_miss_rate_topck=float((COV[ix, vi, pi, li, 0] < 1).mean()),
                                         coverage_full_ivf_top10=COV[ix, vi, pi, li, 1].mean(),
                                         miss_rate_v65_final10=COV[ix, vi, pi, li, 2].mean(),
                                         queries_losing_v65_final=float((COV[ix, vi, pi, li, 2] > 0).mean())))
        for vi, (pol, lam) in enumerate(VARIANTS):
            for pi, p in enumerate(PS):
                ix = S["test"]
                rows_imp.append(dict(tag=tag, policy=pol, lam=lam, p=p, words_read=max(1, int(round(p * W))), W=W,
                                     query_energy_share=IMP[ix, vi, pi, 0].mean(), pearson_r_estimate_vs_full=IMP[ix, vi, pi, 1].mean(),
                                     L_need_topck_over_ck_median=float(np.median(LNEED[ix, vi, pi, 0] / ck)),
                                     L_need_topck_over_ck_p90=float(np.quantile(LNEED[ix, vi, pi, 0] / ck, 0.9)),
                                     L_need_final10_over_ck_median=float(np.median(LNEED[ix, vi, pi, 1] / ck)),
                                     L_need_final10_over_ck_p90=float(np.quantile(LNEED[ix, vi, pi, 1] / ck, 0.9)),
                                     L_need_final10_over_ck_max=float((LNEED[ix, vi, pi, 1] / ck).max())))
        # gates
        tstar_key = lambda pol: "%s/T%d" % (pol, Tstar)
        a0 = ops.get(tstar_key("A0"))
        bestn = min((ops[tstar_key(p)] for p in ("A1", "A1c", "A2", "A3", "A0c") if ops.get(tstar_key(p))),
                    key=lambda o: o["ideal_bytes_ratio"], default=None)
        summary["workloads"][tag] = dict(
            N=meta["N"], d=meta["d"], M=meta["M"], W=W, levels=meta["levels"], nprobe=nprobe, ck=ck, T_star=Tstar, T_grid=Tg,
            n_cand_mean=float(nvec.mean()), n_queries=len(res), split_sizes={s: int(len(v)) for s, v in S.items()},
            prim_reconstruction_max_rel_dev=float(max(o["prim_rel_dev"] for o in res)),
            topck_positions_match_v65=float(np.mean([o["topck_pos_match"] for o in res])),
            sim_equals_v65_final=float(np.mean([o["sim_equals_v65"] for o in res])),
            recall_v65_actual_test=float(np.mean([res[i]["recall_v65_actual"] for i in S["test"]])),
            recall_full={str(T): {s: float(fullR[T][S[s]].mean()) for s in S} for T in Tfin},
            full_tstar_vs_ck_test=dict(delta=float((fullR[Tstar] - fullR[ck])[S["test"]].mean()),
                                       ci95=list(boot_ci((fullR[Tstar] - fullR[ck])[S["test"]], bidx["test"])),
                                       p1=qlow((fullR[Tstar] - fullR[ck])[S["test"]], 0.01),
                                       worst=float((fullR[Tstar] - fullR[ck])[S["test"]].min())),
            B_old_mean=float(Bold.mean()), B_old_sector_mean=float(Bsec.mean()), mu=mu_stats,
            operating_points=ops,
            best_novel_at_Tstar=bestn, A0_at_Tstar=a0,
            novelty_margin_vs_A0=(None if (bestn is None) else
                                  (1.0 if a0 is None else a0["ideal_bytes_ratio"]) - bestn["ideal_bytes_ratio"]),
            seconds=time.time() - t0)
        print(tag, "T*=%d" % Tstar, "ops:", {k2: (None if v is None else (v["p"], v["L_over_ck"], round(v["delta"], 4),
                                                                          round(v["ideal_bytes_ratio"], 3), v["non_inferior"]))
                                             for k2, v in ops.items()}, "%.0fs" % (time.time() - t0), flush=True)
    for fn, rows in (("RECALL_VS_BYTES.csv", rows_rb), ("COVERAGE_VS_READ_RATIO.csv", rows_cov),
                     ("WORD_IMPORTANCE_ANALYSIS.csv", rows_imp), ("PER_QUERY_OPERATING_POINTS.csv", rows_pq)):
        with open(os.path.join(R, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(R, "analysis_summary.json"), "w"), indent=1, default=float)
    print("done:", ", ".join(tags))


if __name__ == "__main__":
    main()
