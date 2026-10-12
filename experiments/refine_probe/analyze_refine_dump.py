#!/usr/bin/env python3
"""How many refinement survivors can a partial residual sum rule out?

Reads one JHQ_DUMP_REFINE file (jhq_v60_refine_probe) and answers the
feasibility question for bound-guided refinement before anyone writes a
kernel for it. Nothing here runs on the GPU and nothing is estimated: every
survivor's composite distance is recomputed from its codes and checked against
the kernel's before any bound is evaluated.

The kernel's score (search.cu, residual_refine_fused_kernel; encode.cu stores
corr = 2<y_hat, r_hat>):

    comp = D_P + corr + sum_j (q'_j - r_hat_j)^2  =  ||q' - x_hat||^2 + ||q'||^2

Every per-dimension term carries q'_j^2, so a partial sum over t of d
dimensions misses (d-t)/d of ||q'||^2 -- the same amount for every candidate,
and large next to the gaps that decide the top-k. Three lower bounds, all
valid (L <= comp exactly), compared:

  B0  D_P + corr + sum_{j<t} term_j                       (the proposal; free)
  B1  B0 + sum_{j>=t} min_c (q'_j - c)^2 over the level   (free: per query)
      set of j's residual subspace
  B2  ||q'||^2 + D_P + ||r||^2 - 2<u,r>_{<t}              (u = q' - y_hat; needs
      - 2 ||u_{>=t}|| ||r_{>=t}||                          ||r||^2 and tail norms
                                                           stored per vector, and
                                                           y_hat per dimension)

  B2 at t=0 is ||q'||^2 + (||u|| - ||r||)^2: a test before any residual byte.

Thresholds tau: "final" is the k-th best composite over all ck survivors (the
most any rule could prune with); "first-k" fully refines the k best survivors
by primary distance and uses their k-th best -- what a kernel can actually
have in hand. A candidate is pruned only if L > tau * (1 + 1e-5) + 1e-6, and
the script counts any pruned candidate that belongs to the final top-k (it
must be zero).

  python3 analyze_refine_dump.py <dump> [--share S] [--json out.json]

--share is the measured fraction of search time spent in refinement at this
operating point; with it the script also prints the end-to-end ceiling,
1 / (1 - S * saved_work), which assumes refinement time is proportional to the
dimensions processed (optimistic at small ck, where the kernel is latency-bound).
"""
import argparse
import json

import numpy as np

CUTS = (0.0, 0.125, 0.25, 0.5, 0.75)
REL_MARGIN, ABS_MARGIN = 1e-5, 1e-6


def read_dump(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=14)
        magic, ver, nqd, ck, d, M, Ds, K, Kr, Br, bpv, k, q_first, nq = (int(x) for x in hdr)
        if magic != 0x4A485150 or ver != 1:
            raise SystemExit("not a JHQ_DUMP_REFINE v1 file: %s" % path)
        rd = lambda dt, n, shape: np.fromfile(f, dtype=dt, count=n).reshape(shape)
        D = dict(nqd=nqd, ck=ck, d=d, M=M, Ds=Ds, K=K, Kr=Kr, Br=Br, bpv=bpv, k=k,
                 q_first=q_first, nq=nq)
        D["qr"] = rd(np.float32, nqd * d, (nqd, d))
        D["rcb"] = rd(np.float32, M * Kr, (M, Kr))
        D["cen"] = rd(np.float32, M * K * Ds, (M, K, Ds))
        D["pos"] = rd(np.int32, nqd * ck, (nqd, ck))
        D["ids"] = rd(np.int32, nqd * ck, (nqd, ck))
        D["prim"] = rd(np.float32, nqd * ck, (nqd, ck))
        D["corr"] = rd(np.float32, nqd * ck, (nqd, ck))
        D["comp"] = rd(np.float32, nqd * ck, (nqd, ck))
        D["fid"] = rd(np.int32, nqd * k, (nqd, k))
        D["rc"] = rd(np.uint8, nqd * ck * bpv, (nqd, ck, bpv))
        D["pc"] = rd(np.uint8, nqd * ck * M, (nqd, ck, M))
        if f.read(1):
            raise SystemExit("trailing bytes: the dump and this reader disagree on the format")
    return D


def decode_residual_codes(rc, Br, d):
    if Br == 8:
        return rc.astype(np.int64)
    out = np.empty(rc.shape[:-1] + (d,), dtype=np.int64)
    out[..., 0::2] = rc & 0x0F                    # even dimension: low nibble
    out[..., 1::2] = rc >> 4                      # odd dimension: high nibble
    return out


def analyze(D):
    d, M, Ds, k, ck = D["d"], D["M"], D["Ds"], D["k"], D["ck"]
    m_of_j = np.arange(d) // Ds
    rcb = D["rcb"].astype(np.float64)
    cen = D["cen"].astype(np.float64)
    tcut = [int(round(c * d / Ds)) * Ds for c in CUTS]           # cuts on subspace boundaries
    taus = ("final", "first-k")
    bounds = ("B0", "B1", "B2")
    check = dict(prim_rel=0.0, corr_rel=0.0, comp_rel=0.0, ident_rel=0.0, topk_mismatch=0)
    # pruned[bound][tau][cut] = (#pruned, #eligible); staged work per bound/tau
    pruned = {b: {t: np.zeros((len(CUTS), 2)) for t in taus} for b in bounds}
    work = {b: {t: [0.0, 0.0] for t in taus} for b in bounds}   # dims processed, dims total
    unsafe = {b: {t: 0 for t in taus} for b in bounds}
    for q in range(D["nqd"]):
        live = D["pos"][q] >= 0
        qv = D["qr"][q].astype(np.float64)
        codes = decode_residual_codes(D["rc"][q][live], D["Br"], d)            # (c, d)
        rhat = rcb[m_of_j[None, :], codes]
        yhat = cen[np.arange(M)[None, :], D["pc"][q][live].astype(np.int64)].reshape(-1, d)
        prim = D["prim"][q][live].astype(np.float64)
        corr = D["corr"][q][live].astype(np.float64)
        comp = D["comp"][q][live].astype(np.float64)
        ids = D["ids"][q][live]
        term = (qv[None, :] - rhat) ** 2
        u = qv[None, :] - yhat
        # -- the kernel's arithmetic, recomputed ------------------------------
        prim_off = (u ** 2).sum(1)
        corr_off = 2.0 * (yhat * rhat).sum(1)
        comp_off = prim + term.sum(1) + corr
        ident = (qv ** 2).sum() + ((u - rhat) ** 2).sum(1)
        rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        check["prim_rel"] = max(check["prim_rel"], rel(prim_off, prim))
        check["corr_rel"] = max(check["corr_rel"], float(np.max(np.abs(corr_off - corr))
                                                         / max(np.max(np.abs(comp)), 1e-12)))
        check["comp_rel"] = max(check["comp_rel"], rel(comp_off, comp))
        check["ident_rel"] = max(check["ident_rel"], rel(ident, comp_off))
        order = np.argsort(comp, kind="stable")
        if set(ids[order[:k]].tolist()) != set(D["fid"][q].tolist()) - {-1}:
            check["topk_mismatch"] += 1
        in_topk = np.zeros(len(comp), dtype=bool)
        in_topk[order[:k]] = True
        # -- partial quantities at each cut -----------------------------------
        cs_term = np.concatenate([np.zeros((len(comp), 1)), np.cumsum(term, 1)], 1)
        tailmin_j = ((qv[:, None] - rcb[m_of_j]) ** 2).min(1)                  # (d,)
        tailmin = np.concatenate([np.cumsum(tailmin_j[::-1])[::-1], [0.0]])    # suffix sums
        ur = u * rhat
        cs_ur = np.concatenate([np.zeros((len(comp), 1)), np.cumsum(ur, 1)], 1)
        u2s = np.concatenate([np.cumsum((u ** 2)[:, ::-1], 1)[:, ::-1], np.zeros((len(comp), 1))], 1)
        r2s = np.concatenate([np.cumsum((rhat ** 2)[:, ::-1], 1)[:, ::-1], np.zeros((len(comp), 1))], 1)
        q2 = (qv ** 2).sum()
        R2 = (rhat ** 2).sum(1)
        L = {}
        for b in bounds:
            L[b] = np.empty((len(CUTS), len(comp)))
        for i, t in enumerate(tcut):
            L["B0"][i] = prim + corr + cs_term[:, t]
            L["B1"][i] = L["B0"][i] + tailmin[t]
            L["B2"][i] = q2 + prim + R2 - 2.0 * cs_ur[:, t] - 2.0 * np.sqrt(u2s[:, t] * r2s[:, t])
        # -- thresholds --------------------------------------------------------
        by_prim = np.argsort(prim, kind="stable")
        first = np.zeros(len(comp), dtype=bool)
        first[by_prim[:k]] = True
        tau = {"final": np.sort(comp)[k - 1], "first-k": np.sort(comp[first])[k - 1]}
        for b in bounds:
            for tn in taus:
                thr = tau[tn] * (1 + REL_MARGIN) + ABS_MARGIN
                elig = ~first if tn == "first-k" else np.ones(len(comp), dtype=bool)
                exit_dim = np.full(len(comp), float(d))                       # dims processed
                for i, t in enumerate(tcut):
                    p = (L[b][i] > thr) & elig
                    pruned[b][tn][i] += (p.sum(), elig.sum())
                    unsafe[b][tn] += int((p & in_topk).sum())
                    newly = p & (exit_dim == d)
                    exit_dim[newly] = t
                work[b][tn][0] += exit_dim.sum()
                work[b][tn][1] += float(d) * len(comp)
    return check, pruned, work, unsafe, tcut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--share", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    D = read_dump(a.dump)
    check, pruned, work, unsafe, tcut = analyze(D)
    print("dump %s: %d queries (from query %d of %d), ck=%d, d=%d, M=%d, Ds=%d, Br=%d" % (
        a.dump, D["nqd"], D["q_first"], D["nq"], D["ck"], D["d"], D["M"], D["Ds"], D["Br"]))
    # A tie at the k-th distance can resolve differently here (stable order) and
    # in the kernel (lower id wins), so allow it on at most 1% of queries.
    ok = (check["prim_rel"] < 1e-4 and check["comp_rel"] < 1e-4 and check["ident_rel"] < 1e-4
          and check["topk_mismatch"] <= max(1, D["nqd"] // 100))
    print("check: primary rel err %.2e | corr err/|comp| %.2e | composite rel err %.2e | "
          "comp = ||q'-x||^2+||q'||^2 rel err %.2e | top-k mismatches %d -> %s" % (
              check["prim_rel"], check["corr_rel"], check["comp_rel"], check["ident_rel"],
              check["topk_mismatch"], "OK" if ok else "FAILED: the bounds below are not to be trusted"))
    out = {"check": check, "ok": ok, "cuts": tcut, "rows": []}
    print("pruned share of eligible survivors at a single cut (t = dims read), and staged work:")
    print("  %-4s %-8s" % ("bnd", "tau") + "".join("%9s" % ("t=%d" % t) for t in tcut)
          + "   dims read   unsafe" + ("   e2e ceiling" if a.share is not None else ""))
    for b in ("B0", "B1", "B2"):
        for tn in ("final", "first-k"):
            p = pruned[b][tn]
            frac = [100.0 * p[i, 0] / max(p[i, 1], 1) for i in range(len(tcut))]
            read = work[b][tn][0] / work[b][tn][1]
            line = "  %-4s %-8s" % (b, tn) + "".join("%8.1f%%" % x for x in frac) + \
                   "   %8.1f%%   %6d" % (100 * read, unsafe[b][tn])
            row = {"bound": b, "tau": tn, "pruned_pct": frac, "dims_read_frac": read,
                   "unsafe": unsafe[b][tn]}
            if a.share is not None:
                ceil = 1.0 / (1.0 - a.share * (1.0 - read))
                line += "   x%.3f" % ceil
                row["e2e_ceiling"] = ceil
            print(line)
            out["rows"].append(row)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
