#!/usr/bin/env python3
"""Round 3: 1-bit thresholds (and, in one variant, reconstruction levels) learned by a pairwise rank loss.

  python3 rankloss.py <tag> <results dir>

Training data come from the database only (the evaluation queries and their ground truth are never read):
  pseudo-queries   20,000 base vectors drawn at random (seeded); 2,000 more held out for a sanity check
  positives        each pseudo-query's exact top-10 among all base vectors (itself excluded; GPU, FP32)
  negatives        its exact ranks 11..110 (hard) and 200 members of its 128 nearest lists (the routing the
                   index does, emulated with the list means); a routed draw that is a positive is dropped from the loss
Score: the asymmetric primary distance D(q, x) = sum_j (q_j - l_j(b_j(x)))^2, b_j(x) = 1[x_j > t_j], with
soft bits sigmoid((x_j - t_j) / tau_j) during training (tau annealed from 0.3 sd_j to 0.01 sd_j).
Loss (the brief's Candidate B): mean softplus(g (D(q, x+) - D(q, x-))) over positive/negative pairs.
Variants:
  rl_t     t learned; levels the conditional means on each side (recomputed from a 100K database sample)
  rl_tl    t, lo and hi all learned
Each is trained from two starts, t = 0 (JHQ) and t = median, to show where the loss takes it. Output
RQ/<tag>/learned_<variant>_<init>.npz (thr, lo, hi), read by quantize.py's learned_* variants, and a summary.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roq_common as C  # noqa: E402

NQ, NHOLD, K, HARD, RAND, NPROBE = 20000, 2000, 10, 100, 200, 128


def cond_means_t(x, t):
    a = x > t
    hi = torch.where(a, x, 0).sum(0) / a.sum(0).clamp(min=1)
    lo = torch.where(~a, x, 0).sum(0) / (~a).sum(0).clamp(min=1)
    return lo, hi


def main():
    tag, R = sys.argv[1], sys.argv[2]
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    m = C.meta(tag)
    N, d = m["N"], m["d"]
    xp = C.xpos(tag)
    X = torch.from_numpy(np.ascontiguousarray(xp[:])).to(dev)                  # [N, d], position order
    off = C.list_offsets(tag)
    lmean = torch.from_numpy(np.fromfile(os.path.join(C.RQ, tag, "lmean.f32"), dtype=np.float32).reshape(-1, d)).to(dev)
    lpos = torch.from_numpy(C.list_of_pos(tag).astype(np.int64)).to(dev)
    rng = np.random.default_rng(C.SEED + 3)
    qi = torch.from_numpy(rng.choice(N, NQ + NHOLD, replace=False)).to(dev)
    # exact neighbours (unit vectors: inner product order = L2 order) and routed random negatives
    pos_l, hard_l, rnd_l = [], [], []
    offs = torch.from_numpy(off).to(dev)
    for a in range(0, len(qi), 256):
        q = X[qi[a:a + 256]]
        ip = q @ X.T                                                             # [b, N]
        ip[torch.arange(len(q)), qi[a:a + 256]] = -1e9                          # drop self
        top = torch.topk(ip, K + HARD, dim=1).indices
        pos_l.append(top[:, :K]); hard_l.append(top[:, K:])
        dl = torch.cdist(q, lmean)                                               # routing
        pr = torch.topk(dl, NPROBE, largest=False).indices                      # [b, nprobe]
        st, ln = offs[pr], offs[pr + 1] - offs[pr]
        tot = ln.sum(1, keepdim=True)
        u = (torch.rand(len(q), RAND, device=dev) * tot).long()                 # a uniform draw over the routed candidates
        cl = torch.cumsum(ln, 1)
        li = torch.searchsorted(cl, u, right=True)
        before = torch.cat([torch.zeros_like(cl[:, :1]), cl[:, :-1]], 1)
        rnd_l.append(torch.gather(st, 1, li) + (u - torch.gather(before, 1, li)))
        del ip
    P, H, Rn = torch.cat(pos_l), torch.cat(hard_l), torch.cat(rnd_l)
    knn_s = time.time() - t0
    # database sample for the conditional means and the starting points
    smp = X[torch.from_numpy(rng.choice(N, 100000, replace=False)).to(dev)]
    sd = smp.std(0)
    med = smp.median(0).values
    out = dict(tag=tag, knn_seconds=knn_s, runs={})

    def run(variant, init):
        t = (med.clone() if init == "median" else torch.zeros(d, device=dev)).requires_grad_(True)
        lo0, hi0 = cond_means_t(smp, t.detach())
        lo = lo0.clone().requires_grad_(variant == "rl_tl"); hi = hi0.clone().requires_grad_(variant == "rl_tl")
        params = [t] + ([lo, hi] if variant == "rl_tl" else [])
        opt = torch.optim.Adam(params, lr=0.02 * float(sd.mean()))
        steps, B = 1500, 256
        g = None
        hist = []
        for s in range(steps):
            tau = sd * (0.3 * (0.01 / 0.3) ** (s / (steps - 1)))
            b = torch.randint(0, NQ, (B,), device=dev)
            q = X[qi[b]]
            idx = torch.cat([P[b], H[b], Rn[b]], 1)                              # [B, K + HARD + RAND]
            x = X[idx]
            if variant == "rl_t" and s % 25 == 0:
                lo_c, hi_c = cond_means_t(smp, t.detach())
            l_lo, l_hi = (lo_c, hi_c) if variant == "rl_t" else (lo, hi)
            bt = torch.sigmoid((x - t) / tau)
            lv = l_lo + bt * (l_hi - l_lo)
            D = ((q[:, None, :] - lv) ** 2).sum(-1)
            diff = D[:, :K, None] - D[:, None, K:]                                # [B, K, negatives]
            # a routed random negative can be one of the query's positives: those pairs are dropped
            keep = ~(idx[:, K:][:, None, :] == P[b][:, :, None]).any(1)           # [B, negatives]
            w = keep[:, None, :].expand_as(diff).float()
            if g is None:
                g = 1.0 / float(diff.detach()[w > 0].std().clamp(min=1e-12))
            loss = (F.softplus(g * diff) * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 100 == 0 or s == steps - 1:
                hist.append((s, float(loss)))
        with torch.no_grad():
            tt = t.detach()
            if variant == "rl_t":
                lo_f, hi_f = cond_means_t(smp, tt)
            else:
                lo_f, hi_f = lo.detach(), hi.detach()
            # held-out pseudo-queries, hard bits: coverage of the exact top-10 within the primary top-T among
            # the routed negatives plus positives (a cheap sanity check, not the evaluation)
            hb = torch.arange(NQ, NQ + NHOLD, device=dev)
            q = X[qi[hb]]
            idx = torch.cat([P[hb], H[hb], Rn[hb]], 1)
            x = X[idx]
            lv = torch.where(x > tt, hi_f, lo_f)
            D = ((q[:, None, :] - lv) ** 2).sum(-1)
            rk = torch.argsort(torch.argsort(D, 1), 1)[:, :K].float()
            cov = {T: float((rk < T).float().mean()) for T in (10, 20, 40, 80)}
            shift = float(((tt - med) / sd).abs().mean())
            shift0 = float((tt / sd).abs().mean())
        name = "%s_%s" % (variant, init)
        np.savez(os.path.join(C.RQ, tag, "learned_%s.npz" % name), thr=tt.cpu().numpy(), lo=lo_f.cpu().numpy(), hi=hi_f.cpu().numpy())
        out["runs"][name] = dict(loss=hist, holdout_coverage_of_top10_within_T=cov,
                                 mean_abs_threshold_minus_median_over_sd=shift, mean_abs_threshold_over_sd=shift0)
        print(tag, name, json.dumps(out["runs"][name]), flush=True)

    # reference points with hard bits, no training: JHQ's zero threshold and the median, conditional-mean levels
    with torch.no_grad():
        for nm, tt in (("t0", torch.zeros(d, device=dev)), ("median", med)):
            lo_f, hi_f = cond_means_t(smp, tt)
            hb = torch.arange(NQ, NQ + NHOLD, device=dev)
            q = X[qi[hb]]
            idx = torch.cat([P[hb], H[hb], Rn[hb]], 1)
            x = X[idx]
            D = ((q[:, None, :] - torch.where(x > tt, hi_f, lo_f)) ** 2).sum(-1)
            rk = torch.argsort(torch.argsort(D, 1), 1)[:, :K].float()
            out["runs"]["reference_" + nm] = dict(holdout_coverage_of_top10_within_T={T: float((rk < T).float().mean()) for T in (10, 20, 40, 80)})
    for variant in ("rl_t", "rl_tl"):
        for init in ("zero", "median"):
            run(variant, init)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(R, "rankloss_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
