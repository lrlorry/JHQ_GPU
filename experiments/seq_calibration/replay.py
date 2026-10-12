#!/usr/bin/env python3
"""Offline replay: fixed-S bisection, enumeration, and a sequential controller.

Protocol (the paper's, demo_jhq_budget_arms.cu): grid A = 200,100,...,2 with
A[0] = alpha_max the reference; split A = even queries calibrate / odd
evaluate (the paper's split); split B = the reverse, run only to see how much
the answer depends on which 500 queries form the pool. Every draw is a random
permutation of the 500-query calibration pool; a fixed-S policy reads its
first S entries, the sequential policy reads growing prefixes of the same
permutation, so all policies in one draw see the same queries (paired).

Data separation. A policy receives only DIS[a][calibration queries] -- top-k
overlap with the reference, no labels. Ground truth enters only in score(),
on the held-out half, after the policy has returned.

------------------------------------------------------------------------------
Sequential controller SEQ, fixed before any replay was run
------------------------------------------------------------------------------
Unit: a calibration query q is "dirty" at alpha if R_a(q) != R_ref(q) as sets
(d_q > 0). pi(alpha) = P(dirty). Since recall@k(a) >= recall@k(ref) - d_q/k,
and d_q is almost always 1 near the boundary, a held-out loss of 1e-3 against
the reference corresponds to pi of roughly 0.01. Hence the indifference zone

    H0 (alpha good): pi <= pi0 = 0.0025        H1 (alpha bad): pi >= pi1 = 0.02

tested per alpha with Wald's SPRT, error targets a_err = 0.05 (rejecting a
good alpha), b_err = 0.10 (accepting a bad one):

    LLR = x ln(pi1/pi0) + (m - x) ln((1-pi1)/(1-pi0))     x dirty of m tested
    reject  if LLR >= ln((1-b)/a)        accept  if LLR <= ln(b/(1-a))

Stages n = 16, 32, 64, 128 (nested prefixes). At each stage:
  1. search the new queries at alpha_max (one launch);
  2. bisect over the alphas not yet rejected, smallest first; an alpha probed
     here is searched only on queries it has not seen (one launch, cached
     otherwise); at an intermediate stage "pass" = not rejected, at the last
     stage "pass" = LLR <= 0 (evidence leans to H0);
  3. stop if the bisection's answer is alpha_max (everything below rejected) or
     is SPRT-accepted; otherwise go to the next stage. At the last stage
     return the answer regardless (truncation).
Bisection is only a probing order here, as in the shipped rule; nothing is
inferred about an alpha from its neighbours, so nesting is not assumed.

Declared sensitivity variants (all reported, none selected on held-out data):
  pi1 in {0.02 (primary), 0.05, 0.10}  -- looser zones allow earlier acceptance
  schedules (16,32,64,128) primary; (32,128) fewer launches; (16,...,256) can
  escalate past 128 when undecided.
Arithmetic consequence of the primary setting, before any data: with zero
dirty queries acceptance needs m >= ln(b/(1-a)) / ln((1-pi1)/(1-pi0))
= 2.2513 / 0.0177 = 127.2, i.e. m >= 128, so it can only stop at the 128
stage; with one dirty query it needs m >= 246. At truncation, LLR <= 0 allows
at most 1 dirty query in 128 (2 in 256).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIGS, GRID, HERE, Config  # noqa: E402
import cost_model  # noqa: E402

OUT = os.path.join(HERE, "out")
DRAWS = int(os.environ.get("SEQCAL_DRAWS", 2000))
SEED = 20261012
AMAX = GRID[0]
B_LIFE = [1, 10, 100, 1000, 10000]          # 500-query batches served after calibration


# ------------------------------------------------------------------ policies --
class Draw:
    """One permutation of the calibration pool, with O(1) prefix statistics."""

    def __init__(self, dis_cal, perm):
        self.slots = {a: np.concatenate([[0], np.cumsum(v[perm])]) for a, v in dis_cal.items()}
        self.dirty = {a: np.concatenate([[0], np.cumsum(v[perm] > 0)]) for a, v in dis_cal.items()}


def fixed_bisect(dr, S, eps=1):
    L = [(AMAX, S)]
    lo, hi = 0, len(GRID) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        L.append((GRID[mid], S))
        if dr.slots[GRID[mid]][S] <= eps:
            lo = mid
        else:
            hi = mid - 1
    return GRID[lo], S, L


def fixed_enum(dr, S, eps=1):
    L = [(AMAX, S)] + [(a, S) for a in GRID[1:]]
    best = AMAX
    for a in GRID[1:]:
        if dr.slots[a][S] <= eps:
            best = a
    return best, S, L


def seq(dr, stages, pi0=0.0025, pi1=0.02, a_err=0.05, b_err=0.10):
    up, dn = math.log((1 - b_err) / a_err), math.log(b_err / (1 - a_err))
    w1, w0 = math.log(pi1 / pi0), math.log((1 - pi1) / (1 - pi0))
    llr = lambda x, m: x * w1 + (m - x) * w0
    m = {a: 0 for a in GRID[1:]}
    rejected, L, prev = set(), [], 0
    for st, n in enumerate(stages):
        final = st == len(stages) - 1
        L.append((AMAX, n - prev)); prev = n
        C = sorted(a for a in GRID[1:] if a not in rejected)
        lo, hi = 0, len(C)                      # index len(C) stands for alpha_max
        while lo < hi:
            mid = (lo + hi) // 2
            a = C[mid]
            if m[a] < n:
                L.append((a, n - m[a])); m[a] = n
            v = llr(int(dr.dirty[a][n]), n)
            if v >= up:
                rejected.add(a); ok = False
            else:
                ok = v <= 0 if final else True
            if ok:
                hi = mid
            else:
                lo = mid + 1
        if lo == len(C):
            return AMAX, n, L
        c = C[lo]
        if llr(int(dr.dirty[c][n]), n) <= dn or final:
            return c, n, L


def policies():
    P = {}
    for S in (16, 32, 64, 128, 256):
        P["FIXED-BISECT S=%d" % S] = (lambda S: lambda dr: fixed_bisect(dr, S))(S)
    # Control added after the first full run, before writing anything up: eps=1
    # at S=256 is a stricter per-query tolerance than eps=1 at S=128, so S=256
    # alone confounds "more samples" with "stricter criterion". eps=2 at S=256
    # holds the tolerance per query at the S=128 level and isolates the samples.
    P["FIXED-BISECT S=256 eps=2"] = lambda dr: fixed_bisect(dr, 256, eps=2)
    P["FIXED-ENUM S=128"] = lambda dr: fixed_enum(dr, 128)
    P["FULL-POOL S=500 (bisect)"] = lambda dr: fixed_bisect(dr, 500)
    for pi1 in (0.02, 0.05, 0.10):
        for sched in ((16, 32, 64, 128), (32, 128), (16, 32, 64, 128, 256)):
            name = "SEQ pi1=%.2f stages=%s" % (pi1, ",".join(map(str, sched)))
            P[name] = (lambda s, p: lambda dr: seq(dr, s, pi1=p))(sched, pi1)
    return P


PRIMARY = "SEQ pi1=0.02 stages=16,32,64,128"
BASE = "FIXED-BISECT S=128"


# ------------------------------------------------------------------- scoring --
def run_config(cfg, split):
    cal = np.arange(0, cfg.n, 2) if split == "A" else np.arange(1, cfg.n, 2)
    held = np.arange(1, cfg.n, 2) if split == "A" else np.arange(0, cfg.n, 2)
    dis = cfg.dis(AMAX)
    dis_cal = {a: v[cal] for a, v in dis.items()}          # the only thing policies see
    hrec = {a: float(cfg.rec[a][held].mean()) for a in GRID}  # labels: scoring only
    ceil = max(hrec.values())
    # M2 (fit on S <= 128) is the primary estimate; M1 (all seven points) is
    # reported beside it. Neither is GPU time.
    model = cost_model.fit(cfg, cal, include_full=False)
    model1 = cost_model.fit(cfg, cal, include_full=True)
    qps = model["qps"]
    t_batch = {a: 500.0 / qps[a] * 1e3 for a in GRID}       # ms per 500-query batch
    # ground-truth oracle, as the demo defines it: fastest alpha within 1e-3 of ceiling
    orc = max((a for a in GRID if hrec[a] >= ceil - 1e-3), key=lambda a: qps[a])

    P = policies()
    rng = np.random.default_rng(SEED + (0 if split == "A" else 1))
    rec = {name: {"pick": [], "n": [], "L": [], "Q": [], "ms": [], "ms1": [], "per_alpha_L": [],
                  "per_alpha_Q": []} for name in P}
    for _ in range(DRAWS):
        dr = Draw(dis_cal, rng.permutation(len(cal)))
        for name, pol in P.items():
            a, n, L = pol(dr)
            r = rec[name]
            r["pick"].append(a); r["n"].append(n); r["L"].append(len(L))
            r["Q"].append(sum(s for _, s in L)); r["ms"].append(cost_model.cal_ms(model, L))
            r["ms1"].append(cost_model.cal_ms(model1, L))
            r["per_alpha_L"].append([sum(1 for b, _ in L if b == g) for g in GRID])
            r["per_alpha_Q"].append([sum(s for b, s in L if b == g) for g in GRID])

    out = {"config": cfg.name, "split": split, "held_recall": hrec, "ceiling": ceil,
           "oracle_alpha": orc, "qps": qps, "cost_model": {"M2": model, "M1": model1},
           "policies": {}}
    for name, r in rec.items():
        picks = np.array(r["pick"]); ms = np.array(r["ms"]); ms1 = np.array(r["ms1"])
        loss = np.array([ceil - hrec[a] for a in picks])
        tb = np.array([t_batch[a] for a in picks])
        vals, cnt = np.unique(picks, return_counts=True)
        nv, nc = np.unique(r["n"], return_counts=True)
        out["policies"][name] = {
            "picks": {int(v): int(c) for v, c in zip(vals, cnt)},
            "loss_mean": float(loss.mean()), "loss_p95": float(np.percentile(loss, 95)),
            "loss_max": float(loss.max()), "frac_over_1e-3": float((loss > 1e-3).mean()),
            "n_used_mean": float(np.mean(r["n"])), "n_used_dist": {int(v): int(c) for v, c in zip(nv, nc)},
            "launches_mean": float(np.mean(r["L"])), "query_searches_mean": float(np.mean(r["Q"])),
            "per_alpha_launches_mean": dict(zip(map(str, GRID), np.mean(r["per_alpha_L"], 0).tolist())),
            "per_alpha_queries_mean": dict(zip(map(str, GRID), np.mean(r["per_alpha_Q"], 0).tolist())),
            "est_cal_ms_mean": float(ms.mean()), "est_cal_ms_p95": float(np.percentile(ms, 95)),
            "est_cal_ms_M1_mean": float(ms1.mean()),
            "batch_ms_mean": float(tb.mean()),
            "lifetime_ms_mean": {str(B): float((ms + B * tb).mean()) for B in B_LIFE},
            "lifetime_ms_M1_mean": {str(B): float((ms1 + B * tb).mean()) for B in B_LIFE},
            "_picks": picks.tolist(), "_loss": loss.tolist(), "_ms": ms.tolist(),
        }
    # no-calibration references (deterministic)
    for a in (AMAX, 100):
        out["policies"]["NO-CAL alpha=%d" % a] = {
            "picks": {a: DRAWS}, "loss_mean": ceil - hrec[a], "loss_p95": ceil - hrec[a],
            "loss_max": ceil - hrec[a], "frac_over_1e-3": float(ceil - hrec[a] > 1e-3),
            "n_used_mean": 0, "launches_mean": 0, "query_searches_mean": 0,
            "est_cal_ms_mean": 0.0, "est_cal_ms_p95": 0.0, "est_cal_ms_M1_mean": 0.0,
            "batch_ms_mean": t_batch[a],
            "lifetime_ms_mean": {str(B): B * t_batch[a] for B in B_LIFE},
            "lifetime_ms_M1_mean": {str(B): B * t_batch[a] for B in B_LIFE}}
    # paired comparison of the primary controller against fixed S=128
    p, b = out["policies"][PRIMARY], out["policies"][BASE]
    pl, bl = np.array(p["_loss"]), np.array(b["_loss"])
    pm, bm = np.array(p["_ms"]), np.array(b["_ms"])
    out["paired_primary_vs_base"] = {
        "same_pick": float(np.mean(np.array(p["_picks"]) == np.array(b["_picks"]))),
        "seq_lower_loss": float(np.mean(pl < bl - 1e-12)),
        "seq_higher_loss": float(np.mean(pl > bl + 1e-12)),
        "seq_cheaper_cal": float(np.mean(pm < bm)),
        "cal_ms_diff_mean": float((pm - bm).mean()),
    }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    allres = []
    for split in ("A", "B"):
        for ds, np_ in CONFIGS:
            cfg = Config(ds, np_)
            r = run_config(cfg, split)
            allres.append(r)
            print("done %s split %s" % (cfg.name, split), flush=True)
    json.dump(allres, open(os.path.join(OUT, "replay.json"), "w"))
    print("wrote out/replay.json")


if __name__ == "__main__":
    main()
