#!/usr/bin/env python3
"""Paired statistics from the evaluator's per-query recall curves (eval_<label>_<tag>_perquery.npz). Local, no data.

  python3 analyze_perquery.py <results dir> <label> [split ...]      (default splits: calib val; test only when asked)

Per workload and method, against JHQ on the same queries:
  paired recall difference at T = ck/8, ck/4, ck/2, ck with a 95% bootstrap CI (queries resampled, paired)
  T_min at the strict target (JHQ's mean recall at T = ck on that split) and, as a POST-HOC secondary reading added
  because the strict target sits on a plateau where a single neighbour moves T_min a grid step, at 99.5% of it
Output: perquery_<label>_<splits>.json and a printed table.
"""
import json
import os
import sys

import numpy as np

TAGS = ("arxiv-768_np128_a100", "vogue-768_np128_a64", "openai3-3072_np128_a4")
CK = {"arxiv-768_np128_a100": 1000, "vogue-768_np128_a64": 640, "openai3-3072_np128_a4": 40}


def boot(x, rng, B=10000):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def tmin(curve, Ts, target):
    ok = [t for t, r in zip(Ts, curve) if r >= target - 1e-12]
    return min(ok) if ok else None


def main():
    R, label = sys.argv[1], sys.argv[2]
    splits = sys.argv[3:] or ["calib", "val"]
    rng = np.random.default_rng(20261013)
    out = {}
    for tag in TAGS:
        p = os.path.join(R, "eval_%s_%s_perquery.npz" % (label, tag))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        Ts = [int(t) for t in z["Ts"]]
        sel = np.isin(z["split"], splits)
        meths = [k[2:] for k in z.files if k.startswith("R_")]
        J = z["R_JHQ"][sel].astype(np.float64)
        ck = CK[tag]
        tgt = J[:, Ts.index(ck)].mean()
        out[tag] = dict(n=int(sel.sum()), target=float(tgt), methods={})
        print("== %s  splits=%s  n=%d  target(JHQ@ck)=%.4f" % (tag, "+".join(splits), sel.sum(), tgt))
        for m in meths:
            X = z["R_" + m][sel].astype(np.float64)
            cur = X.mean(0)
            ent = dict(T_min_strict=tmin(cur, Ts, tgt), T_min_995=tmin(cur, Ts, 0.995 * tgt), diffs={})
            row = []
            for t in sorted({max(10, ck // 8), max(10, ck // 4), max(10, ck // 2), ck}):
                if t not in Ts:
                    continue
                d = X[:, Ts.index(t)] - J[:, Ts.index(t)]
                lo, hi = boot(d, rng)
                ent["diffs"][str(t)] = dict(mean=float(d.mean()), ci95=[lo, hi])
                row.append("T=%d %+.4f [%+.4f,%+.4f]" % (t, d.mean(), lo, hi))
            out[tag]["methods"][m] = ent
            print("   %-22s Tmin strict=%-5s 99.5%%=%-5s | %s" % (m, ent["T_min_strict"], ent["T_min_995"], " | ".join(row)))
    json.dump(out, open(os.path.join(R, "perquery_%s_%s.json" % (label, "+".join(splits))), "w"), indent=1)


if __name__ == "__main__":
    main()
