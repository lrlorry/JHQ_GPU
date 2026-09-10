#!/usr/bin/env python3
"""Offline validation of the alpha rule: held-out queries, and a distribution.

Section 6.3.2. Everything the rule has been credited with so far comes from a
single draw of S=32 queries, scored on the whole batch -- *including* the 32 it
calibrated on. That is one number with no error bar and mild contamination,
and it is not evidence that the rule generalises.

Fixing it needed no smarter experiment, only the per-query answers. With one
frozen index and the full alpha grid dumped a query at a time
(data/alpha_perquery/, from demo_jhq_v36's out_prefix, which has always
written them), every calibration draw can be replayed here: sample S queries,
run the rule on exactly what it would have seen, and score the alpha it picks
on the queries it did not. Thousands of draws, no GPU.

This file writes data/alpha_resample.json for fig_calibration to read.
"""
import sys, os, json, struct, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import DATA

PQ = os.path.join(DATA, "alpha_perquery")
GRID = [2, 4, 8, 16, 32, 64, 100]          # the rule's grid; alpha_max = 100
SLOTS, K = 1, 10                            # the shipped tolerance, and top-k
DRAWS = 2000
RNG = np.random.default_rng(20260910)


def ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:]


def load(ds, np_):
    ans = {}
    for a in GRID + [200]:
        p = os.path.join(PQ, "%s_np%d_a%d.ivecs" % (ds, np_, a))
        if os.path.exists(p):
            ans[a] = ivecs(p)[:, :K]
    gt = ivecs(os.path.join(PQ, "gt_%s.ivecs" % ds))[:, :K]
    n = min(len(gt), min(len(v) for v in ans.values()))
    return {a: v[:n] for a, v in ans.items()}, gt[:n], n


def recall_per_query(ids, gt):
    """Recall@10 for each query: |returned ∩ true| / k."""
    return np.array([len(set(r) & set(g)) for r, g in zip(ids, gt)]) / float(K)


def misses(cand, ref, rows):
    """Slots of the reference top-k that this alpha failed to return, summed
    over the sampled queries -- the rule's own criterion, set intersection per
    query rather than position equality."""
    return sum(K - len(set(cand[i]) & set(ref[i])) for i in rows)


def rule(ans, rows, slots=SLOTS):
    """The shipped rule: bisect the grid for the smallest alpha whose sampled
    top-k agrees with the alpha_max reference to within `slots`."""
    ref = ans[GRID[-1]]
    lo, hi = 0, len(GRID) - 1                 # index into GRID, ascending
    while lo < hi:
        mid = (lo + hi) // 2
        if misses(ans[GRID[mid]], ref, rows) <= slots:
            hi = mid
        else:
            lo = mid + 1
    return GRID[lo]


out = {}
for ds, np_ in [("vogue-768", 128), ("vogue-768", 512),
                ("openai3-3072", 128), ("openai3-3072", 512)]:
    ans, gt, n = load(ds, np_)
    if len(ans) < len(GRID):
        continue
    rec = {a: recall_per_query(v, gt) for a, v in ans.items()}
    full = {a: float(r.mean()) for a, r in rec.items()}

    # the reference the paper compares against, and the best the grid holds
    fixed = full[100]
    best_a = min(a for a in GRID if fixed - full[a] <= 1e-4)   # oracle constant

    for S in (8, 16, 32, 64, 128):
        picks, losses = [], []
        for _ in range(DRAWS):
            rows = RNG.choice(n, size=S, replace=False)
            held = np.setdiff1d(np.arange(n), rows, assume_unique=False)
            a = rule(ans, rows)
            picks.append(a)
            # scored only on queries the rule never saw
            losses.append(float(rec[100][held].mean() - rec[a][held].mean()))
        losses = np.array(losses)
        cnt = collections.Counter(picks)
        out["%s|%d|%d" % (ds, np_, S)] = dict(
            n=n, S=S, draws=DRAWS,
            picks={str(k): v for k, v in sorted(cnt.items())},
            pick_mode=int(cnt.most_common(1)[0][0]),
            oracle_alpha=int(best_a),
            recall_fixed=fixed,
            heldout_loss_mean=float(losses.mean()),
            heldout_loss_p95=float(np.percentile(losses, 95)),
            heldout_loss_max=float(losses.max()),
            frac_within_1e3=float((losses <= 1e-3).mean()))
        print("%-14s np=%-4d S=%-4d picks=%-22s oracle=%-4d "
              "held-out loss mean=%+.5f p95=%+.5f  within 1e-3: %.1f%%"
              % (ds, np_, S, dict(cnt.most_common(3)), best_a,
                 losses.mean(), np.percentile(losses, 95),
                 100 * (losses <= 1e-3).mean()))

with open(os.path.join(DATA, "alpha_resample.json"), "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("\nwrote data/alpha_resample.json  (%d cells)" % len(out))
