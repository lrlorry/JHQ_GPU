"""An *estimated* calibration-time model, fitted to measured GPU timings.

This is not GPU time. The offline replay counts what a policy would launch;
this file turns counts into milliseconds with a model fitted to the only
calibration timings that exist for these configurations,
data/budget_arms_2ds.log (same harness, same configs as the dumps):

    T_cal = F + a * L + kappa * sum_i s_i / qps(alpha_i)

    F      fixed per calibration (set_calibrating, first-call effects)
    L      number of search launches (one per alpha per batch of queries)
    s_i    queries in launch i, alpha_i its budget
    qps()  the fixed-arm 500-query throughput from the same log

Seven observations per configuration (bisection and enumeration at
S = 32, 64, 128, and the 500-query full pool) fit three parameters. Bisection
on an 8-point grid always makes 3 probes; which alphas it probes is taken from
the offline replay of the same rule, so the per-query term uses the expected
1/qps of the probed alphas.

The launch term is what matters for this experiment: on these logs a launch
costs ~0.6-6 ms whether it searches 32 queries or 128, so a policy that adds
launches to save queries can come out slower.
"""
import os

import numpy as np

from common import DATA, GRID, parse_arms_log


def _probe_inv_qps(dis_cal, rng, S, inv, draws=400):
    """Mean over draws of sum_{probed alpha} 1/qps(alpha) for the fixed bisection."""
    tot = 0.0
    n = len(dis_cal[GRID[0]])
    for _ in range(draws):
        rows = rng.choice(n, size=S, replace=False)
        lo, hi = 0, len(GRID) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            tot += inv[GRID[mid]]
            if dis_cal[GRID[mid]][rows].sum() <= 1:
                lo = mid
            else:
                hi = mid - 1
    return tot / draws


def fit(cfg, cal_idx, seed=7, include_full=True):
    """Fit (F, a, kappa) for one Config from budget_arms_2ds.log.

    include_full=False fits on the S <= 128 points only (model M2). On these
    logs cost barely grows between S=32 and S=128 while the 500-query pool is
    2-3x dearer, so one line through all seven points (M1) overstates the
    marginal cost of a small launch; M2 describes the S <= 128 regime the
    controllers live in and understates anything larger. Both are reported."""
    logs = parse_arms_log(os.path.join(DATA, "budget_arms_2ds.log"))
    g = [v for (d, p, _), v in logs.items() if d == cfg.ds and p == cfg.np][0]
    qps = {a: g["fixed"][a][1] for a in GRID}
    inv = {a: 1.0 / qps[a] * 1e3 for a in GRID}          # ms per query at batch 500
    dis = cfg.dis(GRID[0])
    dis_cal = {a: v[cal_idx] for a, v in dis.items()}
    rng = np.random.default_rng(seed)
    rows, ys, labels = [], [], []
    for S, kv in sorted(g["rule"].items()):
        pin = _probe_inv_qps(dis_cal, rng, S, inv)
        rows.append([1.0, 4.0, S * (inv[GRID[0]] + pin)])   # ref + 3 probes
        ys.append(float(kv["cal_ms"])); labels.append("bisect S=%d" % S)
        e = g["enum"][S]
        rows.append([1.0, 8.0, S * sum(inv.values())])      # ref + 7
        ys.append(float(e["enum_ms"])); labels.append("enum S=%d" % S)
    # full pool: 500 queries, the bisection's deterministic probes
    full_rows = np.arange(len(cal_idx))
    lo, hi, pin = 0, len(GRID) - 1, 0.0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        pin += inv[GRID[mid]]
        if dis_cal[GRID[mid]][full_rows].sum() <= 1:
            lo = mid
        else:
            hi = mid - 1
    n_full = len(cal_idx)
    if include_full:
        rows.append([1.0, 4.0, n_full * (inv[GRID[0]] + pin)])
        ys.append(float(g["full"]["cal_ms"])); labels.append("full S=%d" % n_full)
    X, y = np.array(rows), np.array(ys)
    coef = _nnls(X, y)                                   # no negative overheads
    pred = X @ coef
    return {"F": float(coef[0]), "a": float(coef[1]), "kappa": float(coef[2]),
            "qps": qps, "inv_ms": inv,
            "fit": [(l, float(o), float(p)) for l, o, p in zip(labels, y, pred)]}


def _nnls(X, y):
    """Least squares with coef >= 0, by enumerating active sets (3 unknowns)."""
    best, bc = None, None
    n = X.shape[1]
    for mask in range(1, 1 << n):
        cols = [j for j in range(n) if mask >> j & 1]
        c, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        if (c < 0).any():
            continue
        full = np.zeros(n); full[cols] = c
        r = float(((X @ full - y) ** 2).sum())
        if best is None or r < best:
            best, bc = r, full
    return bc


def cal_ms(model, launches):
    """launches: list of (alpha, n_queries). Returns the estimated ms."""
    t = model["F"]
    for a, s in launches:
        t += model["a"] + model["kappa"] * s * model["inv_ms"][a]
    return t
