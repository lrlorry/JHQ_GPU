#!/usr/bin/env python3
"""Step 2: reproduce the existing fixed-S calibration results before anything new.

R1  Exact. figures/alpha_resample.py is re-run unmodified in a sandbox whose
    `style` module points DATA at a copy-free symlink of alpha_perquery/, and
    its JSON is compared key by key with the committed data/alpha_resample.json.
    That script's protocol: grid 2..100, alpha_max = 100, eps = 1, the S
    calibration queries drawn from all 1000, held-out = the other 1000-S,
    loss = rec(100) - rec(pick).

R2  Statistical. The paper's protocol (demo_jhq_budget_arms.cu, Table
    budget_current): grid 200..2, alpha_max = 200, eps = 1, even queries
    calibrate, odd evaluate, loss = best held-out recall on the grid minus the
    pick's. The GPU run drew its 64 samples with std::rand and dumped no ids,
    and it trained its own index (training is not bit-reproducible), so the
    target is budget_arms_2ds.log's statistics, not its draws. The replay here
    is the same rule on the dumped index.

Writes out/baseline_repro.json and prints both comparisons.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIGS, DATA, GRID, HERE, K, PQ, Config, parse_arms_log  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- R1 ----------
def r1_exact():
    box = os.path.join(OUT, "r1_sandbox")
    shutil.rmtree(box, ignore_errors=True)
    os.makedirs(os.path.join(box, "figures"))
    os.makedirs(os.path.join(box, "data"))
    os.symlink(PQ, os.path.join(box, "data", "alpha_perquery"))
    shutil.copy(os.path.join(DATA, "..", "figures", "alpha_resample.py"),
                os.path.join(box, "figures"))
    with open(os.path.join(box, "figures", "style.py"), "w") as f:
        f.write("import os\nDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),"
                " '..', 'data')\n")
    subprocess.run([sys.executable, "-I", os.path.join(box, "figures", "alpha_resample.py")],
                   check=True, stdout=open(os.path.join(OUT, "r1_stdout.txt"), "w"))
    new = json.load(open(os.path.join(box, "data", "alpha_resample.json")))
    old = json.load(open(os.path.join(DATA, "alpha_resample.json")))
    mism = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            mism.append(key)
    print("R1 alpha_resample.json: %d cells, %d identical, %d differ %s"
          % (len(old), len(old) - len(mism), len(mism), mism[:5]))
    return {"cells": len(old), "mismatched": mism}


# ---------------------------------------------------------------- R2 ----------
def bisect(dis, rows, slots=1):
    """demo_jhq_budget_arms.cu run_rule, verbatim: descending grid, A[0] the
    reference, mid = ceil((lo+hi)/2), accept -> lo = mid."""
    lo, hi, probes = 0, len(GRID) - 1, []
    while lo < hi:
        mid = (lo + hi + 1) // 2
        probes.append(GRID[mid])
        if dis[GRID[mid]][rows].sum() <= slots:
            lo = mid
        else:
            hi = mid - 1
    return GRID[lo], probes


def enumerate_grid(dis, rows, slots=1):
    best = 0
    for gi in range(1, len(GRID)):
        if dis[GRID[gi]][rows].sum() <= slots:
            best = gi
    return GRID[best]


def r2_protocol(draws=2000, seed=20261012):
    logs = parse_arms_log(os.path.join(DATA, "budget_arms_2ds.log"))
    rng = np.random.default_rng(seed)
    res = {}
    for ds, np_ in CONFIGS:
        c = Config(ds, np_)
        cal = np.arange(0, c.n, 2)
        held = np.arange(1, c.n, 2)
        dis = c.dis(GRID[0])
        hrec = {a: float(c.rec[a][held].mean()) for a in GRID}
        ceil = max(hrec.values())
        gpu = [v for (d, p, nl), v in logs.items() if d == ds and p == np_][0]
        print("\nR2 %s  held-out recall  replay | GPU log (different training of the same config)" % c.name)
        print("   " + "  ".join("a%d %.4f|%.4f" % (a, hrec[a], gpu["fixed"][a][0]) for a in GRID))
        full_rows = cal
        full_pick, _ = bisect(dis, full_rows)
        print("   FULL pool pick: replay %d | GPU %s" % (full_pick, gpu["full"]["alpha"]))
        cell = {"held_recall": hrec, "ceiling": ceil, "full_pick": full_pick,
                "gpu_full_pick": int(float(gpu["full"]["alpha"])), "S": {}}
        for S in (32, 64, 128):
            picks, enum_dis = [], 0
            for _ in range(draws):
                rows = cal[rng.choice(len(cal), size=S, replace=False)]
                a, _ = bisect(dis, rows)
                picks.append(a)
                enum_dis += enumerate_grid(dis, rows) != a
            loss = np.array([ceil - hrec[a] for a in picks])
            frac = float((loss > 1e-3).mean())
            g = gpu["rule"][S]
            gf = float(g["frac_over_1e-3"])
            # binomial 95% interval of the GPU's 64-draw frequency, normal approx.
            half = 1.96 * np.sqrt(max(gf * (1 - gf), 1.0 / 64) / 64)
            ok = abs(frac - gf) <= half
            print("   S=%-4d replay: median_a=%-3d mean_loss=%.4f p95=%.4f frac>1e-3=%.3f enum!=bisect=%d/%d"
                  "  | GPU(64): median_a=%s mean_loss=%s p95=%s frac=%s  [%s]"
                  % (S, int(np.median(picks)), loss.mean(), np.percentile(loss, 95), frac,
                     enum_dis, draws, g["median_alpha"], g["mean_loss"], g["p95_loss"],
                     g["frac_over_1e-3"], "within 64-draw CI" if ok else "OUTSIDE 64-draw CI"))
            vals, cnt = np.unique(picks, return_counts=True)
            cell["S"][S] = {"picks": {int(v): int(n) for v, n in zip(vals, cnt)},
                            "mean_loss": float(loss.mean()),
                            "p95_loss": float(np.percentile(loss, 95)),
                            "frac_over_1e-3": frac, "enum_disagree": int(enum_dis),
                            "gpu": {k: g[k] for k in ("median_alpha", "mean_loss", "p95_loss",
                                                      "frac_over_1e-3")},
                            "frac_within_gpu_ci": bool(ok)}
        res[c.name] = cell
    return res


if __name__ == "__main__":
    out = {"R1": r1_exact(), "R2": r2_protocol()}
    json.dump(out, open(os.path.join(OUT, "baseline_repro.json"), "w"), indent=1)
    print("\nwrote out/baseline_repro.json")
