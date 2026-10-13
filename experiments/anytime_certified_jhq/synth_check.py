#!/usr/bin/env python3
"""Brute-force check of the bounds used by acs_pilot.py on synthetic populations (exact, all permutations).

  python3 synth_check.py <out dir>

For populations of W = 8 values with known per-word ranges [-h_w, h_w] (plus offsets c_w for the control-variate
form), every one of the 8! orders is enumerated, so the miscoverage P(UB_t < S) is computed exactly, not sampled.
Requirements: C0 never fails; C1 fails with probability <= delta at every t; C2/C3 with probability <= 5 delta'
(= delta_c as used by the pilot). Several population shapes, including adversarial ones (one large value).
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acs_pilot as A  # noqa: E402


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261013)
    W = 8
    perms = np.array(list(itertools.permutations(range(W))))
    rows = []
    shapes = {
        "uniform": lambda h: rng.uniform(-h, h),
        "extreme": lambda h: np.where(rng.random(W) < 0.5, -h, h),
        "one_large": lambda h: np.concatenate([[h[0]], -h[1:] * 0.1]),
        "gaussian_clip": lambda h: np.clip(rng.normal(0, h / 3), -h, h),
    }
    for name, gen in shapes.items():
        for trial in range(5):
            h = rng.uniform(0.5, 1.5, W)
            y = gen(h)
            cw = rng.uniform(-0.3, 0.3, W)
            S = y.sum()
            for kind in ("C0", "C1", "C2", "C3"):
                for dc in (0.3, 0.1, 0.03):
                    for t in (1, 2, 3, 4, 6, 7):
                        yp = y[perms]
                        cv = (cw[perms] if kind == "C3" else np.zeros_like(yp))
                        hp = h[perms]
                        ub = A.bounds(yp, cv, hp, t, dc, kind)
                        rate = float((ub < S - 1e-12).mean())
                        nominal = 0.0 if kind == "C0" else dc
                        rows.append(dict(shape=name, trial=trial, kind=kind, delta_c=dc, t=t, W=W, miscoverage=rate, nominal=nominal,
                                         ok=bool(rate <= nominal + 1e-12)))
    bad = [r for r in rows if not r["ok"]]
    json.dump(dict(rows=len(rows), failures=len(bad), worst=sorted(rows, key=lambda r: r["miscoverage"] - r["nominal"])[-5:]),
              open(os.path.join(out, "synth_check.json"), "w"), indent=1)
    import csv
    with open(os.path.join(out, "CS_VALIDITY_SYNTH.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("SYNTH", "OK" if not bad else "FAILED", len(rows), "cases,", len(bad), "above nominal")


if __name__ == "__main__":
    main()
