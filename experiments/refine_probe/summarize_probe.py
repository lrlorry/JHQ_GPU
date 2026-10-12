#!/usr/bin/env python3
"""Summary and decisions for run_refine_probe.sh. Reads the demo outputs
(<OUT>/<ds>_a<alpha>_<variant>.txt) and the bound analyses
(<OUT>/<ds>_a<alpha>.bounds.json); measures nothing itself.

Decision rules, fixed before the run (they are conventions, not theorems):

  share     refinement share of search time at an operating point =
            1 - QPS(v60, lanes 0) / QPS(v58_jq, refinement compiled out).
            It caps any refinement-side gain at 1 / (1 - share).

  lanes     GO if, at the adaptive alpha, the best of L = 8, 16, 32 is >= 10%
            faster end to end than the v58 kernel (mean of the two L=0 runs) on
            at least 2 of the 3 datasets, with recall within 1e-4 of it.
            L=1 must reproduce the L=0 recall exactly; if not, the kernel is
            wrong and nothing about it is reported.

  bounds    GO if, with the threshold a kernel can actually hold (first-k),
            a bound that needs no extra storage (B0 or B1) leaves <= 75% of the
            residual dimensions to read, and the end-to-end ceiling that
            implies is >= 1.10, on at least 2 of the 3 datasets. B2 (needs
            stored norms and y_hat per dimension) is reported, not decided on.

  python3 summarize_probe.py <OUT>
"""
import glob
import json
import os
import re
import sys

ADAPTIVE = {"vogue-768": 64, "openai3-3072": 4, "arxiv-768": 200}


def parse(path):
    rec = qps = None
    if os.path.exists(path):
        for ln in open(path, errors="ignore"):
            m = re.match(r"Recall@\d+\s*:\s*([\d.]+)", ln)
            if m:
                rec = float(m.group(1))
            m = re.match(r"QPS\s*:\s*([\d.]+)", ln)
            if m:
                qps = float(m.group(1))
    return rec, qps


def main():
    out = sys.argv[1]
    lanes_go = bounds_go = 0
    lanes_bug = False
    for ds, aa in ADAPTIVE.items():
        print("\n=== %s (np=128) ===" % ds)
        for a in (aa, 100):
            g = lambda v: parse(os.path.join(out, "%s_a%d_%s.txt" % (ds, a, v)))
            base = [x for x in (g("L0"), g("L0b")) if x[1]]
            if not base:
                print("  a=%d: no L=0 result" % a)
                continue
            r0 = base[0][0]
            q0 = sum(x[1] for x in base) / len(base)
            drift = (abs(base[0][1] - base[-1][1]) / q0) if len(base) == 2 else float("nan")
            rj, qj = g("jq")
            share = (1 - q0 / qj) if qj else None
            print("  a=%-4d L=0 recall %.4f  QPS %.0f (two runs differ by %.1f%%)  refine share %s  ceiling %s"
                  % (a, r0, q0, 100 * drift, "%.1f%%" % (100 * share) if share is not None else "NA",
                     "x%.2f" % (1 / (1 - share)) if share is not None and share < 1 else "NA"))
            best = None
            for L in ("L1", "L8", "L16", "L32"):
                r, q = g(L)
                if not q:
                    continue
                dr = r - r0
                print("        %-4s recall %.4f (%+.4f)  QPS %.0f  x%.3f" % (L, r, dr, q, q / q0))
                if L == "L1" and a == aa and abs(dr) > 0:
                    lanes_bug = True
                if L != "L1" and abs(dr) <= 1e-4 + 1e-9 and (best is None or q / q0 > best):
                    best = q / q0
            if a == aa and best is not None and best >= 1.10:
                lanes_go += 1
        bj = os.path.join(out, "%s_a%d.bounds.json" % (ds, aa))
        if os.path.exists(bj):
            B = json.load(open(bj))
            if not B.get("ok"):
                print("  bounds: dump check FAILED -- not counted")
                continue
            free = [r for r in B["rows"] if r["bound"] in ("B0", "B1") and r["tau"] == "first-k"]
            bestb = min(free, key=lambda r: r["dims_read_frac"])
            ceil = bestb.get("e2e_ceiling")
            print("  bounds (first-k): best free bound %s reads %.1f%% of dims, e2e ceiling %s" % (
                bestb["bound"], 100 * bestb["dims_read_frac"], "x%.3f" % ceil if ceil else "NA"))
            if bestb["dims_read_frac"] <= 0.75 and ceil and ceil >= 1.10:
                bounds_go += 1
    print("\nDECISIONS (pre-registered)")
    if lanes_bug:
        print("  lanes : L=1 did not reproduce L=0 recall -- kernel bug, no result")
    else:
        print("  lanes : %d of 3 datasets >= +10%% at the adaptive alpha -> %s" % (
            lanes_go, "GO" if lanes_go >= 2 else "STOP"))
    print("  bounds: %d of 3 datasets pass -> %s" % (bounds_go, "GO" if bounds_go >= 2 else "STOP"))


if __name__ == "__main__":
    main()
