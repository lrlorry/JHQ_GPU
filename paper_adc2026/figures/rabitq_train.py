#!/usr/bin/env python3
"""IVF-RaBitQ with the coarse-training budget and the search kernel as variables.

Two things the archived comparison could not settle.

**The training budget was not controlled.** cuVS defaults
max_train_points_per_cluster to 256 while JHQ trains at 39, so the archived
RaBitQ runs trained on 5.8x to 6.6x JHQ's points -- the whole dataset at
vogue-768.  That inflates its build time and improves its centroids, so the
archived comparison is unfair to it on build and generous to it on search.
Here the budget is a variable: 39, matching JHQ, and cuVS's own 256.

**The search kernel was not recorded.** cuVS picks between LUT16, LUT32,
QUANT4 and QUANT8, and paper_rabitq.log kept only its wrapper's parsed summary,
so which kernel the published curves used is not recoverable.  The sweep here
tries all four at one probe depth per dataset and records the mode on every row.
A baseline measured at a kernel that is not its best is not a baseline.
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style

MODE = {0: "LUT16", 1: "LUT32", 2: "QUANT4", 3: "QUANT8"}


def load():
    rows = []
    p = style.datafile("rq_train.log")
    if not os.path.exists(p):
        return rows
    for ln in open(p, errors="ignore"):
        m = re.match(r"\s*RQ\s+(\S+)\s+nlist=(\d+)\s+tpl=(\d+)\s+used=(\S+)\s+"
                     r"mode=(\d+)\s+np=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)\s+"
                     r"build_ms=(\S+)", ln)
        if m:
            rows.append(dict(ds=m.group(1), nlist=int(m.group(2)),
                             tpl=int(m.group(3)), used=m.group(4),
                             mode=int(m.group(5)), np=int(m.group(6)),
                             recall=float(m.group(7)), qps=int(m.group(8)),
                             build=m.group(9)))
    return rows


def main():
    rows = load()
    if not rows:
        print("  rq_train.log not present yet"); return

    print("  search kernel, at nprobe=128 and the library's own budget:")
    by = collections.defaultdict(dict)
    for r in rows:
        if r["np"] == 128 and r["tpl"] == 256:
            by[r["ds"]][r["mode"]] = (r["recall"], r["qps"])
    for ds in sorted(by):
        best = max(by[ds].items(), key=lambda kv: kv[1][1])
        cells = "  ".join("%s %d@%.4f" % (MODE[m], q, rc)
                          for m, (rc, q) in sorted(by[ds].items()))
        print("     %-14s %s   best %s" % (ds, cells, MODE[best[0]]))

    print("\n  what the coarse-training budget costs and buys:")
    for ds in sorted({r["ds"] for r in rows}):
        for np_ in sorted({r["np"] for r in rows if r["ds"] == ds}):
            v = {r["tpl"]: r for r in rows
                 if r["ds"] == ds and r["np"] == np_ and r["mode"] == 2}
            if 39 in v and 256 in v:
                a, b = v[39], v[256]
                print("     %-14s np=%-5d 39: %.4f/%s  256: %.4f/%s  "
                      "qps %+.1f%%  recall %+.4f"
                      % (ds, np_, a["recall"], a["qps"], b["recall"], b["qps"],
                         100 * (b["qps"] / a["qps"] - 1),
                         b["recall"] - a["recall"]))

    builds = [(r["ds"], r["tpl"], float(r["build"]) / 1000)
              for r in rows if r["build"] not in ("?", "")]
    if builds:
        print("\n  build seconds by training budget:")
        agg = collections.defaultdict(list)
        for ds, tpl, s in builds:
            agg[(ds, tpl)].append(s)
        import statistics as st
        for k in sorted(agg):
            print("     %-14s tpl=%-4d %.2f s (median of %d)"
                  % (k[0], k[1], st.median(agg[k]), len(agg[k])))


if __name__ == "__main__":
    main()
