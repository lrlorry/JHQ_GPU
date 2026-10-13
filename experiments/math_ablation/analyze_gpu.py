#!/usr/bin/env python3
"""Matched-recall comparison on the measured frontiers (local, from RECALL_QPS_FRONTIER.csv).

  python3 analyze_gpu.py <dir with RECALL_QPS_FRONTIER.csv>

Per workload and batch size:
  B0 frontier      the Pareto set of v65 over every measured (alpha, nprobe): highest QPS at each recall
  table cost       V1 arm A (paper index through the full 256-entry table) against B0 at the same alpha and nprobe:
                   recall should match, the QPS ratio is the cost of the unfactorised table alone
  E8 effect        arm B (E8) against arm A at the same alpha: recall and QPS change from the representation alone
  E8 vs B0         arm B against the B0 frontier at B's recall (log-QPS interpolated between frontier points;
                   points outside the frontier's recall range are reported as such)
Output: GPU_COMPARISON.json and a printed table.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict


def pareto(pts):
    """points (recall, qps) -> frontier sorted by recall, QPS non-increasing."""
    pts = sorted(pts)
    out = []
    for r, q in sorted(pts, key=lambda x: (-x[0], -x[1])):
        if not out or q > out[-1][1]:
            out.append((r, q))
    return sorted(out)


def qps_at(front, rec):
    """B0's best QPS at recall >= rec, log-interpolated between neighbouring frontier points."""
    if not front or rec > front[-1][0] + 1e-12:
        return None
    for i, (r, q) in enumerate(front):
        if r >= rec - 1e-12:
            if i == 0:
                return q
            r0, q0 = front[i - 1]
            t = (rec - r0) / max(r - r0, 1e-12)
            return math.exp(math.log(q0) + t * (math.log(q) - math.log(q0)))
    return None


def main():
    D = sys.argv[1]
    rows = list(csv.DictReader(open(os.path.join(D, "RECALL_QPS_FRONTIER.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["dataset"], int(r["batch"]))].append(r)
    out = {}
    for (var, ds, b), rs in sorted(by.items()):
        if var != "B0 v65":
            continue
        front = pareto([(float(r["recall_at_10"]), float(r["qps_median"])) for r in rs])
        A = {(int(r["nprobe"]), float(r["alpha"])): r for r in by.get(("V1 GPU A: paper index, full 256 table", ds, b), [])}
        Bm = {(int(r["nprobe"]), float(r["alpha"])): r for r in by.get(("V1 GPU B: E8, full 256 table", ds, b), [])}
        b0 = {(int(r["nprobe"]), float(r["alpha"])): r for r in rs}
        ent = dict(frontier=front, table_cost={}, e8_vs_A={}, e8_vs_B0={})
        for key, a in sorted(A.items()):
            if key in b0:
                ent["table_cost"]["np%d a%g" % key] = dict(recall_B0=float(b0[key]["recall_at_10"]), recall_A=float(a["recall_at_10"]),
                                                           qps_ratio_A_over_B0=float(a["qps_median"]) / float(b0[key]["qps_median"]))
            if key in Bm:
                ent["e8_vs_A"]["np%d a%g" % key] = dict(recall_A=float(a["recall_at_10"]), recall_B=float(Bm[key]["recall_at_10"]),
                                                        qps_ratio_B_over_A=float(Bm[key]["qps_median"]) / float(a["qps_median"]))
        for key, bb in sorted(Bm.items()):
            rec, q = float(bb["recall_at_10"]), float(bb["qps_median"])
            qb = qps_at(front, rec)
            ent["e8_vs_B0"]["np%d a%g" % key] = dict(recall_B=rec, qps_B=q, qps_B0_frontier_at_recall=qb,
                                                     ratio=None if qb is None else q / qb)
        out["%s b%d" % (ds, b)] = ent
        tc = [v["qps_ratio_A_over_B0"] for v in ent["table_cost"].values()]
        ev = [v["ratio"] for v in ent["e8_vs_B0"].values() if v["ratio"] is not None]
        print("%-13s b=%-4d  table cost A/B0 QPS %s | E8 vs B0 frontier at matched recall %s" % (
            ds, b, " ".join("%.3f" % x for x in tc) or "-", " ".join("%.3f" % x for x in ev) or "-"))
    json.dump(out, open(os.path.join(D, "GPU_COMPARISON.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
