#!/usr/bin/env python3
"""What a smaller refinement budget at matched recall buys end to end (b = 8 and b = 1000), derived.

  python3 roq_cost_model.py <results dir> <eval json> [<eval json> ...]

For JHQ at its matched budget T_J and a method at its matched budget T_V (both read on the VALIDATION split of
the eval json, or the test-confirmed ones when the json has them), the end-to-end time is
  front + scan(T) + merge(T) + refine(T) + topk(T) + host + extra(method)
with every stage a measured curve in T [M]: v65 at the workload alpha (pipeline_campaign exp2, v63 builds: b = 8 at
G = 8 with the shared bound, b = 1000 at G = 0) and the v62 reduced-alpha sweep (scan, merge, refine and top-k against
ck) interpolated in log T and normalised to v65's measurement at ck [D]. The primary scan's dependence on T is the
selection buffer only: the bits read, the lookups and the table size are those of the paper index for every
per-coordinate rule (the levels differ, the table does not). extra:
  per-coordinate rules   0 (the table builder reads 2d levels instead of one) [T]
  per-list rules         one table per (query, probed list): measured single-table build time x nprobe as an upper
                         bound, and the build folded into the scan (about a quarter more scan work) as a lower one [T]
No new kernel ran. Output: roq_cost_model.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(HERE, "..", "pipeline_campaign", "exp2")
WL = {"arxiv-768_np128_a100": ("arxiv-768", 100.0), "vogue-768_np128_a64": ("vogue-768", 64.0),
      "openai3-3072_np128_a4": ("openai3-3072", 4.0)}
PATHOLOGICAL = {("arxiv-768", 50.0)}
NPROBE = 128


def curve(points, c):
    cs = np.array([p[0] for p in points], float); ts = np.array([p[1] for p in points], float)
    if c >= cs[-1]:
        c0, c1, t0, t1 = cs[-2], cs[-1], ts[-2], ts[-1]
        return float(t1 + max(t1 - t0, 0.0) * (np.log(c) - np.log(c1)) / (np.log(c1) - np.log(c0)))
    if c <= cs[0]:
        return float(ts[0])
    return float(np.interp(np.log(c), np.log(cs), ts))


def main():
    R = sys.argv[1]
    evals = {}
    for p in sys.argv[2:]:
        for tag, e in json.load(open(p)).items():
            evals.setdefault(tag, {}).update(e["methods"])
    meas = json.load(open(os.path.join(CAMP, "v63_raw", "perf", "stages_summary.json")))
    sweep = json.load(open(os.path.join(CAMP, "raw", "nsys", "stages_summary.json")))
    out = {}
    for tag, (ds, alpha) in WL.items():
        if tag not in evals:
            continue
        ck = int(alpha * 10)
        out[tag] = {}
        for b, G, Gm in ((8, 8, 108), (1000, 0, 0)):
            m = next(s for s in meas if s["tag"] == ds and float(s["alpha"]) == alpha and s["batch"] == b and s["G"] == Gm)
            u = m["stages_us"]
            total = m["latency_ms"] * 1e3
            front = u["rotation"] + u["centroid"] + u["probe_select"] + u["lut"]
            host = total - sum(u.values())
            sw = sorted((float(s["alpha"]) * 10, s["stages_us"]) for s in sweep
                        if s["tag"] == ds and s["batch"] == b and s["G"] == G and (ds, float(s["alpha"])) not in PATHOLOGICAL)
            pts = {k: [(c, st[k]) for c, st in sw] for k in ("scan", "merge", "refine", "topk")}
            norm = {k: (u[k] / curve(pts[k], ck)) if curve(pts[k], ck) > 0 else 1.0 for k in pts}

            def e2e(T, extra=0.0):
                st = dict(front=front, scan=curve(pts["scan"], T) * norm["scan"],
                          merge=(curve(pts["merge"], T) * norm["merge"]) if b == 8 else 0.0,
                          refine=curve(pts["refine"], T) * norm["refine"], topk=curve(pts["topk"], T) * norm["topk"],
                          host=host, extra=extra)
                return sum(st.values()), st
            res = dict(v65_measured_at_ck=total, methods={})
            for meth, ent in evals[tag].items():
                src = ent.get("test_matched")
                if src:
                    TV, TJ = src["T_variant"], src["T_JHQ"]
                else:
                    TV, TJ = ent.get("val", {}).get("T_min"), ent.get("val", {}).get("T_min_JHQ")
                if TV is None or TJ is None:
                    res["methods"][meth] = dict(note="no matched budget")
                    continue
                tj, stj = e2e(TJ)
                per_list = meth.startswith("list_")
                lo_extra = 0.25 * stj["scan"] if per_list else 0.0
                hi_extra = NPROBE * u["lut"] if per_list else 0.0
                tv_lo, stv = e2e(TV, lo_extra)
                tv_hi, _ = e2e(TV, hi_extra)
                res["methods"][meth] = dict(T_method=TV, T_JHQ=TJ, jhq_us=tj, method_us=tv_lo, method_us_upper=tv_hi,
                                            time_saving=1 - tv_lo / tj, time_saving_upper_cost=1 - tv_hi / tj,
                                            stages_jhq=stj, stages_method=stv)
            out[tag]["b%d" % b] = res
            print(tag, "b=%d" % b, {k: (v.get("T_method"), v.get("T_JHQ"), None if "time_saving" not in v else round(v["time_saving"], 3))
                                     for k, v in res["methods"].items()})
    json.dump(out, open(os.path.join(R, "roq_cost_model.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
