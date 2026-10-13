#!/usr/bin/env python3
"""GPU cost model for progressive primary screening, b = 8 and b = 1000 (RTX 5090).

Configurations are priced against v65 at T* (alpha lowered to T*/k, the fair baseline):
  ops       each policy's VALIDATION-chosen operating point (analysis_summary.json, both rules,
            final top-T*) -- the decision basis
  envelope  the best of every configuration that is non-inferior on TEST -- selected on the
            test split, so an optimistic envelope and never a decision basis
Labels:
  [M] measured: nsys GPU time per search from experiments/pipeline_campaign/exp2 -- v63 builds at
      the workload alpha (b = 8, G = 8 with the shared bound = v65's b = 8 execution; b = 1000,
      G = 0 = v65's), and v62 at reduced alpha (the scan, merge, refine and top-k time as a
      function of ck at b = 8 G = 8 and b = 1000 G = 0)
  [D] derived: a measured curve evaluated at another candidate count, or scaled by the byte ratios
      the offline pilot measured
  [T] theoretical: a stated assumption
Nothing here ran as a new kernel.

The scan at ck is split into A + S(ck): A = the scan at the smallest measured ck (reads, table
lookups, per-candidate bookkeeping), S = what the selection adds as ck grows. Stage 1 costs
A (f s1 + 1 - f) + S(L): f is the part of A that shrinks with the words read, s1 the stage-1
sector ratio. Stage 2 reads s2 B_old_sec bytes per query as scattered 32-byte sectors, does the
table lookups for those words, and selects T* among L (priced as the final top-k kernel over L
candidates, which selects only 10: optimistic). Scenarios:
  U  ceiling:     f = 1; S(L) = S(T*) and merge(L) = merge(T*) (a free top-L); scattered sectors
                  at the DRAM peak (1.79 TB/s); no stage-2 launch floor
  O  optimistic:  f = 1; S, merge and the stage-2 selection beyond the measured range extrapolated
                  in log(c) from the last two points (sublinear); sectors at the peak; 3 us floor
  C  central:     f = 0.8; S, merge and the stage-2 selection extrapolated linearly in c, at no
                  less than arxiv's measured growth per candidate (the widest measured range) where
                  a workload's own curve is flat; scattered sectors at 50% of the peak; 5 us floor
Front-end, refinement, final top-k and host time are v65's at T* in every case: they cancel.

  python3 cost_model.py <results dir>  ->  cost_model.json, COST_MODEL_TABLE.csv
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(HERE, "..", "pipeline_campaign", "exp2")
WL = {"arxiv-768_np128_a100": ("arxiv-768", 100.0), "vogue-768_np128_a64": ("vogue-768", 64.0),
      "openai3-3072_np128_a4": ("openai3-3072", 4.0)}
PATHOLOGICAL = {("arxiv-768", 50.0)}      # ck = 500: 12 free buffer slots (pipeline_campaign Exp 2)
BW = 1.79e12                              # RTX 5090 DRAM peak, B/s
SCEN = {"U": dict(f=1.0, eff=1.0, ext="free", floor=0.0),
        "O": dict(f=1.0, eff=1.0, ext="log", floor=3.0),
        "C": dict(f=0.8, eff=0.5, ext="lin", floor=5.0)}
POLS = ("A0", "A0c", "A1", "A1c", "A2", "A3")


def curve(points, c, ext):
    """a measured stage time at candidate count c: interpolation in log(c) inside the measured
    range; outside, 'log' continues the last segment in log(c), 'lin' in c."""
    cs = np.array([p[0] for p in points], float); ts = np.array([p[1] for p in points], float)
    if c <= cs[-1]:
        return float(np.interp(np.log(max(c, cs[0])), np.log(cs), ts))
    c0, c1, t0, t1 = cs[-2], cs[-1], ts[-2], ts[-1]
    if ext == "log":
        return float(t1 + max(t1 - t0, 0.0) * (np.log(c) - np.log(c1)) / (np.log(c1) - np.log(c0)))
    return float(t1 + max(t1 - t0, 0.0) * (c - c1) / (c1 - c0))


def main():
    R = sys.argv[1]
    S = json.load(open(os.path.join(R, "analysis_summary.json")))["workloads"]
    rows = list(csv.DictReader(open(os.path.join(R, "RECALL_VS_BYTES.csv"))))
    meas = json.load(open(os.path.join(CAMP, "v63_raw", "perf", "stages_summary.json")))
    sweep = json.load(open(os.path.join(CAMP, "raw", "nsys", "stages_summary.json")))
    ref_slope = {}
    for b, G in ((8, 8), (1000, 0)):
        sw = sorted((float(s["alpha"]) * 10, s["stages_us"]) for s in sweep
                    if s["tag"] == "arxiv-768" and s["batch"] == b and s["G"] == G and ("arxiv-768", float(s["alpha"])) not in PATHOLOGICAL)
        ref_slope[b] = {k: (sw[-1][1][k] - sw[-2][1][k]) / (sw[-1][0] - sw[-2][0]) for k in ("scan", "merge", "topk")}
    out, table = dict(scenarios=SCEN, ref_slope_us_per_candidate=ref_slope, workloads={}), []
    for tag, (ds, alpha) in WL.items():
        w = S[tag]
        ck, Ts, Bsec, n = w["ck"], w["T_star"], w["B_old_sector_mean"], w["n_cand_mean"]
        mine = {(r["policy"], r["p"], r["L_over_ck"]): r for r in rows if r["tag"] == tag and int(r["T_final"]) == Ts}
        out["workloads"][tag] = {}
        for b, G, Gm in ((8, 8, 108), (1000, 0, 0)):
            m = next(s for s in meas if s["tag"] == ds and float(s["alpha"]) == alpha and s["batch"] == b and s["G"] == Gm)
            u = m["stages_us"]
            total = m["latency_ms"] * 1e3
            front = u["rotation"] + u["centroid"] + u["probe_select"] + u["lut"]
            host = total - sum(u.values())
            sw = sorted((float(s["alpha"]) * 10, s["stages_us"]) for s in sweep
                        if s["tag"] == ds and s["batch"] == b and s["G"] == G and (ds, float(s["alpha"])) not in PATHOLOGICAL)
            pts = {k: [(c, st[k]) for c, st in sw] for k in ("scan", "merge", "refine", "topk")}
            norm = {k: (u[k] / curve(pts[k], ck, "log")) if curve(pts[k], ck, "log") > 0 else 1.0 for k in pts}

            def at(k, c, ext="log"):
                v = curve(pts[k], c, ext) * norm[k]
                cmax = pts[k][-1][0]
                if ext == "lin" and c > cmax and k in ref_slope[b]:
                    v = max(v, curve(pts[k], cmax, ext) * norm[k] + ref_slope[b][k] * (c - cmax))
                return v
            A = at("scan", pts["scan"][0][0])                      # [D] the scan at the smallest measured ck
            base = dict(front=front, scan=at("scan", Ts), merge=at("merge", Ts) if b == 8 else 0.0,
                        refine=at("refine", Ts), topk=at("topk", Ts), host=host)          # v65 at T*   [D]
            base_total = sum(base.values())

            def price(r):
                L = float(r["L_mean"]); s1, s2 = float(r["sector_stage1"]), float(r["sector_stage2"])
                meta_b = float(r["metadata_ratio"]) * Bsec; scr_b = float(r["scratch_ratio"]) * Bsec
                res = {}
                for sc, P in SCEN.items():
                    f, eff = P["f"], P["eff"]
                    sel_L = (at("scan", Ts) - A) if P["ext"] == "free" else (at("scan", L, P["ext"]) - A)
                    stage1 = A * (f * s1 + 1.0 - f) + max(sel_L, 0.0)                                # [D]
                    merge = (at("merge", Ts) if P["ext"] == "free" else at("merge", L, P["ext"])) if b == 8 else 0.0
                    gather = s2 * Bsec * b / (BW * eff) * 1e6                                         # [T]
                    lookups = A * f * (L / n) * (1.0 - float(r["p"]))                                # [D]
                    select = at("topk", L, P["ext"]) if P["ext"] != "free" else at("topk", Ts)       # [D]
                    stage2 = max(P["floor"], gather + lookups) + select
                    extra = (meta_b + scr_b) * b / (BW * eff) * 1e6                                   # [T]
                    prog = dict(front=front, stage1=stage1, merge=merge, stage2=stage2, extra=extra,
                                refine=base["refine"], topk=base["topk"], host=host)
                    pt = sum(prog.values())
                    res[sc] = dict(stages=prog, total=pt, gain=1.0 - pt / base_total)
                return res

            res = dict(v65_measured_us=total, v65_measured_stages=dict(u, host=host), v65_at_Tstar=base,
                       v65_at_Tstar_total=base_total, A_us=A, ops={}, envelope={})
            for rule in ("ideal", "realistic"):
                res["ops"][rule] = {}
                for pol in POLS:
                    o = w["operating_points"][rule].get("%s/T%d" % (pol, Ts))
                    if o is None:
                        continue
                    r = mine[(pol, str(o["p"]), str(o["L_over_ck"]))]
                    pr = price(r)
                    res["ops"][rule][pol] = dict(config="p=%s L=%sck" % (o["p"], o["L_over_ck"]), non_inferior=o["non_inferior"],
                                                 delta=o["delta"], total_bytes_ratio=o["total_bytes_ratio"], price=pr)
                    for sc, v in pr.items():
                        table.append(dict(tag=tag, batch=b, basis="validation op (%s rule)" % rule, scenario=sc, policy=pol,
                                          p=o["p"], L_over_ck=o["L_over_ck"], non_inferior=o["non_inferior"], delta=o["delta"],
                                          total_bytes_ratio=o["total_bytes_ratio"], v65_Tstar_us=base_total, progressive_us=v["total"],
                                          predicted_e2e_gain=v["gain"], stage1_us=v["stages"]["stage1"], merge_us=v["stages"]["merge"],
                                          stage2_us=v["stages"]["stage2"], extra_us=v["stages"]["extra"],
                                          v65_scan_us=base["scan"], v65_merge_us=base["merge"]))
            for pol in POLS:
                cand = [r for (p_, _, _), r in mine.items() if p_ == pol and r["non_inferior"] == "True" and float(r["p"]) < 1.0]
                best = {}
                for r in cand:
                    pr = price(r)
                    for sc, v in pr.items():
                        if sc not in best or v["gain"] > best[sc]["gain"]:
                            best[sc] = dict(config="p=%s L=%sck" % (r["p"], r["L_over_ck"]), gain=v["gain"], total=v["total"], stages=v["stages"])
                res["envelope"][pol] = best
            out["workloads"][tag]["b%d" % b] = res
            show = {rule: {pol: (v["config"], v["non_inferior"], round(v["price"]["O"]["gain"], 3), round(v["price"]["C"]["gain"], 3))
                           for pol, v in res["ops"][rule].items()} for rule in res["ops"]}
            print(tag, "b=%d" % b, "v65(T*) %.1f us (measured at ck %.1f)" % (base_total, total))
            for rule, d in show.items():
                print("   ops/%s" % rule, d)
            print("   envelope(O)", {pol: (v["O"]["config"], round(v["O"]["gain"], 3)) for pol, v in res["envelope"].items() if v})
    json.dump(out, open(os.path.join(R, "cost_model.json"), "w"), indent=1, default=float)
    with open(os.path.join(R, "COST_MODEL_TABLE.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        wr.writeheader(); wr.writerows(table)


if __name__ == "__main__":
    main()
