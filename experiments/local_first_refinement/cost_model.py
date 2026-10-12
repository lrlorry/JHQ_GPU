#!/usr/bin/env python3
"""GPU cost model for local-first refinement at matched recall (batch 8, G = 8).

Every term is labelled:
  [M] measured: nsys GPU time per search from the pipeline campaign -- v63 G8 at
      the workload alpha (= what v65 runs at b = 8), and v62 G8 at reduced
      alpha (experiments/pipeline_campaign/exp2), which gives the scan and merge
      time as a function of the per-group buffer ck
  [D] derived: a measured time scaled to another candidate count
  [T] theoretical: an assumption, stated
Nothing here ran as a new pipeline.

  python3 cost_model.py <results dir>   ->  cost_model.json, figures/fig4_cost_model.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(HERE, "..", "pipeline_campaign", "exp2")
WL = {"arxiv-768_np128_a100": ("arxiv-768", 100.0), "vogue-768_np128_a64": ("vogue-768", 64.0),
      "openai3-3072_np128_a4": ("openai3-3072", 4.0)}
PATHOLOGICAL = {("arxiv-768", 50.0)}     # ck = 500: 12 free buffer slots, a separate artefact (pipeline_campaign Exp 2)


def main():
    R = sys.argv[1]
    S = json.load(open(os.path.join(R, "summary.json")))
    v63 = json.load(open(os.path.join(CAMP, "v63_raw", "perf", "stages_summary.json")))
    red = json.load(open(os.path.join(CAMP, "raw", "nsys", "stages_summary.json")))
    out = {}
    for tag, (ds, alpha) in WL.items():
        ck = S[tag]["ck"]
        m = next(s for s in v63 if s["tag"] == ds and float(s["alpha"]) == alpha and s["batch"] == 8 and s["G"] == 108)
        u = m["stages_us"]
        front = u["rotation"] + u["centroid"] + u["probe_select"] + u["lut"]
        host = m["latency_ms"] * 1e3 - sum(v for v in u.values())
        # scan and merge against the per-group buffer, from the reduced-alpha run (v62 G8), relative to its own ck point
        pts = sorted((float(s["alpha"]) * 10, s["stages_us"]["scan"], s["stages_us"]["merge"]) for s in red
                     if s["tag"] == ds and s["batch"] == 8 and s["G"] == 8 and (ds, float(s["alpha"])) not in PATHOLOGICAL)
        cks = np.array([p[0] for p in pts]); sc = np.array([p[1] for p in pts]); mg = np.array([p[2] for p in pts])
        sc0 = np.interp(np.log(ck), np.log(cks), sc); mg0 = np.interp(np.log(ck), np.log(cks), mg)
        scan_at = lambda c: u["scan"] * np.interp(np.log(max(c, 1)), np.log(cks), sc) / sc0      # [D]
        merge_at = lambda c: u["merge"] * np.interp(np.log(max(c, 1)), np.log(cks), mg) / mg0    # [D]
        per_cand_refine = u["refine"] / ck          # one block per query: time per candidate it walks   [M -> D]
        per_cand_topk = u["topk"] / ck
        matched = S[tag]["matched"]
        rows = {}
        def add(name, T, per_group_ck, merge, refine_block_cands):
            t1 = front + scan_at(per_group_ck) + merge + per_cand_refine * T + per_cand_topk * T + host
            t2 = front + scan_at(per_group_ck) + merge + per_cand_refine * refine_block_cands + per_cand_topk * T + host
            rows[name] = dict(T=T, per_group_buffer=per_group_ck, merge_us=merge, est_us_one_block_refine=t1,
                              est_us_group_parallel_refine=t2, speedup_vs_v65_measured=m["latency_ms"] * 1e3 / t1,
                              speedup_vs_v65_parallel=m["latency_ms"] * 1e3 / t2)
        rows["v65 (measured)"] = dict(T=ck, measured_us=m["latency_ms"] * 1e3, front_us=front, scan_us=u["scan"], merge_us=u["merge"],
                                      refine_us=u["refine"], topk_us=u["topk"], host_us=host)
        g = matched.get("global/G8")
        if g:
            T = g["T"]
            add("global top-T* (alpha lowered)", T, T, merge_at(T), T / 8)
        for pol in ("probe", "calib"):
            g = matched.get("%s/G8" % pol)
            if not g:
                rows["local %s" % pol] = dict(note="does not reach v65 recall within T <= 1.5 ck at G = 8")
                continue
            T = g["T"]
            if pol == "calib":
                b = np.array(S[tag]["calib_tables"]["8"][str(T)], dtype=float)
            else:
                w = 1.0 / (np.arange(8) + 1); b = np.floor(T * w / w.sum())
            add("local %s (T*)" % pol, T, float(b.max()), 0.0, float(b.max()))
        out[tag] = dict(ck=ck, rows=rows,
                        bounds=dict(merge_share=u["merge"] / (m["latency_ms"] * 1e3),
                                    full_overlap_share=min(u["scan"], u["refine"]) / (m["latency_ms"] * 1e3),
                                    refine_share=u["refine"] / (m["latency_ms"] * 1e3)))
        print(tag)
        for k2, v in rows.items():
            print("   %-32s %s" % (k2, {a: (round(b2, 1) if isinstance(b2, float) else b2) for a, b2 in v.items()}))
    json.dump(out, open(os.path.join(R, "cost_model.json"), "w"), indent=1, default=float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (tag, D) in zip(axes, out.items()):
        names, v1, v2 = [], [], []
        for k2, v in D["rows"].items():
            if "measured_us" in v:
                names.append("v65\n(measured)"); v1.append(v["measured_us"]); v2.append(np.nan)
            elif "est_us_one_block_refine" in v:
                names.append(k2.replace(" (", "\n(").replace("top-T* ", "top-T*\n")); v1.append(v["est_us_one_block_refine"]); v2.append(v["est_us_group_parallel_refine"])
            else:
                names.append(k2 + "\n(no T reaches\nv65 recall)"); v1.append(np.nan); v2.append(np.nan)
        x = np.arange(len(names))
        a.bar(x - 0.2, v1, 0.38, color="#7570b3", label="refine as today: one block per query [D]")
        a.bar(x + 0.2, v2, 0.38, color="#1b9e77", label="refine split over G blocks, both sides [T]")
        a.set_xticks(x); a.set_xticklabels(names, fontsize=7)
        a.set_ylabel("µs per batch of 8 (matched recall)"); a.set_title(tag.replace("_np", " np").replace("_a", " α="), fontsize=10)
        if tag.startswith("arxiv"):
            a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig4_cost_model.png"), dpi=110)


if __name__ == "__main__":
    main()
