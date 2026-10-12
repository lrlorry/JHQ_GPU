#!/usr/bin/env python3
"""End-to-end cost model of a bound-screened pipeline, against v65 as measured.

  T_new = T_front + T_bound_LUT + T_bound_scan + T_seed + T_survivor_refine
          + T_final_selection + T_coordination + T_host

Every term is labelled by where it comes from:
  [M]  measured: nsys GPU time per search from the pipeline campaign
       (experiments/pipeline_campaign/exp2/v63_raw/perf, exp2/raw/nsys)
  [D]  derived: a measured rate times this pilot's measured survivor counts
  [T]  theoretical: an assumption, stated
Nothing here was run as a new pipeline. Run from experiments/residual_aware_bound:
  python3 cost_model.py   ->  cost_model.json, figures/fig7_cost_model.png
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(HERE, "..", "pipeline_campaign")
BW = 1.79e12          # RTX 5090 DRAM bandwidth, bytes/s (spec)
SEEDS = 64
WL = {"arxiv-768": (100, 12.5, 768, "arxiv-768_np128_a100"), "vogue-768": (64, 8, 768, "vogue-768_np128_a64"),
      "openai3-3072": (4, 1, 3072, "openai3-3072_np128_a4")}


def main():
    st = json.load(open(os.path.join(CAMP, "exp2", "v63_raw", "perf", "stages_summary.json")))
    red = json.load(open(os.path.join(CAMP, "exp2", "raw", "nsys", "stages_summary.json")))
    surv = json.load(open(os.path.join(HERE, "survivor_summary.json")))
    find = lambda L, tag, a, b, G: next(s for s in L if s["tag"] == tag and float(s["alpha"]) == a and s["batch"] == b and s["G"] == G)
    out = {}
    for ds, (alpha, a_small, d, stag) in WL.items():
        out[ds] = {}
        for b, G in ((8, 108), (1000, 0)):
            v = find(st, ds, alpha, b, G)
            u = v["stages_us"]
            ck = surv[stag]["full_oracle/cond"]["median"] / surv[stag]["full_oracle/cond"]["over_ck_median"]
            n_s = surv[stag]["prog_probe_L8_s64/cond"]["median"]           # refined per query, seeded progressive
            n_c = surv[stag]["_n_cand"]["median"]
            front = u["rotation"] + u["centroid"] + u["probe_select"]                    # [M]
            lut = u["lut"]                                                                # [M] same 16+16 shape
            scan = find(red, ds, a_small, b, 8 if b == 8 else 0)["stages_us"]["scan"]   # [M] proxy: smallest ck
            # refinement rate: the batch-1000 kernel, the most efficient measured, per candidate
            r1000 = find(st, ds, alpha, 1000, 0)["stages_us"]["refine"] / (1000.0 * ck)  # [M] us per candidate
            refine = b * n_s * r1000                                                      # [D]
            refine_floor = b * n_s * d / BW * 1e6                                         # [T] residual bytes / bandwidth
            seed = b * SEEDS * u["refine"] / (b * ck)                                     # [D] at this batch's rate
            topk_scaled = u["topk"] * n_s / ck                                            # [D] one block per query, linear
            coord = 5.0                                                                   # [T] two more graph nodes + compaction
            host = v["host_gaps_us"]                                                      # [M]
            t_v65 = v["latency_ms"] * 1e3                                                 # [M]
            opt = front + lut + scan + seed + refine + 0.0 + coord + host
            pes = front + lut + scan + seed + refine + topk_scaled + coord + host
            out[ds][b] = dict(ck=ck, n_cand_median=n_c, refined_median=n_s, v65_us=t_v65,
                              terms_us=dict(front_M=front, bound_lut_M=lut, bound_scan_Mproxy=scan, seed_D=seed,
                                            survivor_refine_D=refine, survivor_refine_floor_T=refine_floor,
                                            final_selection_D=topk_scaled, coordination_T=coord, host_M=host),
                              T_new_optimistic_us=opt, T_new_measured_rates_us=pes,
                              speedup_optimistic=t_v65 / opt, speedup_measured_rates=t_v65 / pes)
            print("%-13s b=%-4d ck=%5.0f refined=%7.0f | v65 %8.1f us | new %8.1f .. %9.1f us | x%.2f .. x%.2f" % (
                ds, b, ck, n_s, t_v65, opt, pes, t_v65 / opt, t_v65 / pes))
    json.dump(out, open(os.path.join(HERE, "cost_model.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, b in zip(axes, (8, 1000)):
        names = list(out)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [out[n][b]["v65_us"] for n in names], 0.38, color="#1b9e77", label="v65, measured")
        bottoms = np.zeros(len(names))
        parts = (("front_M", "front end [M]"), ("bound_scan_Mproxy", "bound scan [M, proxy]"),
                 ("survivor_refine_D", "refine survivors [D]"), ("final_selection_D", "top-k over survivors [D]"),
                 ("host_M", "host [M]"))
        cols = ("#bbbbbb", "#7570b3", "#d95f02", "#e7298a", "#666666")
        for (key, lab), c in zip(parts, cols):
            h = np.array([out[n][b]["terms_us"][key] for n in names])
            ax.bar(x + 0.2, h, 0.38, bottom=bottoms, color=c, label=lab)
            bottoms += h
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_yscale("log")
        ax.set_ylabel("µs per batch"); ax.set_title("batch %d: v65 (left) vs bound-screened (right)" % b, fontsize=10)
        if b == 8:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig7_cost_model.png"), dpi=110)


if __name__ == "__main__":
    main()
