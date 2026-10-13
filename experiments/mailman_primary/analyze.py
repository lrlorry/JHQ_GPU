#!/usr/bin/env python3
"""Tables for experiments/mailman_primary from run_box.sh's raw outputs.

    python3 analyze.py <results dir> [--md out.md]

<results dir> holds bench_<tag>.json (timing, correctness), bench_<tag>_profile.json (per-phase clock64
cycles of the Mailman kernels), stats_<tag>.json (structure of the codes), atomics_sass.txt.
The end-to-end share of scoring comes from v70's nsys stage times
(experiments/practical_opt/v70_results/stages_v70/stages_summary.json, G = -1, same indexes): the share
f of the search's latency that B0's scoring at the scan's own grid takes, and the model
S = 1 / ((1 - f) + f / s) for a scorer s times as fast as B0 (eta = 0, the most favourable case).
"""
import json
import os
import sys

TAGS = [("vogue-768_np128_a64", "vogue-768"), ("arxiv-768_np128_a100", "arxiv-768"),
        ("openai3-3072_np128_a4", "openai3-3072")]
MAILMAN = ["H1_r8", "H1z_r8", "H1_r4", "H2_r8", "H2_r6", "H2_r4"]


def main():
    d = sys.argv[1]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stages = json.load(open(os.path.join(root, "experiments/practical_opt/v70_results/stages_v70/stages_summary.json")))
    st = {(r["tag"], r["batch"]): r for r in stages if r["G"] == -1}
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    out = {}
    emit("## Isolated scoring (median us per launch; B0/impl > 1 = faster than B0)")
    for tag, ds in TAGS:
        p = os.path.join(d, f"bench_{tag}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        out[ds] = {"layout_bytes": j["layout_bytes"], "build_s": j["build_s"]}
        for run in j["runs"]:
            b = run["batch"]
            imps = {x["impl"]: x for x in run["impls"]}
            b0 = imps["B0"]["median_us"]
            emit(f"\n### {ds}, b = {b} ({run['candidates']} candidates a pass, {run['batches_timed']} batches)")
            emit("| impl | median us | B0/impl | max abs err vs FP64 | p99 | top-T != FP64 (queries) | != B0 |")
            emit("|---|---:|---:|---:|---:|---:|---:|")
            for name in ["B0"] + MAILMAN:
                x = imps.get(name)
                if not x:
                    continue
                emit(f"| {name} | {x['median_us']:.1f} | {b0 / x['median_us']:.3f} | {x['err_fp64_max']:.2e} | "
                     f"{x['err_fp64_p99']:.2e} | {x['topT_diff_fp64_queries']} | {x['topT_diff_b0_queries']} |")
            sg = imps["B0_scangrid"]["median_us"]
            best = max(MAILMAN, key=lambda n: b0 / imps[n]["median_us"] if n in imps else 0)
            s_best = b0 / imps[best]["median_us"]
            emit(f"| B0 at the scan's grid | {sg:.1f} | | | | | |")
            emit(f"| query preparation: B0 table / Mailman C0 | {imps['B0_prep_lut']['median_us']:.1f} / "
                 f"{imps['MM_prep_C0']['median_us']:.1f} | | | | | |")
            r = st.get((ds, b))
            if r:
                lat = r["latency_ms"] * 1e3
                scan = r["stages_us"]["scan"]
                f = sg / lat
                S = 1.0 / ((1 - f) + f / s_best)
                emit(f"\nscan stage (v70 nsys) {scan:.0f} us of {lat:.0f} us latency; B0's scoring at the scan's grid "
                     f"is {sg / scan * 100:.0f}% of the scan and f = {f * 100:.0f}% of the latency. Best Mailman here: "
                     f"{best} at s = {s_best:.3f} -> modelled end-to-end x{S:.3f}")
                out[ds][b] = {"b0_us": b0, "b0_scangrid_us": sg, "scan_us": scan, "latency_us": lat, "f": f,
                              "best": best, "s_best": s_best, "S_model": S,
                              "impls": {n: imps[n]["median_us"] for n in imps}}
    emit("\n## Where the Mailman kernels spend their cycles (clock64 per warp, MB_PROFILE=1)")
    emit("| dataset | b | impl | zero | accumulate | recover | write |")
    emit("|---|---:|---|---:|---:|---:|---:|")
    for tag, ds in TAGS:
        p = os.path.join(d, f"bench_{tag}_profile.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for run in j["runs"]:
            for x in run["impls"]:
                ph = x.get("phase_cycles")
                if not ph or sum(ph) == 0:
                    continue
                t = sum(ph)
                emit(f"| {ds} | {run['batch']} | {x['impl']} | {ph[0] / t * 100:.1f}% | {ph[1] / t * 100:.1f}% | "
                     f"{ph[2] / t * 100:.1f}% | {ph[3] / t * 100:.1f}% |")
    emit("\n## Structure of the codes (top 512 lists by scan reads)")
    for tag, ds in TAGS:
        p = os.path.join(d, f"stats_{tag}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        bb = s["bit_balance"]
        emit(f"\n### {ds}: {s['lists_analysed']} lists, {s['share_of_scan_reads'] * 100:.1f}% of the scan's reads; "
             f"|p1 - 1/2| per (list, dim) mean {bb['mean_abs_dev_from_half']:.3f}, share > 0.25 {bb['share_dev_gt_0.25'] * 100:.0f}%")
        for r in ("r4", "r8"):
            m = s["mailman"][r]
            emit(f"- Mailman {r}: pattern entropy {m['pattern_entropy_bits']:.2f}/{m['max_entropy_bits']} bits, "
                 f"all-zero {m['share_all_zero'] * 100:.1f}% all-one {m['share_all_one'] * 100:.1f}%; a warp over 32 "
                 f"dims: {m['warp32']['distinct_mean']:.1f} distinct patterns, largest multiplicity "
                 f"{m['warp32']['max_multiplicity_mean']:.1f} (random bits {m['warp32']['random_bits_max_multiplicity_mean']:.1f})")
        best = min(s["dictionary"].items(), key=lambda kv: kv[1]["implementable_bytes_ratio_with_fallback"])
        bestw = min(s["dictionary"].items(), key=lambda kv: kv[1]["work_ratio_scan_weighted"])
        emit(f"- dictionary: best bytes {best[0]} -> {best[1]['implementable_bytes_ratio_with_fallback']:.3f} of raw "
             f"({best[1]['share_blocks_worth_dictionary_scan_weighted'] * 100:.0f}% of blocks use it); best table work "
             f"{bestw[0]} -> {bestw[1]['work_ratio_scan_weighted']:.3f}")
        out.setdefault(ds, {})["stats"] = {"bit_balance": bb, "dict_best_bytes": [best[0], best[1]],
                                           "dict_best_work": [bestw[0], bestw[1]]}
    p = os.path.join(d, "atomics_sass.txt")
    if os.path.exists(p):
        emit("\n## Shared FP32 atomicAdd in SASS\n```\n" + open(p).read().strip() + "\n```")
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    if md:
        open(md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
