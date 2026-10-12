#!/usr/bin/env python3
"""Bound-aware residual quantization pilot: tables, IO model and figures.

  python3 analyze.py <results dir with raw/<tag>_{rows.csv,ratios.bin,summary.json}>

Writes BOUND_TIGHTNESS.csv, SURVIVOR_VS_BITS.csv, io_model.json and
figures/fig{A,B,C}_*.png into the results dir.

Bytes per query (what the scan and the refinement must read):
  v65  n_cand * M  (every candidate's primary code)  +  ck * (d + 4)  (residual codes
       and the correction of the alpha*k it refines)
  new  n_cand * (M + extra)  (primary code + the bound's extra bits, packed)  +
       refined * (d + 4)
extra is the bound's information per vector: family A/Ac p bits per dimension
(d*p/8 bytes; a bitplane sidecar, or the same bits re-laid out as bitplanes),
family B b bits per 4-d half-subspace (d*b/32 bytes). The break-even extra,
where even refining nothing at all only ties v65, is d*ck/n_cand bytes.
Time (derived, [D]): v65's measured scan time scaled by (M + extra)/M -- the
scan reads that many bytes per candidate -- plus its measured refinement time
scaled by refined/ck; the measured front end and top-k unchanged. That is an
assumption about the scan, stated as one; nothing new was run on the GPU.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(HERE, "..", "pipeline_campaign", "exp2", "v63_raw", "perf", "stages_summary.json")
TIGHT = os.path.join(HERE, "..", "residual_aware_bound", "tightness.json")
TAGS = {"arxiv-768_np128_a100": ("arxiv-768", 100), "vogue-768_np128_a64": ("vogue-768", 64),
        "openai3-3072_np128_a4": ("openai3-3072", 4)}


def q(x, f):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, f)) if len(x) else float("nan")


def main():
    R = sys.argv[1]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    st = json.load(open(STAGES))
    tight = json.load(open(TIGHT))
    T_rows, S_rows, IO = [], [], {}
    data = {}
    for tag, (ds, alpha) in TAGS.items():
        rows = list(csv.DictReader(open(os.path.join(R, "raw", tag + "_rows.csv"))))
        Sm = json.load(open(os.path.join(R, "raw", tag + "_summary.json")))
        b = open(os.path.join(R, "raw", tag + "_ratios.bin"), "rb").read()
        nS, NB, NSAMP = np.frombuffer(b[:12], dtype=np.int32)
        ratios = np.frombuffer(b[12:], dtype=np.float32).reshape(nS, NB, NSAMP)
        d, M, ck = Sm["d"], Sm["M"], Sm["ck"]
        bounds = Sm["bounds"]
        v8 = next(s for s in st if s["tag"] == ds and float(s["alpha"]) == alpha and s["batch"] == 8 and s["G"] == 108)
        v1k = next(s for s in st if s["tag"] == ds and float(s["alpha"]) == alpha and s["batch"] == 1000 and s["G"] == 0)
        data[tag] = dict(d=d, M=M, ck=ck, bounds=[], v8=v8, v1k=v1k)
        for bi, B in enumerate(bounds):
            name = B["name"]
            R_b = [r for r in rows if r["bound"] == name]
            nc = np.array([float(r["n_cand"]) for r in R_b])
            so = np.array([float(r["survivors_oracle"]) for r in R_b])
            s8 = np.array([float(r["survivors_seed_L8_s64"]) for r in R_b])
            s1 = np.array([float(r["survivors_seed_L1_s32"]) for r in R_b])
            pr = np.array([float(r["prog_refined_L8_s64"]) for r in R_b])
            rr = ratios[:, bi, :].ravel()
            extra_bytes = B["extra_bits"] / 8.0
            # bytes per query, median query
            v65_bytes = nc * M + ck * (d + 4)
            new_bytes = nc * (M + extra_bytes) + pr * (d + 4)
            # derived time: scan scaled by bytes per candidate, refine by refined/ck
            def t_new(v):
                u = v["stages_us"]
                rest = v["latency_ms"] * 1e3 - u["scan"] - u["merge"] - u["refine"]
                return rest + (u["scan"] + u["merge"]) * (M + extra_bytes) / M + u["refine"] * float(np.median(pr)) / ck
            tr = dict(dataset=ds, bound=name, family=B["family"], param=B["param"], extra_bits_per_vector=B["extra_bits"],
                      extra_bytes_per_vector=extra_bytes, extra_bits_per_dim=B["extra_bits"] / d,
                      LB_over_DH_p10=q(rr, 0.1), LB_over_DH_p50=q(rr, 0.5), LB_over_DH_p90=q(rr, 0.9), LB_over_DH_p99=q(rr, 0.99),
                      violations=B["violations"], max_LB_minus_DH=B["max_excess"])
            T_rows.append(tr)
            sr = dict(dataset=ds, bound=name, family=B["family"], param=B["param"], extra_bits_per_vector=B["extra_bits"],
                      extra_bytes_per_vector=extra_bytes, ck=ck, n_cand_median=float(np.median(nc)),
                      oracle_survivors_median=float(np.median(so)), oracle_survivors_p95=q(so, 0.95), oracle_survivors_max=float(so.max()),
                      oracle_survivors_over_ck_median=float(np.median(so / ck)), oracle_prune_rate_median=float(np.median(1 - so / nc)),
                      seed_L8_s64_survivors_median=float(np.median(s8)), seed_L8_s64_survivors_p95=q(s8, 0.95),
                      seed_L1_s32_survivors_median=float(np.median(s1)),
                      progressive_refined_median=float(np.median(pr)), progressive_refined_p95=q(pr, 0.95), progressive_refined_max=float(pr.max()),
                      progressive_refined_over_ck_median=float(np.median(pr / ck)),
                      queries_with_progressive_at_or_below_ck=float(np.mean(pr <= ck)),
                      bytes_per_query_new_median=float(np.median(new_bytes)), bytes_per_query_v65_median=float(np.median(v65_bytes)),
                      bytes_ratio_new_over_v65_median=float(np.median(new_bytes / v65_bytes)),
                      breakeven_extra_bytes=float(np.median(d * ck / nc)),
                      derived_us_b8_new=t_new(v8), measured_us_b8_v65=v8["latency_ms"] * 1e3,
                      derived_us_b1000_new=t_new(v1k), measured_us_b1000_v65=v1k["latency_ms"] * 1e3)
            S_rows.append(sr)
            data[tag]["bounds"].append(dict(tr, **sr))
        IO[tag] = dict(d=d, M=M, ck=ck, v65_us_b8=v8["latency_ms"] * 1e3, v65_us_b1000=v1k["latency_ms"] * 1e3,
                       breakeven_extra_bytes=float(np.median([r["breakeven_extra_bytes"] for r in S_rows if r["dataset"] == ds])),
                       required_tightness=tight[tag]["required_tightness"] if tag in tight else None)
    for fn, rows in (("BOUND_TIGHTNESS.csv", T_rows), ("SURVIVOR_VS_BITS.csv", S_rows)):
        with open(os.path.join(R, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    json.dump(IO, open(os.path.join(R, "io_model.json"), "w"), indent=1)
    figures(R, data, IO)
    print("bounds per dataset:", {t: len(v["bounds"]) for t, v in data.items()})


def figures(R, data, IO):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = os.path.join(R, "figures")
    fam_style = {"A": ("prefix p (A)", "o-", "#1b9e77"), "Ac": ("prefix p ∩ half-code envelope (Ac)", "s--", "#d95f02"),
                 "B_max": ("coarse b, max(box, ball) (B)", "^-", "#7570b3"), "B_maxcond": ("coarse b, max(box ∩ env, ball) (B)", "v--", "#e7298a")}

    def series(bl):
        out = {}
        for x in bl:
            key = x["family"] if x["family"] != "B" else ("B_" + x["bound"].split("_")[-1])
            if key not in fam_style:
                continue
            out.setdefault(key, []).append(x)
        for k in out:
            out[k].sort(key=lambda x: x["extra_bits_per_dim"])
        return out
    # A: tightness against bits
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (tag, D) in zip(axes, data.items()):
        for key, xs in series(D["bounds"]).items():
            lab, sty, col = fam_style[key]
            xx = [x["extra_bits_per_dim"] for x in xs]
            md = np.array([x["LB_over_DH_p50"] for x in xs])
            lo = md - np.array([x["LB_over_DH_p10"] for x in xs]); hi = np.array([x["LB_over_DH_p90"] for x in xs]) - md
            ax.errorbar(xx, md, yerr=[lo, hi], fmt=sty, color=col, ms=4, lw=1.2, capsize=2, label=lab)
        rt = IO[tag]["required_tightness"]
        if rt:
            for kf, ls in (("prune_50%", ":"), ("prune_90%", "--"), ("prune_99%", "-")):
                ax.axhline(rt[kf], color="r", ls=ls, lw=0.9, label="needed to prune %s (oracle τ)" % kf.split("_")[1])
        ax.set_xscale("symlog", linthresh=0.5); ax.set_xticks([0, 0.5, 1, 2, 4, 6, 8]); ax.set_xticklabels(["0", "0.5", "1", "2", "4", "6", "8"])
        ax.set_xlabel("extra bits per dimension read by the bound"); ax.set_ylabel("LB / D_H (P10–P50–P90)")
        ax.set_ylim(0, 1.02); ax.set_title(tag.replace("_np", " np").replace("_a", " α="), fontsize=10)
        if tag.startswith("arxiv"):
            ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "figA_bound_tightness.png"), dpi=110); plt.close(fig)
    # B: survivors against bits
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (tag, D) in zip(axes, data.items()):
        for key, xs in series(D["bounds"]).items():
            lab, sty, col = fam_style[key]
            xx = [x["extra_bits_per_dim"] for x in xs]
            ax.plot(xx, [x["oracle_survivors_over_ck_median"] for x in xs], sty, color=col, ms=4, lw=1.2, label=lab + ", oracle τ")
            ax.plot(xx, [x["progressive_refined_over_ck_median"] for x in xs], sty.replace("-", ":").replace("::", ":"), color=col, ms=3, lw=0.9, alpha=0.7,
                    label=lab + ", progressive (seeds)")
        ax.axhline(1.0, color="r", lw=1, label="= ck (what v65 refines)")
        ax.set_xscale("symlog", linthresh=0.5); ax.set_xticks([0, 0.5, 1, 2, 4, 6, 8]); ax.set_xticklabels(["0", "0.5", "1", "2", "4", "6", "8"])
        ax.set_yscale("log"); ax.set_xlabel("extra bits per dimension read by the bound"); ax.set_ylabel("candidates to refine / ck (median query)")
        ax.set_title(tag.replace("_np", " np").replace("_a", " α="), fontsize=10)
        if tag.startswith("arxiv"):
            ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "figB_pruning_vs_bits.png"), dpi=110); plt.close(fig)
    # C: bytes and derived time against bits
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for col_i, (tag, D) in enumerate(data.items()):
        ax, ax2 = axes[0][col_i], axes[1][col_i]
        for key, xs in series(D["bounds"]).items():
            lab, sty, col = fam_style[key]
            xx = [x["extra_bits_per_dim"] for x in xs]
            ax.plot(xx, [x["bytes_per_query_new_median"] / 1e6 for x in xs], sty, color=col, ms=4, lw=1.2, label=lab)
            ax2.plot(xx, [x["derived_us_b8_new"] for x in xs], sty, color=col, ms=4, lw=1.2, label=lab + ", b=8")
            ax2.plot(xx, [x["derived_us_b1000_new"] / 10 for x in xs], sty.replace("-", ":"), color=col, ms=3, lw=0.9, alpha=0.6)
        v65b = D["bounds"][0]["bytes_per_query_v65_median"] / 1e6
        ax.axhline(v65b, color="k", lw=1.2, label="v65 (primary scan + αk refinement)")
        be = IO[tag]["breakeven_extra_bytes"] * 8 / D["d"]
        ax.axvline(be, color="r", ls="--", lw=1, label="break-even extra = d·ck/n_cand (%.2f bits/dim)" % be)
        ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("log")
        ax.set_xlabel("extra bits per dimension"); ax.set_ylabel("MB read per query (median)")
        ax.set_title(tag.replace("_np", " np").replace("_a", " α=") + ": bytes", fontsize=10)
        ax2.axhline(IO[tag]["v65_us_b8"], color="k", lw=1.2, label="v65 measured, b=8")
        ax2.axhline(IO[tag]["v65_us_b1000"] / 10, color="k", ls=":", lw=1, label="v65 measured, b=1000 (÷10)")
        ax2.set_xscale("symlog", linthresh=0.5); ax2.set_yscale("log")
        ax2.set_xlabel("extra bits per dimension"); ax2.set_ylabel("µs per batch [derived]")
        ax2.set_title("derived time (solid b=8, dotted b=1000 ÷10)", fontsize=9)
        if col_i == 0:
            ax.legend(fontsize=6.5); ax2.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(F, "figC_gpu_cost.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
