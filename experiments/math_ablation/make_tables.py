#!/usr/bin/env python3
"""The study's tables from the results directory (local, no data access).

  python3 make_tables.py <results dir> <out dir>

ABLATION_RESULTS.csv     B0 / V1 / V2 / V3 per workload and split: recall at the paper's operating point (T = ck),
                         T_min at the matched recall, coverage, misranking, full-IVF recall, MSE, candidates
                         scanned (V2), the measured QPS where a GPU run exists, and what kind of evidence each is
RECALL_QPS_FRONTIER.csv  the measured v65 frontier (alpha x nprobe x batch) and any GPU prototype's
STAGE_BREAKDOWN.csv      per-stage GPU time (nsys) of the measured configurations
INDEX_MEMORY.csv         bytes per vector, global metadata, per-query LUT, build time, GPU memory
"""
import csv
import glob
import json
import os
import sys

TAGS = {"arxiv-768_np128_a100": ("arxiv-768", 768, 96, 2253000), "vogue-768_np128_a64": ("vogue-768", 768, 96, 932328),
        "openai3-3072_np128_a4": ("openai3-3072", 3072, 384, 999000)}
PAPER_ALPHA = {"arxiv-768": 100, "vogue-768": 64, "openai3-3072": 4}


def rows_of(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def write(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(os.path.basename(path), len(rows), "rows")


def bench_rows(path, variant):
    j = jload(path)
    if not j:
        return []
    out = []
    for r in j["runs"]:
        for v in r["variants"]:
            q = sorted(v["qps"])
            lat = sorted(v["lat_med_ms"])
            b = j["base"]
            ds = "vogue-768" if "vogue" in b else "arxiv-768" if "arxiv" in b else "openai3-3072" if "openai3-3072" in b else b
            out.append(dict(variant=variant, dataset=ds,
                            nprobe=j["nprobe"], alpha=r["alpha"], ck=r["ck"], batch=r["batch"], G=v["G"], recall_at_10=v["recall"],
                            qps_median=q[len(q) // 2], qps_min=q[0], qps_max=q[-1], latency_median_ms=lat[len(lat) // 2],
                            latency_p95_ms=sorted(v["lat_p95_ms"])[len(v["lat_p95_ms"]) // 2], gpu_used_mib=v.get("used_mib"),
                            rounds=len(q), evidence="GPU measured"))
    return out


def e8_bench_rows(path):
    """demo_jhq_v68_e8_bench: runs [{batch, rep A|B, alpha, recall, qps[], lat_med_ms[]}], file <dataset>_np<nprobe>.json"""
    j = jload(path)
    if not j:
        return []
    ds, npb = os.path.basename(path)[:-5].rsplit("_np", 1)
    name = {"A": "V1 GPU A: paper index, full 256 table", "B": "V1 GPU B: E8, full 256 table"}
    out = []
    for r in j["runs"]:
        q = sorted(r["qps"]); lat = sorted(r["lat_med_ms"])
        out.append(dict(variant=name[r["rep"]], dataset=ds, nprobe=int(npb), alpha=r["alpha"], ck=int(round(10 * r["alpha"])),
                        batch=r["batch"], G=-1, recall_at_10=r["recall"], qps_median=q[len(q) // 2], qps_min=q[0], qps_max=q[-1],
                        latency_median_ms=lat[len(lat) // 2], latency_p95_ms=None, gpu_used_mib=None, rounds=len(q),
                        evidence="GPU measured"))
    return out


def main():
    R, O = sys.argv[1], sys.argv[2]
    os.makedirs(O, exist_ok=True)
    # frontier: B0 and any prototype
    fr = []
    for p in sorted(glob.glob(os.path.join(R, "b0", "*.json"))):
        fr += bench_rows(p, "B0 v65")
    for p in sorted(glob.glob(os.path.join(R, "v1_gpu", "*.json"))):
        fr += e8_bench_rows(p)
    write(os.path.join(O, "RECALL_QPS_FRONTIER.csv"), fr)
    # stage breakdown
    st = []
    for sub, var in (("b0_stages", "B0 v65"),):
        s = jload(os.path.join(R, sub, "stages_summary.json"))
        for x in s or []:
            st.append(dict(variant=var, **{k: v for k, v in x.items() if k != "stages_us"}, **{"us_" + k: v for k, v in x.get("stages_us", {}).items()}))
    write(os.path.join(O, "STAGE_BREAKDOWN.csv"), st)
    # ablation table
    ab = []
    e8 = rows_of(os.path.join(R, "v1_e8", "eval_e8.csv")) + rows_of(os.path.join(R, "v1_e8", "eval_e8_test.csv"))
    rd = rows_of(os.path.join(R, "v3_rd", "eval_rd.csv")) + rows_of(os.path.join(R, "v3_rd", "eval_rd_test.csv"))
    evt = rows_of(os.path.join(R, "v2_evt", "EVT_POLICIES.csv")) + rows_of(os.path.join(R, "v2_evt", "EVT_POLICIES_test.csv"))
    b0q = {}
    for r in fr:
        if r["variant"] == "B0 v65" and r["nprobe"] == 128 and r["alpha"] == PAPER_ALPHA.get(r["dataset"]):
            b0q[(r["dataset"], r["batch"])] = r["qps_median"]
    for tag, (ds, d, M, N) in TAGS.items():
        for r in e8:
            if r["tag"] != tag:
                continue
            var = {"JHQ": "B0 v65 (paper index)", "jhq_re": "B0 re-implemented (retrained residual)", "E8|JHQ-DH": "V1 E8 primary + paper D_H (isolation only)",
                   "E8": "V1 E8-JHQ (global scale)", "E8g": "V1 E8-JHQ (per-subspace scale)"}.get(r["method"], r["method"])
            ab.append(dict(variant=var, dataset=ds, split=r["split"], recall_at_ck=r["recall_at_ck"], T_min=r["T_min"], target=r["target_recall"],
                           coverage_at_ck=r["coverage_at_ck"], full_ivf_recall=r["full_ivf_recall"], misrank_random=r["misrank_random"],
                           frac_candidates_scanned=1.0, **({"qps_b8": b0q.get((ds, 8)), "qps_b1000": b0q.get((ds, 1000))} if r["method"] == "JHQ" else {}),
                           evidence="offline (CPU, FP64) on the paper index's candidates" + ("; QPS GPU measured" if r["method"] == "JHQ" else "")))
        for r in rd:
            if r["tag"] != tag or r["method"] == "JHQ":
                continue
            ab.append(dict(variant="V3 RD-JHQ " + r["method"][3:], dataset=ds, split=r["split"], recall_at_ck=r["recall_at_ck"], T_min=r["T_min"],
                           target=r["target_recall"], coverage_at_ck=r["coverage_at_ck"], full_ivf_recall=r["full_ivf_recall"],
                           misrank_random=r["misrank_random"], frac_candidates_scanned=1.0, evidence="offline (CPU, FP64)"))
        for r in evt:
            if r["tag"] != tag or r["policy"] in ("nprobe",):
                continue
            ab.append(dict(variant="V2 EVT-JHQ " + (r["policy"] if r["policy"] == "oracle" else "%s L1=%s eps=%s" % (r["policy"], r["L1"], r["eps"])),
                           dataset=ds, split=r["split"], recall_at_ck=r["recall"], frac_candidates_scanned=r["frac_candidates"], lists_scanned=r["lists"],
                           evidence="offline simulation (CPU) on the paper index's candidates"))
        for r in evt:
            if r["tag"] == tag and r["policy"] == "nprobe":
                ab.append(dict(variant="B0 nprobe cut to %s" % r["nprobe_eff"], dataset=ds, split=r["split"], recall_at_ck=r["recall"],
                               frac_candidates_scanned=r["frac_candidates"], lists_scanned=r["lists"], evidence="offline simulation (CPU)"))
    write(os.path.join(O, "ABLATION_RESULTS.csv"), ab)
    # index memory
    mem = []
    for tag, (ds, d, M, N) in TAGS.items():
        b8 = [r for r in fr if r["variant"] == "B0 v65" and r["dataset"] == ds and r["nprobe"] == 128 and r["batch"] == 1000]
        gpu = b8[0]["gpu_used_mib"] if b8 else None
        e8b = jload(os.path.join(R, "v1_e8", "build_%s.json" % tag)) or {}
        e8g = jload(os.path.join(R, "v1_e8", "build_%s_grouped.json" % tag)) or {}
        rdb = {a: jload(os.path.join(R, "v3_rd", "build_%s_%s.json" % (tag, a))) or {} for a in ("uniform", "waterfill", "empirical", "queryweighted")}
        mem.append(dict(variant="B0 v65", dataset=ds, primary_bytes_per_vector=d // 8, residual_bytes_per_vector=d, corr_bytes_per_vector=4,
                        global_metadata="levels +-c, residual codebooks M x 256", lut_bytes_per_query_primary=M * 32 * 4,
                        gpu_used_mib_b1000=gpu, build="paper index (cached)"))
        mem.append(dict(variant="V1 E8-JHQ", dataset=ds, primary_bytes_per_vector=d // 8, residual_bytes_per_vector=d, corr_bytes_per_vector=4,
                        global_metadata="E8 codebook 256 x 8, scale(s) 1 or M, residual codebooks M x 256", lut_bytes_per_query_primary=M * 256 * 4,
                        primary_mse_ratio_vs_jhq=e8b.get("primary_mse_ratio_e8_over_jhq"), primary_mse_ratio_grouped=e8g.get("primary_mse_ratio_e8_over_jhq"),
                        encode_seconds_cpu_python=e8b.get("encode_seconds_cpu"), build_seconds_total_cpu=e8b.get("seconds")))
        mem.append(dict(variant="V2 EVT-JHQ", dataset=ds, primary_bytes_per_vector=d // 8, residual_bytes_per_vector=d, corr_bytes_per_vector=4,
                        global_metadata="per-list size and centre (already in the index); risk model: a few floats", lut_bytes_per_query_primary=M * 32 * 4))
        for a, b in rdb.items():
            mem.append(dict(variant="V3 RD-JHQ " + a, dataset=ds, primary_bytes_per_vector=d // 8, residual_bytes_per_vector=b.get("bytes_per_vector", d),
                            corr_bytes_per_vector=4, global_metadata="residual codebooks M x 2^b_m (<= 1024)", bits_histogram=json.dumps(b.get("bits_histogram")),
                            hierarchical_mse_per_vector=b.get("hierarchical_mse_per_vector"), pack_seconds_cpu=b.get("pack_seconds_cpu"),
                            build_seconds_total_cpu=b.get("seconds")))
    write(os.path.join(O, "INDEX_MEMORY.csv"), mem)


if __name__ == "__main__":
    main()
