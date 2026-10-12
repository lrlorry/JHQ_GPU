#!/usr/bin/env python3
"""Per-stage GPU time of every search, from an nsys GPU trace, joined with the
harness's end-to-end latency. Used for Phase 0 (v61) and Phase 3 (v62).

Inputs in <dir>, per configuration tag <t>:
  <t>.json                 demo_jhq_{v61,v62}_batch, plain run (no profiler):
                           per (alpha, b, G) rounds of latency median / p95, QPS
  <t>_trace*.csv           nsys stats -r cuda_gpu_trace:nvtx-name of the same
                           harness run under nsys (one round): every kernel and
                           memcpy with its duration and enclosing NVTX range
                           "a<alpha>|b<b>|G<G>|r<round>"

Each NVTX range holds back-to-back searches (nsys writes the range in front of
the kernel name, "a64|b8|G-1|r0/<kernel>"). They are split in time order: a
search starts at its first kernel after the previous search's final top-k; an
H2D copy is charged to the search that follows it and a D2H copy to the one
before. Within a search the kernels before select_probes are cuBLAS GEMMs: the
first is the query rotation, any others the IVF centroid scoring.

Per (alpha, b, G): median over searches of each stage's GPU time, the number of
kernels a search launches, and -- from the plain run -- the median batch
latency. "host+gaps" = latency - sum of GPU stage times: launch, copy setup,
synchronisation and any idle time between kernels.

Phase 0 bound: if the primary scan took no time at all, latency would fall by
scan; if it were split over 8 groups with perfect scaling and no merge, by
7/8 of scan. Both are printed as end-to-end speedups.

  python3 stage_times.py <dir> [--md out.md]
"""
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

STAGES = ["h2d", "rotation", "centroid", "probe_select", "lut", "scan", "merge", "refine", "topk", "d2h"]


def stage_of(name):
    n = name
    if "memcpy" in n.lower() or "memset" in n.lower():
        if "HtoD" in n or "Host-to-Device" in n:
            return "h2d"
        if "DtoH" in n or "Device-to-Host" in n:
            return "d2h"
        return "copy_other"
    if "select_probes" in n:
        return "probe_select"
    if "build_byte_lut" in n:
        return "lut"
    if "merge_group_topk" in n:
        return "merge"
    if "scan_ivf" in n:
        return "scan"
    if "residual_refine" in n:
        return "refine"
    if "batched_topk_final" in n:
        return "topk"
    return "gemm"


def read_trace(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return []
    cols = rows[0].keys()
    kstart = next(c for c in cols if c.startswith("Start"))
    kdur = next(c for c in cols if c.startswith("Duration"))
    kname = next(c for c in cols if c == "Name" or c.endswith("Name") and "NVTX" not in c)
    knvtx = next((c for c in cols if "NVTX" in c), None)
    out = []
    for r in rows:
        name = r[kname]
        rng = (r.get(knvtx) or "") if knvtx else ""
        # nsys 2025 puts the enclosing NVTX range in front of the name,
        # "a64|b8|G-1|r0/void cutlass::...", rather than in a column of its own.
        m = re.match(r"^(a[\d.]+\|b\d+\|G-?\d+\|r\d+)/(.*)$", name)
        if m:
            rng, name = m.group(1), m.group(2)
        out.append((float(r[kstart]), float(r[kdur]), name, rng))
    out.sort()
    return out


def searches_by_range(trace):
    """range -> list of searches, each a dict stage -> ns, plus kernel count."""
    by = defaultdict(list)
    for t, dur, name, rng in trace:
        m = re.search(r"a([\d.]+)\|b(\d+)\|G(-?\d+)\|r(\d+)", rng)
        if m:
            by[(float(m.group(1)), int(m.group(2)), int(m.group(3)))].append((t, dur, name))
    res = {}
    for key, rows in by.items():
        S, cur, pending_h2d, after_topk = [], None, 0.0, True
        for t, dur, name in rows:
            st = stage_of(name)
            if st == "h2d":
                pending_h2d += dur
                continue
            if st == "d2h" or st == "copy_other":
                if cur is not None:
                    cur["d2h"] += dur
                continue
            if after_topk:
                cur = defaultdict(float)
                cur["h2d"] += pending_h2d
                pending_h2d = 0.0
                cur["_kernels"] = 0
                cur["_gemms"] = 0
                S.append(cur)
                after_topk = False
            cur["_kernels"] += 1
            if st == "gemm":
                # One cublasSgemm can launch a GEMM kernel and a split-K reduce;
                # the reduce belongs to the call before it. The first call is the
                # query rotation, the second the IVF centroid scoring.
                if "reduce" not in name.lower() or cur["_gemms"] == 0:
                    cur["_gemms"] += 1
                cur["rotation" if cur["_gemms"] == 1 else "centroid"] += dur
            else:
                cur[st] += dur
            if st == "topk":
                after_topk = True
        res[key] = S
    return res


def main():
    d = sys.argv[1]
    md_out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    lines = []
    summary = []
    for jf in sorted(glob.glob(os.path.join(d, "*.json"))):
        tag = os.path.basename(jf)[:-5]
        if tag.endswith("_stages"):
            continue
        J = json.load(open(jf))
        if not isinstance(J, dict) or "runs" not in J:
            continue
        traces = sorted(glob.glob(os.path.join(d, tag + "_trace*cuda_gpu_trace*.csv")))
        T = searches_by_range(read_trace(traces[0])) if traces else {}
        lines += ["", "### %s (np=%d, %s, %d SMs)" % (tag, J["nprobe"], J["gpu"], J["sms"]), "",
                  "GPU time per search, median us. latency: plain run, median over rounds of the "
                  "per-round median batch latency.", "",
                  "| alpha | b | G | latency ms | QPS | " + " | ".join(STAGES) +
                  " | GPU sum | host+gaps | kernels | scan % of latency | e2e if scan=0 | e2e if scan/8 |",
                  "|" + "---:|" * (len(STAGES) + 11)]
        for R in J["runs"]:
            for V in R["variants"]:
                key = (float(R["alpha"]), R["batch"], V["G"])
                lat = statistics.median(V["lat_med_ms"])
                qps = statistics.median(V["qps"])
                S = T.get(key, [])
                med = {s: (statistics.median([x[s] for x in S]) / 1e3 if S else float("nan")) for s in STAGES}
                kern = statistics.median([x["_kernels"] for x in S]) if S else float("nan")
                gpu = sum(v for v in med.values() if v == v)
                host = lat * 1e3 - gpu
                scan = med["scan"] if med["scan"] == med["scan"] else 0.0
                share = 100.0 * scan / (lat * 1e3)
                b0 = lat * 1e3 / max(lat * 1e3 - scan, 1e-9)
                b8 = lat * 1e3 / max(lat * 1e3 - scan * 7.0 / 8.0, 1e-9)
                lines.append("| %g | %d | %d | %.3f | %.0f | " % (R["alpha"], R["batch"], V["G"], lat, qps) +
                             " | ".join("%.1f" % med[s] if med[s] == med[s] else "-" for s in STAGES) +
                             " | %.1f | %.1f | %s | %.0f%% | x%.2f | x%.2f |" % (
                                 gpu, host, "%g" % kern if kern == kern else "-", share, b0, b8))
                summary.append(dict(tag=tag, alpha=R["alpha"], batch=R["batch"], G=V["G"], latency_ms=lat,
                                    qps=qps, stages_us=med, kernels=kern, gpu_sum_us=gpu, host_gaps_us=host,
                                    scan_share=share / 100, bound_scan0=b0, bound_scan8=b8,
                                    searches_traced=len(S)))
    text = "\n".join(lines) + "\n"
    print(text)
    json.dump(summary, open(os.path.join(d, "stages_summary.json"), "w"), indent=1)
    if md_out:
        open(md_out, "w").write(text)


if __name__ == "__main__":
    main()
