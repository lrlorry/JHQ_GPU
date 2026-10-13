#!/usr/bin/env python3
"""Summary of v69_run.sh: exactness, end-to-end identity, interleaved timing, stage breakdown.

    python3 analyze_v69.py results/practical_opt/v69      (writes <dir>/v69_summary.json)

Timing: per workload, two v65 and two v69 runs in the order v65 v69 v69 v65, 5 rounds each; for the
scheduled pipeline (G = -1) the QPS and median batch latency are pooled over each binary's 10 rounds
(median, and the min-max across rounds). Stage times are the nsys medians of experiments/pipeline_v62/
stage_times.py, one traced round per binary.
"""
import glob
import json
import os
import re
import statistics
import sys

DS = ["arxiv-768", "vogue-768", "openai3-3072"]
BATCHES = [8, 32, 256, 1000]


def main():
    R = sys.argv[1]
    out = {}
    t = open(os.path.join(R, "probe_test.txt")).read() if os.path.exists(os.path.join(R, "probe_test.txt")) else ""
    m = re.search(r"exactness: (\d+) cases, (\d+) failing", t)
    print("## Kernel exactness (probe_select_test)")
    print(m.group(0) if m else "no result")
    out["exactness"] = dict(cases=int(m.group(1)), failing=int(m.group(2))) if m else None
    print("\n## Kernel alone (us per launch)")
    kt = []
    for line in t.splitlines():
        mm = re.match(r"\s+nlist=\s*(\d+) block=\s*(\d+) b=\s*(\d+)\s+v65\s+([\d.]+)\s+v69\s+([\d.]+)", line)
        if mm:
            kt.append(dict(nlist=int(mm.group(1)), block=int(mm.group(2)), batch=int(mm.group(3)),
                           v65_us=float(mm.group(4)), v69_us=float(mm.group(5))))
            print(line.strip())
    out["kernel_us"] = kt

    print("\n## End-to-end identity (same index, G = 1 dump + recall of every variant)")
    out["identity"] = {}
    for ds in DS:
        p = os.path.join(R, f"id_{ds}_compare.txt")
        if not os.path.exists(p):
            print(ds, "missing"); continue
        lines = open(p).read().splitlines()
        res = lines[-1] if lines else "empty"
        out["identity"][ds] = res
        print(f"{ds}: {res}; " + "; ".join(l for l in lines if l.startswith(("probe ids", "primary top-ck"))))

    print("\n## Timing, scheduled pipeline (G = -1), pooled over 10 rounds per binary")
    print("| workload | b | v65 QPS | v69 QPS | x | v65 latency ms | v69 latency ms | saved us |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    out["timing"] = []
    for ds in DS:
        pool = {65: {}, 69: {}}
        for f in sorted(glob.glob(os.path.join(R, f"t_{ds}_*_v*.json"))):
            v = int(re.search(r"_v(\d+)\.json$", f).group(1))
            j = json.load(open(f))
            for r in j["runs"]:
                for var in r["variants"]:
                    if var["G"] != -1:
                        continue
                    e = pool[v].setdefault(r["batch"], dict(qps=[], lat=[], recall=set()))
                    e["qps"] += var["qps"]; e["lat"] += var["lat_med_ms"]; e["recall"].add(var["recall"])
        for b in BATCHES:
            a, c = pool[65].get(b), pool[69].get(b)
            if not a or not c:
                continue
            q65, q69 = statistics.median(a["qps"]), statistics.median(c["qps"])
            l65, l69 = statistics.median(a["lat"]), statistics.median(c["lat"])
            row = dict(workload=ds, batch=b, v65_qps=q65, v69_qps=q69, speedup=q69 / q65,
                       v65_qps_range=[min(a["qps"]), max(a["qps"])], v69_qps_range=[min(c["qps"]), max(c["qps"])],
                       v65_lat_ms=l65, v69_lat_ms=l69, rounds=[len(a["qps"]), len(c["qps"])],
                       recall_v65=sorted(a["recall"]), recall_v69=sorted(c["recall"]))
            out["timing"].append(row)
            print(f"| {ds} | {b} | {q65:.0f} | {q69:.0f} | {q69 / q65:.3f} | {l65:.4f} | {l69:.4f} | "
                  f"{(l65 - l69) * 1000:.1f} |")

    print("\n## Stage times (nsys, G = -1, us per search)")
    out["stages"] = []
    st = {}
    for v in (65, 69):
        p = os.path.join(R, f"stages_v{v}", "stages_summary.json")
        if os.path.exists(p):
            for row in json.load(open(p)):
                if row["G"] == -1:
                    st[(v, row["tag"], row["batch"])] = row
    print("| workload | b | probe_select v65 | v69 | centroid | rotation | scan | refine | GPU sum v65 | v69 | latency v65 | v69 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in DS:
        for b in BATCHES:
            a, c = st.get((65, ds, b)), st.get((69, ds, b))
            if not a or not c:
                continue
            sa, sc = a["stages_us"], c["stages_us"]
            out["stages"].append(dict(workload=ds, batch=b, v65=a, v69=c))
            print(f"| {ds} | {b} | {sa['probe_select']:.1f} | {sc['probe_select']:.1f} | {sc['centroid']:.1f} | "
                  f"{sc['rotation']:.1f} | {sc['scan']:.1f} | {sc['refine']:.1f} | {a['gpu_sum_us']:.1f} | "
                  f"{c['gpu_sum_us']:.1f} | {a['latency_ms'] * 1000:.1f} | {c['latency_ms'] * 1000:.1f} |")
    json.dump(out, open(os.path.join(R, "v69_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
