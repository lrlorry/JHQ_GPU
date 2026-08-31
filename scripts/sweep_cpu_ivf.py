#!/usr/bin/env python3
"""
Run the CPU JHQ-IVF benchmark (JHQ_repro/examples/demo_jhq_ivf) on a given
dataset and save a CSV in the same {method,nprobe,recall,qps,build_time}
shape sweep_gpu.py writes, so plot_v12_vs_cpu_all.py can read both directly.

demo_jhq_ivf.cpp already sweeps nprobe internally (one process, one build,
loops nprobe in-process) and prints a table -- this script just runs it
once and parses that table, it does not re-invoke per nprobe like
sweep_gpu.py has to (the GPU demo doesn't sweep internally).

demo_jhq_ivf builds the index once, then runs TWO sweeps against it:
full JHQ (primary + residual refine) and JQ-only (primary-code ablation,
no residual stage) -- each preceded by a "=== <label> ===" marker line
that this script uses to tag the rows that follow it, so one CSV ends up
with two "method" values and plot_v12_vs_cpu_all.py plots both lines.

Resumable: if --output already exists and looks complete (has at least one
data row), skips the run entirely. Delete the file to force a rerun.

Usage:
  python3 scripts/sweep_cpu_ivf.py --base .../base.fvecs --query .../query.fvecs \
      --gt .../groundtruth.ivecs --output results/jhq_cpu_ivf_openai3-1536.csv
"""
import argparse
import csv
import os
import re
import struct
import subprocess
import sys

# demo_*_ivf now prints 4 columns: nprobe | Recall@10 | Pre-fix | QPS.
# The 3-column form is what those demos printed before the evaluator fix;
# accept both so old logs still parse.
ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?\s*$")
BUILD_RE = re.compile(r"Index build:\s*([\d.]+)\s*s")
METHOD_RE = re.compile(r"^===\s*(\S+)\s*===$")


def fvecs_dim(path):
    with open(path, "rb") as f:
        (d,) = struct.unpack("<i", f.read(4))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--demo", default=os.path.expanduser(
        "~/JHQ_repro/build/examples/demo_jhq_ivf"))
    ap.add_argument("--method-name", default="JHQ-CPU-IVF",
                     help="fallback method label for rows printed before any "
                          "'=== label ===' marker (current demo_jhq_ivf always "
                          "prints one, so this is effectively unused)")
    ap.add_argument("--M", type=int, default=None,
                     help="defaults to d/B (same rule the binary itself uses,"
                          " computed here explicitly so the wrapper doesn't"
                          " depend on positional-arg defaulting)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--Br", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if os.path.exists(args.output):
        with open(args.output) as f:
            if sum(1 for _ in f) > 1:
                print(f"[skip] {args.output} already has results, not rerunning "
                      f"(delete it to force a rerun)", flush=True)
                return

    d = fvecs_dim(args.base)
    M = args.M if args.M is not None else max(1, d // args.B)

    cmd = [args.demo, args.base, args.query, args.gt,
           str(M), str(args.B), str(args.Br), str(args.nlist),
           str(args.alpha), str(args.k)]
    print("CMD:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demo_jhq_ivf failed with code {proc.returncode}")

    bt_m = BUILD_RE.search(out)
    if not bt_m:
        raise RuntimeError("could not parse 'Index build: ... s' from output")
    build_time = float(bt_m.group(1))

    rows = []
    current_method = args.method_name
    for line in out.splitlines():
        mm = METHOD_RE.match(line.strip())
        if mm:
            current_method = mm.group(1)
            continue
        m = ROW_RE.match(line)
        if m:
            nprobe, recall, third, fourth = m.groups()
            # 4 columns -> (recall, pre_fix, qps); 3 -> (recall, qps), pre_fix unknown
            legacy, qps = (third, fourth) if fourth else ("nan", third)
            rows.append((current_method, int(nprobe), float(recall),
                         float(qps), build_time, float(legacy)))

    if not rows:
        raise RuntimeError("parsed 0 result rows from demo_jhq_ivf output -- "
                            "output format may not match what this script expects, "
                            "check the printed table above")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nprobe", "recall", "qps", "build_time",
                    "pre_fix_score"])
        w.writerows(rows)
    print(f"\nSaved -> {args.output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
