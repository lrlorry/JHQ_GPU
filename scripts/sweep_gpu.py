#!/usr/bin/env python3
"""
Run Vogue-768 GPU sweeps and write a CSV accepted by plot_gpu_comparison.py.

Defaults match the existing vogue768_gpu_comparison figure:
  dataset = Vogue-768, d=768, k=10
  JHQ/JQ parameters = M=96, B=8, Br=4

Examples:
  python3 scripts/sweep_gpu.py --version jhq_v12_transposed --output results/jhq_gpu_v12.csv
  python3 scripts/sweep_gpu.py --version hblock_v1 --alpha 4.0 --alpha2 2.0
"""
import argparse
import csv
import os
import re
import subprocess
import sys


DEFAULT_BASE = "/root/data/vogue-768_base.fvecs"
DEFAULT_QUERY = "/root/data/vogue-768_query.fvecs"
DEFAULT_GT = "/root/data/vogue-768_groundtruth.ivecs"

DEFAULT_ALPHAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0]
DEFAULT_NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]

# JHQ series: IVF-scan pipeline optimization (v1–v12)
# HBlock series: hierarchical block index for massive datasets
METHOD_NAME = {
    "jhq_v1_plain":        "JHQ-GPU-v1",
    "jhq_v2_topk":         "JHQ-GPU-v2",
    "jhq_v3_ivf":          "JHQ-GPU-v3-IVF",
    "jhq_v4_batched_query":"JHQ-GPU-v4-Batched",
    "jhq_v5_cuda_graph":   "JHQ-GPU-v5-CUDAGraph",
    "jhq_v6_async_h2d":    "JHQ-GPU-v6-AsyncH2D",
    "jhq_v7_spin_sync":    "JHQ-GPU-v7-SpinSync",
    "jhq_v8_timing":       "JHQ-GPU-v8-Timing",
    "jhq_v10_bytelut":     "JHQ-GPU-v10-ByteLUT",
    "jhq_v11_outerlut":    "JHQ-GPU-v11-OuterLUT",
    "jhq_v12_transposed":  "JHQ-GPU-v12-Transposed",
    "hblock_v1":           "HBlock-v1",
}

JHQ_IVF_VERSIONS = {
    "jhq_v3_ivf", "jhq_v4_batched_query", "jhq_v5_cuda_graph",
    "jhq_v6_async_h2d", "jhq_v7_spin_sync", "jhq_v8_timing",
    "jhq_v10_bytelut", "jhq_v11_outerlut", "jhq_v12_transposed",
}
JHQ_BATCHED_VERSIONS = {
    "jhq_v4_batched_query", "jhq_v5_cuda_graph", "jhq_v6_async_h2d",
    "jhq_v7_spin_sync", "jhq_v8_timing",
    "jhq_v10_bytelut", "jhq_v11_outerlut", "jhq_v12_transposed",
}
HBLOCK_VERSIONS = {"hblock_v1"}
ALL_IVF_VERSIONS = JHQ_IVF_VERSIONS | HBLOCK_VERSIONS


def parse_list(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def parse_metric(pattern, text, name):
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"could not parse {name}")
    return float(m.group(1))


def run_one(cmd):
    print("CMD:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}\n{out[-4000:]}")
    recall = parse_metric(r"Recall@\d+\s*:\s*([\d.]+)", out, "recall")
    qps = parse_metric(r"QPS\s*:\s*([\d.]+)", out, "qps")
    train_ms = parse_metric(r"train:\s*([\d.]+)", out, "train time")
    add_ms = parse_metric(r"add:\s*([\d.]+)", out, "add time")
    return recall, qps, (train_ms + add_ms) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        choices=sorted(METHOD_NAME.keys()),
        default="jhq_v12_transposed",
    )
    ap.add_argument("--output", default=None)
    ap.add_argument("--demo", default=None)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--Br", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--alpha2", type=float, default=2.0,
                    help="hblock_v1 coarse cascade ratio (ck2/k)")
    ap.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobes", default=",".join(str(x) for x in DEFAULT_NPROBES))
    ap.add_argument("--ivf-iters", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    repo = os.path.expanduser("~/JHQ_GPU")
    demo = args.demo or os.path.join(repo, "build", f"demo_{args.version}")
    output = args.output or f"gpu_{args.version}_vogue768.csv"
    method = METHOD_NAME[args.version]

    sweep_values = (parse_list(args.nprobes, int)
                    if args.version in ALL_IVF_VERSIONS
                    else parse_list(args.alphas, float))

    rows = []
    build_time = None
    for val in sweep_values:
        if args.version in ALL_IVF_VERSIONS:
            nprobe = int(val)
            if args.version in HBLOCK_VERSIONS:
                cmd = [
                    demo, args.base, args.query, args.gt,
                    str(args.M), str(args.B), str(args.Br),
                    str(args.alpha), str(args.alpha2),
                    str(args.k),
                    str(args.nlist), str(nprobe), str(args.ivf_iters),
                    str(args.batch_size),
                ]
            else:
                cmd = [
                    demo, args.base, args.query, args.gt,
                    str(args.M), str(args.B), str(args.Br),
                    str(args.alpha), str(args.k),
                    str(args.nlist), str(nprobe), str(args.ivf_iters),
                ]
                if args.version in JHQ_BATCHED_VERSIONS:
                    cmd.append(str(args.batch_size))
            x_value = nprobe
            print(f"\nversion={args.version} nlist={args.nlist} nprobe={nprobe} alpha={args.alpha}",
                  flush=True)
        else:
            alpha = float(val)
            cmd = [
                demo, args.base, args.query, args.gt,
                str(args.M), str(args.B), str(args.Br), str(alpha), str(args.k),
            ]
            x_value = alpha
            print(f"\nversion={args.version} alpha={alpha}", flush=True)

        try:
            recall, qps, bt = run_one(cmd)
            if build_time is None:
                build_time = bt
            rows.append((method, x_value, recall, qps, build_time))
            print(f"  recall={recall:.4f}  qps={qps:.0f}  build={build_time:.1f}s", flush=True)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
            raise

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nprobe", "recall", "qps", "build_time"])
        w.writerows(rows)

    print(f"\nSaved -> {output}", flush=True)


if __name__ == "__main__":
    main()
