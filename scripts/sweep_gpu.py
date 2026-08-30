#!/usr/bin/env python3
"""
Run Vogue-768 GPU sweeps and write a CSV accepted by plot_gpu_comparison.py.

Defaults match the existing vogue768_gpu_comparison figure:
  dataset = Vogue-768, d=768, k=10
  JHQ/JQ parameters = M=96, B=8, Br=4

Examples:
  python3 scripts/sweep_gpu.py --version jhq_v12_transposed --output results/jhq_v12_vogue.csv
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
    "jhq_v14_streaming_add": "JHQ-GPU-v14-StreamingAdd",
    "jhq_v15_eval_fix":      "JHQ-GPU-v15-EvalFix",
    "hblock_v1":           "HBlock-v1",
}

JHQ_IVF_VERSIONS = {
    "jhq_v3_ivf", "jhq_v4_batched_query", "jhq_v5_cuda_graph",
    "jhq_v6_async_h2d", "jhq_v7_spin_sync", "jhq_v8_timing",
    "jhq_v10_bytelut", "jhq_v11_outerlut", "jhq_v12_transposed",
    "jhq_v14_streaming_add", "jhq_v15_eval_fix",
}
JHQ_BATCHED_VERSIONS = {
    "jhq_v4_batched_query", "jhq_v5_cuda_graph", "jhq_v6_async_h2d",
    "jhq_v7_spin_sync", "jhq_v8_timing",
    "jhq_v10_bytelut", "jhq_v11_outerlut", "jhq_v12_transposed",
    "jhq_v14_streaming_add", "jhq_v15_eval_fix",
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
    # Post-v15 demos also print the score the old evaluator produced. Carrying
    # it in the CSV is what lets a re-run be checked against the pre-v15 files
    # in results/ instead of just replacing them.
    m = re.search(r"Pre-v15 score\s*:\s*([\d.]+)", out)
    legacy = float(m.group(1)) if m else float("nan")
    return recall, qps, (train_ms + add_ms) / 1000.0, legacy


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
    ap.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobes", default=",".join(str(x) for x in DEFAULT_NPROBES))
    ap.add_argument("--ivf-iters", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dump-prefix", default=None,
                    help="v15 only: write <prefix>_np<N>.{ivecs,json} per point")
    # hblock_v1-specific
    ap.add_argument("--hblock-K1", type=int, default=64, dest="hblock_K1")
    ap.add_argument("--hblock-K2", type=int, default=128, dest="hblock_K2")
    ap.add_argument("--hblock-ck1", type=int, default=4, dest="hblock_ck1")
    ap.add_argument("--hblock-ck3", type=int, default=64, dest="hblock_ck3")
    args = ap.parse_args()

    repo = os.path.expanduser("~/JHQ_GPU")
    demo = args.demo or os.path.join(repo, "build", f"demo_{args.version}")
    output = args.output or f"gpu_{args.version}_vogue768.csv"
    method = METHOD_NAME[args.version]

    sweep_values = (parse_list(args.nprobes, int)
                    if args.version in ALL_IVF_VERSIONS
                    else parse_list(args.alphas, float))

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Resume support: each nprobe/alpha value redoes a full train+add from
    # scratch (the GPU demo doesn't sweep internally the way the CPU one
    # does), so an interrupted run losing that work is expensive to redo.
    # If `output` already has rows for some of `sweep_values` (matched on
    # the x-axis value: nprobe or alpha), skip re-running those and reuse
    # what's on disk; write the CSV after every point, not just at the end,
    # so a kill mid-sweep only loses the point in flight.
    rows = []
    done_x = set()
    build_time = None
    if os.path.exists(output):
        with open(output, newline="") as f:
            for row in csv.DictReader(f):
                x = int(row["nprobe"]) if args.version in ALL_IVF_VERSIONS else float(row["nprobe"])
                rows.append((row["method"], x, float(row["recall"]),
                             float(row["qps"]), float(row["build_time"]),
                             float(row.get("pre_v15_score") or "nan")))
                done_x.add(x)
                build_time = float(row["build_time"])
        if done_x:
            print(f"[resume] {output} already has {len(done_x)} point(s): "
                  f"{sorted(done_x)} -- skipping those", flush=True)

    def save():
        with open(output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "nprobe", "recall", "qps", "build_time",
                        "pre_v15_score"])
            w.writerows(rows)

    for val in sweep_values:
        x_probe = int(val) if args.version in ALL_IVF_VERSIONS else float(val)
        if x_probe in done_x:
            continue
        if args.version in ALL_IVF_VERSIONS:
            nprobe = int(val)
            if args.version in HBLOCK_VERSIONS:
                # hblock_v1: sweep ck2 (= nprobe in this context)
                cmd = [
                    demo, args.base, args.query, args.gt,
                    str(args.hblock_K1), str(args.hblock_K2),
                    str(args.hblock_ck1), str(nprobe), str(args.hblock_ck3),
                    str(args.k), str(args.batch_size),
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
                # v15 takes a 13th arg: dump returned ids + a run record
                # (params, gt_width, both metrics) next to the CSV.
                if args.dump_prefix:
                    cmd.append(f"{args.dump_prefix}_np{nprobe}")
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
            recall, qps, bt, legacy = run_one(cmd)
            if build_time is None:
                build_time = bt
            rows.append((method, x_value, recall, qps, build_time, legacy))
            print(f"  recall={recall:.4f}  pre_v15={legacy:.4f}  "
                  f"qps={qps:.0f}  build={build_time:.1f}s", flush=True)
            save()  # persist after every point, not just at the end -- a
                     # kill mid-sweep should only lose the point in flight
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
            raise

    print(f"\nSaved -> {output}", flush=True)


if __name__ == "__main__":
    main()
