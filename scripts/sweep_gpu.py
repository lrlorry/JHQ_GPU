#!/usr/bin/env python3
"""
Sweep alpha on demo_v1_plain to generate speed-accuracy curve.
Run on the GPU server after building demo_v1_plain.

Usage:
    python3 scripts/sweep_gpu.py [output.csv]
"""
import subprocess, re, csv, sys, os

DEMO  = os.path.expanduser("~/JHQ_GPU/build/demo_v1_plain")
BASE  = "/root/data/vogue-768_base.fvecs"
QUERY = "/root/data/vogue-768_query.fvecs"
GT    = "/root/data/vogue-768_groundtruth.ivecs"
M, B, Br, k = 96, 8, 4, 10
ALPHAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0]
OUT = sys.argv[1] if len(sys.argv) > 1 else "jhq_gpu_v1_results.csv"

rows = []
build_time = None

for alpha in ALPHAS:
    print(f"\nalpha={alpha} ...", flush=True)
    r = subprocess.run(
        [DEMO, BASE, QUERY, GT, str(M), str(B), str(Br), str(alpha), str(k)],
        capture_output=True, text=True
    )
    out = r.stdout + r.stderr
    try:
        recall = float(re.search(r'Recall@\d+\s*:\s*([\d.]+)', out).group(1))
        qps    = float(re.search(r'QPS\s*:\s*([\d.]+)', out).group(1))
        t_ms   = float(re.search(r'train:\s*([\d.]+)', out).group(1))
        a_ms   = float(re.search(r'add:\s*([\d.]+)', out).group(1))
        if build_time is None:
            build_time = (t_ms + a_ms) / 1000.0
        rows.append((alpha, recall, qps, build_time))
        print(f"  recall={recall:.4f}  qps={qps:.0f}  build={build_time:.1f}s")
    except Exception as e:
        print(f"  PARSE ERROR: {e}\n{out[:500]}")

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'nprobe', 'recall', 'qps', 'build_time'])
    for alpha, recall, qps, bt in rows:
        w.writerow(['JHQ-GPU-v1', alpha, recall, qps, f"{bt:.2f}"])

print(f"\nSaved → {OUT}")
