#!/usr/bin/env python3
"""
Sweep graph_budget for HBlock v22 (L1+L2 beam + L3 graph traversal).
Usage:
    python scripts/sweep_hblock_v22.py \
        --base /root/data/vogue-768/base.fvecs \
        --query /root/data/vogue-768/query.fvecs \
        --gt /root/data/vogue-768/groundtruth.ivecs \
        --bin ./build/demo_hblock_v22 \
        [--out results/hblock_v22_vogue.csv]
"""
import argparse
import subprocess
import re
import csv
import sys

BUDGETS  = [4, 8, 16, 32, 64, 96, 128]
CK1, CK2 = 2, 4   # L1/L2 beam width (fixed)
GRAPH_DEG = 32
D_PROJ    = 64
RERANK_R  = 128
BATCH     = 1024
KM_ITERS  = 30
K         = 10


def run_one(bin_path, base, query, gt, graph_budget):
    cmd = [
        bin_path, base, query, gt,
        "16", "16", "16",           # K1 K2 K3
        str(CK1), str(CK2),         # ck1 ck2
        str(K), str(BATCH),         # k batch
        str(D_PROJ), str(RERANK_R), # d_proj rerank_r
        str(KM_ITERS),              # km_iters
        str(GRAPH_DEG),             # graph_degree
        str(graph_budget),          # graph_budget
    ]
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    print(out)

    recall = qps = None
    for line in out.splitlines():
        m = re.search(r'Recall@\d+\s*:\s*([0-9.]+)', line)
        if m:
            recall = float(m.group(1))
        m = re.search(r'QPS\s*:\s*([0-9.]+)', line)
        if m:
            qps = float(m.group(1))

    return recall, qps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',  required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--gt',    required=True)
    parser.add_argument('--bin',   default='./build/demo_hblock_v22')
    parser.add_argument('--out',   default='results/hblock_v22.csv')
    args = parser.parse_args()

    rows = []
    for budget in BUDGETS:
        print(f"\n{'='*60}")
        print(f"graph_budget={budget}  ck1={CK1}  ck2={CK2}  degree={GRAPH_DEG}  rerank_r={RERANK_R}")
        recall, qps = run_one(args.bin, args.base, args.query, args.gt, budget)
        rows.append({
            'graph_budget': budget,
            'ck1': CK1, 'ck2': CK2,
            'graph_degree': GRAPH_DEG,
            'rerank_r': RERANK_R,
            'recall': recall,
            'qps': qps,
        })
        print(f"  → recall={recall:.4f}  qps={qps:.0f}")

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {args.out}")
    print(f"\n{'budget':>8}  {'recall':>8}  {'qps':>10}")
    for r in rows:
        print(f"{r['graph_budget']:>8}  {r['recall']:>8.4f}  {r['qps']:>10.0f}")


if __name__ == '__main__':
    main()
