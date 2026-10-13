#!/usr/bin/env python3
"""Item 1: the median-threshold representation (B) against the paper index (A) on the GPU.

    python3 analyze_median.py <dir with <ds>_np128.json>      (writes <dir>/median_summary.json)

Both representations run in one process through v65's search (demo_jhq_v67_repr_bench): same
rotation, lists and positions, alpha swept on each. For each batch size the comparison is at
matched recall: the best QPS among a representation's points whose recall is at least the
target. Targets are A's recall at the paper alpha and at each of A's other alphas.
QPS is the median of 3 rounds.
"""
import json
import os
import statistics
import sys

PAPER_ALPHA = {"arxiv-768": 100, "vogue-768": 64, "openai3-3072": 4}


def best_at(points, target):
    ok = [p for p in points if p["recall"] >= target - 1e-12]
    return max(ok, key=lambda p: p["qps"]) if ok else None


def main():
    d = sys.argv[1]
    out = {}
    for ds, pa in PAPER_ALPHA.items():
        path = os.path.join(d, f"{ds}_np128.json")
        if not os.path.exists(path):
            continue
        j = json.load(open(path))
        pts = {}
        for r in j["runs"]:
            pts.setdefault(r["batch"], {}).setdefault(r["rep"], []).append(
                dict(alpha=r["alpha"], recall=r["recall"], qps=statistics.median(r["qps"]),
                     lat_ms=statistics.median(r["lat_med_ms"])))
        print(f"\n## {ds} (paper alpha {pa})")
        print("| b | target recall (A alpha) | A best: alpha, recall, QPS | B best: alpha, recall, QPS | B/A |")
        print("|---:|---|---|---|---:|")
        rows = []
        for b in sorted(pts):
            A, B = pts[b].get("A", []), pts[b].get("B", [])
            for a_pt in sorted(A, key=lambda p: p["alpha"]):
                t = a_pt["recall"]
                ba, bb = best_at(A, t), best_at(B, t)
                ratio = bb["qps"] / ba["qps"] if (ba and bb) else None
                rows.append(dict(batch=b, target_alpha=a_pt["alpha"], target_recall=t, paper=(a_pt["alpha"] == pa),
                                 A=ba, B=bb, ratio=ratio))
                bs = f"{bb['alpha']}, {bb['recall']:.4f}, {bb['qps']:.0f}" if bb else "not reached"
                print(f"| {b} | {t:.4f} ({a_pt['alpha']}{', paper' if a_pt['alpha'] == pa else ''}) | "
                      f"{ba['alpha']}, {ba['recall']:.4f}, {ba['qps']:.0f} | {bs} | "
                      f"{'' if ratio is None else f'{ratio:.2f}'} |")
            ra = max(p["recall"] for p in A); rb = max(p["recall"] for p in B)
            print(f"| {b} | highest recall | A {ra:.4f} | B {rb:.4f} | |")
        out[ds] = dict(points=pts, matched=rows)
    json.dump(out, open(os.path.join(d, "median_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
