#!/usr/bin/env python3
"""
Sweep hblock_v21 recall-QPS tradeoff.
v21 = v17 + sparse subtree bitmask routing (valid_c2/valid_c3 per parent node).

Usage (run on server from /root/JHQ_GPU):
  python3 scripts/sweep_hblock_v21.py --dataset vogue
  python3 scripts/sweep_hblock_v21.py --dataset arxiv
  python3 scripts/sweep_hblock_v21.py --dataset both
"""
import argparse, csv, os, re, subprocess, sys

REPO   = os.path.expanduser("~/JHQ_GPU")
BINARY = os.path.join(REPO, "build", "demo_hblock_v21")

DATASETS = {
    "vogue": {
        "base":  "/root/data/vogue-768_base.fvecs",
        "query": "/root/data/vogue-768_query.fvecs",
        "gt":    "/root/data/vogue-768_groundtruth.ivecs",
        "label": "HBlock-v21 (Vogue-768)",
    },
    "arxiv": {
        "base":  "/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
        "query": "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs",
        "gt":    "/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs",
        "label": "HBlock-v21 (Arxiv-768)",
    },
}

CK_SWEEP  = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
K1, K2, K3 = 16, 16, 16
BATCH      = 1024
D_PROJ     = 64   # v21: same as v17, bitmask is the only change
K_TOPK     = 10
KM_ITERS   = 30


def run_one(base, query, gt, ck, rerank_r):
    cmd = [
        BINARY, base, query, gt,
        str(K1), str(K2), str(K3),
        str(ck), str(ck), str(ck),
        str(K_TOPK), str(BATCH), str(D_PROJ),
        str(rerank_r), str(KM_ITERS),
    ]
    print("  CMD:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out  = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"binary failed (rc={proc.returncode})\n{out[-3000:]}")

    def parse(pat, name):
        m = re.search(pat, out)
        if not m:
            raise RuntimeError(f"cannot parse {name} from output:\n{out[-1000:]}")
        return float(m.group(1))

    recall   = parse(r"Recall@\d+\s*:\s*([\d.]+)", "recall")
    qps      = parse(r"QPS\s*:\s*([\d.]+)", "qps")
    train_ms = parse(r"train:\s*([\d.]+)\s*ms", "train")
    add_ms   = parse(r"add:\s*([\d.]+)\s*ms", "add")
    return recall, qps, (train_ms + add_ms) / 1000.0


def sweep_dataset(ds_name, ds_info, out_csv):
    rows = []
    build_time = None
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  →  {out_csv}")
    print(f"{'='*60}")
    for ck in CK_SWEEP:
        rerank_r = min(128, max(64, ck * ck * ck))
        print(f"\n  ck={ck}  probed_cells={ck**3}  rerank_r={rerank_r}", flush=True)
        try:
            recall, qps, bt = run_one(
                ds_info["base"], ds_info["query"], ds_info["gt"],
                ck, rerank_r,
            )
            if build_time is None:
                build_time = bt
            rows.append({
                "method":     ds_info["label"],
                "ck":         ck,
                "probed":     ck ** 3,
                "rerank_r":   rerank_r,
                "recall":     recall,
                "qps":        qps,
                "build_time": build_time,
            })
            print(f"  recall={recall:.4f}  qps={qps:.0f}  build={build_time:.1f}s", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","ck","probed","rerank_r","recall","qps","build_time"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vogue","arxiv","both"], default="both")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results"))
    args = ap.parse_args()

    targets = ["vogue","arxiv"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        out = os.path.join(args.out_dir, f"hblock_v21_{ds}.csv")
        sweep_dataset(ds, DATASETS[ds], out)


if __name__ == "__main__":
    main()
