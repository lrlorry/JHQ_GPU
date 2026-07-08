#!/usr/bin/env python3
"""
Run CPU baselines (FAISS IVFPQ, JQ+IVF, JHQ+IVF) on vogue-768 and arxiv-768.
Auto-detects server vs local. Use --server to force server mode.
Use --dataset vogue|arxiv|both to select dataset (default: both on server, vogue locally).

Server: python3 scripts/bench_cpu_baselines.py --server
Local:  python3 scripts/bench_cpu_baselines.py
"""
import os, sys, time, csv, re, subprocess
import numpy as np

ON_SERVER = os.path.exists("/root/data/vogue-768_base.fvecs") or "--server" in sys.argv
ds_arg    = next((sys.argv[sys.argv.index("--dataset") + 1]
                  for i, a in enumerate(sys.argv) if a == "--dataset"), None)

DATASETS = {}
if ON_SERVER:
    BUILD_BIN = "/root/JHQ_repro/build/examples"
    RESULTS_DIR = "/root/JHQ_GPU/results"
    DATASETS["vogue"] = {
        "base":  "/root/data/vogue-768_base.fvecs",
        "query": "/root/data/vogue-768_query.fvecs",
        "gt":    "/root/data/vogue-768_groundtruth.ivecs",
        "out":   f"{RESULTS_DIR}/cpu_vogue.csv",
        "nlist": 1024,
    }
    DATASETS["arxiv"] = {
        "base":  "/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
        "query": "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs",
        "gt":    "/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs",
        "out":   f"{RESULTS_DIR}/cpu_arxiv.csv",
        "nlist": 4096,
    }
    targets = ["vogue", "arxiv"] if ds_arg is None else [ds_arg]
else:
    _JHQ_REPRO = os.path.expanduser("~/github/JHQ_repro")
    BUILD_BIN = os.path.join(_JHQ_REPRO, "build", "examples")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    DATASETS["vogue"] = {
        "base":  os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_base.fvecs"),
        "query": os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_query.fvecs"),
        "gt":    os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_groundtruth.ivecs"),
        "out":   os.path.join(RESULTS_DIR, "cpu_vogue.csv"),
        "nlist": 1024,
    }
    targets = ["vogue"] if ds_arg is None else [ds_arg]

NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
K = 10


def read_fvecs(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.int32)
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.int32)
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def recall_at_k(pred, gt, k):
    hits = sum(len(set(pred[i, :k]) & set(gt[i, :k])) for i in range(len(pred)))
    return hits / (len(pred) * k)


def bench_dataset(ds_name, ds):
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  →  {ds['out']}")
    print(f"{'='*60}")
    print(f"Loading {ds_name} …")
    xb = read_fvecs(ds["base"]);  print(f"  base  {xb.shape}")
    xq = read_fvecs(ds["query"]); print(f"  query {xq.shape}")
    gt = read_ivecs(ds["gt"]);    print(f"  gt    {gt.shape}")
    d  = xb.shape[1]
    nlist = ds["nlist"]
    results = []

    # ── FAISS IVFPQ ──────────────────────────────────────────────────────────
    try:
        import faiss
        def bench_faiss(M, label):
            print(f"\n{label}  M={M}")
            quant = faiss.IndexFlatL2(d)
            idx   = faiss.IndexIVFPQ(quant, d, nlist, M, 8)
            t0 = time.time()
            idx.train(xb[:200000])
            idx.add(xb)
            build = time.time() - t0
            print(f"  build: {build:.2f}s")
            for np_ in NPROBES:
                if np_ > nlist: break
                idx.nprobe = np_
                t1 = time.time()
                _, I = idx.search(xq, K)
                elapsed = time.time() - t1
                rec = recall_at_k(I, gt, K)
                qps = len(xq) / elapsed
                print(f"  nprobe={np_:<4}  recall={rec:.4f}  qps={qps:.0f}")
                results.append({"method": label, "ck": np_, "probed": np_,
                                "rerank_r": 0, "recall": rec, "qps": qps,
                                "build_time": build})

        bench_faiss(96,  "FAISS-IVFPQ-96B")
        bench_faiss(192, "FAISS-IVFPQ-192B")
        bench_faiss(384, "FAISS-IVFPQ-384B")
    except ImportError:
        print("faiss not found, skipping FAISS baselines")

    # ── JQ+IVF / JHQ+IVF ─────────────────────────────────────────────────────
    def run_binary(exe, extra_args, label):
        if not os.path.exists(exe):
            print(f"  binary not found: {exe}, skipping")
            return
        cmd = [exe, ds["base"], ds["query"], ds["gt"]] + [str(a) for a in extra_args]
        print(f"\n{label}\n  cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            print(f"  ERROR: {proc.stderr[:1000]}")
            return
        out = proc.stdout
        build_time = None
        m = re.search(r"Index build[:\s]+([\d.]+)\s*s", out)
        if m: build_time = float(m.group(1))
        m2 = re.search(r"(train|add)[:\s]+([\d.]+)\s*ms", out)
        if m2 and build_time is None: build_time = float(m2.group(2)) / 1000.0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3:
                try:
                    np_, rec, qps = int(parts[0]), float(parts[1]), float(parts[2])
                    results.append({"method": label, "ck": np_, "probed": np_,
                                    "rerank_r": 0, "recall": rec, "qps": qps,
                                    "build_time": build_time or 0})
                    print(f"  nprobe={np_:<4}  recall={rec:.4f}  qps={qps:.0f}")
                except ValueError:
                    pass

    run_binary(f"{BUILD_BIN}/demo_jq_ivf",
               ["96", "8", str(nlist)], "JQ+IVF")
    run_binary(f"{BUILD_BIN}/demo_jhq_ivf",
               ["96", "8", "4", str(nlist), "4.0"], "JHQ+IVF")

    os.makedirs(os.path.dirname(os.path.abspath(ds["out"])), exist_ok=True)
    with open(ds["out"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","ck","probed","rerank_r","recall","qps","build_time"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {ds['out']}")


print(f"Environment: {'server' if ON_SERVER else 'local'}  |  targets: {targets}")
for ds_name in targets:
    bench_dataset(ds_name, DATASETS[ds_name])
