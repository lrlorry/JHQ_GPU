#!/usr/bin/env python3
"""
Run CPU baselines (FAISS IVFPQ, JQ+IVF, JHQ+IVF) on vogue-768.
Auto-detects server vs local environment.

Local:  python3 scripts/bench_cpu_baselines.py
Server: python3 scripts/bench_cpu_baselines.py
"""
import os, sys, time, csv, re, subprocess
import numpy as np

# ── Environment detection ─────────────────────────────────────────────────────
_SERVER_BASE = "/root/data/vogue-768_base.fvecs"
ON_SERVER    = os.path.exists(_SERVER_BASE)

if ON_SERVER:
    BASE_F    = "/root/data/vogue-768_base.fvecs"
    QUERY_F   = "/root/data/vogue-768_query.fvecs"
    GT_F      = "/root/data/vogue-768_groundtruth.ivecs"
    BUILD_BIN = "/root/JHQ_repro/build/examples"   # may not exist
    OUT_CSV   = "/root/JHQ_GPU/results/cpu_vogue.csv"
else:
    # Local Mac paths
    _JHQ_REPRO = os.path.expanduser("~/github/JHQ_repro")
    BASE_F    = os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_base.fvecs")
    QUERY_F   = os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_query.fvecs")
    GT_F      = os.path.join(_JHQ_REPRO, "data", "vogue768", "vogue768_groundtruth.ivecs")
    BUILD_BIN = os.path.join(_JHQ_REPRO, "build", "examples")
    OUT_CSV   = os.path.join(os.path.dirname(__file__), "..", "results", "cpu_vogue.csv")

print(f"Environment: {'server' if ON_SERVER else 'local'}")
print(f"  base:  {BASE_F}")
print(f"  query: {QUERY_F}")

NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
NLIST   = 1024
K       = 10


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


print("\nLoading vogue-768 …")
xb = read_fvecs(BASE_F);  print(f"  base  {xb.shape}")
xq = read_fvecs(QUERY_F); print(f"  query {xq.shape}")
gt = read_ivecs(GT_F);    print(f"  gt    {gt.shape}")
d  = xb.shape[1]

RESULTS = []

# ── FAISS IVFPQ ───────────────────────────────────────────────────────────────
try:
    import faiss
    def bench_faiss(M, label):
        print(f"\n{label}  M={M}")
        quant = faiss.IndexFlatL2(d)
        idx   = faiss.IndexIVFPQ(quant, d, NLIST, M, 8)
        t0 = time.time()
        idx.train(xb[:65536])
        idx.add(xb)
        build = time.time() - t0
        print(f"  build: {build:.2f}s")
        for np_ in NPROBES:
            if np_ > NLIST: break
            idx.nprobe = np_
            t1 = time.time()
            _, I = idx.search(xq, K)
            elapsed = time.time() - t1
            rec = recall_at_k(I, gt, K)
            qps = len(xq) / elapsed
            print(f"  nprobe={np_:<4}  recall={rec:.4f}  qps={qps:.0f}")
            RESULTS.append({"method": label, "ck": np_, "probed": np_,
                            "rerank_r": 0, "recall": rec, "qps": qps,
                            "build_time": build})

    bench_faiss(96,  "FAISS-IVFPQ-96B")
    bench_faiss(192, "FAISS-IVFPQ-192B")
    bench_faiss(384, "FAISS-IVFPQ-384B")
except ImportError:
    print("faiss not found, skipping FAISS baselines")

# ── JQ+IVF / JHQ+IVF binaries ────────────────────────────────────────────────
def run_binary(exe, extra_args, label):
    if not os.path.exists(exe):
        print(f"  binary not found: {exe}, skipping")
        return
    cmd = [exe, BASE_F, QUERY_F, GT_F] + [str(a) for a in extra_args]
    print(f"\n{label}\n  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
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
                RESULTS.append({"method": label, "ck": np_, "probed": np_,
                                "rerank_r": 0, "recall": rec, "qps": qps,
                                "build_time": build_time or 0})
                print(f"  nprobe={np_:<4}  recall={rec:.4f}  qps={qps:.0f}")
            except ValueError:
                pass

run_binary(f"{BUILD_BIN}/demo_jq_ivf",  ["96", "8", str(NLIST)],          "JQ+IVF")
run_binary(f"{BUILD_BIN}/demo_jhq_ivf", ["96", "8", "4", str(NLIST), "4.0"], "JHQ+IVF")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","ck","probed","rerank_r","recall","qps","build_time"])
    w.writeheader()
    w.writerows(RESULTS)
print(f"\nSaved → {OUT_CSV}")
