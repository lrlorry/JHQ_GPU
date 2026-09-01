#!/usr/bin/env python3
"""FAISS GPU IVF-PQ and brute-force, scored with the same Recall@10 the GPU
demos now use (first k ground-truth entries, set semantics).

The comparison JHQ has to survive is IVF-PQ at matched bytes: same IVF, same
nprobe grid, a product quantiser in both cases, differing only in the JL
rotation, the residual level and the analytical codebook init. Two matchings
are reported because they answer different questions:

  matched primary  FAISS M bytes vs JHQ's M-byte primary code. Isolates the
                   quantiser, ignoring that JHQ also stores a residual.
  matched total    FAISS M bytes vs JHQ's M + d*Br/8 + 8 bytes. What a fixed
                   memory budget actually buys.
"""
import argparse, csv, time
import numpy as np
import faiss


def read_fvecs(p):
    a = np.fromfile(p, dtype=np.int32)
    d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))


def read_ivecs(p):
    a = np.fromfile(p, dtype=np.int32)
    d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


def recall_at_k(I, gt, k):
    """Standard Recall@k: returned top-k against the true top-k, as sets."""
    return float(np.mean([len(set(I[i, :k]) & set(gt[i, :k])) for i in range(len(I))]) / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ms", default="48,96,192,488", help="PQ subspace counts = code bytes at 8 bits")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    xb, xq, gt = read_fvecs(a.base), read_fvecs(a.query), read_ivecs(a.gt)
    d = xb.shape[1]
    print(f"{a.name}: base={xb.shape} query={xq.shape} gt={gt.shape}", flush=True)
    if gt.shape[1] < a.k:
        raise SystemExit(f"ground truth is {gt.shape[1]} wide, need >= k={a.k}")

    res = faiss.StandardGpuResources()
    rows = []

    # Exact reference. Also the recall ceiling any approximate row is measured against.
    flat = faiss.GpuIndexFlatL2(res, d)
    flat.add(xb)
    t = time.time(); _, I = flat.search(xq, a.k); el = time.time() - t
    r = recall_at_k(I, gt, a.k)
    print(f"  GPU-Flat            recall={r:.4f}  qps={len(xq)/el:.0f}", flush=True)
    rows.append(dict(method="GPU-Flat", M=d * 4, nprobe=0, recall=r,
                     qps=len(xq) / el, build_s=0.0, bytes_per_vec=d * 4))
    del flat

    for M in [int(x) for x in a.ms.split(",") if x.strip()]:
        if d % M:
            print(f"  skip M={M}: d={d} not divisible", flush=True)
            continue
        cfg = faiss.GpuIndexIVFPQConfig()
        try:
            idx = faiss.GpuIndexIVFPQ(res, d, a.nlist, M, 8, faiss.METRIC_L2, cfg)
        except Exception as e:
            print(f"  skip M={M}: {e}", flush=True)
            continue
        t = time.time()
        idx.train(xb[:100000])
        idx.add(xb)
        build = time.time() - t
        print(f"  IVFPQ M={M} ({M} B/vec) built in {build:.1f}s", flush=True)
        for np_ in [int(x) for x in a.nprobes.split(",")]:
            idx.nprobe = np_
            t = time.time(); _, I = idx.search(xq, a.k); el = time.time() - t
            r = recall_at_k(I, gt, a.k)
            qps = len(xq) / el
            print(f"    nprobe={np_:<4} recall={r:.4f}  qps={qps:.0f}", flush=True)
            rows.append(dict(method=f"FAISS-GPU-IVFPQ-{M}B", M=M, nprobe=np_,
                             recall=r, qps=qps, build_s=build, bytes_per_vec=M))
        del idx

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "M", "nprobe", "recall", "qps",
                                          "build_s", "bytes_per_vec"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
