#!/usr/bin/env python3
"""FAISS IVF-PQ at code lengths the GPU path cannot configure.

GpuIndexIVFPQ holds the PQ lookup table in shared memory, so M=96 already asks
for 98304 bytes against 49152 available, and M>=192 fails a supported-length
check even with interleavedLayout. Recall is a property of the index, not of
the device that scans it, so running those code lengths on CPU gives exactly
the number the equal-total-bytes comparison needs -- JHQ at M=48 stores
48 + d*Br/8 + 8 = 440 bytes per vector on 768-d, so the honest comparison is
against FAISS at a few hundred bytes, not at 48.

QPS from this script is CPU QPS and is not comparable to the GPU numbers; only
the recall column is used.
"""
import argparse, csv, time
import numpy as np
import faiss


def read_fvecs(p):
    a = np.fromfile(p, dtype=np.int32); d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))


def read_ivecs(p):
    a = np.fromfile(p, dtype=np.int32); d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


def recall_at_k(I, gt, k):
    return float(np.mean([len(set(I[i, :k]) & set(gt[i, :k])) for i in range(len(I))]) / k)


def main():
    ap = argparse.ArgumentParser()
    for x in ("base", "query", "gt", "name", "out"):
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--ms", default="192,384")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobes", default="8,32,128")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        faiss.omp_set_num_threads(a.threads)

    xb, xq, gt = read_fvecs(a.base), read_fvecs(a.query), read_ivecs(a.gt)
    d = xb.shape[1]
    print(f"{a.name}: base={xb.shape} threads={faiss.omp_get_max_threads()}", flush=True)

    rows = []
    for M in [int(x) for x in a.ms.split(",") if x.strip()]:
        if d % M:
            print(f"  skip M={M}: d={d} not divisible", flush=True); continue
        idx = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, a.nlist, M, 8)
        t = time.time(); idx.train(xb[:100000]); idx.add(xb); build = time.time() - t
        print(f"  IVFPQ M={M} ({M} B/vec) built in {build:.1f}s", flush=True)
        for np_ in [int(x) for x in a.nprobes.split(",")]:
            idx.nprobe = np_
            t = time.time(); _, I = idx.search(xq, a.k); el = time.time() - t
            r = recall_at_k(I, gt, a.k)
            print(f"    nprobe={np_:<4} recall={r:.4f}  cpu_qps={len(xq)/el:.0f}", flush=True)
            rows.append(dict(method=f"FAISS-CPU-IVFPQ-{M}B", M=M, nprobe=np_, recall=r,
                             cpu_qps=len(xq) / el, build_s=build, bytes_per_vec=M))
        del idx

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "M", "nprobe", "recall", "cpu_qps",
                                          "build_s", "bytes_per_vec"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
