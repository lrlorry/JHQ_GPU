#!/usr/bin/env python3
"""cuVS CAGRA / IVF-PQ and HNSW, on the same footing as the JHQ-GPU runs.

Scored with the same Recall@10 the GPU demos use: the returned top-k against
the true top-k, as sets. Index size is reported alongside recall and QPS
because that, not raw throughput, is where a quantised index stands against a
graph -- CAGRA keeps the raw vectors, so on 768-d it needs about 3.2 KB per
vector against roughly 490 B for JHQ at M=96, Br=4.
"""
import argparse, csv, time
import numpy as np


def read_fvecs(p):
    a = np.fromfile(p, dtype=np.int32); d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))


def read_ivecs(p):
    a = np.fromfile(p, dtype=np.int32); d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


def recall_at_k(I, gt, k):
    return float(np.mean([len(set(np.asarray(I)[i, :k]) & set(gt[i, :k]))
                          for i in range(len(I))]) / k)


def run_cagra(xb, xq, gt, k, rows, degrees, itopks):
    import cupy as cp
    from cuvs.neighbors import cagra
    d = xb.shape[1]
    xb_d, xq_d = cp.asarray(xb), cp.asarray(xq)
    for deg in degrees:
        t = time.time()
        idx = cagra.build(cagra.IndexParams(graph_degree=deg), xb_d)
        build = time.time() - t
        # raw vectors stay resident, plus one int32 per edge
        bpv = d * 4 + deg * 4
        print(f"  CAGRA degree={deg} built in {build:.1f}s ({bpv} B/vec)", flush=True)
        for it in itopks:
            sp = cagra.SearchParams(itopk_size=it)
            el, (_, I) = _timed_search(lambda: cagra.search(sp, idx, xq_d, k))
            r = recall_at_k(cp.asnumpy(I), gt, k)
            qps = len(xq) / el
            print(f"    itopk={it:<5} recall={r:.4f}  qps={qps:.0f}", flush=True)
            rows.append(dict(method=f"cuVS-CAGRA-deg{deg}", param=it, recall=r,
                             qps=qps, build_s=build, bytes_per_vec=bpv))


def run_cuvs_ivfpq(xb, xq, gt, k, rows, ms, nlist, nprobes):
    import cupy as cp
    from cuvs.neighbors import ivf_pq
    d = xb.shape[1]
    xb_d, xq_d = cp.asarray(xb), cp.asarray(xq)
    for M in ms:
        if d % M:
            continue
        t = time.time()
        idx = ivf_pq.build(ivf_pq.IndexParams(n_lists=nlist, pq_dim=M, pq_bits=8), xb_d)
        build = time.time() - t
        print(f"  cuVS IVF-PQ pq_dim={M} built in {build:.1f}s", flush=True)
        for np_ in nprobes:
            sp = ivf_pq.SearchParams(n_probes=np_)
            el, (_, I) = _timed_search(lambda: ivf_pq.search(sp, idx, xq_d, k))
            r = recall_at_k(cp.asnumpy(I), gt, k)
            qps = len(xq) / el
            print(f"    nprobe={np_:<4} recall={r:.4f}  qps={qps:.0f}", flush=True)
            rows.append(dict(method=f"cuVS-IVFPQ-{M}B", param=np_, recall=r, qps=qps,
                             build_s=build, bytes_per_vec=M))


def _timed_search(fn, reps=5):
    """Time a cuVS search with the device actually drained.

    cuVS's Python search is asynchronous: it enqueues work and hands back
    device arrays, so a time.time() taken straight after it can stop before the
    GPU has finished. The sync in this benchmark used to arrive one line too
    late, inside cp.asnumpy(I) after the clock had already been read, which
    overstates QPS. Drain once after the warm-up and once after the timed runs,
    and average over reps the way the JHQ demo does.
    """
    import cupy as cp   # imported per-function elsewhere in this file, so the
                       # module-level name does not exist here
    out = fn()                                   # warm-up
    cp.cuda.Stream.null.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    cp.cuda.Stream.null.synchronize()
    return (time.time() - t) / reps, out


def run_hnsw(xb, xq, gt, k, rows, Ms, efs, threads):
    import hnswlib
    d = xb.shape[1]
    for M in Ms:
        idx = hnswlib.Index(space="l2", dim=d)
        idx.init_index(max_elements=len(xb), ef_construction=200, M=M)
        idx.set_num_threads(threads)
        t = time.time(); idx.add_items(xb); build = time.time() - t
        # vectors plus, roughly, M links per node on the upper layers and 2M on layer 0
        bpv = d * 4 + M * 2 * 4
        print(f"  HNSW M={M} built in {build:.1f}s ({threads} threads, ~{bpv} B/vec)", flush=True)
        for ef in efs:
            idx.set_ef(ef)
            idx.knn_query(xq, k=k)                     # warm-up
            t = time.time(); I, _ = idx.knn_query(xq, k=k); el = time.time() - t
            r = recall_at_k(I, gt, k)
            qps = len(xq) / el
            print(f"    ef={ef:<5} recall={r:.4f}  qps={qps:.0f}", flush=True)
            rows.append(dict(method=f"HNSW-M{M}", param=ef, recall=r, qps=qps,
                             build_s=build, bytes_per_vec=bpv))


def main():
    ap = argparse.ArgumentParser()
    for x in ("base", "query", "gt", "name", "out"):
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--which", default="cagra,ivfpq,hnsw")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()

    xb, xq, gt = read_fvecs(a.base), read_fvecs(a.query), read_ivecs(a.gt)
    print(f"{a.name}: base={xb.shape} query={xq.shape}", flush=True)
    if gt.shape[1] < a.k:
        raise SystemExit(f"ground truth {gt.shape[1]} wide, need >= {a.k}")

    which = {w.strip() for w in a.which.split(",")}
    rows = []
    for name, fn in (("cagra", lambda: run_cagra(xb, xq, gt, a.k, rows, [32, 64], [32, 64, 128, 256])),
                     ("ivfpq", lambda: run_cuvs_ivfpq(xb, xq, gt, a.k, rows, [48, 96, 192, 384],
                                                      1024, [1, 8, 32, 128])),
                     ("hnsw",  lambda: run_hnsw(xb, xq, gt, a.k, rows, [16, 32],
                                                [32, 64, 128, 256], a.threads))):
        if name not in which:
            continue
        try:
            fn()
        except Exception as e:                     # one missing dependency must not lose the rest
            print(f"  {name} unavailable: {type(e).__name__}: {e}", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "param", "recall", "qps",
                                          "build_s", "bytes_per_vec"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
