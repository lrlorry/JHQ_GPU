#!/usr/bin/env python3
"""COMP8800 poster, experiment E1: the cuVS baselines on the protocol JHQ v69's harness measures.

    python3 bench_cuvs.py --dataset arxiv-768 --method cagra --out RAW/cuvs_arxiv-768_cagra.jsonl

Methods: cagra (fp32 dataset), cagra-int8 (cuVS scalar quantiser), ivfpq, ivfpq-refine (IVF-PQ
candidates re-ranked exactly against the fp32 dataset held on the device), bruteforce.

Protocol A, host to host -- the poster's. One timed call answers the whole query set and runs:
    copy the original host fp32 queries into a pinned staging buffer (JHQ's search() makes the
    same memcpy) -> H2D -> [cagra-int8: scalar-quantise the queries on the device] -> search
    [ivfpq-refine: -> exact re-rank] -> D2H of the ids and the distances -> synchronise.
Protocol B, device resident -- auxiliary. The queries are already on the device (for cagra-int8
already quantised): search (and re-rank) only, then synchronise.

What changed against scripts/bench_all.py, every item in the baseline's favour or neutral:
  * each index is built once and every search parameter is swept on it (bench_all rebuilt the
    index for every search parameter, so a sweep cost one build per point);
  * one cuVS Resources handle on one stream and preallocated outputs serve every call, as JHQ
    reuses its workspace (bench_all let every call create a handle and allocate its outputs);
  * the queries go up from pinned memory, as JHQ's do (bench_all copied from pageable memory);
  * cagra-int8 now quantises the queries inside the timed call (bench_all quantised them once,
    outside it, so its "host to host" number left out a step the method cannot skip).

Each point: WARMUP untimed calls, then ROUNDS rounds of CALLS timed calls (JHQ's harness: one
untimed call, then 24 timed, per round), every call's time kept. Slow points (a call over
SLOW_MS) run fewer calls a round so the sweep stays bounded; the record says how many.

Output: one JSON object per line -- "meta", "index", "point" and "failure" records. Failures
are results and are never dropped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

DATASETS = {
    # name: (base, query, groundtruth, d, N)
    "vogue-768": ("/root/data/vogue-768_base.fvecs", "/root/data/vogue-768_query.fvecs",
                  "/root/data/vogue-768_groundtruth.ivecs", 768, 932328),
    "arxiv-768": ("/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
                  "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs",
                  "/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs", 768, 2253000),
    "openai3-1536": ("/root/autodl-tmp/openai3-1536/base.fvecs",
                     "/root/autodl-tmp/openai3-1536/query.fvecs",
                     "/root/autodl-tmp/openai3-1536/groundtruth.ivecs", 1536, 999000),
    "openai3-3072": ("/root/autodl-tmp/openai3-3072/base.fvecs",
                     "/root/autodl-tmp/openai3-3072/query.fvecs",
                     "/root/autodl-tmp/openai3-3072/groundtruth.ivecs", 3072, 999000),
}
K = 10


def fvecs(path):
    """One float32 copy of an .fvecs file (memmap, then row blocks), as bench_all's _fvecs."""
    mm = np.memmap(path, dtype=np.int32, mode="r")
    d = int(mm[0])
    n = mm.size // (d + 1)
    out = np.empty((n, d), dtype=np.float32)
    step = 1 << 19
    for i in range(0, n, step):
        j = min(i + step, n)
        out[i:j] = np.asarray(mm[i * (d + 1):j * (d + 1)]).reshape(-1, d + 1)[:, 1:].view(np.float32)
    del mm
    return out


def ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


def recall_at_k(ids, gt, k=K):
    """|returned top-k ∩ true top-k| / k averaged over queries -- common/recall.cuh's definition."""
    hits = 0
    for i in range(ids.shape[0]):
        hits += len(set(ids[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return hits / (ids.shape[0] * k)


def id_checks(ids, n):
    """Out-of-range ids (including cuVS's sentinels) and rows that repeat an id."""
    bad = int(((ids < 0) | (ids >= n)).sum())
    dup_rows = int(sum(1 for r in ids if len(set(r.tolist())) != len(r)))
    return bad, dup_rows


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--method", required=True,
                    choices=["cagra", "cagra-int8", "ivfpq", "ivfpq-refine", "bruteforce"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--graph-degrees", default="32,64")
    ap.add_argument("--itopk", default="32,64,96,128,192,256,384,512")
    ap.add_argument("--search-width", default="1,2,4,8")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--pq-dims", default="192,384,768")
    ap.add_argument("--nprobe", default="16,32,64,96,128,192,256,384,512")
    ap.add_argument("--lut", default="f32,f16", help="IVF-PQ lut/internal dtype pairs: f32 or f16")
    ap.add_argument("--refine-ratios", default="2,4,8")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--slow-ms", type=float, default=40.0)
    ap.add_argument("--no-b", action="store_true", help="skip protocol B")
    a = ap.parse_args()

    import cupy as cp
    from cuvs.common import Resources
    from cuvs.neighbors import brute_force, cagra, ivf_pq
    from cuvs.neighbors import refine as cuvs_refine
    from cuvs.preprocessing.quantize import scalar

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fh = open(a.out, "a")

    def emit(rec):
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def mem_used_mib():
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 2**20

    def mem_note():
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return f"device free {free / 2**30:.2f} of {total / 2**30:.2f} GiB"
        except Exception:
            return "device memory unreadable"

    base, query, gtp, d, N = DATASETS[a.dataset]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stream = cp.cuda.Stream(non_blocking=True)
    res = Resources(stream=stream.ptr)
    cp.cuda.runtime.deviceSynchronize()
    ctx_mib = mem_used_mib()          # CUDA context and the libraries, before anything of ours

    xq = fvecs(query)
    gt = ivecs(gtp)
    if gt.shape[1] < K or gt.shape[0] < xq.shape[0]:
        raise SystemExit(f"ground truth {gt.shape} cannot score {xq.shape[0]} queries at k={K}")
    nq = xq.shape[0]
    t_load = time.perf_counter()
    xb = fvecs(base)
    t_load = time.perf_counter() - t_load
    if xb.shape != (N, d):
        raise SystemExit(f"base {xb.shape} != expected {(N, d)}")
    emit({"type": "meta", "dataset": a.dataset, "method": a.method, "n": N, "d": d, "nq": nq,
          "gt_width": int(gt.shape[1]), "k": K, "argv": sys.argv,
          "commit": sh(f"git -C {repo} rev-parse --short HEAD"),
          "dirty": bool(sh(f"git -C {repo} status --porcelain -- experiments/comp8800_poster_e1")),
          "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
          "cuvs": __import__("cuvs").__version__, "cupy": cp.__version__, "numpy": np.__version__,
          "loadavg": sh("cat /proc/loadavg"), "ctx_mib": round(ctx_mib, 1),
          "base_load_s": round(t_load, 2), "protocol_A": "pinned staging + H2D + [quantise] + search "
          "[+ refine] + D2H ids and distances + sync", "protocol_B": "resident queries, search only + sync"})

    def pinned(shape, dtype):
        count = int(np.prod(shape))
        mem = cp.cuda.alloc_pinned_memory(count * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype, count).reshape(shape)

    # buffers shared by every method and point, allocated before any index is measured
    pin_q = pinned((nq, d), np.float32)
    q_dev = cp.empty((nq, d), cp.float32)          # protocol A's upload target
    q_res = cp.asarray(xq)                          # protocol B's resident queries
    q8_dev = cp.empty((nq, d), cp.int8)
    q8_res = cp.empty((nq, d), cp.int8)
    ids_dev = cp.empty((nq, K), cp.int64)
    dist_dev = cp.empty((nq, K), cp.float32)
    idsB_dev = cp.empty((nq, K), cp.int64)
    distB_dev = cp.empty((nq, K), cp.float32)
    pin_ids = pinned((nq, K), np.int64)
    pin_dist = pinned((nq, K), np.float32)
    cp.cuda.runtime.deviceSynchronize()

    def measure(label, index_rec, sparams, search_into, quantise=None):
        """Time one search configuration under protocols A and B and emit its record.

        search_into(qin, ids_out, dist_out) enqueues the search on `stream` via `res`.
        quantise(src, dst), when given, is part of A's timed call (cagra-int8)."""
        def call_a():
            np.copyto(pin_q, xq)
            q_dev.set(pin_q, stream=stream)
            if quantise is not None:
                quantise(q_dev, q8_dev)
                search_into(q8_dev, ids_dev, dist_dev)
            else:
                search_into(q_dev, ids_dev, dist_dev)
            ids_dev.get(stream=stream, out=pin_ids, blocking=False)
            dist_dev.get(stream=stream, out=pin_dist, blocking=False)
            stream.synchronize()

        def call_b():
            search_into(q8_res if quantise is not None else q_res, idsB_dev, distB_dev)
            stream.synchronize()

        try:
            t = time.perf_counter()
            call_a()
            first_ms = (time.perf_counter() - t) * 1e3
            for _ in range(a.warmup - 1):
                call_a()
            ref = pin_ids.copy()
            rec = recall_at_k(ref, gt)
            bad, dup = id_checks(ref, N)
            per_call = (time.perf_counter() - t) * 1e3 / max(1, a.warmup)
            calls = a.calls if per_call <= a.slow_ms else max(3, int(a.calls * a.slow_ms / per_call))

            def rounds(fn):
                out = []
                for _ in range(a.rounds):
                    ts = []
                    for _ in range(calls):
                        t0 = time.perf_counter()
                        fn()
                        ts.append((time.perf_counter() - t0) * 1e3)
                    out.append(ts)
                return out

            ta = rounds(call_a)
            deterministic = bool(np.array_equal(ref, pin_ids))
            tb, b_same = None, None
            if not a.no_b:
                call_b()
                tb = rounds(call_b)
                b_same = bool(np.array_equal(ref, idsB_dev.get()))

            def summ(tt):
                q = [calls * nq / (sum(r) / 1e3) for r in tt]
                lat_med = [float(np.median(r)) for r in tt]
                lat_p95 = [float(np.percentile(r, 95)) for r in tt]
                return {"qps_rounds": [round(x, 1) for x in q], "qps_median": round(float(np.median(q)), 1),
                        "qps_min": round(min(q), 1), "qps_max": round(max(q), 1),
                        "lat_med_ms_rounds": [round(x, 4) for x in lat_med],
                        "lat_p95_ms_rounds": [round(x, 4) for x in lat_p95],
                        "call_ms": [[round(x, 4) for x in r] for r in tt]}

            point = {"type": "point", "dataset": a.dataset, "method": a.method, "label": label,
                     "index": index_rec, "search": sparams, "recall": round(rec, 6),
                     "invalid_ids": bad, "dup_rows": dup, "deterministic": deterministic,
                     "b_matches_a": b_same, "first_call_ms": round(first_ms, 3),
                     "warmup": a.warmup, "rounds": a.rounds, "calls_per_round": calls,
                     "mem_used_mib": round(mem_used_mib(), 1), "A": summ(ta)}
            if tb is not None:
                point["B"] = summ(tb)
            emit(point)
            print(f"  {label} {sparams} R={rec:.4f} A={point['A']['qps_median']:.0f} "
                  f"B={point['B']['qps_median'] if tb else 0:.0f} calls={calls} bad={bad} dup={dup} "
                  f"det={deterministic}", file=sys.stderr, flush=True)
        except Exception as ex:
            emit({"type": "failure", "dataset": a.dataset, "method": a.method, "label": label,
                  "index": index_rec, "search": sparams, "error": f"{type(ex).__name__}: {str(ex)[:400]}",
                  "trace": traceback.format_exc()[-1500:], "mem": mem_note()})
            print(f"  FAILED {label} {sparams}: {ex}", file=sys.stderr, flush=True)

    def built(kind, params, build_fn):
        """Build once, record time and the device memory the index holds."""
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        m0 = mem_used_mib()
        t0 = time.perf_counter()
        try:
            idx = build_fn()
            res.sync()
            cp.cuda.runtime.deviceSynchronize()
        except Exception as ex:
            emit({"type": "failure", "dataset": a.dataset, "method": a.method, "label": kind,
                  "index": params, "search": None, "error": f"{type(ex).__name__}: {str(ex)[:400]}",
                  "trace": traceback.format_exc()[-1500:], "mem": mem_note()})
            print(f"  BUILD FAILED {kind} {params}: {ex}", file=sys.stderr, flush=True)
            return None, None
        build_s = time.perf_counter() - t0
        cp.get_default_memory_pool().free_all_blocks()
        m1 = mem_used_mib()
        rec = dict(params, kind=kind, build_s=round(build_s, 3), index_mib=round(m1 - m0, 1),
                   mem_used_after_build_mib=round(m1, 1), ctx_mib=round(ctx_mib, 1))
        emit(dict(rec, type="index", dataset=a.dataset, method=a.method))
        print(f"  built {kind} {params}: {build_s:.1f} s, {m1 - m0:.0f} MiB", file=sys.stderr, flush=True)
        return idx, rec

    ints = lambda s: [int(x) for x in s.split(",") if x]
    dtypes = {"f32": np.float32, "f16": np.float16}

    if a.method in ("cagra", "cagra-int8"):
        quantise = None
        data = xb
        extra = {}
        if a.method == "cagra-int8":
            # the quantiser belongs to the dataset, not to a search parameter: trained once,
            # bench_all's choice of quantile kept
            t0 = time.perf_counter()
            qz = scalar.train(scalar.QuantizerParams(quantile=0.99), xb)
            data = scalar.transform(qz, xb)
            data = np.asarray(data) if not hasattr(data, "__cuda_array_interface__") else data
            extra = {"quantile": 0.99, "quantise_dataset_s": round(time.perf_counter() - t0, 3)}
            scalar.transform(qz, q_res, output=q8_res, resources=res)
            res.sync()

            def quantise(src, dst):
                scalar.transform(qz, src, output=dst, resources=res)
        for gd in ints(a.graph_degrees):
            p = dict(graph_degree=gd, intermediate_graph_degree=128, build_algo="ivf_pq", **extra)
            idx, irec = built(a.method, p, lambda: cagra.build(
                cagra.IndexParams(graph_degree=gd, intermediate_graph_degree=128), data, resources=res))
            if idx is None:
                continue
            for it in ints(a.itopk):
                for sw in ints(a.search_width):
                    sp = cagra.SearchParams(itopk_size=it, search_width=sw)

                    def search_into(qin, ids_out, dist_out, sp=sp, idx=idx):
                        cagra.search(sp, idx, qin, K, neighbors=ids_out, distances=dist_out, resources=res)
                    measure(f"{a.method}-gd{gd}", irec, {"itopk_size": it, "search_width": sw},
                            search_into, quantise)
            del idx
    elif a.method in ("ivfpq", "ivfpq-refine"):
        ratios = ints(a.refine_ratios) if a.method == "ivfpq-refine" else [1]
        raw = {}
        xb_dev = None
        if a.method == "ivfpq-refine":
            # the exact re-rank reads the fp32 dataset on the device: part of this method's
            # footprint, uploaded once and recorded beside every index it serves
            m0 = mem_used_mib()
            xb_dev = cp.asarray(xb)
            cp.cuda.runtime.deviceSynchronize()
            raw = {"raw_fp32_on_device_mib": round(mem_used_mib() - m0, 1)}
        cand = {r: (cp.empty((nq, K * r), cp.int64), cp.empty((nq, K * r), cp.float32)) for r in ratios}
        for pd in ints(a.pq_dims):
            p = dict(n_lists=a.nlist, pq_dim=pd, pq_bits=8, code_bytes_per_vec=pd, **raw)
            idx, irec = built(a.method, p, lambda pd=pd: ivf_pq.build(
                ivf_pq.IndexParams(n_lists=a.nlist, pq_dim=pd, pq_bits=8), xb, resources=res))
            if idx is None:
                continue
            for lut in a.lut.split(","):
                for np_ in ints(a.nprobe):
                    for r in ratios:
                        sp = ivf_pq.SearchParams(n_probes=np_, lut_dtype=dtypes[lut],
                                                 internal_distance_dtype=dtypes[lut])
                        if a.method == "ivfpq":
                            def search_into(qin, ids_out, dist_out, sp=sp, idx=idx):
                                ivf_pq.search(sp, idx, qin, K, neighbors=ids_out, distances=dist_out,
                                              resources=res)
                        else:
                            def search_into(qin, ids_out, dist_out, sp=sp, idx=idx, r=r):
                                cv, cd = cand[r]
                                ivf_pq.search(sp, idx, qin, K * r, neighbors=cv, distances=cd, resources=res)
                                cuvs_refine(xb_dev, qin, cv, K, indices=ids_out, distances=dist_out,
                                            resources=res)
                        measure(f"{a.method}-pq{pd}-{lut}", irec,
                                {"n_probes": np_, "lut": lut, "refine_ratio": r}, search_into)
            del idx
    elif a.method == "bruteforce":
        holder = {}

        def build_fn():
            holder["xb"] = cp.asarray(xb)              # the index reads it: keep it alive
            return brute_force.build(holder["xb"], metric="sqeuclidean", resources=res)
        idx, irec = built("bruteforce", {"metric": "sqeuclidean", "dtype": "fp32"}, build_fn)
        if idx is not None:
            def search_into(qin, ids_out, dist_out):
                brute_force.search(idx, qin, K, neighbors=ids_out, distances=dist_out, resources=res)
            measure("bruteforce-fp32", irec, {}, search_into)
    print("BENCH_CUVS_DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
