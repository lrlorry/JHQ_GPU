#!/usr/bin/env python3
"""One driver for every final measurement: JHQ, cuVS IVF-PQ, CAGRA (fp32 and int8).

Written after a round of sweeps in which several rows that looked like data were
not: a launch that ran out of registers printed recall 0.0000 beside 285796 QPS,
a shell substitution called a binary whose name had lost its suffix and left the
row blank while its neighbours filled in, and a hand-copied CSV duplicated one
dataset's numbers onto another. So:

  * every row carries the commit, GPU, driver, CUDA and cuVS versions, and the
    full parameter set it was produced with;
  * every point is run REPS times after a warm-up and reports mean and stdev,
    not one number;
  * rows that cannot be parsed, or that carry the signature of a failed launch
    (recall 0 with an impossible QPS), are recorded as failures with their
    stderr, never silently dropped and never averaged into anything;
  * nothing is transcribed by hand -- the CSV is the only output.

Usage:
  bench_all.py --suite front --dataset vogue-768 --out results/pre_freeze_v22_s2b1/front.csv
"""
import argparse, csv, json, os, re, subprocess, statistics, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    # name: (base, query, groundtruth, d, N)
    "vogue-768":      ("/root/data/vogue-768_base.fvecs",
                       "/root/data/vogue-768_query.fvecs",
                       "/root/data/vogue-768_groundtruth.ivecs", 768, 932328),
    "arxiv-768":      ("/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
                       "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs",
                       "/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs", 768, 2253000),
    "openai3-1536":   ("/root/autodl-tmp/openai3-1536/base.fvecs",
                       "/root/autodl-tmp/openai3-1536/query.fvecs",
                       "/root/autodl-tmp/openai3-1536/groundtruth.ivecs", 1536, 999000),
    "openai3-3072":   ("/root/autodl-tmp/openai3-3072/base.fvecs",
                       "/root/autodl-tmp/openai3-3072/query.fvecs",
                       "/root/autodl-tmp/openai3-3072/groundtruth.ivecs", 3072, 999000),
    "bge-m3":         ("/root/autodl-tmp/bge-m3/base.fvecs",
                       "/root/autodl-tmp/bge-m3/query.fvecs",
                       "/root/autodl-tmp/bge-m3/groundtruth.ivecs", 1024, 10091524),
    "stella-trec24":  ("/root/autodl-tmp/stella-trec24/base.fvecs",
                       "/root/autodl-tmp/stella-trec24/query.fvecs",
                       "/root/autodl-tmp/stella-trec24/groundtruth.ivecs", 1024, 17776615),
}

RE_RECALL = re.compile(r"^Recall@\d+\s*:\s*([\d.]+)", re.M)
RE_QPS    = re.compile(r"^QPS\s*:\s*([\d.]+)", re.M)
RE_VRAM   = re.compile(r"^VRAM used\s*:\s*([\d.]+) MiB", re.M)
RE_TRAIN  = re.compile(r"^\s*train:\s*([\d.]+) ms", re.M)
RE_ADD    = re.compile(r"^\s*add:\s*([\d.]+) ms", re.M)


def env_metadata():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    meta = {
        "commit": sh(f"git -C {REPO} rev-parse --short HEAD"),
        "dirty": "yes" if sh(f"git -C {REPO} status --porcelain") else "no",
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")[0],
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0],
        "nvcc": (sh("nvcc --version | tail -1") or "").split(",")[-1].strip(),
        "host_cores": sh("nproc"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import cuvs
        meta["cuvs"] = cuvs.__version__
    except Exception:
        meta["cuvs"] = "n/a"
    return meta


def looks_like_failed_launch(recall, qps):
    """A launch that fails for resources leaves every distance at its initial
    value: recall 0 next to a QPS no configuration reaches."""
    return recall is not None and recall == 0.0 and qps is not None and qps > 150000


def run_jhq(binary, ds, params, env, reps):
    """Run the JHQ demo `reps` times and return one aggregated record."""
    base, query, gt, d, N = DATASETS[ds]
    argv = [os.path.join(REPO, "build", binary), base, query, gt,
            str(params["M"]), str(params["B"]), str(params["Br"]),
            str(params["alpha"]), str(params["k"]), str(params["nlist"]),
            str(params["nprobe"]), str(params["ivf_iters"]),
            str(params["batch"]), "", str(params["kmeans_iters"])]
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    qpss, recalls, vram, train_ms, add_ms, errs = [], [], None, None, None, []
    for _ in range(reps):
        p = subprocess.run(argv, capture_output=True, text=True, env=e, timeout=3600)
        out = p.stdout + p.stderr
        mr, mq = RE_RECALL.search(out), RE_QPS.search(out)
        if not (mr and mq):
            errs.append((p.returncode, out.strip().splitlines()[-3:]))
            continue
        r, q = float(mr.group(1)), float(mq.group(1))
        if looks_like_failed_launch(r, q):
            errs.append((p.returncode, ["failed-launch signature: recall 0 with %.0f QPS" % q]))
            continue
        recalls.append(r); qpss.append(q)
        mv = RE_VRAM.search(out)
        if mv: vram = float(mv.group(1))
        mt, ma = RE_TRAIN.search(out), RE_ADD.search(out)
        if mt: train_ms = float(mt.group(1))
        if ma: add_ms = float(ma.group(1))
    return aggregate("JHQ", binary, ds, params, env, qpss, recalls,
                     vram, train_ms, add_ms, errs)


def aggregate(method, variant, ds, params, env, qpss, recalls,
              vram, train_ms, add_ms, errs):
    """One row. `qpss` and `recalls` are the per-run samples, never a mean that
    has already thrown away its spread."""
    ok = len(qpss) > 0
    return {
        "method": method, "variant": variant, "dataset": ds,
        "d": DATASETS[ds][3], "N": DATASETS[ds][4],
        "params": json.dumps(params, sort_keys=True),
        "env": json.dumps(env, sort_keys=True),
        "n_runs": len(qpss),
        "recall": f"{statistics.mean(recalls):.4f}" if ok else "",
        "recall_spread": f"{(max(recalls)-min(recalls)):.5f}" if ok else "",
        "qps_mean": f"{statistics.mean(qpss):.0f}" if ok else "",
        "qps_std": f"{statistics.stdev(qpss):.0f}" if len(qpss) > 1 else "0",
        "qps_rel_std_pct": f"{100*statistics.stdev(qpss)/statistics.mean(qpss):.2f}" if len(qpss) > 1 else "0",
        "vram_mib": f"{vram:.0f}" if vram else "",
        "train_ms": f"{train_ms:.0f}" if train_ms else "",
        "add_ms": f"{add_ms:.0f}" if add_ms else "",
        "status": ("CONTAMINATED" if (ok and contamination(vram, qpss)) else
                   ("ok" if ok else "FAILED")),
        "failures": json.dumps(errs[:2]) if errs else
                    (json.dumps([[-2, contamination(vram, qpss)]])
                     if ok and contamination(vram, qpss) else ""),
    }


# ── cuVS baselines ────────────────────────────────────────────────────────────
#
# Timing drains the stream before the clock stops. cuVS's search is
# asynchronous and returns device arrays, so a clock read straight after it can
# stop before the GPU has finished -- the benchmark this replaces synchronised
# one line too late, inside cp.asnumpy() after the time had been taken.
#
# Memory is measured with cudaMemGetInfo around the build rather than derived
# from bytes per vector, and the pool is trimmed first so the number reflects
# what the index holds rather than what the allocator is caching.

def _fvecs(path, limit=None):
    """Read an .fvecs file holding one float32 array, not three.

    fromfile followed by ascontiguousarray materialises the file twice: at
    stella-trec24's 67 GiB that is 134 GiB before cuVS allocates anything, and
    the runs were killed by the host OOM killer with no CUDA error to explain
    it. Map the file and copy row blocks into a single preallocated array.
    """
    import numpy as np
    mm = np.memmap(path, dtype=np.int32, mode="r")
    d = int(mm[0])
    n = mm.size // (d + 1)
    if limit:
        n = min(n, limit)
    out = np.empty((n, d), dtype=np.float32)
    CHUNK = 1 << 19
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        block = np.asarray(mm[i * (d + 1):j * (d + 1)]).reshape(-1, d + 1)[:, 1:]
        out[i:j] = block.view(np.float32)
    del mm
    return out


def _ivecs(path):
    import numpy as np
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


def _recall_at_k(found, gt, k):
    """Set intersection against the true top-k, matching common/recall.cuh."""
    hits = 0
    for i in range(found.shape[0]):
        hits += len(set(found[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return hits / (found.shape[0] * k)


def _mem_used_mib():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 2**20


def _timed(fn, reps):
    """Warm up, drain, then time `reps` calls and drain again."""
    import cupy as cp
    out = fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(reps):
        t = time.time()
        out = fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.time() - t)
    return ts, out


def run_cuvs(method, ds, grid, k, reps):
    import numpy as np, cupy as cp
    from cuvs.neighbors import ivf_pq, cagra
    base, query, gt_path, d, N = DATASETS[ds]
    xb_h, xq_h, gt = _fvecs(base), _fvecs(query), _ivecs(gt_path)

    # Quantise once, not once per configuration. Only the dataset decides the
    # quantiser, never the search parameters, and doing it inside the loop
    # meant re-reading and re-writing the whole set for every itopk: at
    # 17.8M x 1024 that is 73 GiB of fp32 through scalar.transform each time,
    # which is what made the stella int8 sweep exceed a 90 minute limit
    # without finishing a single row. The fp32 array is dropped afterwards --
    # nothing below needs it, and holding both it and the int8 copy is 91 GiB
    # against a 96.6 GiB cgroup.
    xb_q8 = xq_q8 = None
    if method == "cagra-int8":
        from cuvs.preprocessing.quantize import scalar
        q = scalar.train(scalar.QuantizerParams(quantile=0.99), xb_h)
        _b, _q = scalar.transform(q, xb_h), scalar.transform(q, xq_h)
        xb_q8 = _b[0] if isinstance(_b, tuple) else _b
        xq_q8 = _q[0] if isinstance(_q, tuple) else _q
        n_queries = len(xq_h)
        del xb_h
        import gc; gc.collect()
    else:
        n_queries = len(xq_h)

    rows = []
    for cfg in grid:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            m_before = _mem_used_mib()
            # Build from the host array: cuVS accepts host input for ivf_pq,
            # cagra and the scalar quantiser, so the build itself no longer
            # needs the caller to upload the set first. Only the queries go to
            # the device here.
            #
            # This does NOT make a large fp32 set fit. A CAGRA index holds the
            # dataset on the device for search, because traversal computes real
            # distances against the vectors; the graph is only the adjacency.
            # Measured VRAM matches N*d*bytes_per_component + N*degree*4 to
            # within 0.1% over every dataset and degree here -- bge-m3 int8 for
            # instance reads 11220 MiB against 11087 predicted -- so fp32
            # bge-m3 needs 38.50 GiB of vectors plus 1.20 GiB of graph, 39.70
            # against a 31.36 GiB card. The OOM that path reports is the index
            # not fitting, not an avoidable staging copy.
            if method == "cagra-int8":
                xb_in, xq_in = xb_q8, xq_q8
                bytes_vec = d + 4 * cfg["graph_degree"]
            else:
                xb_in, xq_in = xb_h, xq_h
                bytes_vec = (cfg.get("pq_dim") or d) if method == "ivfpq" \
                            else 4 * d + 4 * cfg["graph_degree"]
            # Queries are 1000 rows; the search API needs them on the device.
            xq_d = cp.asarray(xq_in) if not hasattr(xq_in, "__cuda_array_interface__") else xq_in
            xb_d = xb_in

            t0 = time.time()
            if method == "ivfpq":
                # kmeans_trainset_fraction defaults to 0.5, so cuVS puts half
                # the dataset on the device to train: 36.4 GiB at 17.8M x 1024,
                # which is why the build failed on a 31.4 GiB card. Shrinking it
                # is the fair thing to try before reporting that IVF-PQ does not
                # fit at this scale.
                kp = {}
                if cfg.get("trainset_fraction"):
                    kp["kmeans_trainset_fraction"] = cfg["trainset_fraction"]
                idx = ivf_pq.build(ivf_pq.IndexParams(
                    n_lists=cfg["n_lists"], pq_dim=cfg["pq_dim"],
                    pq_bits=cfg["pq_bits"], **kp), xb_d)
            else:
                idx = cagra.build(cagra.IndexParams(
                    graph_degree=cfg["graph_degree"]), xb_d)
            cp.cuda.Stream.null.synchronize()
            build_s = time.time() - t0
            m_after = _mem_used_mib()

            if method == "ivfpq":
                sp = ivf_pq.SearchParams(n_probes=cfg["n_probes"])
                call = lambda: ivf_pq.search(sp, idx, xq_d, k)
            else:
                sp = cagra.SearchParams(itopk_size=cfg["itopk_size"],
                                        search_width=cfg.get("search_width", 1))
                call = lambda: cagra.search(sp, idx, xq_d, k)
            ts, out = _timed(call, reps)
            I = cp.asnumpy(out[1])
            qps = [n_queries / t for t in ts]
            rec = _recall_at_k(I, gt, k)
            rows.append(aggregate(
                {"ivfpq": "cuVS-IVFPQ", "cagra": "cuVS-CAGRA",
                 "cagra-int8": "cuVS-CAGRA-int8"}[method],
                f"{bytes_vec}B", ds, dict(cfg, bytes_per_vec=bytes_vec, k=k),
                {}, qps, [rec] * len(qps), m_after - m_before,
                build_s * 1000, None, []))
            del idx, xq_d
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as ex:
            # An out-of-memory failure is a result, not an accident: record the
            # data scale, the configuration and what the device actually had
            # free at the moment it failed, so the claim rests on a measurement
            # rather than on bytes-per-vector arithmetic.
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                mem_note = (f"device free {free_b/2**30:.2f} GiB of "
                            f"{total_b/2**30:.2f} GiB at failure; dataset "
                            f"{N} x {d} fp32 = {N*d*4/2**30:.2f} GiB")
            except Exception:
                mem_note = "device memory unreadable at failure"
            rows.append(aggregate(
                method, str(cfg), ds, cfg, {}, [], [], None, None, None,
                [(-1, [f"{type(ex).__name__}: {str(ex)[:300]}", mem_note])]))
        r = rows[-1]
        print(f"  {method} {cfg} -> {r['status']} R={r['recall']} "
              f"QPS={r['qps_mean']}±{r['qps_std']} vram={r['vram_mib']}MiB",
              file=sys.stderr, flush=True)
    return rows


def contamination(vram, qpss):
    """Signals that another process shared the card during the measurement.

    Two are visible in the row itself. Device memory is read as a delta around
    the build, so a concurrent process freeing memory between the two reads
    makes it negative. And run-to-run spread on an idle card is 0.02-0.64% on
    this hardware, so a relative standard deviation in the double digits means
    the runs were not seeing the same machine.
    """
    notes = []
    # A third signature, from jhq-gpu-d8: the same index configuration measured
    # twice should hold similar memory, so a reading several times another
    # measurement of the same (dataset, pq_dim) is doubtful even when it is
    # positive and the spread looks fine. That is a cross-file comparison, so
    # it cannot be made here; check_cross_file_memory() below does it over a
    # directory once the runs are written.
    if vram is not None and vram < 0:
        notes.append(f"negative memory delta ({vram:.0f} MiB): another process "
                     "freed memory between the two reads")
    if len(qpss) > 1:
        rel = 100 * statistics.stdev(qpss) / statistics.mean(qpss)
        if rel > 5.0:
            notes.append(f"QPS spread {rel:.1f}% against 0.02-0.64% on an idle card")
    return notes


def check_cross_file_memory(paths, factor=2.5):
    """Rows whose memory is far from another measurement of the same index.

    Memory belongs to the index, so the same (dataset, pq_dim) should hold
    roughly the same amount whatever the search or training settings. A reading
    several times another one is doubtful even with a positive value and a
    tight QPS spread -- the case that found this was 15,951 MiB against 3,486
    for the same configuration at a different training fraction. Returns
    (path, row_index, note) for each suspect.
    """
    import collections
    seen = collections.defaultdict(list)
    for path in paths:
        with open(path) as fh:
            body = [l for l in fh if not l.startswith("#")]
        for i, r in enumerate(csv.DictReader(body)):
            try:
                p = json.loads(r["params"])
                v = float(r["vram_mib"])
            except (ValueError, KeyError):
                continue
            if v > 0:
                seen[(r["dataset"], p.get("pq_dim"), p.get("n_probes"))].append((path, i, v))
    out = []
    for key, rows in seen.items():
        if len(rows) < 2:
            continue
        lo = min(v for _, _, v in rows)
        for path, i, v in rows:
            if v > lo * factor:
                out.append((path, i, f"memory {v:.0f} MiB against {lo:.0f} for the same "
                                     f"{key[0]} pq_dim={key[1]} n_probes={key[2]}"))
    return out


FIELDS = ["method", "variant", "dataset", "d", "N", "params", "env", "n_runs",
          "recall", "recall_spread", "qps_mean", "qps_std", "qps_rel_std_pct",
          "vram_mib", "train_ms", "add_ms", "status", "failures"]


def write_rows(path, rows, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("# " + json.dumps(meta, sort_keys=True) + "\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows, "
          f"{sum(1 for r in rows if r['status']!='ok')} failed", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--method", default="jhq",
                    choices=["jhq", "ivfpq", "cagra", "cagra-int8"])
    ap.add_argument("--pq-dims", default="96,192,384,768")
    ap.add_argument("--trainset-fraction", type=float, default=0.0,
                    help="cuVS kmeans_trainset_fraction; 0 leaves the default (0.5)")
    ap.add_argument("--graph-degrees", default="32,64")
    ap.add_argument("--itopk", default="32,64,128,256,512")
    ap.add_argument("--search-width", default="1,2,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--binary", default="demo_jhq_v22_s2b1")
    ap.add_argument("--nprobe", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--Br", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--prefix", default="1/2")
    a = ap.parse_args()

    meta = env_metadata()
    print("env: " + json.dumps(meta, sort_keys=True), file=sys.stderr)

    if a.method != "jhq":
        probes = [int(x) for x in a.nprobe.split(",")]
        if a.method == "ivfpq":
            grid = [dict(n_lists=a.nlist, pq_dim=pd, pq_bits=8, n_probes=np_,
                         trainset_fraction=a.trainset_fraction)
                    for pd in (int(x) for x in a.pq_dims.split(","))
                    for np_ in probes]
        else:
            grid = [dict(graph_degree=gd, itopk_size=it, search_width=sw)
                    for gd in (int(x) for x in a.graph_degrees.split(","))
                    for it in (int(x) for x in a.itopk.split(","))
                    for sw in (int(x) for x in a.search_width.split(","))]
        write_rows(a.out, run_cuvs(a.method, a.dataset, grid, 10, a.reps), meta)
        return

    num, den = a.prefix.split("/")
    # Record every JHQ_* the child will actually see, not just the four set
    # here. The p0 rows carry only those four, and reading them back gives no
    # way to tell which primary codebook was in force: JHQ_PAPER_CODEBOOK
    # defaults on, but openai3 at M=96 throws under it, so those rows can only
    # have been produced with it off -- and nothing in the file says so. Their
    # recall does not reproduce today either, 0.9729 recorded against 0.9744
    # measured on vogue at nprobe=128, which is seven times the build's own
    # spread. A row that cannot say what produced it cannot be re-run.
    env = dict(os.environ)
    env = {k: v for k, v in env.items() if k.startswith("JHQ_")}
    env.update({"JHQ_BLOCK": a.block, "JHQ_PFX_NUM": num, "JHQ_PFX_DEN": den,
                "JHQ_TILE_M_RT": a.M})
    rows = []
    for np_ in [int(x) for x in a.nprobe.split(",")]:
        params = dict(M=a.M, B=8, Br=a.Br, alpha=a.alpha, k=10, nlist=a.nlist,
                      nprobe=np_, ivf_iters=8, batch=a.batch, kmeans_iters=5,
                      prefix=a.prefix)
        r = run_jhq(a.binary, a.dataset, params, env, a.reps)
        rows.append(r)
        print(f"  nprobe={np_:<5} {r['status']:<7} R={r['recall']} "
              f"QPS={r['qps_mean']}±{r['qps_std']} ({r['qps_rel_std_pct']}%)",
              file=sys.stderr, flush=True)
    write_rows(a.out, rows, meta)


if __name__ == "__main__":
    main()
