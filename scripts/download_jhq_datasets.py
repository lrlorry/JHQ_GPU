#!/usr/bin/env python3
"""Download + convert the 4 remaining JHQ paper datasets (OpenAI3-1536,
OpenAI3-3072, BGE-M3-1024, Stella-TREC24) into the same
/root/autodl-tmp/<name>/{base.fvecs,query.fvecs,groundtruth.ivecs} layout
already used for arxiv-abstracts-768 in the hblock run scripts.

Run ON the AutoDL server (needs `numpy`, `huggingface_hub`, `pyarrow`,
and a `curl` binary -- pip install numpy huggingface_hub pyarrow; curl
is already present on basically every Linux box). No FAISS dependency --
ground truth is computed via a plain-numpy tiled brute-force scan, see
below.

Usage:
    python3 download_jhq_datasets.py openai3-1536
    python3 download_jhq_datasets.py openai3-3072
    python3 download_jhq_datasets.py bge-m3
    python3 download_jhq_datasets.py stella-trec24
    python3 download_jhq_datasets.py all          # do all four, in ascending size order
    python3 download_jhq_datasets.py openai3-1536 --prepare-only

Each dataset is capped at TARGET_ROWS (matching the JHQ paper's reported
dataset sizes) via lazy iteration over parquet shards -- avoids
materializing the full upstream dataset (BGE-M3's English config alone is
~47M rows) before subsampling, and stops fetching shards entirely once
TARGET_ROWS is reached (see iter_shard_batches).

Download mechanism (v5, see curl_download()): two Python HTTP paths were
tried and both failed on this network in ways that don't self-recover.
(1) `datasets.load_dataset(..., streaming=True)` reads parquet over HTTP
range requests via `aiohttp`, and `aiohttp.ClientSession` does NOT read
`http_proxy`/`https_proxy` env vars by default -- timed out even with a
proxy independently confirmed reachable via curl (AutoDL's
network_turbo). (2) Switching to `huggingface_hub.hf_hub_download()`
(`requests`-backed, does respect those env vars) got further -- it
started downloading -- but then stalled dead at a fixed byte count with
zero throughput for 30+ seconds and no exception, confirmed by comparing
`du -sb` before/after rather than guessing. Now uses a `curl` subprocess
straight to hf-mirror.com (independently confirmed reachable and fast --
0.57s round trip, no proxy needed for that domain) with
`--speed-limit 1000 --speed-time 30` (abort a stall instead of hanging
forever) and `--retry 5 -C -` (auto-retry with resume from wherever the
partial file left off). Each shard's local path gets a sibling `.done`
marker file once fully fetched, so re-running the whole script after a
partial/failed attempt doesn't re-download shards already complete.

Memory note (v2): rows are written straight to base.fvecs/query.fvecs as
they stream in -- never held as one big (target_rows, dim) array. The
first version of this script pre-allocated that array up front (e.g.
~6GB for openai3-1536, ~70GB for stella-trec24) and got silently SIGKILLed
by the OOM killer on openai3-1536 before it even finished downloading.

Ground truth (v3): computed via a tiled brute-force scan (see
compute_ground_truth_tiled) instead of a FAISS flat index holding the
whole base set -- a flat index for bge-m3 (10M x 1024) or stella-trec24
(17.8M x 1024) would need ~41GB / ~73GB resident (GPU or CPU), which
exceeds a 32GB GPU and plausibly system RAM too. The tiled scan reads
base.fvecs back in BASE_CHUNK-row pieces, computes exact L2 distances
from every query to that chunk via one GEMM, and folds the result into a
running per-query top-k -- peak memory is one chunk's distance matrix
regardless of how large the base set is. Recall is still exact (this is
brute force, not an approximation), just bounded in memory. No FAISS
dependency needed for this step anymore.

Dataset config fixes (v3, per a second review that caught what streaming
schema-detection alone couldn't): BGE-M3's config was "en" (47M rows) --
JHQ's paper-reported "10M" is actually the *Italian* config, which has
exactly 10,092,524 rows. Stella-TREC24's split was "train", which
doesn't exist on that dataset -- the real splits are "corpus" (17.8M,
what we want as the base set) and "test_query" (65 rows, the dataset's
own official query set). This script still self-samples its own held-out
queries from "corpus" rather than using "test_query" directly -- 65
queries is a very small eval set relative to the QUERY_SIZE convention
used for the other datasets here, and it isn't confirmed yet whether
test_query's embeddings are directly comparable to corpus's (same model,
same normalization). Worth revisiting before this dataset's numbers go
in the paper; flagging rather than silently deciding.
"""
import argparse
import os
import struct
import sys

import numpy as np

# AutoDL frequently has poor direct connectivity to huggingface.co.  Users can
# still select the official endpoint explicitly with HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

OUT_ROOT = "/root/autodl-tmp"
QUERY_SIZE = 1000   # held out from TARGET_ROWS, matching JHQ_official README's default
GT_K = 20
SEED = 42
BASE_CHUNK = 200_000   # rows per tiled ground-truth scan step
PARQUET_BATCH_ROWS = 4096

DATASETS = {
    "openai3-1536": dict(
        hf_path="Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        config=None, split="train", shard_prefix="data/train-",
        embedding_col="embedding", dim=1536, target_rows=1_000_000,
    ),
    "openai3-3072": dict(
        hf_path="Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        config=None, split="train", shard_prefix="data/train-",
        embedding_col="embedding", dim=3072, target_rows=1_000_000,
    ),
    "bge-m3": dict(
        # JHQ's "BGE-M3-1024, 10M" is the Italian config, not English --
        # confirmed against the dataset's own README: en=47,018,430,
        # it=10,092,524.
        hf_path="Upstash/wikipedia-2024-06-bge-m3",
        config="it", split="train", shard_prefix="data/it/",
        embedding_col="embedding", dim=1024, target_rows=10_092_524,
    ),
    "stella-trec24": dict(
        # "train" doesn't exist on this dataset; real splits are
        # corpus (17.8M) and test_query (65) -- see module docstring.
        hf_path="ielabgroup/stella_trec24_biogen_embedding",
        config=None, split="corpus", shard_prefix="data/corpus-",
        embedding_col="embedding", dim=1024, target_rows=17_800_000,
    ),
}


def write_fvecs_batch(f, vecs):
    """Write a full batch without one Python file write per vector."""
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    dim = vecs.shape[1]
    dtype = np.dtype([("dim", "<i4"), ("vec", "<f4", (dim,))])
    records = np.empty(vecs.shape[0], dtype=dtype)
    records["dim"] = dim
    records["vec"] = vecs
    f.write(records.tobytes())


def write_ivecs(path, arr):
    arr = arr.astype(np.int32)
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", len(v)))
            f.write(v.tobytes())


def iter_fvecs_chunks(path, dim, chunk_rows):
    """Read a .fvecs file back in chunks of `chunk_rows` (each yielded as a
    (chunk_rows_actual, dim) float32 array) without ever holding the whole
    file in memory at once."""
    rec_bytes = 4 + dim * 4
    dtype = np.dtype([("dim", "<i4"), ("vec", "<f4", (dim,))])
    with open(path, "rb") as f:
        while True:
            buf = f.read(rec_bytes * chunk_rows)
            if not buf:
                return
            if len(buf) % rec_bytes:
                raise RuntimeError(f"truncated fvecs record in {path}")
            records = np.frombuffer(buf, dtype=dtype)
            if not np.all(records["dim"] == dim):
                raise RuntimeError(f"invalid vector dimension in {path}")
            yield np.ascontiguousarray(records["vec"])


def compute_ground_truth_tiled(base_path, dim, n_base, query_vecs, k,
                                base_chunk=BASE_CHUNK):
    """Exact brute-force top-k nearest neighbors via a tiled L2 scan --
    memory bounded by (nq x base_chunk), never by n_base. Replaces holding
    the whole base set in one FAISS flat index (41GB/73GB for bge-m3/
    stella-trec24, more than a 32GB GPU and plausibly system RAM too).

    Vectors are already L2-normalized (see download()), so squared L2
    distance reduces to 2 - 2*dot; using the general q_norm+b_norm-2*dot
    form anyway so this still works if that assumption ever changes.
    """
    nq = query_vecs.shape[0]
    best_dist = np.full((nq, k), np.inf, dtype=np.float32)
    best_idx = np.full((nq, k), -1, dtype=np.int64)
    q_norm = (query_vecs ** 2).sum(axis=1)

    scanned = 0
    for base in iter_fvecs_chunks(base_path, dim, base_chunk):
        bc = base.shape[0]
        b_norm = (base ** 2).sum(axis=1)
        dots = query_vecs @ base.T                       # [nq, bc]
        d2 = q_norm[:, None] + b_norm[None, :] - 2.0 * dots
        cand_idx = np.arange(scanned, scanned + bc)

        cat_dist = np.concatenate([best_dist, d2], axis=1)
        cat_idx = np.concatenate(
            [best_idx, np.broadcast_to(cand_idx, (nq, bc))], axis=1)
        keep = np.argpartition(cat_dist, k - 1, axis=1)[:, :k]
        best_dist = np.take_along_axis(cat_dist, keep, axis=1)
        best_idx = np.take_along_axis(cat_idx, keep, axis=1)

        scanned += bc
        print(f"  [ground truth, tiled scan] {scanned:,}/{n_base:,} base vectors")

    order = np.argsort(best_dist, axis=1)
    return np.take_along_axis(best_idx, order, axis=1)


def find_parquet_shards(repo_id, shard_prefix):
    """List parquet shards under the dataset's known upstream directory."""
    from huggingface_hub import HfApi
    all_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shards = sorted(
        f for f in all_files
        if f.startswith(shard_prefix) and f.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(
            f"no parquet files matched prefix={shard_prefix!r} among "
            f"{len(all_files)} files in {repo_id} -- inspect "
            f"huggingface.co/datasets/{repo_id}/tree/main and hardcode the shard list."
        )
    return shards


MIRROR_BASE = os.environ["HF_ENDPOINT"].rstrip("/")
SHARD_CACHE_DIR = f"{OUT_ROOT}/.jhq_shard_dl"  # OUT_ROOT is the big data
# volume (/root/autodl-tmp), not /root/.cache -- /root itself sits on
# AutoDL's small system disk (confirmed 30GB total on this box) and a
# real run filled it to 100% with just two datasets' worth of raw parquet
# shard cache. bge-m3 (10M rows) and stella-trec24 (17.8M rows) would
# need far more than that.


def curl_download(repo_id, filename, repo_type="datasets", revision="main"):
    """Fetch one repo file via a `curl` subprocess against hf-mirror.com --
    NOT hf_hub_download()/requests, NOT datasets/aiohttp. Both Python HTTP
    paths were tried first and both failed on this network in ways that
    don't self-recover: aiohttp silently ignores http_proxy/https_proxy
    entirely, and a requests-based hf_hub_download() download stalled dead
    at a fixed byte count with zero throughput for 30+s and no exception --
    confirmed via repeated `du -sb` before/after checks, not a guess.

    curl, independently confirmed reachable and fast against hf-mirror.com
    (0.57s round trip, tested directly, no proxy needed for the mirror
    domain), has two things neither Python path gave us for free:
      --speed-limit/--speed-time: abort if throughput drops below 1000B/s
        for 30s straight -- turns a silent stall into a real error.
      --retry/--retry-delay + -C -: automatic retry with resume from
        wherever the partial file left off, instead of restarting cold.
    """
    import subprocess

    cache_dir = f"{SHARD_CACHE_DIR}/{repo_id.replace('/', '--')}"
    os.makedirs(cache_dir, exist_ok=True)
    local_path = f"{cache_dir}/{filename.replace('/', '--')}"
    done_marker = local_path + ".done"
    if os.path.exists(done_marker):
        return local_path

    url = f"{MIRROR_BASE}/{repo_type}/{repo_id}/resolve/{revision}/{filename}"
    print(f"  curl: {url}", flush=True)
    subprocess.run(
        ["curl", "-fSL",
         "--connect-timeout", "15",
         "--speed-limit", "1000", "--speed-time", "30",
         "--retry", "5", "--retry-delay", "5",
         "-C", "-",
         "-o", local_path,
         url],
        check=True,
    )
    open(done_marker, "w").close()
    return local_path


def iter_shard_batches(repo_id, shards, embedding_col):
    """Yield embedding-only Arrow batches, downloading one shard at a time.

    curl_download()
    for shard N+1 only happens once the caller actually asks for it, so a
    consumer that stops early (got >= need) never fetches shards it doesn't
    need. curl_download() marks each shard done on disk, so re-running the
    whole script doesn't re-download shards already fetched. Reading only the
    embedding column avoids converting text metadata into Python objects.
    """
    import pyarrow.parquet as pq

    for shard_no, shard in enumerate(shards, 1):
        print(f"  [shard {shard_no}/{len(shards)}] {shard}", flush=True)
        local_path = curl_download(repo_id, shard, repo_type="datasets")
        size_mib = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  [ready] {size_mib:.1f} MiB: {local_path}", flush=True)
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            columns=[embedding_col],
        ):
            yield batch.column(0)


def arrow_embeddings_to_numpy(column, dim):
    """Convert a regular Arrow list column to a dense float32 matrix."""
    values = column.values.to_numpy(zero_copy_only=False)
    if hasattr(column, "offsets"):
        offsets = column.offsets.to_numpy(zero_copy_only=False)
        if offsets.size != len(column) + 1 or not np.all(np.diff(offsets) == dim):
            raise ValueError("embedding lists do not all have the expected dimension")
        values = values[int(offsets[0]):int(offsets[-1])]
    if values.size != len(column) * dim:
        raise ValueError(
            f"expected {len(column)} x {dim} embedding values, got {values.size}"
        )
    return np.asarray(values, dtype=np.float32).reshape(len(column), dim)


def download(name, prepare_only=False):
    cfg = DATASETS[name]
    dim = cfg["dim"]
    need = cfg["target_rows"]
    repo_id = cfg["hf_path"]
    print(f"[{name}] listing parquet shards for {repo_id} "
          f"(config={cfg['config']}, split={cfg['split']}), target={need:,} rows, dim={dim}")
    print(f"[{name}] endpoint: {MIRROR_BASE}")
    shards = find_parquet_shards(repo_id, cfg["shard_prefix"])
    print(f"[{name}] {len(shards)} shard(s): {shards[0]}"
          + (f" .. {shards[-1]}" if len(shards) > 1 else ""))

    # Pick query row-positions up front (only needs the target count, not the
    # data itself) so we can route each streamed row to base or query as it
    # arrives, instead of buffering everything then splitting.
    query_size = min(QUERY_SIZE, need // 10)
    rng = np.random.default_rng(SEED)
    query_idx = set(rng.choice(need, query_size, replace=False).tolist())
    query_vecs = np.empty((query_size, dim), dtype=np.float32)

    out_dir = f"{OUT_ROOT}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    base_path, query_path = f"{out_dir}/base.fvecs", f"{out_dir}/query.fvecs"

    got = n_base = n_query = 0
    query_positions = np.fromiter(query_idx, dtype=np.int64)
    next_report = 200_000
    with open(base_path, "wb") as fb, open(query_path, "wb") as fq:
        for column in iter_shard_batches(repo_id, shards, cfg["embedding_col"]):
            if got >= need:
                break   # generator is lazy -- breaking here means any shard
                        # not yet fetched never gets downloaded at all

            vecs = arrow_embeddings_to_numpy(column, dim)
            take = min(vecs.shape[0], need - got)
            vecs = np.ascontiguousarray(vecs[:take])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            np.divide(vecs, norms, out=vecs, where=norms > 1e-12)

            positions = np.arange(got, got + take, dtype=np.int64)
            query_mask = np.isin(positions, query_positions, assume_unique=True)
            base_vecs = vecs[~query_mask]
            batch_queries = vecs[query_mask]
            write_fvecs_batch(fb, base_vecs)
            write_fvecs_batch(fq, batch_queries)
            query_vecs[n_query:n_query + batch_queries.shape[0]] = batch_queries

            n_base += base_vecs.shape[0]
            n_query += batch_queries.shape[0]
            got += take
            if got >= next_report:
                print(f"  [converted] {got:,}/{need:,}", flush=True)
                next_report = ((got // 200_000) + 1) * 200_000

    if got < need:
        print(f"[{name}] WARNING: only found {got:,} rows, wanted {need:,} "
              f"(upstream dataset smaller than expected)")
    if n_query < query_size:
        query_vecs = query_vecs[:n_query]
    print(f"[{name}] base={n_base:,} query={n_query:,} written -> {out_dir}/ "
          f"(streamed straight to disk, no full-array buffering)")

    if prepare_only:
        print(f"[{name}] prepare-only requested; ground truth was not computed.")
        return

    gt = compute_ground_truth_tiled(base_path, dim, n_base, query_vecs, GT_K)
    write_ivecs(f"{out_dir}/groundtruth.ivecs", gt.astype(np.int32))
    print(f"[{name}] groundtruth {gt.shape} -> {out_dir}/groundtruth.ivecs  done.")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASETS) + ["all"])
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="download and convert vectors, but do not compute ground truth",
    )
    args = parser.parse_args()
    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    # ascending size order so a failure on the big ones doesn't waste time
    targets.sort(key=lambda n: DATASETS[n]["target_rows"])

    # One dataset's failure (e.g. an unverified column name on stella-trec24)
    # must not stop the others from running -- catch per-dataset, print a
    # loud, unambiguous FAILED marker with the full traceback (so the log
    # alone is enough to see which dataset broke and why), and keep going.
    results = {}
    for name in targets:
        print(f"\n{'='*70}\n[{name}] starting\n{'='*70}")
        try:
            download(name, prepare_only=args.prepare_only)
            results[name] = "OK"
        except Exception:
            import traceback
            print(f"\n!!!!! [{name}] FAILED !!!!!")
            traceback.print_exc()
            print(f"!!!!! [{name}] FAILED -- see traceback above !!!!!\n")
            results[name] = "FAILED"

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name in targets:
        print(f"  {name:20s} {results.get(name, 'SKIPPED')}")
    if any(v == "FAILED" for v in results.values()):
        sys.exit(1)
