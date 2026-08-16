#!/usr/bin/env python3
"""Download + convert the 4 remaining JHQ paper datasets (OpenAI3-1536,
OpenAI3-3072, BGE-M3-1024, Stella-TREC24) into the same
/root/autodl-tmp/<name>/{base.fvecs,query.fvecs,groundtruth.ivecs} layout
already used for arxiv-abstracts-768 in the hblock run scripts.

Run ON the AutoDL server (needs `datasets`, `numpy`, `scikit-learn`,
`faiss-cpu` or `faiss-gpu` -- pip install datasets numpy scikit-learn faiss-gpu).

Usage:
    python3 download_jhq_datasets.py openai3-1536
    python3 download_jhq_datasets.py openai3-3072
    python3 download_jhq_datasets.py bge-m3
    python3 download_jhq_datasets.py stella-trec24
    python3 download_jhq_datasets.py all          # do all four, in ascending size order

Each dataset is capped at TARGET_ROWS (matching the JHQ paper's reported
dataset sizes) via streaming iteration -- avoids materializing the full
upstream dataset (BGE-M3's English config alone is ~42.5M rows) before
subsampling.

Memory note (v2): rows are written straight to base.fvecs/query.fvecs as
they stream in -- never held as one big (target_rows, dim) array. The
first version of this script pre-allocated that array up front (e.g.
~6GB for openai3-1536, ~70GB for stella-trec24) and got silently SIGKILLed
by the OOM killer on openai3-1536 before it even finished downloading.
The ground-truth FAISS index is still built from the full base set (a
flat/exact index inherently holds everything -- no way around that for
an exact ground truth), but it's added in chunks read back off disk
instead of coexisting with a second full in-RAM copy during download.
For stella-trec24 (17M x 1024 floats, ~70GB for the flat index alone)
this may still be too much for a single machine's RAM/VRAM -- if that
step OOMs, tell me and we'll switch that one dataset to an IVF-based
approximate ground truth instead of exact flat search.
"""
import ast
import os
import struct
import sys

import numpy as np

OUT_ROOT = "/root/autodl-tmp"
QUERY_SIZE = 1000   # held out from TARGET_ROWS, matching JHQ_official README's default
GT_K = 20
SEED = 42
GT_ADD_CHUNK = 200_000  # rows per FAISS index.add() call when building ground truth

DATASETS = {
    "openai3-1536": dict(
        hf_path="Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        config=None, split="train", dim=1536, target_rows=1_000_000,
    ),
    "openai3-3072": dict(
        hf_path="Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        config=None, split="train", dim=3072, target_rows=1_000_000,
    ),
    "bge-m3": dict(
        hf_path="Upstash/wikipedia-2024-06-bge-m3",
        config="en", split="train", dim=1024, target_rows=10_000_000,
    ),
    "stella-trec24": dict(
        hf_path="ielabgroup/stella_trec24_biogen_embedding",
        config=None, split="train", dim=1024, target_rows=17_000_000,
    ),
}


def parse_embedding(val):
    if isinstance(val, str):
        try:
            return [float(x) for x in ast.literal_eval(val)]
        except (ValueError, SyntaxError):
            s = val.strip("[]")
            return [float(x.strip().strip('"').strip("'")) for x in s.split(",") if x.strip()]
    return [float(x) for x in val]


def write_vec(f, vec_f32):
    f.write(struct.pack("<i", vec_f32.shape[0]))
    f.write(vec_f32.tobytes())


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
    with open(path, "rb") as f:
        while True:
            buf = f.read(rec_bytes * chunk_rows)
            if not buf:
                return
            n = len(buf) // rec_bytes
            arr = np.empty((n, dim), dtype=np.float32)
            for i in range(n):
                off = i * rec_bytes
                # skip the 4-byte length prefix, read dim float32s
                arr[i] = np.frombuffer(buf, dtype=np.float32, count=dim,
                                        offset=off + 4)
            yield arr


def download(name):
    from datasets import load_dataset

    cfg = DATASETS[name]
    dim = cfg["dim"]
    need = cfg["target_rows"]
    print(f"[{name}] streaming {cfg['hf_path']} (config={cfg['config']}), "
          f"target={need:,} rows, dim={dim}")

    ds = load_dataset(cfg["hf_path"], cfg["config"], split=cfg["split"], streaming=True)

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

    emb_col = None
    got = n_base = n_query = 0
    with open(base_path, "wb") as fb, open(query_path, "wb") as fq:
        for ex in ds:
            if emb_col is None:
                for k, v in ex.items():
                    try:
                        if len(parse_embedding(v)) == dim:
                            emb_col = k
                            break
                    except Exception:
                        continue
                if emb_col is None:
                    raise RuntimeError(
                        f"[{name}] couldn't find a {dim}-dim embedding column "
                        f"among fields {list(ex.keys())} -- inspect the dataset schema "
                        f"manually (huggingface.co/datasets/{cfg['hf_path']}) and hardcode emb_col."
                    )
                print(f"[{name}] using column '{emb_col}'")

            vec = np.asarray(parse_embedding(ex[emb_col]), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            if got in query_idx:
                write_vec(fq, vec)
                query_vecs[n_query] = vec
                n_query += 1
            else:
                write_vec(fb, vec)
                n_base += 1
            got += 1
            if got % 200_000 == 0:
                print(f"  {got:,}/{need:,}")
            if got >= need:
                break

    if got < need:
        print(f"[{name}] WARNING: only found {got:,} rows, wanted {need:,} "
              f"(upstream dataset smaller than expected)")
    if n_query < query_size:
        query_vecs = query_vecs[:n_query]
    print(f"[{name}] base={n_base:,} query={n_query:,} written -> {out_dir}/ "
          f"(streamed straight to disk, no full-array buffering)")

    import faiss
    if faiss.get_num_gpus() > 0:
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dim)
    else:
        index = faiss.IndexFlatL2(dim)
    added = 0
    for chunk in iter_fvecs_chunks(base_path, dim, GT_ADD_CHUNK):
        index.add(chunk)
        added += chunk.shape[0]
        print(f"  [ground truth] indexed {added:,}/{n_base:,} base vectors")
    _, gt = index.search(query_vecs.astype(np.float32), GT_K)
    write_ivecs(f"{out_dir}/groundtruth.ivecs", gt.astype(np.int32))
    print(f"[{name}] groundtruth {gt.shape} -> {out_dir}/groundtruth.ivecs  done.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in list(DATASETS) + ["all"]:
        print(f"usage: {sys.argv[0]} <{'|'.join(DATASETS)}|all>")
        sys.exit(1)
    targets = list(DATASETS) if sys.argv[1] == "all" else [sys.argv[1]]
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
            download(name)
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
