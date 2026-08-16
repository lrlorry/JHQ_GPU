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
"""
import ast
import struct
import sys

import numpy as np

OUT_ROOT = "/root/autodl-tmp"
QUERY_SIZE = 1000   # held out from TARGET_ROWS, matching JHQ_official README's default
GT_K = 20
SEED = 42

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


def write_fvecs(path, arr):
    arr = arr.astype(np.float32)
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", len(v)))
            f.write(v.tobytes())


def write_ivecs(path, arr):
    arr = arr.astype(np.int32)
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", len(v)))
            f.write(v.tobytes())


def download(name):
    from datasets import load_dataset

    cfg = DATASETS[name]
    print(f"[{name}] streaming {cfg['hf_path']} (config={cfg['config']}), "
          f"target={cfg['target_rows']:,} rows, dim={cfg['dim']}")

    ds = load_dataset(cfg["hf_path"], cfg["config"], split=cfg["split"], streaming=True)

    need = cfg["target_rows"]
    rows = np.empty((need, cfg["dim"]), dtype=np.float32)
    got = 0
    # find the embedding column name on the fly (usually "embedding"; fall back to
    # any field whose value looks like a dim-length float sequence)
    emb_col = None
    for ex in ds:
        if emb_col is None:
            for k, v in ex.items():
                try:
                    if len(parse_embedding(v)) == cfg["dim"]:
                        emb_col = k
                        break
                except Exception:
                    continue
            if emb_col is None:
                raise RuntimeError(
                    f"[{name}] couldn't find a {cfg['dim']}-dim embedding column "
                    f"among fields {list(ex.keys())} -- inspect the dataset schema "
                    f"manually (huggingface.co/datasets/{cfg['hf_path']}) and hardcode emb_col."
                )
            print(f"[{name}] using column '{emb_col}'")
        rows[got] = parse_embedding(ex[emb_col])
        got += 1
        if got % 200_000 == 0:
            print(f"  {got:,}/{need:,}")
        if got >= need:
            break
    if got < need:
        print(f"[{name}] WARNING: only found {got:,} rows, wanted {need:,} "
              f"(upstream dataset smaller than expected)")
        rows = rows[:got]

    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(min=1e-12)

    rng = np.random.default_rng(SEED)
    n = rows.shape[0]
    q_idx = rng.choice(n, min(QUERY_SIZE, n // 10), replace=False)
    mask = np.ones(n, dtype=bool)
    mask[q_idx] = False
    base, query = rows[mask], rows[q_idx]

    out_dir = f"{OUT_ROOT}/{name}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    write_fvecs(f"{out_dir}/base.fvecs", base)
    write_fvecs(f"{out_dir}/query.fvecs", query)
    print(f"[{name}] base={base.shape} query={query.shape} -> {out_dir}/")

    import faiss
    d = base.shape[1]
    if faiss.get_num_gpus() > 0:
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(base.astype(np.float32))
    _, gt = index.search(query.astype(np.float32), GT_K)
    write_ivecs(f"{out_dir}/groundtruth.ivecs", gt.astype(np.int32))
    print(f"[{name}] groundtruth {gt.shape} -> {out_dir}/groundtruth.ivecs  done.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in list(DATASETS) + ["all"]:
        print(f"usage: {sys.argv[0]} <{'|'.join(DATASETS)}|all>")
        sys.exit(1)
    targets = list(DATASETS) if sys.argv[1] == "all" else [sys.argv[1]]
    # ascending size order so a failure on the big ones doesn't waste time
    targets.sort(key=lambda n: DATASETS[n]["target_rows"])
    for name in targets:
        download(name)
