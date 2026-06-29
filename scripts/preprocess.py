"""
Dataset preprocessing for JHQ reproduction.

Downloads datasets from HuggingFace, converts to .fvecs/.ivecs format,
and computes ground-truth k-NN labels using FAISS.

Usage:
    # Quick test (100K vectors, ~1 min)
    python scripts/preprocess.py --dataset openai-1024-100k --output_dir datasets

    # Full datasets
    python scripts/preprocess.py --dataset vogue-768 --output_dir datasets
    python scripts/preprocess.py --dataset openai-1536 --output_dir datasets
    python scripts/preprocess.py --dataset arxiv-768 --output_dir datasets
    python scripts/preprocess.py --dataset bge-1024 --output_dir datasets
"""

import argparse
import ast
import os
import struct

import faiss
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "openai-1024-100k": {
        "hf_name": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "openai-3072-100k": {
        "hf_name": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-100K",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "openai-1536": {
        "hf_name": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "openai-3072": {
        "hf_name": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "vogue-768": {
        "hf_name": "tonyassi/vogue933k-embeddings",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "arxiv-768": {
        "hf_name": "macrocosm/arxiv_abstracts",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
    "bge-1024": {
        "hf_name": "Upstash/wikipedia-2024-06-bge-m3",
        "split": "train",
        "embed_col": "embedding",
        "normalize": True,
    },
}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_fvecs(filename: str, vecs: np.ndarray) -> None:
    vecs = vecs.astype(np.float32)
    with open(filename, "wb") as f:
        for v in vecs:
            f.write(struct.pack("i", len(v)))
            f.write(v.tobytes())


def read_fvecs(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        data = f.read()
    n = 0
    offset = 0
    d = struct.unpack_from("i", data, offset)[0]
    n = len(data) // (4 + 4 * d)
    vecs = np.frombuffer(data, dtype=np.float32).reshape(n, d + 1)
    return np.ascontiguousarray(vecs[:, 1:])


def write_ivecs(filename: str, vecs: np.ndarray) -> None:
    vecs = vecs.astype(np.int32)
    with open(filename, "wb") as f:
        for v in vecs:
            f.write(struct.pack("i", len(v)))
            f.write(v.tobytes())


# ---------------------------------------------------------------------------
# Embedding parsing
# ---------------------------------------------------------------------------

def parse_embedding(value):
    if isinstance(value, (list, np.ndarray)):
        return [float(x) for x in value]
    if isinstance(value, str):
        try:
            return [float(x) for x in ast.literal_eval(value)]
        except Exception:
            clean = value.strip("[]")
            return [float(x.strip().strip("\"'")) for x in clean.split(",") if x.strip()]
    raise ValueError(f"Unknown embedding type: {type(value)}")


def extract_embeddings(dataset, embed_col: str) -> np.ndarray:
    print("Extracting embeddings...")
    all_vecs = []
    for example in dataset:
        all_vecs.append(parse_embedding(example[embed_col]))
    return np.array(all_vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def compute_ground_truth(
    base: np.ndarray,
    query: np.ndarray,
    k: int = 100,
    use_gpu: bool = False,
) -> np.ndarray:
    d = base.shape[1]
    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Computing ground truth on GPU ({faiss.get_num_gpus()} GPUs)...")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        print("Computing ground truth on CPU (this may take a while for large datasets)...")
        index = faiss.IndexFlatL2(d)
    index.add(base)
    _, labels = index.search(query, k)
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(
    dataset_key: str,
    output_dir: str,
    query_size: int = 1000,
    k: int = 100,
    use_gpu: bool = False,
    seed: int = 42,
) -> None:
    cfg = DATASETS[dataset_key]
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, dataset_key)

    # Skip if already processed
    if all(os.path.exists(f"{prefix}_{s}.fvecs") for s in ("base", "query")) and \
       os.path.exists(f"{prefix}_groundtruth.ivecs"):
        print(f"[{dataset_key}] Already preprocessed, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_key}")
    print(f"  HuggingFace: {cfg['hf_name']}")
    print(f"{'='*60}")

    # Download
    print("Downloading from HuggingFace...")
    ds_dict = load_dataset(cfg["hf_name"])
    split = cfg["split"]
    ds = ds_dict.get(split, ds_dict.get("train", list(ds_dict.values())[0]))
    print(f"Loaded {len(ds):,} records.")

    # Extract
    vecs = extract_embeddings(ds, cfg["embed_col"])
    print(f"Embedding shape: {vecs.shape}")

    # Normalize
    if cfg.get("normalize", True):
        print("Normalizing (L2)...")
        vecs = normalize(vecs, axis=1, norm="l2")

    # Split
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(vecs), size=query_size, replace=False)
    base_mask = np.ones(len(vecs), dtype=bool)
    base_mask[query_idx] = False
    base = vecs[base_mask]
    query = vecs[query_idx]
    print(f"Base: {len(base):,}  Query: {len(query):,}")

    # Ground truth
    gt = compute_ground_truth(base, query, k=k, use_gpu=use_gpu)

    # Save
    write_fvecs(f"{prefix}_base.fvecs", base)
    write_fvecs(f"{prefix}_query.fvecs", query)
    write_ivecs(f"{prefix}_groundtruth.ivecs", gt)
    print(f"Saved to {prefix}_{{base,query}}.fvecs  +  _groundtruth.ivecs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for JHQ reproduction")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to process",
    )
    parser.add_argument("--output_dir", default="datasets", help="Output directory")
    parser.add_argument("--query_size", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--k", type=int, default=100, help="Ground truth k-NN depth")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for ground truth computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    preprocess(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        query_size=args.query_size,
        k=args.k,
        use_gpu=args.gpu,
        seed=args.seed,
    )
