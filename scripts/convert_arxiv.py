"""
Convert arXiv-Abstracts-768 parquet files to fvecs/ivecs format for HBlock demo.
Usage: python3 convert_arxiv.py <parquet_dir> <output_dir>
"""
import os, sys, struct
import numpy as np
import glob

def write_fvecs(path, X):
    n, d = X.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(X[i].astype('float32').tobytes())
    print(f"Wrote {n}×{d} → {path}")

def write_ivecs(path, X):
    n, d = X.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(X[i].astype('int32').tobytes())
    print(f"Wrote {n}×{d} → {path}")

def main():
    parquet_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/arxiv-abstracts-768'
    out_dir     = sys.argv[2] if len(sys.argv) > 2 else '/root/autodl-tmp/arxiv-abstracts-768'
    nq = 1000
    gt_k = 100

    import pandas as pd

    # ── Load all parquet files ────────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(parquet_dir, 'data', '*.parquet')))
    if not files:
        files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
    print(f"Found {len(files)} parquet files")

    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        # find embedding column
        emb_col = [c for c in df.columns if 'embed' in c.lower() or 'vector' in c.lower()]
        if not emb_col:
            emb_col = [df.columns[0]]
        col = emb_col[0]
        vecs = np.stack(df[col].values).astype('float32')
        chunks.append(vecs)
        print(f"  {f}: {vecs.shape}")

    all_vecs = np.concatenate(chunks, axis=0)
    print(f"Total: {all_vecs.shape}")

    # ── Split query / base ────────────────────────────────────────────────────
    np.random.seed(42)
    idx = np.random.choice(len(all_vecs), nq, replace=False)
    query = all_vecs[idx]
    mask  = np.ones(len(all_vecs), dtype=bool)
    mask[idx] = False
    base  = all_vecs[mask]

    write_fvecs(os.path.join(out_dir, 'base.fvecs'),  base)
    write_fvecs(os.path.join(out_dir, 'query.fvecs'), query)

    # ── Brute-force ground truth (batched to avoid OOM) ──────────────────────
    print("Computing ground truth (brute force)...")
    try:
        import faiss
        index = faiss.IndexFlatL2(base.shape[1])
        index.add(base)
        _, gt = index.search(query, gt_k)
        print("Used FAISS for ground truth")
    except ImportError:
        # fallback: numpy batched
        print("FAISS not found, using numpy (slow)...")
        gt = np.zeros((nq, gt_k), dtype='int32')
        bs = 50
        for i in range(0, nq, bs):
            q = query[i:i+bs]
            dists = np.sum((base[None] - q[:, None])**2, axis=2)  # (bs, nb)
            gt[i:i+bs] = np.argsort(dists, axis=1)[:, :gt_k]
            print(f"  {i+bs}/{nq}")

    write_ivecs(os.path.join(out_dir, 'groundtruth.ivecs'), gt)
    print("Done.")

if __name__ == '__main__':
    main()
