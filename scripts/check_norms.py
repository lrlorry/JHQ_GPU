#!/usr/bin/env python3
"""L2 norms of the vectors that actually enter JHQ, and the Eq. 4 scale.

The demo hands what load_fvecs_mmap returns straight to train() and add() with
no transformation, so the file contents are the vectors JHQ consumes. This
reads those files rather than trusting the preparation scripts.

Eq. 4 writes the codeword scale as sqrt(2/d). The implementation uses
sigma*sqrt(2) with sigma^2 = E[||x||^2]/d, which is Lemma 2's definition and
what train() computes as sqrt(sum_sq / (n*d)). The two agree exactly when
E[||x||^2] = 1. Whether that holds is measured here, not assumed.
"""
import sys, os, numpy as np

DATASETS = [
    ("vogue-768",     "/root/data/vogue-768_base.fvecs",
                      "/root/data/vogue-768_query.fvecs"),
    ("arxiv-768",     "/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
                      "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs"),
    ("openai3-1536",  "/root/autodl-tmp/openai3-1536/base.fvecs",
                      "/root/autodl-tmp/openai3-1536/query.fvecs"),
    ("openai3-3072",  "/root/autodl-tmp/openai3-3072/base.fvecs",
                      "/root/autodl-tmp/openai3-3072/query.fvecs"),
    ("bge-m3",        "/root/autodl-tmp/bge-m3/base.fvecs",
                      "/root/autodl-tmp/bge-m3/query.fvecs"),
    ("stella-trec24", "/root/autodl-tmp/stella-trec24/base.fvecs",
                      "/root/autodl-tmp/stella-trec24/query.fvecs"),
]

SAMPLE = 200_000   # deterministic stride sample; whole file when smaller


def norms(path, sample=SAMPLE):
    """Read the fvecs header for d, then a strided sample of whole vectors."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        d = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    rec = 4 + 4 * d
    n = size // rec
    step = max(1, n // sample)
    idx = np.arange(0, n, step)[:sample]
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = np.empty(len(idx), dtype=np.float64)
    for i, r in enumerate(idx):
        off = int(r) * rec + 4
        v = np.frombuffer(mm[off:off + 4 * d].tobytes(), dtype=np.float32)
        out[i] = float(np.linalg.norm(v.astype(np.float64)))
    del mm
    return d, n, len(idx), out


def report(tag, d, n, checked, v):
    print(f"  {tag:<10} d={d:<5} n={n:<10,} checked={checked:<8,} "
          f"min={v.min():.8f} mean={v.mean():.8f} max={v.max():.8f} std={v.std():.3e}")
    return v


print("=== L2 norms of the vectors entering JHQ ===")
rows = []
for name, base, query in DATASETS:
    if not os.path.exists(base):
        print(f"  {name}: base missing"); continue
    print(f"{name}:")
    d, n, c, bn = norms(base)
    report("base", d, n, c, bn)
    qn = None
    if os.path.exists(query):
        dq, nq, cq, qn = norms(query, sample=100_000)
        report("query", dq, nq, cq, qn)
    rows.append((name, d, bn, qn))

print()
print("=== Equation 4 scale: sigma*sqrt(2) against sqrt(2/d) ===")
print(f"  {'dataset':<15}{'sigma':>12}{'sigma*sqrt2':>14}{'sqrt(2/d)':>14}"
      f"{'abs diff':>12}{'rel diff':>12}")
for name, d, bn, _ in rows:
    # sigma^2 = E[||x||^2]/d, the same reduction train() performs.
    sigma = float(np.sqrt((bn ** 2).mean() / d))
    a = sigma * np.sqrt(2.0)
    b = np.sqrt(2.0 / d)
    print(f"  {name:<15}{sigma:>12.8f}{a:>14.8f}{b:>14.8f}"
          f"{abs(a-b):>12.3e}{abs(a-b)/b:>12.3e}")
