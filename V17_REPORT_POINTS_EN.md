# HBlock-v17 Report Points

## 1. HBlock-v17 vs CPU Baselines

1. In `v17_vs_cpu_linear`, HBlock-v17 is above JHQ CPU and FAISS-IVFPQ CPU across most medium-to-high recall points.
2. In the ultra-high recall range (`recall > 0.998`), JHQ CPU is higher, reaching `1.0000 recall @ 8.1K QPS`, because JHQ CPU uses full scan.
3. HBlock can also reach this regime with a full-scan version.

## 2. Limitation of the Current Tree + Beam/IVF Route

1. The current method is essentially a tree + beam search over leaf cells.
2. Its candidate complexity is approximately:

```text
O(P * N / L)
```

Here, `P` is the number of probed leaves, `N` is the dataset size, and `L` is the number of leaves.

3. At the 1B scale, higher recall requires probing more leaves, increasing both candidate count and data reads; graph-based methods are closer to `O(V)`, where `V` is the number of actually visited graph nodes.

## 3. Next Step: Block-Level Graph Search

1. New design: use the HBlock tree to find entry leaves, then treat leaf/block IDs as graph nodes for local expansion.
2. Because the graph is built at block level rather than vector level, the graph size is much smaller and graph construction should also be faster.
3. The complexity changes from scanning large leaves:

```text
O(P * N / L)
```

to expanding fixed-size physical blocks:

```text
O(Vb * B)
```

Here, `Vb` is the number of visited leaf blocks, and `B` is the block size, e.g. 128 vectors.

4. `Vb` should be controllable in practice: similar vectors usually form local neighborhoods, so a few graph hops from the entry leaves should cover the relevant blocks.
5. Each graph node is a contiguous physical block, so the search transfers and scans a small set of sequential tiles instead of many random vector-level accesses.
6. This reduces the transfer bottleneck and is expected to significantly improve throughput. The tree structure itself can also be further optimized.
