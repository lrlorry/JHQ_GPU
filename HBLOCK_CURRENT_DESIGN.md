# HBlock Current Design, Evidence, and Roadmap

This is the main design document for HBlock. Historical version details stay
in `VERSIONS.md`; per-version build instructions stay in each version README.

## 1. Research Goal

HBlock targets high-throughput GPU ANN search, including datasets whose PQ
codes and raw vectors do not fit in GPU memory.

The central co-design is:

~~~text
residual/JL tree          global routing and multiple entry regions
block-level graph         local correction and boundary crossing
physical vector blocks    coalesced GPU scan and transferable I/O units
batch-region scheduler    reuse one loaded region across multiple queries
exact rerank              recover final distance accuracy
~~~

The graph node is a physical block of at most 128 vectors, not an individual
vector. This reduces graph metadata by roughly two orders of magnitude and
turns irregular vector accesses into block scans. The cost is that search
must estimate the distance from a query to a *set* of vectors using compact
block metadata.

## 2. Current Query Pipeline

~~~text
L1/L2/L3 residual tree beam
        -> selected L3 cells
        -> block entry selection
        -> block-graph beam traversal
        -> PQ top-p inside every visited block
        -> exact L2 for those candidates
        -> global top-k merge
~~~

Current defaults in the SPACEV branch are approximately:

~~~text
K1=K2=K3=16              4096 L3 cells
ck1=2, ck2=2, ck3=4      16 selected L3 cells/query
block capacity=128
graph degree=32
entry_per_cell=4
d_proj=64
per_block_r=16
ef<=256
~~~

The tree, projected block representatives, graph adjacency, and block mapping
metadata are navigation data. PQ codes and raw vectors are payload data.

## 3. Current Evidence

### Vogue-768

The full-resident graph pipeline demonstrates that the basic GPU execution
path can achieve both high recall and high throughput:

~~~text
ef=128: recall@10=0.9904, QPS=129826
ef=256: recall@10=0.9968, QPS=72033
~~~

### SPACEV-100M

The current full-resident `v36_1` run builds about 1.113M blocks in about
254 seconds. Its search result is:

| ef | Recall@10 | QPS |
|---:|----------:|----:|
| 8 | 0.2935 | 458697 |
| 16 | 0.3434 | 421943 |
| 32 | 0.3936 | 363892 |
| 64 | 0.4409 | 295992 |
| 128 | 0.4893 | 216449 |
| 256 | 0.5365 | 141882 |

This is not a compute-throughput failure. The main unresolved problem is
finding the few blocks that contain the true neighbors.

### Region experiments

`v39` validates block-to-region mapping, bounded GPU region pools, fetch
accounting, and cross-query region deduplication. It also shows an important
tradeoff: small batches reduce the working-set footprint but lose throughput;
large random-query batches can touch most naively packed regions.

`v40` separates PQ candidate generation from exact rerank. `v41` introduces
region waves. `v42` moves region storage to mmap-backed files, and `v43`
experiments with graph-aware packing. These are experimental branches, not
yet a validated 1B end-to-end result.

## 4. Main Recall Problem

At 100M scale, 4096 fixed L3 cells contain about 272 blocks per cell on
average in the observed build. At 1B, the same design would contain roughly
1900 to 2700 blocks per cell, depending on packing efficiency.

The current entry selection, graph construction, and graph traversal all use
one projected block mean. The score is

~~~text
centroid_score(q, B) = ||Pq - P mean(B)||^2
~~~

but ANN search needs an estimate of

~~~text
set_distance(q, B) = min_{x in B} ||q - x||^2.
~~~

The block mean is an exact mean, but it is not an accurate nearest-point
representative. A block can contain one true neighbor while its mean remains
far from the query. The same representation error is currently reused at
three points:

1. selecting four entry blocks from each selected L3 cell;
2. selecting the graph's nearest 32 block neighbors;
3. ordering the graph traversal beam.

Scale does not imply that top-10 neighbors occupy hundreds of blocks; they
can occupy at most ten. Scale makes those few blocks harder to identify among
an increasingly large number of blocks with similar centroid scores.

### Other unresolved recall factors

The centroid hypothesis is plausible but is not yet the only proven cause:

1. The SPACEV demo trains the tree on the first 200K vectors instead of a
   reproducible random sample from the full corpus.
2. Graph candidate generation is hierarchical but truncated; the final graph
   keeps the nearest projected-centroid edges without diversity pruning,
   reciprocal repair, or explicit boundary portals.
3. Coarse parent-cell neighbors are represented using a particular child
   centroid, which is not a sound parent representative.
4. One residual PQ LUT from the best tree route is reused for blocks reached
   in other cells. Per-block top-16 may therefore discard a true neighbor
   before exact rerank.
5. A fixed 4096-cell tree becomes too coarse as the dataset grows.

### What cumulative residuals do and do not solve

For a block in tree path center `C`, the mean residual is

~~~text
mean_r(B) = mean(x - C).
~~~

Therefore:

~~~text
C + mean_r(B) = mean(B).
~~~

Replacing one block centroid with one cumulative mean residual is only a
change of coordinates and cannot improve ranking. A useful replacement is a
small set of residual medoids/prototypes:

~~~text
score(q, B) = min_j ||P(q - C_B) - P r_(B,j)||^2.
~~~

Two 64D FP16 prototypes per block require about the same storage as one 64D
FP32 centroid and can represent block boundaries or multiple local modes.

## 5. Diagnosis Before Redesign

The next diagnostic must use actual search entries and must not classify a
miss only by whether its L3 cell was selected, because the graph can cross
cell boundaries.

For every query and ground-truth neighbor, record:

~~~text
GT distinct block count
GT cell selected by tree?
GT block among actual entries?
shortest graph hop from actual entries to GT block
GT block visited?
GT vector survives per-block PQ top-p?
~~~

Run these controlled oracles:

1. **Visited-block exact oracle:** exact-scan every vector in visited blocks.
   This isolates PQ/top-p loss.
2. **Oracle block score:** replace centroid score offline with the true
   `min distance(query, vector in block)` while keeping entries, graph, and
   `ef` fixed. This measures block-representation loss.
3. **Unrestricted graph reachability:** measure shortest hops from actual
   entry blocks. This measures graph-topology loss.
4. **Random-training control:** rebuild with a fixed random 200K sample.

Decision rules:

| Observation | Primary fix |
|---|---|
| exact oracle is high, normal recall is low | per-cell LUT or larger/adaptive per-block top-p |
| oracle block score is much better | residual medoids/prototypes |
| GT block is unreachable or many hops away | graph reconstruction and portal edges |
| random training greatly improves entry coverage | representative tree training |
| selected-cell coverage is poor at all widths | finer/adaptive tree |

Do not combine all fixes in one version; each change needs an ablation.

## 6. Recommended Recall Fixes

### Stage A: Correct the controls

1. Train from a fixed uniform sample over the full dataset.
2. Repair the diagnostic to start BFS from actual entry blocks.
3. Add visited-block exact and oracle-block-score results.

### Stage B: Improve block representation

Compare, with identical graph budget:

~~~text
one projected mean
one medoid
two residual medoids
four residual medoids
~~~

Use the same representation for entry selection, edge construction, and
traversal. Otherwise topology and search scores remain inconsistent.

### Stage C: Improve graph topology

Candidate generation should preserve multiple scales. For degree 32, an
initial allocation to test is:

~~~text
20 local/same-cell edges
 6 cross-L3 boundary edges
 4 cross-L2 edges
 2 cross-L1 edges
~~~

Within each category, use a Vamana-style diversity/RobustPrune rule rather
than plain nearest-32. Add reciprocal edges and re-prune overflow. Boundary
blocks may be registered as logical portals in neighboring cells without
duplicating their vector payload.

### Stage D: Scale the tree

Keep L1-L3 as coarse routing, then split oversized L3 cells into adaptive
block groups or an L4 residual layer. The goal is to bound the number of
entry candidates per terminal routing unit. This is more robust than keeping
4096 cells while allowing thousands of blocks per cell.

Routing margin between the first and second choices can control additional
probes or portal insertion for boundary queries. It is an uncertainty signal,
not a replacement for block-set distance.

## 7. PRR Result

Certified PQ racing with the current Vogue `Br=4` codes is a no-go for the
uniform search path:

~~~text
best mean exact-work ratio vs fixed top-16: 2.563x
median ratio at ef=256, two seeds/block:     0.815x
p90 ratio:                                  7.607x
candidate-set agreement:                    100%
~~~

The bounds are safe, but a tail of unprunable blocks dominates mean work.
Do not spend more time tuning seed count. Revisit only for out-of-core I/O,
Br=8, or an easy/hard fallback policy after the main recall path works.

## 8. 1B Storage and Query Design

For SPACEV `d=100`, `Br=4`:

~~~text
PQ payload:       about 50 GB for 1B vectors
raw int8 vectors: about 100 GB
vector IDs:       about 4 GB
~~~

These payloads must be host/NVMe resident and loaded by logical region. The
following compact metadata should remain on GPU where capacity permits:

~~~text
tree centroids and entry tables
block graph adjacency
block prototypes/medoids
block -> code_region/raw_region/slot mapping
region directory and cache state
~~~

For 7.8M minimum-capacity blocks, a degree-32 graph is about 1 GB. Two 64D
FP16 prototypes are about 2 GB. Metadata residency is therefore plausible on
a 32 GB GPU, unlike the vector payload.

### Logical region layout

OS page IDs are transparent to HBlock. HBlock owns stable logical IDs:

~~~text
block_id -> code_region_id, code_slot
vector_id/block_id -> raw_region_id, raw_slot
~~~

Logical regions are stored in regular files or mmap. The OS maps those bytes
to physical pages. Region packing should place graph-neighbor and frequently
co-accessed blocks together, but graph edges remain global and may cross
regions.

Code and raw stores remain separate so PQ scanning never transfers raw vectors
that will not be reranked.

### Query pipeline

~~~text
Phase 0: resident tree + metadata graph traversal -> visited block IDs
Phase 1: map blocks to code regions, sort/RLE across query batch,
         fetch each missing region once, run per-block PQ top-p
Phase 2: map surviving IDs to raw regions, sort/RLE, fetch missing regions,
         exact L2 and per-query global top-k merge
~~~

Graph traversal may run completely before payload fetch only while its score
uses resident metadata. Code and raw transfers are processed in waves with
double buffering:

~~~text
copy wave i+1  ||  compute wave i
~~~

The scheduler groups work by `(region_id, query_id, block_id)`. Loading a
region once can serve multiple blocks and multiple queries. The benefit must
be measured under both random and locality-heavy query workloads.

### Required scale fixes

1. Replace the per-query bitmap over all blocks with an `ef`-proportional
   hash/set structure.
2. Avoid scanning thousands of blocks to select entries; use adaptive block
   groups and precomputed portal/entry metadata.
3. Build by external partitioning: stream encode, write cell shards, pack one
   shard at a time, and construct graph metadata incrementally.
4. Measure region overfetch, cache hit rate, useful blocks per loaded byte,
   H2D bandwidth, compute overlap, and tail latency. Do not infer locality
   only from graph distance.

## 9. Execution Order and Estimate

~~~text
1-2 days: correct oracle diagnostics and random-training control
3-5 days: first block-representation or graph fix, with ablation
~1 week:  credible SPACEV-100M recall-QPS curve
2-3 weeks: validated out-of-core region pipeline and initial 1B run
~~~

The schedule assumes no new correctness problem in the region-wave path. A
claim against BANG should wait until recall is matched, complete 100M/1B data
is used, and hardware plus batch/latency conditions are comparable.

## 10. Source of Truth

~~~text
HBLOCK_CURRENT_DESIGN.md         current architecture and roadmap
VERSIONS.md                      historical implementation record
results/hblock_v39_findings.md   region experiment evidence
results/timing_breakdown_all_versions.md historical profiling evidence
hblock_v40/README.md             v40 run and staging boundary
hblock_v41/README.md             v41 run and region-wave boundary
~~~

