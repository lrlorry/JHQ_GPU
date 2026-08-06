#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <list>
#include <vector>

#include "hblock_v39/search.cuh"
#include "common/spacev_io.cuh"
#include "common/bvecs_io.cuh"

namespace hblock_v39 {

// HBlock v39: v38 (SPACEV-100M int8 baseline) + logical region partitioning
// and a bounded GPU region pool. PQ codes and raw int8 vectors are NOT held
// as flat permanently-GPU-resident arrays; add() packs them into fixed-size
// logical regions (block-major layout, region = contiguous group of blocks)
// in a host-resident region store, and search() plans + stages only the
// regions a query batch actually needs into a capacity-bounded GPU region
// pool via an indirection table (region_id -> pool slot, -1 if absent).
// Same tree routing + block graph + per-block fixed top-16 exact rerank as
// v36/v38 — this version changes payload residency and addressing only, not
// recall semantics.
// ef maps internally to (graph_depth=ef, beam_size=ef, no cap).
// Routing fixed: ck1=2, ck2=2, ck3=4.
class HBlockIndex {
public:
    struct Params {
        int K1           = 16;
        int K2           = 16;
        int K3           = 16;
        int Kr           = 16;
        int Br           = 4;
        int leaf_size    = 128;
        int ck1          = 2;
        int ck2          = 2;
        int ck3          = 4;
        int d_proj       = 64;
        int per_block_r  = 16;
        int klocal       = 10;
        int km_iters     = 30;
        int batch_size   = 1024;   // query search batch
        int graph_degree   = 32;
        int max_ef         = 256;  // sets workspace buffer sizes; must be >= any ef passed to search()
        int entry_per_cell = 4;
        int n_c2_nbrs      = 4;
        int n_c1_nbrs      = 2;
        int max_cand_blocks = 2048;
        int mini_km_iters  = 5;
        int add_batch_size = 2000000;  // streamed add() batch size (vectors)

        // ── v39: logical region partitioning ────────────────────────────
        int region_bytes        = 1 << 20;  // 1 MiB logical region size (partition granularity)
        int gpu_code_region_cap = 512;       // max resident code regions in the GPU pool
        int gpu_raw_region_cap  = 512;       // max resident raw regions in the GPU pool
    };

    // Diagnostics returned by diagnose_missed_gt(): routing-level metrics
    // computed from the *restricted* ground truth (see common/spacev_io.cuh),
    // no raw-vector access required.
    struct RoutingDiag {
        double routing_recall = 0.0;  // fraction of (query,gt) pairs whose designated
                                       // routing cell WAS selected by L1/L2/L3 routing
                                       // (i.e. NOT a pure routing miss == cnt_A)
        double graph_coverage = 0.0;  // fraction of restricted-GT ids present among the
                                       // union of vector-ids across all blocks visited by
                                       // beam search, averaged over evaluated queries
        long long cnt_total = 0, cnt_found = 0, cnt_A = 0, cnt_B = 0, cnt_C = 0;
        int n_eval = 0;                // queries actually contributing to graph_coverage
                                        // (num_survivors > 0)
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train(const float* h_x, int n_train);

    // Streamed add(): reads the base vectors from `reader` in add_batch_size
    // chunks (never materializes the full n*d array as float), casting each
    // batch to float transiently on GPU to feed the existing v36 routing +
    // PQ-encode pipeline. Raw int8 vectors and PQ codes are packed into
    // block-major logical regions in a host-resident region store (see
    // build_region_layout()) rather than uploaded as permanent flat
    // GPU-resident arrays; search() stages only the regions it needs.
    // n < 0 (default) means "use reader.npts"; otherwise min(n, reader.npts).
    void add(I8BinReader& reader, int n = -1);

    // Same streamed path, reading SIFT/BIGANN .bvecs instead (see
    // common/bvecs_io.cuh — BVecsReader already applies the uint8->int8
    // -128 shift, the same convention hblock_v40 uses, so results are
    // directly comparable). Lets v39's region-partition work run against
    // SIFT100M immediately, without waiting on a SPACEV .i8bin conversion.
    void add(BVecsReader& reader, int n = -1);

    // ef -> depth=ef, beam=ef (no cap, v35/v36 semantics)
    // h_q: raw int8 query vectors [nq * d_]; cast to float per search-batch
    // internally before feeding the (unchanged, float-based) routing kernels.
    void search(const int8_t* h_q, int nq, int k,
                float* h_dists, int* h_ids, int ef = 64) const;

    // Routing-level diagnostics (no raw-vector access): routing_recall +
    // graph_coverage against a restricted ground truth (local ids, -1 padded,
    // see common/spacev_io.cuh::RestrictedGT::ids/num_survivors). gt_k is the
    // stride of h_gt (== RestrictedGT::k); ef selects the beam width used for
    // the diagnostic traversal (independent of any prior search() call).
    // If verbose, also prints the full v36-style missed-GT breakdown.
    RoutingDiag diagnose_missed_gt(const int8_t* h_q, int nq, int k,
                                   const int32_t* h_gt, int gt_k, int ef,
                                   bool verbose = false) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int per_block_r_, klocal_, km_iters_, batch_size_;
    int graph_degree_, max_ef_, entry_per_cell_;
    int n_c2_nbrs_, n_c1_nbrs_, max_cand_blocks_;
    int mini_km_iters_, add_batch_size_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    float* d_Pi1_ = nullptr;
    float* d_Pi2_ = nullptr;
    float* d_Pi3_ = nullptr;

    float* d_route1_cents_proj_ = nullptr;
    float* d_route1_cents_full_ = nullptr;
    float* d_route1_norms_      = nullptr;
    float* d_route2_cents_proj_ = nullptr;
    float* d_route2_cents_full_ = nullptr;
    float* d_route2_norms_      = nullptr;
    float* d_route3_cents_proj_ = nullptr;
    float* d_route3_cents_full_ = nullptr;
    float* d_route3_norms_      = nullptr;

    float* d_fine_c1d_ = nullptr;

    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    // block_sizes is small (n_leaf_blocks_ ints) and stays resident as
    // block-to-region metadata; PQ codes, vector ids, and raw vectors do
    // NOT — see the v39 region layer below.
    int*     d_leaf_sizes_     = nullptr;

    std::vector<float> h_block_cent_;
    std::vector<float> h_block_cent_proj_;
    std::vector<float> h_block_cent_norm_;
    std::vector<float> h_Pi_blk_;
    std::vector<int>   h_block_cell_id_;
    std::vector<int>   h_block_adj_;
    std::vector<float> h_leaf_abs_cents_;
    std::vector<int>   h_pair_blk_start_cpu_;
    std::vector<int>   h_pair_blk_count_cpu_;

    std::vector<int>   h_leaf_ids_cpu_;
    std::vector<int>   h_leaf_sizes_cpu_;

    int*   d_block_adj_gpu_  = nullptr;
    float* d_blk_proj_gpu_   = nullptr;
    float* d_blk_norm_gpu_   = nullptr;

    // ── v39: logical region partitioning + bounded GPU region pool ───────
    // Design invariant: navigation metadata above stays permanently
    // GPU-resident; PQ codes + vector ids ("code" payload) and raw int8
    // vectors ("raw" payload) do not. add() packs both into block-major
    // logical regions of ~region_bytes_ each; search() plans + stages only
    // the regions a batch actually needs into a capacity-bounded GPU pool.
    int region_bytes_        = 0;
    int gpu_code_region_cap_ = 0;
    int gpu_raw_region_cap_  = 0;
    int n_code_regions_      = 0;
    int n_raw_regions_       = 0;

    // block_id -> region placement (built once in build_region_layout(),
    // called from add() after leaf packing). Host copies drive fetch
    // planning; device copies let the leaf kernel resolve addresses through
    // the pool without any host round-trip per query.
    std::vector<int> h_block_code_region_;   // [n_leaf_blocks_]
    std::vector<int> h_block_code_offset_;   // [n_leaf_blocks_] byte offset of (codes+ids) within its region
    std::vector<int> h_block_raw_region_;    // [n_leaf_blocks_]
    std::vector<int> h_block_raw_offset_;    // [n_leaf_blocks_] byte offset of raw vectors within its region
    int* d_block_code_region_ = nullptr;
    int* d_block_code_offset_ = nullptr;
    int* d_block_raw_region_  = nullptr;
    int* d_block_raw_offset_  = nullptr;

    // original vector id -> local position (0..leaf_size_-1) within its
    // home block. Lets the leaf kernel locate a candidate's raw vector
    // inside the block-major raw region store without a second,
    // id-indexed region table (a candidate id belongs to exactly one block,
    // which the kernel already knows).
    std::vector<int> h_vec_local_pos_;       // [ntotal_]
    int* d_vec_local_pos_ = nullptr;

    // Host-resident region stores: stand in for the eventual mmap'd
    // hblock_codes.region / hblock_raw.region files. Same addressing scheme
    // either way, so the file-backed version can replace this later without
    // touching the GPU pool or fetch-planning code.
    std::vector<uint8_t> h_code_region_store_;  // n_code_regions_ * region_bytes_
    std::vector<uint8_t> h_raw_region_store_;   // n_raw_regions_  * region_bytes_

    // Bounded GPU region pool + indirection (region_id -> pool slot,
    // -1 = not resident). Capacity is deliberately bounded below what would
    // fit the whole store, so the fetch/evict path is always exercised
    // rather than degenerating into "everything is resident anyway".
    uint8_t* d_code_region_pool_ = nullptr;   // [gpu_code_region_cap_ * region_bytes_]
    uint8_t* d_raw_region_pool_  = nullptr;   // [gpu_raw_region_cap_  * region_bytes_]
    // Mutable: fetch_code_regions()/fetch_raw_regions() update this
    // bookkeeping from search() (const, matching the existing public
    // interface) — same rationale as ws_ below, this is cache state, not
    // index identity.
    mutable std::vector<int> h_code_region_slot_;     // [n_code_regions_], -1 or pool slot idx
    mutable std::vector<int> h_raw_region_slot_;      // [n_raw_regions_]
    int* d_code_region_slot_ = nullptr;
    int* d_raw_region_slot_  = nullptr;
    mutable std::vector<int> code_pool_region_of_slot_;  // [gpu_code_region_cap_], -1 or resident region id
    mutable std::vector<int> raw_pool_region_of_slot_;   // [gpu_raw_region_cap_]
    // std::list (not vector) so move-to-front / evict-tail are O(1) once
    // the node is found; find is a linear scan but capacity is small
    // (hundreds of regions), not the 1B-scale block/region count.
    mutable std::list<int> code_lru_;               // resident code region ids, most-recently-used first
    mutable std::list<int> raw_lru_;                // resident raw region ids, most-recently-used first
    // region_id -> iterator into code_lru_/raw_lru_, for O(1) erase on hit
    // instead of a linear std::find over the LRU list.
    mutable std::vector<std::list<int>::iterator> code_lru_pos_;  // [n_code_regions_]
    mutable std::vector<std::list<int>::iterator> raw_lru_pos_;   // [n_raw_regions_]

    // Running transfer counters (reset per search() call) for reporting.
    // stat_*_unique_ sums each *internal* qstart-batch's own within-batch
    // dedup count -- the same region touched by two different internal
    // batches (batch_size_ queries each) is counted twice. That is the
    // right denominator for "how many (batch, region) touch events did
    // fetch planning have to resolve", but it is NOT "how many distinct
    // regions this whole search() call needed" -- see stat_*_true_unique_.
    mutable long long stat_code_bytes_h2d_ = 0;
    mutable long long stat_raw_bytes_h2d_  = 0;
    mutable long long stat_code_region_reqs_ = 0, stat_code_region_unique_ = 0;
    mutable long long stat_raw_region_reqs_  = 0, stat_raw_region_unique_  = 0;
    // True cross-internal-batch-deduped counts for this search() call --
    // the number actually comparable to the plan docs' region_reuse metric
    // (total_query_block_tasks / unique_regions, "unique" meaning across
    // the whole batch, not per sub-batch).
    mutable long long stat_code_region_true_unique_ = 0;
    mutable long long stat_raw_region_true_unique_  = 0;
    mutable long long call_id_ = 0;
    mutable std::vector<long long> code_region_last_call_;  // [n_code_regions_], -1 or call_id_ last seen
    mutable std::vector<long long> raw_region_last_call_;   // [n_raw_regions_]

    // Packs PQ codes+ids and raw vectors into block-major logical regions
    // (host-resident stores) and allocates the bounded GPU pool + slot
    // indirection. Called once from add(), after leaf packing produces
    // h_leaf_codes/h_leaf_ids/h_leaf_sizes and h_raw_all holds every raw
    // vector indexed by original id.
    void build_region_layout(const std::vector<uint8_t>& h_leaf_codes,
                              const std::vector<int>& h_leaf_ids,
                              const std::vector<int>& h_leaf_sizes,
                              const std::vector<int8_t>& h_raw_all,
                              int total_blocks);

    // Ensures every region in `needed` is resident in the GPU pool
    // (LRU-evicting as needed) and up to date in the slot indirection
    // table. Returns bytes actually copied H2D (0 if already resident) —
    // an real transfer count, not an estimate.
    long long fetch_code_regions(const std::vector<int>& needed_region_ids) const;
    long long fetch_raw_regions (const std::vector<int>& needed_region_ids) const;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();

    // Shared body for both add() overloads -- I8BinReader and BVecsReader
    // expose the same read_batch(start, count, out)/dim/npts shape, so one
    // template covers both without duplicating the ~150-line encode loop.
    // Defined and both instantiated in jhq_gpu_index.cu; never called
    // outside that translation unit.
    template<class Reader>
    void add_impl(Reader& reader, int n);

    static void init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi);

    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    void upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                      const std::vector<bool>& h_valid, int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);

    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);
};

} // namespace hblock_v39
