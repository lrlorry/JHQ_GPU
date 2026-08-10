#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v36_2/search.cuh"
#include "common/spacev_io.cuh"

namespace hblock_v36_2 {

// HBlock v36_2: hblock_v38 (plain SPACEV-100M int8, fully GPU-resident, no
// region partitioning) with the balanced-kmeans re-clustering step inside
// add() -- splitting any routing cell with more than leaf_size_ vectors
// into ~leaf_size_-sized blocks -- replaced by a recursive bisection
// algorithm. Named v36_2 because the original code traces back unchanged
// to hblock_v36 and was never revisited for scale; it went unnoticed at
// vogue-768/arxiv-768 scale (hundreds of ms) and became the dominant cost
// at SPACEV-100M scale.
//
// History (see git log for the two intermediate, insufficient attempts):
// the original flat K-way split (K = ceil(N_cell/leaf_size_) sub-
// clusters via k-means, then a capacity-constrained greedy assignment
// over ALL N_cell*K distance pairs) is fundamentally O(N_cell^2/leaf_size_)
// -- K itself grows linearly with N_cell, so the "build all pairs, sort
// them" step is quadratic in cell size. On the real 100M SPACEV run the
// single largest oversized cell had N_cell=216310, i.e. ~365M pairs to
// sort -- moving the distance computation to GPU (attempt 1) and hoisting
// its allocations out of the per-cell loop (attempt 2) both left this
// quadratic term untouched and barely changed the measured per-cell rate.
//
// recursive_bisect_cell() instead repeatedly splits a cell in half (exact
// median split along a cheap "farthest-pair" direction estimate, so
// balance is guaranteed regardless of data distribution, not just
// approximate) until every piece is <= leaf_size_. Total cost is
// O(N_cell * d_proj * log(N_cell/leaf_size_)) -- no K-dependence, no
// pairwise sort -- which for the same 216310-vector outlier is ~150M
// float ops, not ~365M sorted pairs. This is intentionally plain CPU code,
// not GPU: once the algorithm is O(N log N) instead of O(N^2), the
// absolute work per cell is small enough that GPU kernel-launch/sync
// overhead (the actual lesson from the two earlier attempts) would cost
// more than it saves; each cell's bisection is independent of every other
// cell's, so cross-cell parallelism (not attempted here) is the more
// promising lever if this still isn't fast enough.
//
// One correctness note this version had to fix: recursive halving does
// NOT always produce exactly ceil(N_cell/leaf_size_) pieces (it can
// produce more, e.g. N_cell=641 with leaf_size_=128 yields 8 pieces via
// halving vs the ceil-formula's 6) -- add()'s downstream block-count
// bookkeeping, which used to independently recompute the ceil formula and
// relied on it exactly matching what balanced-kmeans produced, now reads
// the actual per-cell piece count instead.
//
// Same tree routing + block graph + per-block fixed top-16 exact rerank as
// v36/v38 -- this version changes how blocks get formed inside add() (and
// slightly how many, per the note above), not recall semantics.
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
        int add_batch_size = 2000000;  // v38: streamed add() batch size (vectors)
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
    // PQ-encode pipeline, while permanently storing the raw int8 vectors on
    // GPU (d_base_vecs_i8_) for exact rerank at query time.
    // n < 0 (default) means "use reader.npts"; otherwise min(n, reader.npts).
    void add(I8BinReader& reader, int n = -1);

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
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
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

    // v38: permanent GPU-resident raw base storage, int8 instead of float32.
    // Size (long long)ntotal_ * d_ bytes (~9.3 GiB for SPACEV-100M @ d=100).
    int8_t* d_base_vecs_i8_ = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();

    static void init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi);

    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    // Recursively bisects one oversized cell's points (cell_proj, N rows
    // of d_proj_ floats each, contiguous) into pieces of size <=
    // leaf_size_, appending each resulting piece's size to out_sizes in
    // left-to-right order. local_order (initially 0..N-1, indices into
    // cell_proj) is reordered in place so consecutive ranges match
    // consecutive out_sizes entries. See class comment for why this
    // replaced the GPU-accelerated flat K-way approach entirely (the
    // flat approach's complexity was the problem, not where it ran).
    void recursive_bisect_cell(const float* cell_proj, int N,
                               std::vector<int>& local_order, int lo, int hi,
                               std::vector<int>& out_sizes);

    void upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                      const std::vector<bool>& h_valid, int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);

    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);
};

} // namespace hblock_v36_2
