#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v41/search.cuh"
#include "common/spacev_io.cuh"
#include "common/bvecs_io.cuh"

namespace hblock_v41 {

// HBlock v41: v38 search semantics with block-major logical regions and a
// bounded GPU staging pool. Region waves change addresses, not candidates.
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
        int mini_km_max_cell = 4096;
        int add_batch_size = 262144;   // streamed add() batch size (vectors)

        // ── v41: logical region partitioning ────────────────────────────
        int region_bytes        = 1 << 20;  // 1 MiB logical region size (partition granularity)
        int gpu_region_cap      = 512;       // maximum regions staged in one wave
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
    // directly comparable). Lets v41's region-streaming path run against
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
    int mini_km_iters_, mini_km_max_cell_, add_batch_size_;
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
    // NOT — see the v41 region layer below.
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

    int*   d_block_adj_gpu_  = nullptr;
    float* d_blk_proj_gpu_   = nullptr;
    float* d_blk_norm_gpu_   = nullptr;

    // ── v41: complete block records in logical regions ──────────────────
    // Record layout: [PQ codes | vector ids | raw int8 vectors].
    int region_bytes_       = 0;
    int gpu_region_cap_     = 0;
    int n_regions_          = 0;
    int block_record_bytes_ = 0;
    int block_ids_offset_   = 0;
    int block_raw_offset_   = 0;

    // block_id -> region placement (built once in build_region_layout(),
    // called from add() after leaf packing). Host copies drive fetch
    // planning; device copies let the leaf kernel resolve addresses through
    // the pool without any host round-trip per query.
    std::vector<int> h_block_region_;
    std::vector<int> h_block_offset_;
    int* d_block_region_ = nullptr;
    int* d_block_offset_ = nullptr;

    // Host-resident region stores. They bound GPU memory, but are not yet
    // mmap-backed; a later version can replace these vectors without changing
    // device addressing or the wave planner.
    std::vector<uint8_t> h_region_store_;

    uint8_t* d_region_pool_ = nullptr;
    int* d_region_slot_ = nullptr;
    uint8_t* h_region_stage_ = nullptr;
    mutable std::vector<int> h_region_slot_;
    mutable std::vector<int> staged_regions_;

    mutable long long stat_bytes_h2d_ = 0;
    mutable long long stat_region_reqs_ = 0;

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

    // Replaces the staging pool contents with one immutable region wave.
    long long stage_regions(const std::vector<int>& region_ids) const;

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

} // namespace hblock_v41
