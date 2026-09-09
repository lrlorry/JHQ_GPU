// Can cuVS's ivf_rabitq find a vector that is in its own index?
//
// results/rabitq/README.md rules out the reader, the ground truth, the metric,
// the row count, the search mode, bits_per_dim and the architecture, and
// linking the fork's own libivf_rabitq.a instead of the wheel's libcuvs.so
// changes nothing: recall stays 0.0000 with distances below the true minimum.
//
// Every one of those checks still went through a real dataset on disk. This
// one does not. The data is generated here, the queries ARE rows of the index,
// and the expected answer is the row's own id. If the top-1 is not the query
// itself, the library cannot answer the easiest question there is on this
// card, and nothing about our call site is left to blame.
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cstdlib>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/copy.hpp>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

int main(int argc, char** argv) {
    const int  N     = (argc > 1) ? std::atoi(argv[1]) : 50000;
    const int  d     = (argc > 2) ? std::atoi(argv[2]) : 128;
    const int  NQ    = 20;
    const int  k     = 10;
    const uint32_t nlist  = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 256;
    const uint32_t bits   = (argc > 4) ? (uint32_t)std::atoi(argv[4]) : 8;
    const uint32_t nprobe = (argc > 5) ? (uint32_t)std::atoi(argv[5]) : nlist;  // probe everything

    std::printf("N=%d d=%d nlist=%u bits=%u nprobe=%u (all lists probed)\n",
                N, d, nlist, bits, nprobe);

    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> xb((size_t)N * d);
    for (auto& v : xb) v = g(rng);

    raft::device_resources res;
    auto d_base = raft::make_device_matrix<float, int64_t>(res, N, d);
    raft::copy(d_base.data_handle(), xb.data(), (size_t)N * d,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    cuvs::neighbors::ivf_rabitq::index_params ip;
    ip.n_lists = nlist; ip.bits_per_dim = bits;
    ip.metric = cuvs::distance::DistanceType::L2Expanded;
    auto built = cuvs::neighbors::ivf_rabitq::build(
        res, ip, raft::make_device_matrix_view<const float, int64_t>(
                     d_base.data_handle(), N, d));
    raft::resource::sync_stream(res);

    // cuVS's own test (cpp/tests/neighbors/ann_ivf_rabitq.cuh) round-trips the
    // index through serialize/deserialize with the comment "reorganize data
    // for efficient search". So build() does not leave the codes in the layout
    // search reads, which is what ids that are essentially random beside
    // distances below the true minimum look like. JHQ_RQ_NO_ROUNDTRIP=1
    // searches the built index directly, so both are measured here.
    const bool roundtrip = !(std::getenv("JHQ_RQ_NO_ROUNDTRIP"));
    cuvs::neighbors::ivf_rabitq::index<int64_t> loaded(res);
    if (roundtrip) {
        const char* f = "/tmp/rq_selftest.idx";
        cuvs::neighbors::ivf_rabitq::serialize(res, f, built);
        cuvs::neighbors::ivf_rabitq::deserialize(res, f, &loaded);
        raft::resource::sync_stream(res);
    }
    auto& idx = roundtrip ? loaded : built;
    std::printf("roundtrip=%d  index size=%lld dim=%u\n",
                (int)roundtrip, (long long)idx.size(), idx.dim());

    // Queries are rows 0, 1000, 2000, ... of the index itself.
    std::vector<float> xq((size_t)NQ * d);
    std::vector<int>   want(NQ);
    for (int i = 0; i < NQ; ++i) {
        want[i] = i * (N / NQ);
        for (int j = 0; j < d; ++j) xq[(size_t)i * d + j] = xb[(size_t)want[i] * d + j];
    }
    auto d_q  = raft::make_device_matrix<float,   int64_t>(res, NQ, d);
    auto d_id = raft::make_device_matrix<int64_t, int64_t>(res, NQ, k);
    auto d_di = raft::make_device_matrix<float,   int64_t>(res, NQ, k);
    raft::copy(d_q.data_handle(), xq.data(), (size_t)NQ * d,
               raft::resource::get_cuda_stream(res));

    cuvs::neighbors::ivf_rabitq::search_params sp;
    sp.n_probes = nprobe;
    cuvs::neighbors::ivf_rabitq::search(res, sp, idx, d_q.view(),
                                        d_id.view(), d_di.view());
    raft::resource::sync_stream(res);

    std::vector<int64_t> h_id((size_t)NQ * k);
    std::vector<float>   h_di((size_t)NQ * k);
    raft::copy(h_id.data(), d_id.data_handle(), (size_t)NQ * k,
               raft::resource::get_cuda_stream(res));
    raft::copy(h_di.data(), d_di.data_handle(), (size_t)NQ * k,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    int self_top1 = 0, self_topk = 0;
    for (int i = 0; i < NQ; ++i) {
        if (h_id[(size_t)i * k] == want[i]) ++self_top1;
        for (int j = 0; j < k; ++j) if (h_id[(size_t)i * k + j] == want[i]) { ++self_topk; break; }
    }
    std::printf("self in top-1 : %d / %d\nself in top-%d: %d / %d\n",
                self_top1, NQ, k, self_topk, NQ);
    // A query that IS a database row sits at distance 0 from itself. Anything
    // the search reports below that is not a distance in this data.
    for (int i = 0; i < 3; ++i) {
        std::printf("q%-2d want id %-7d  got", i, want[i]);
        for (int j = 0; j < 5; ++j)
            std::printf("  %lld(%.4f)", (long long)h_id[(size_t)i * k + j], h_di[(size_t)i * k + j]);
        std::printf("\n");
    }
    return 0;
}
