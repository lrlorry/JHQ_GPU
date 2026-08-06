// SIFT/BIGANN entry point for v41's bounded, wave-based code/raw region pools.
#include "hblock_v41/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* ids, const int* gt, int nq, int k, int gt_k)
{
    long long hits = 0;
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < k; ++j) {
            const int found = ids[(long long)i*k+j];
            for (int g = 0; g < gt_k; ++g) {
                if (gt[(long long)i*gt_k+g] == found) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return (double)hits / (double)((long long)nq*k);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <base.bvecs> <query.bvecs> <gt.ivecs>\n"
            "         [nbase=-1] [max_ef=256] [K1=16] [K2=16] [K3=16]\n"
            "         [graph_degree=32] [entry_per_cell=4] [d_proj=64]\n"
            "         [per_block_r=16] [batch=1024] [reps=3]\n"
            "         [region_bytes_mib=1] [gpu_code_region_cap=256] [gpu_raw_region_cap=256]\n"
            "         [csv=]\n"
            "\n"
            "The capacities bound GPU payload memory. Larger working sets are split into\n"
            "multiple region waves without changing candidate semantics.\n",
            argv[0]);
        return 1;
    }

    BVecsReader base_reader;
    BVecsReader query_reader;
    if (!base_reader.open(argv[1])) {
        std::fprintf(stderr, "Cannot open bvecs base: %s\n", argv[1]);
        return 1;
    }
    if (!query_reader.open(argv[2])) {
        std::fprintf(stderr, "Cannot open bvecs query: %s\n", argv[2]);
        return 1;
    }
    if (base_reader.dim != query_reader.dim) {
        std::fprintf(stderr, "Base/query dimensions differ: %d vs %d\n",
                     base_reader.dim, query_reader.dim);
        return 1;
    }

    const long long requested_n = (argc > 4) ? std::atoll(argv[4]) : -1;
    const long long nbase = requested_n < 0 ? base_reader.npts :
                            std::min(requested_n, base_reader.npts);
    const int max_ef = (argc > 5) ? std::atoi(argv[5]) : 256;
    const int K1 = (argc > 6) ? std::atoi(argv[6]) : 16;
    const int K2 = (argc > 7) ? std::atoi(argv[7]) : 16;
    const int K3 = (argc > 8) ? std::atoi(argv[8]) : 16;
    const int degree = (argc > 9) ? std::atoi(argv[9]) : 32;
    const int entry_per_cell = (argc > 10) ? std::atoi(argv[10]) : 4;
    const int d_proj = (argc > 11) ? std::atoi(argv[11]) : 64;
    const int per_block_r = (argc > 12) ? std::atoi(argv[12]) : 16;
    const int batch = (argc > 13) ? std::atoi(argv[13]) : 1024;
    const int reps = (argc > 14) ? std::atoi(argv[14]) : 3;
    const int region_mib = (argc > 15) ? std::atoi(argv[15]) : 1;
    const int code_cap   = (argc > 16) ? std::atoi(argv[16]) : 256;
    const int raw_cap    = (argc > 17) ? std::atoi(argv[17]) : 256;
    const char* csv_path = (argc > 18) ? argv[18] : nullptr;
    const int k = 10;

    std::vector<int> gt;
    int gt_k = 0;
    const int ngt = read_ivecs(argv[3], gt, gt_k);
    const int nq = (int)std::min<long long>(query_reader.npts, ngt);
    std::vector<int8_t> query;
    if (query_reader.read_batch(0, nq, query) != nq) {
        std::fprintf(stderr, "Short read from query bvecs\n");
        return 1;
    }

    const int ntrain = (int)std::min<long long>(200000, nbase);
    std::vector<int8_t> train_i8;
    if (base_reader.read_batch(0, ntrain, train_i8) != ntrain) {
        std::fprintf(stderr, "Short read from training prefix\n");
        return 1;
    }
    std::vector<float> train((long long)ntrain*base_reader.dim);
    std::transform(train_i8.begin(), train_i8.end(), train.begin(),
                   [](int8_t x) { return (float)x; });
    train_i8.clear();
    train_i8.shrink_to_fit();

    std::printf("base=%lld x %d  query=%d x %d  gt=%d x %d\n",
                nbase, base_reader.dim, nq, query_reader.dim, ngt, gt_k);
    std::printf("max_ef=%d K1=%d K2=%d K3=%d degree=%d entry_per_cell=%d\n"
                "  d_proj=%d per_block_r=%d batch=%d reps=%d\n"
                "  region_bytes=%dMiB gpu_code_region_cap=%d gpu_raw_region_cap=%d\n",
                max_ef, K1, K2, K3, degree, entry_per_cell,
                d_proj, per_block_r, batch, reps,
                region_mib, code_cap, raw_cap);

    hblock_v41::HBlockIndex::Params p;
    p.K1=K1; p.K2=K2; p.K3=K3;
    p.max_ef=max_ef;
    p.graph_degree=degree;
    p.entry_per_cell=entry_per_cell;
    p.d_proj=d_proj;
    p.per_block_r=per_block_r;
    p.batch_size=batch;
    p.region_bytes = region_mib * (1 << 20);
    p.gpu_code_region_cap = code_cap;
    p.gpu_raw_region_cap  = raw_cap;

    hblock_v41::HBlockIndex index(base_reader.dim, p);

    std::printf("Training on %d vectors...\n", ntrain);
    auto t0 = Clock::now();
    index.train(train.data(), ntrain);
    std::printf("  train: %.1f ms\n", Ms(Clock::now()-t0).count());
    train.clear();
    train.shrink_to_fit();

    std::printf("Adding %lld vectors from bvecs (streamed)...\n", nbase);
    t0 = Clock::now();
    index.add(base_reader, (int)nbase);
    std::printf("  add: %.1f ms\n", Ms(Clock::now()-t0).count());

    std::vector<float> distances((long long)nq*k);
    std::vector<int> ids((long long)nq*k);
    std::vector<int> efs;
    for (int ef=8; ef<=max_ef; ef*=2) efs.push_back(ef);
    if (efs.empty() || efs.back()!=max_ef) efs.push_back(max_ef);

    // Warmup: first search() incurs CUDA/cuBLAS init + first-touch region
    // fetch overhead; keep it out of timing.
    index.search(query.data(), nq, k, distances.data(), ids.data(), efs.back());

    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "ef,recall@10,qps,latency_ms\n");
    std::printf("\n%-8s %-12s %-10s\n", "ef", "recall@10", "QPS");
    std::printf("%-8s %-12s %-10s\n", "----", "---------", "---");

    for (int ef : efs) {
        t0 = Clock::now();
        for (int r=0; r<reps; ++r)
            index.search(query.data(), nq, k, distances.data(), ids.data(), ef);
        const double ms = Ms(Clock::now()-t0).count()/reps;
        const double recall = recall_at_k(ids.data(), gt.data(), nq, k, gt_k);
        const double qps = nq/(ms/1000.0);
        std::printf("%-8d %-12.4f %-10.0f  (%.2f ms)\n", ef, recall, qps, ms);
        if (csv) std::fprintf(csv, "%d,%.6f,%.1f,%.3f\n",
                              ef, recall, qps, ms);
    }
    if (csv) std::fclose(csv);
    return 0;
}
