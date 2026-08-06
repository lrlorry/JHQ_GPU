// SPACEV .i8bin entry point for v41's bounded unified-region staging pool.
#include "hblock_v41/jhq_gpu_index.cuh"
#include "common/spacev_io.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* ids, int nq, int k, const RestrictedGT& gt)
{
    long long hits = 0;
    for (int qi = 0; qi < nq; ++qi) {
        const int* row = ids + (long long)qi * k;
        const int32_t* gt_row = gt.ids.data() + (long long)qi * gt.k;
        for (int j = 0; j < k; ++j) {
            const int found = row[j];
            if (found < 0) continue;
            for (int g = 0; g < gt.k; ++g) {
                if (gt_row[g] == found) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return (double)hits / (double)((long long)nq * k);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: %s <base.i8bin> <query.i8bin> <ids.NNNM.i32bin> <groundtruth.NNK.i32bin>\n"
            "         [nbase=-1] [nquery=-1] [max_ef=256] [K1=16] [K2=16] [K3=16]\n"
            "         [graph_degree=32] [entry_per_cell=4] [d_proj=64]\n"
            "         [per_block_r=16] [batch=32768] [reps=1]\n"
            "         [region_bytes_mib=1] [gpu_region_cap=512]\n"
            "         [csv=]\n",
            argv[0]);
        return 1;
    }

    I8BinReader base_reader;
    I8BinReader query_reader;
    if (!base_reader.open(argv[1])) {
        std::fprintf(stderr, "Cannot open SPACEV base: %s\n", argv[1]);
        return 1;
    }
    if (!query_reader.open(argv[2])) {
        std::fprintf(stderr, "Cannot open SPACEV query: %s\n", argv[2]);
        return 1;
    }
    if (base_reader.dim != query_reader.dim) {
        std::fprintf(stderr, "Base/query dimensions differ: %d vs %d\n",
                     base_reader.dim, query_reader.dim);
        return 1;
    }

    const int requested_nbase = (argc > 5) ? std::atoi(argv[5]) : -1;
    const int requested_nquery = (argc > 6) ? std::atoi(argv[6]) : -1;
    const int nbase = requested_nbase < 0 ? base_reader.npts :
                      std::min(requested_nbase, base_reader.npts);
    const int nquery = requested_nquery < 0 ? query_reader.npts :
                       std::min(requested_nquery, query_reader.npts);
    const int max_ef = (argc > 7) ? std::atoi(argv[7]) : 256;
    const int K1 = (argc > 8) ? std::atoi(argv[8]) : 16;
    const int K2 = (argc > 9) ? std::atoi(argv[9]) : 16;
    const int K3 = (argc > 10) ? std::atoi(argv[10]) : 16;
    const int degree = (argc > 11) ? std::atoi(argv[11]) : 32;
    const int entry_per_cell = (argc > 12) ? std::atoi(argv[12]) : 4;
    const int d_proj = (argc > 13) ? std::atoi(argv[13]) : 64;
    const int per_block_r = (argc > 14) ? std::atoi(argv[14]) : 16;
    const int batch = (argc > 15) ? std::atoi(argv[15]) : 32768;
    const int reps = (argc > 16) ? std::atoi(argv[16]) : 1;
    const int region_mib = (argc > 17) ? std::atoi(argv[17]) : 1;
    const int region_cap = (argc > 18) ? std::atoi(argv[18]) : 512;
    const char* csv_path = (argc > 19) ? argv[19] : nullptr;
    const int k = 10;

    std::vector<int32_t> local_to_global = load_id_map(argv[3]);
    if ((int)local_to_global.size() < nbase) {
        std::fprintf(stderr, "ID map has %zu entries, fewer than nbase=%d\n",
                     local_to_global.size(), nbase);
        return 1;
    }
    if ((int)local_to_global.size() > nbase) local_to_global.resize(nbase);
    RestrictedGT gt = build_restricted_gt(argv[4], local_to_global, k, nquery);
    const int nq = std::min(nquery, gt.nq);

    std::vector<int8_t> query;
    if (query_reader.read_batch(0, nq, query) != nq) {
        std::fprintf(stderr, "Short read from SPACEV query file\n");
        return 1;
    }

    const int ntrain = std::min(200000, nbase);
    std::vector<int8_t> train_i8;
    if (base_reader.read_batch(0, ntrain, train_i8) != ntrain) {
        std::fprintf(stderr, "Short read from SPACEV training prefix\n");
        return 1;
    }
    std::vector<float> train((long long)ntrain * base_reader.dim);
    std::transform(train_i8.begin(), train_i8.end(), train.begin(),
                   [](int8_t x) { return (float)x; });
    train_i8.clear();
    train_i8.shrink_to_fit();

    std::printf("base=%d x %d  query=%d x %d  gt=%d x %d\n",
                nbase, base_reader.dim, nq, query_reader.dim, gt.nq, gt.k);
    std::printf("max_ef=%d K1=%d K2=%d K3=%d degree=%d entry_per_cell=%d\n"
                "  d_proj=%d per_block_r=%d batch=%d reps=%d\n"
                "  region_bytes=%dMiB gpu_region_cap=%d\n",
                max_ef, K1, K2, K3, degree, entry_per_cell,
                d_proj, per_block_r, batch, reps, region_mib, region_cap);

    hblock_v41::HBlockIndex::Params p;
    p.K1 = K1; p.K2 = K2; p.K3 = K3;
    p.max_ef = max_ef;
    p.graph_degree = degree;
    p.entry_per_cell = entry_per_cell;
    p.d_proj = d_proj;
    p.per_block_r = per_block_r;
    p.batch_size = batch;
    p.region_bytes = region_mib * (1 << 20);
    p.gpu_region_cap = region_cap;

    hblock_v41::HBlockIndex index(base_reader.dim, p);

    std::printf("Training on %d vectors...\n", ntrain);
    auto t0 = Clock::now();
    index.train(train.data(), ntrain);
    std::printf("  train: %.1f ms\n", Ms(Clock::now() - t0).count());
    train.clear();
    train.shrink_to_fit();

    std::printf("Adding %d SPACEV vectors (streamed)...\n", nbase);
    t0 = Clock::now();
    index.add(base_reader, nbase);
    std::printf("  add: %.1f ms\n", Ms(Clock::now() - t0).count());

    std::vector<float> distances((long long)nq * k);
    std::vector<int> ids((long long)nq * k);
    std::vector<int> efs;
    for (int ef = 8; ef <= max_ef; ef *= 2) efs.push_back(ef);
    if (efs.empty() || efs.back() != max_ef) efs.push_back(max_ef);

    // Warm only CUDA/cuBLAS and the staged path. A full-query warmup would
    // stream the whole out-of-core working set before measurements begin.
    const int warm_nq = std::min(nq, 1024);
    index.search(query.data(), warm_nq, k,
                 distances.data(), ids.data(), efs.back());

    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "ef,recall@10,qps,latency_ms\n");
    std::printf("\n%-8s %-12s %-10s\n", "ef", "recall@10", "QPS");
    std::printf("%-8s %-12s %-10s\n", "----", "---------", "---");

    for (int ef : efs) {
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            index.search(query.data(), nq, k, distances.data(), ids.data(), ef);
        const double ms = Ms(Clock::now() - t0).count() / reps;
        const double recall = recall_at_k(ids.data(), nq, k, gt);
        const double qps = nq / (ms / 1000.0);
        std::printf("%-8d %-12.4f %-10.0f  (%.2f ms)\n", ef, recall, qps, ms);
        if (csv) std::fprintf(csv, "%d,%.6f,%.1f,%.3f\n", ef, recall, qps, ms);
    }
    if (csv) std::fclose(csv);
    return 0;
}
