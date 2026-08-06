// demo_hblock_v42: SPACEV .i8bin subset evaluation for v42 (v39's region
// pool, now backed by real mmap'd files on disk instead of an in-process
// host RAM buffer -- see hblock_v42/jhq_gpu_index.cuh's class comment).
// Same CLI shape as demo_hblock_v39.cu plus region_file_dir.
//
// Expects the big-ann-benchmarks-style SPACEV subset layout:
//   base.i8bin              -- I8BinReader format (8-byte header, then int8 payload)
//   query.i8bin              -- same format
//   ids.NNNM.i32bin          -- local->global id map for the base subset
//   groundtruth.NNK.i32bin   -- full-corpus top-K groundtruth, global ids
//
// search() prints per-call region fetch-planning diagnostics (request/
// unique-per-batch/unique-this-call/bytes-H2D/region_reuse) to stdout.
#include "hblock_v42/jhq_gpu_index.cuh"
#include "common/spacev_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* ids, int nq, int k, const RestrictedGT& gt)
{
    long long hits = 0;
    for (int qi = 0; qi < nq; qi++) {
        const int* row = ids + (long long)qi * k;
        const int32_t* gt_row = gt.ids.data() + (long long)qi * gt.k;
        for (int j = 0; j < k; j++) {
            int found = row[j];
            if (found < 0) continue;
            for (int g = 0; g < gt.k; g++)
                if (gt_row[g] == found) { hits++; break; }
        }
    }
    return (double)hits / (double)((long long)nq * k);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <base.i8bin> <query.i8bin> <ids.NNNM.i32bin> <groundtruth.NNK.i32bin>\n"
            "         [n_base=-1] [n_query=-1] [max_ef=128] [batch=256]\n"
            "         [region_bytes_mib=1] [gpu_code_region_cap=4096] [gpu_raw_region_cap=4096]\n"
            "         [K1=16] [K2=16] [K3=16] [graph_degree=32] [entry_per_cell=4]\n"
            "         [d_proj=64] [per_block_r=16] [region_file_dir=/tmp] [csv=]\n"
            "\n"
            "gpu_code_region_cap/gpu_raw_region_cap bound the GPU region pool; a single\n"
            "search() batch fails loudly if it touches more distinct regions than that --\n"
            "raise the cap, raise region_bytes_mib, or lower batch/max_ef if you hit it.\n"
            "\n"
            "region_file_dir is where the two real region files get written\n"
            "(hblock_v42_<pid>_codes.region / _raw.region) -- point it at fast local\n"
            "disk, not a slow network mount, or region fetch latency will dominate.\n",
            argv[0]);
        return 1;
    }

    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* ids_path   = argv[3];
    const char* gt_path    = argv[4];

    int n_base_req  = (argc >  5) ? atoi(argv[5])  : -1;
    int n_query_req = (argc >  6) ? atoi(argv[6])  : -1;
    int max_ef      = (argc >  7) ? atoi(argv[7])  : 128;
    int batch       = (argc >  8) ? atoi(argv[8])  : 256;
    int region_mib  = (argc >  9) ? atoi(argv[9])  : 1;
    int code_cap    = (argc > 10) ? atoi(argv[10]) : 4096;
    int raw_cap     = (argc > 11) ? atoi(argv[11]) : 4096;
    int K1          = (argc > 12) ? atoi(argv[12]) : 16;
    int K2          = (argc > 13) ? atoi(argv[13]) : 16;
    int K3          = (argc > 14) ? atoi(argv[14]) : 16;
    int deg         = (argc > 15) ? atoi(argv[15]) : 32;
    int epc         = (argc > 16) ? atoi(argv[16]) : 4;
    int dprj        = (argc > 17) ? atoi(argv[17]) : 64;
    int pbr         = (argc > 18) ? atoi(argv[18]) : 16;
    const char* region_dir = (argc > 19) ? argv[19] : "/tmp";
    const char* csv_path   = (argc > 20) ? argv[20] : nullptr;
    const int k = 10;

    I8BinReader base_reader;
    if (!base_reader.open(base_path)) {
        fprintf(stderr, "failed to open %s\n", base_path);
        return 1;
    }
    I8BinReader query_reader;
    if (!query_reader.open(query_path)) {
        fprintf(stderr, "failed to open %s\n", query_path);
        return 1;
    }

    int nb = (n_base_req  < 0) ? base_reader.npts  : std::min(n_base_req,  base_reader.npts);
    int nq = (n_query_req < 0) ? query_reader.npts : std::min(n_query_req, query_reader.npts);
    int d  = base_reader.dim;
    if (query_reader.dim != d) {
        fprintf(stderr, "base dim (%d) != query dim (%d)\n", d, query_reader.dim);
        return 1;
    }

    printf("base=%d x %d (of %d)  query=%d x %d (of %d)\n",
           nb, d, base_reader.npts, nq, d, query_reader.npts);
    printf("max_ef=%d  batch=%d  region_bytes=%dMiB  gpu_code_region_cap=%d  gpu_raw_region_cap=%d\n"
           "K1=%d K2=%d K3=%d  graph_degree=%d  entry_per_cell=%d  d_proj=%d  per_block_r=%d\n"
           "region_file_dir=%s\n",
           max_ef, batch, region_mib, code_cap, raw_cap,
           K1, K2, K3, deg, epc, dprj, pbr, region_dir);

    std::vector<int32_t> local_to_global = load_id_map(ids_path);
    if ((int)local_to_global.size() < nb) {
        fprintf(stderr, "id map has %zu entries, fewer than nb=%d\n", local_to_global.size(), nb);
        return 1;
    }
    if ((int)local_to_global.size() > nb) local_to_global.resize(nb);
    RestrictedGT gt = build_restricted_gt(gt_path, local_to_global, k, nq);

    hblock_v42::HBlockIndex::Params p;
    p.K1 = K1; p.K2 = K2; p.K3 = K3;
    p.max_ef = max_ef;
    p.d_proj = dprj; p.per_block_r = pbr;
    p.graph_degree = deg; p.entry_per_cell = epc;
    p.batch_size = batch;
    p.region_bytes = region_mib * (1 << 20);
    p.gpu_code_region_cap = code_cap;
    p.gpu_raw_region_cap  = raw_cap;
    p.region_file_dir = region_dir;

    hblock_v42::HBlockIndex idx(d, p);

    // train() still takes a float subsample directly (small: <=200K vectors,
    // never worth streaming) -- read it from the same base file and cast.
    int n_train = std::min(nb, 200000);
    printf("\nTraining on %d vectors...\n", n_train);
    std::vector<int8_t> train_i8;
    int got_train = base_reader.read_batch(0, n_train, train_i8);
    if (got_train != n_train) {
        fprintf(stderr, "short read for training subsample\n");
        return 1;
    }
    std::vector<float> train_f((long long)n_train * d);
    for (long long i = 0; i < (long long)n_train * d; i++) train_f[i] = (float)train_i8[i];
    auto t0 = Clock::now();
    idx.train(train_f.data(), n_train);
    printf("  train: %.1f ms\n", Ms(Clock::now()-t0).count());
    train_i8.clear(); train_i8.shrink_to_fit();
    train_f.clear();  train_f.shrink_to_fit();

    printf("Adding %d vectors (streamed)...\n", nb);
    t0 = Clock::now();
    idx.add(base_reader, nb);
    printf("  add: %.1f ms\n", Ms(Clock::now()-t0).count());

    std::vector<int8_t> h_query;
    int got_q = query_reader.read_batch(0, nq, h_query);
    if (got_q != nq) {
        fprintf(stderr, "short read for query set\n");
        return 1;
    }

    std::vector<float> h_dists((long long)nq * k);
    std::vector<int>   h_ids  ((long long)nq * k);

    std::vector<int> efs;
    for (int e = 8; e <= max_ef; e *= 2) efs.push_back(e);
    if (efs.empty() || efs.back() != max_ef) efs.push_back(max_ef);

    FILE* csv = csv_path ? fopen(csv_path, "w") : nullptr;
    if (csv) fprintf(csv, "ef,recall@10,qps,latency_ms\n");

    // Warmup: first search() incurs CUDA/cuBLAS init + first-touch region
    // fetch (cold page-cache misses against the mmap'd files) overhead;
    // keep it out of timing.
    idx.search(h_query.data(), nq, k, h_dists.data(), h_ids.data(), efs.back());

    printf("\n%-8s %-12s %-10s\n", "ef", "recall@10", "QPS");
    printf("%-8s %-12s %-10s\n", "----", "---------", "---");

    const int REPS = 3;
    for (int ef : efs) {
        t0 = Clock::now();
        for (int r = 0; r < REPS; r++)
            idx.search(h_query.data(), nq, k, h_dists.data(), h_ids.data(), ef);
        double ms  = Ms(Clock::now()-t0).count() / REPS;
        double rec = recall_at_k(h_ids.data(), nq, k, gt);
        double qps = nq / (ms / 1000.0);

        printf("%-8d %-12.4f %-10.0f  (%.2f ms)\n", ef, rec, qps, ms);
        if (csv) fprintf(csv, "%d,%.6f,%.1f,%.3f\n", ef, rec, qps, ms);
    }

    if (csv) fclose(csv);
    return 0;
}
