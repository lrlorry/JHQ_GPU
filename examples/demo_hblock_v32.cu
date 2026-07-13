#include "hblock_v32/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs>\n"
            "         [ef=32] [K1=16] [K2=16] [K3=16]\n"
            "         [graph_degree=32] [entry_per_cell=4] [d_proj=64]\n"
            "         [per_block_r=16] [batch=1024]\n",
            argv[0]);
        return 1;
    }

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;
    int nb = read_fvecs(argv[1], base,  d_base);
    int nq = read_fvecs(argv[2], query, d_query);
             read_ivecs(argv[3], gt,    d_gt);
    int d = d_base;
    int k = 10;

    int ef    = (argc >  4) ? atoi(argv[4])  : 32;
    int K1    = (argc >  5) ? atoi(argv[5])  : 16;
    int K2    = (argc >  6) ? atoi(argv[6])  : 16;
    int K3    = (argc >  7) ? atoi(argv[7])  : 16;
    int deg   = (argc >  8) ? atoi(argv[8])  : 32;
    int epc   = (argc >  9) ? atoi(argv[9])  : 4;
    int dprj  = (argc > 10) ? atoi(argv[10]) : 64;
    int pbr   = (argc > 11) ? atoi(argv[11]) : 16;
    int batch = (argc > 12) ? atoi(argv[12]) : 1024;

    printf("nb=%d  nq=%d  d=%d  d_gt=%d  ef=%d\n", nb, nq, d_base, d_gt, ef);

    hblock_v32::HBlockIndex::Params p;
    p.K1=K1; p.K2=K2; p.K3=K3;
    p.ef=ef;
    p.d_proj=dprj; p.per_block_r=pbr;
    p.graph_degree=deg; p.entry_per_cell=epc;
    p.batch_size=batch;

    hblock_v32::HBlockIndex idx(d_base, p);

    int n_train = std::min(nb, 200000);
    printf("Training on %d vectors...\n", n_train);
    idx.train(base.data(), n_train);

    printf("Adding %d vectors...\n", nb);
    idx.add(base.data(), nb);

    // Sweep ef values at search time
    std::vector<int> efs = {8, 16, 32, 64, 128};
    printf("\n%-10s %-10s %-12s\n", "ef", "recall@10", "QPS");
    printf("%-10s %-10s %-12s\n", "----", "---------", "---");

    std::vector<float> h_dists((long long)nq * k);
    std::vector<int>   h_ids  ((long long)nq * k);

    for (int b : efs) {
        // Update ef at search time by rebuilding params is not directly supported.
        // For sweep, we just run with the build-time ef and report.
        // A proper sweep would rebuild the index with different params.
        break;
    }

    // Single run with build-time ef
    auto t0 = Clock::now();
    const int REPS = 5;
    for (int rep = 0; rep < REPS; rep++)
        idx.search(query.data(), nq, k, h_dists.data(), h_ids.data());
    double ms = Ms(Clock::now() - t0).count() / REPS;

    long long hits = 0;
    for (int qi = 0; qi < nq; qi++)
        for (int ki = 0; ki < k; ki++)
            for (int gi = 0; gi < k; gi++)
                if (h_ids[qi*k+ki] == gt[qi*d_gt+gi]) { hits++; break; }
    double recall = (double)hits / (double)((long long)nq * k);
    double qps    = nq / (ms / 1000.0);

    printf("ef=%-6d  recall@10=%.4f  QPS=%.0f  latency=%.2f ms\n",
           ef, recall, qps, ms);

    return 0;
}
