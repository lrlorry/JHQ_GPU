#include "hblock_v34/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* ids, const int* gt, int nq, int k, int gt_k)
{
    int hits = 0;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < k; j++) {
            int found = ids[i * k + j];
            for (int g = 0; g < gt_k; g++)
                if (gt[i * gt_k + g] == found) { hits++; break; }
        }
    return (double)hits / (double)(nq * k);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs>\n"
            "         [max_ef=128] [K1=16] [K2=16] [K3=16]\n"
            "         [graph_degree=32] [entry_per_cell=4] [d_proj=64]\n"
            "         [per_block_r=16] [batch=1024] [csv=]\n",
            argv[0]);
        return 1;
    }

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;
    int nb = read_fvecs(argv[1], base,  d_base);
    int nq = read_fvecs(argv[2], query, d_query);
             read_ivecs(argv[3], gt,    d_gt);
    int k = 10;

    int max_ef      = (argc >  4) ? atoi(argv[4])  : 128;
    int K1          = (argc >  5) ? atoi(argv[5])  : 16;
    int K2          = (argc >  6) ? atoi(argv[6])  : 16;
    int K3          = (argc >  7) ? atoi(argv[7])  : 16;
    int deg         = (argc >  8) ? atoi(argv[8])  : 32;
    int epc         = (argc >  9) ? atoi(argv[9])  : 4;
    int dprj        = (argc > 10) ? atoi(argv[10]) : 64;
    int pbr         = (argc > 11) ? atoi(argv[11]) : 16;
    int batch       = (argc > 12) ? atoi(argv[12]) : 1024;
    const char* csv_path = (argc > 13) ? argv[13] : nullptr;

    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n",
           nb, d_base, nq, d_query, (int)(gt.size()/d_gt), d_gt);
    printf("max_ef=%d  K1=%d K2=%d K3=%d  graph_degree=%d  entry_per_cell=%d\n"
           "  d_proj=%d  per_block_r=%d  batch=%d\n",
           max_ef, K1, K2, K3, deg, epc, dprj, pbr, batch);

    hblock_v34::HBlockIndex::Params p;
    p.K1=K1; p.K2=K2; p.K3=K3;
    p.max_ef=max_ef;
    p.d_proj=dprj; p.per_block_r=pbr;
    p.graph_degree=deg; p.entry_per_cell=epc;
    p.batch_size=batch;

    hblock_v34::HBlockIndex idx(d_base, p);

    int n_train = std::min(nb, 200000);
    printf("\nTraining on %d vectors...\n", n_train);
    auto t0 = Clock::now();
    idx.train(base.data(), n_train);
    printf("  train: %.1f ms\n", Ms(Clock::now()-t0).count());

    printf("Adding %d vectors...\n", nb);
    t0 = Clock::now();
    idx.add(base.data(), nb);
    printf("  add: %.1f ms\n", Ms(Clock::now()-t0).count());

    std::vector<float> h_dists((long long)nq * k);
    std::vector<int>   h_ids  ((long long)nq * k);

    // Sweep ef at search time — no rebuild needed
    std::vector<int> efs;
    for (int e = 8; e <= max_ef; e *= 2) efs.push_back(e);
    if (efs.empty() || efs.back() != max_ef) efs.push_back(max_ef);

    FILE* csv = csv_path ? fopen(csv_path, "w") : nullptr;
    if (csv) fprintf(csv, "ef,recall@10,qps,latency_ms\n");

    // Warmup: first search() incurs CUDA/cuBLAS init overhead; keep it out of timing.
    idx.search(query.data(), nq, k, h_dists.data(), h_ids.data(), efs.back());

    printf("\n%-8s %-12s %-10s\n", "ef", "recall@10", "QPS");
    printf("%-8s %-12s %-10s\n", "----", "---------", "---");

    const int REPS = 5;
    for (int ef : efs) {
        t0 = Clock::now();
        for (int r = 0; r < REPS; r++)
            idx.search(query.data(), nq, k, h_dists.data(), h_ids.data(), ef);
        double ms  = Ms(Clock::now()-t0).count() / REPS;
        double rec = recall_at_k(h_ids.data(), gt.data(), nq, k, d_gt);
        double qps = nq / (ms / 1000.0);

        printf("%-8d %-12.4f %-10.0f  (%.2f ms)\n", ef, rec, qps, ms);
        if (csv) fprintf(csv, "%d,%.6f,%.1f,%.3f\n", ef, rec, qps, ms);
    }

    if (csv) fclose(csv);
    return 0;
}
