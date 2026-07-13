#include "hblock_v32/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <ctime>
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
            "         [max_ef=128] [K1=16] [K2=16] [K3=16]\n"
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
    int k = 10;

    int max_ef = (argc >  4) ? atoi(argv[4])  : 128;
    int K1     = (argc >  5) ? atoi(argv[5])  : 16;
    int K2     = (argc >  6) ? atoi(argv[6])  : 16;
    int K3     = (argc >  7) ? atoi(argv[7])  : 16;
    int deg    = (argc >  8) ? atoi(argv[8])  : 32;
    int epc    = (argc >  9) ? atoi(argv[9])  : 4;
    int dprj   = (argc > 10) ? atoi(argv[10]) : 64;
    int pbr    = (argc > 11) ? atoi(argv[11]) : 16;
    int batch  = (argc > 12) ? atoi(argv[12]) : 1024;

    printf("nb=%d  nq=%d  d=%d  d_gt=%d  max_ef=%d\n", nb, nq, d_base, d_gt, max_ef);

    hblock_v32::HBlockIndex::Params p;
    p.K1=K1; p.K2=K2; p.K3=K3;
    p.max_ef=max_ef;
    p.d_proj=dprj; p.per_block_r=pbr;
    p.graph_degree=deg; p.entry_per_cell=epc;
    p.batch_size=batch;

    hblock_v32::HBlockIndex idx(d_base, p);

    int n_train = std::min(nb, 200000);
    printf("Training on %d vectors...\n", n_train);
    idx.train(base.data(), n_train);

    printf("Adding %d vectors...\n", nb);
    idx.add(base.data(), nb);

    // Sweep ef at search time — no rebuild needed
    std::vector<int> efs;
    for (int e = 8; e <= max_ef; e *= 2) efs.push_back(e);
    if (efs.empty() || efs.back() != max_ef) efs.push_back(max_ef);

    // CSV output alongside stdout
    char csv_path[256];
    {
        time_t t = time(nullptr);
        struct tm* tm = localtime(&t);
        strftime(csv_path, sizeof(csv_path),
                 "results/hblock_v32_%Y%m%d_%H%M%S.csv", tm);
    }
    FILE* csv = fopen(csv_path, "w");
    if (csv) fprintf(csv, "ef,recall@10,qps,latency_ms\n");

    printf("\n%-8s %-12s %-10s\n", "ef", "recall@10", "QPS");
    printf("%-8s %-12s %-10s\n", "----", "---------", "---");

    std::vector<float> h_dists((long long)nq * k);
    std::vector<int>   h_ids  ((long long)nq * k);

    for (int ef : efs) {
        const int REPS = 5;
        auto t0 = Clock::now();
        for (int rep = 0; rep < REPS; rep++)
            idx.search(query.data(), nq, k, h_dists.data(), h_ids.data(), ef);
        double ms = Ms(Clock::now() - t0).count() / REPS;

        long long hits = 0;
        for (int qi = 0; qi < nq; qi++)
            for (int ki = 0; ki < k; ki++)
                for (int gi = 0; gi < k; gi++)
                    if (h_ids[qi*k+ki] == gt[qi*d_gt+gi]) { hits++; break; }
        double recall = (double)hits / (double)((long long)nq * k);
        double qps    = nq / (ms / 1000.0);

        printf("%-8d %-12.4f %-10.0f  (%.2f ms)\n", ef, recall, qps, ms);
        if (csv) fprintf(csv, "%d,%.6f,%.1f,%.3f\n", ef, recall, qps, ms);
    }

    if (csv) { fclose(csv); printf("\nCSV: %s\n", csv_path); }
    return 0;
}
