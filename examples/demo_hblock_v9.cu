#include "hblock_v9/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* labels, const int* gt,
                           int nq, int k, int gt_k)
{
    int hits = 0;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < k; j++) {
            int found = labels[i * k + j];
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
            "         [K1=64] [K2=128] [ck1=8] [ck2=32] [ck3=256]\n"
            "         [k=10] [batch_size=1024]\n",
            argv[0]);
        return 1;
    }

    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int K1         = (argc > 4)  ? atoi(argv[4])  : 64;
    int K2         = (argc > 5)  ? atoi(argv[5])  : 128;
    int ck1        = (argc > 6)  ? atoi(argv[6])  : 8;
    int ck2        = (argc > 7)  ? atoi(argv[7])  : 32;
    int ck3        = (argc > 8)  ? atoi(argv[8])  : 256;
    int k          = (argc > 9)  ? atoi(argv[9])  : 10;
    int batch_size = (argc > 10) ? atoi(argv[10]) : 1024;

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;

    int nb = read_fvecs(base_path,  base,  d_base);
    int nq = read_fvecs(query_path, query, d_query);
         read_ivecs(gt_path,    gt,    d_gt);

    int d = d_base;
    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n",
           nb, d, nq, d_query, (int)(gt.size()/d_gt), d_gt);
    printf("K1=%d  K2=%d  ck1=%d  ck2=%d  ck3=%d  k=%d  batch_size=%d\n",
           K1, K2, ck1, ck2, ck3, k, batch_size);

    hblock_v9::HBlockIndex::Params p;
    p.K1         = K1;
    p.K2         = K2;
    p.ck1        = ck1;
    p.ck2        = ck2;
    p.ck3        = ck3;
    p.batch_size = batch_size;

    hblock_v9::HBlockIndex idx(d, p);

    int n_train = std::min(nb, 100000);
    printf("Training on %d vectors...\n", n_train);
    auto t0 = Clock::now();
    idx.train(base.data(), n_train);
    printf("  train: %.1f ms\n", Ms(Clock::now() - t0).count());

    printf("Adding %d vectors...\n", nb);
    t0 = Clock::now();
    idx.add(base.data(), nb);
    printf("  add:   %.1f ms\n", Ms(Clock::now() - t0).count());

    std::vector<float> out_dists((long long)nq * k);
    std::vector<int>   out_ids  ((long long)nq * k);

    printf("Warm-up...\n");
    idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());

    const int REPS = 5;
    printf("Benchmarking (%d reps)...\n", REPS);
    t0 = Clock::now();
    for (int r = 0; r < REPS; r++)
        idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());
    double ms = Ms(Clock::now() - t0).count() / REPS;

    double rec = recall_at_k(out_ids.data(), gt.data(), nq, k, d_gt);
    double qps = nq / (ms / 1000.0);

    printf("\nRecall@%d : %.4f\n", k, rec);
    printf("Latency   : %.2f ms  (%d queries)\n", ms, nq);
    printf("QPS       : %.0f\n", qps);

    return 0;
}
