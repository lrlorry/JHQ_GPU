#include "hblock_v18/jhq_gpu_index.cuh"
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
            "         [K1=16] [K2=16] [K3=16] [ck1=4] [ck2=4] [ck3=4]\n"
            "         [k=10] [batch_size=1024] [d_proj=64] [rerank_r=64] [km_iters=30]\n",
            argv[0]);
        return 1;
    }

    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int K1        = (argc >  4) ? atoi(argv[4])  : 16;
    int K2        = (argc >  5) ? atoi(argv[5])  : 16;
    int K3        = (argc >  6) ? atoi(argv[6])  : 16;
    int ck1       = (argc >  7) ? atoi(argv[7])  : 4;
    int ck2       = (argc >  8) ? atoi(argv[8])  : 4;
    int ck3       = (argc >  9) ? atoi(argv[9])  : 4;
    int k         = (argc > 10) ? atoi(argv[10]) : 10;
    int batch_sz  = (argc > 11) ? atoi(argv[11]) : 1024;
    int d_proj    = (argc > 12) ? atoi(argv[12]) : 64;
    int rerank_r  = (argc > 13) ? atoi(argv[13]) : 64;
    int km_iters  = (argc > 14) ? atoi(argv[14]) : 30;

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;

    int nb = read_fvecs(base_path,  base,  d_base);
    int nq = read_fvecs(query_path, query, d_query);
         read_ivecs(gt_path,    gt,    d_gt);

    int d = d_base;
    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n",
           nb, d, nq, d_query, (int)(gt.size()/d_gt), d_gt);
    printf("K1=%d  K2=%d  K3=%d  ck1=%d  ck2=%d  ck3=%d  k=%d  batch=%d"
           "  d_proj=%d  rerank_r=%d  km_iters=%d\n",
           K1, K2, K3, ck1, ck2, ck3, k, batch_sz, d_proj, rerank_r, km_iters);
    printf("JL coarse scan (v18): total leaf blocks per query = %d\n", ck1 * ck2 * ck3);

    hblock_v18::HBlockIndex::Params p;
    p.K1        = K1;
    p.K2        = K2;
    p.K3        = K3;
    p.ck1       = ck1;
    p.ck2       = ck2;
    p.ck3       = ck3;
    p.d_proj    = d_proj;
    p.rerank_r  = rerank_r;
    p.km_iters  = km_iters;
    p.batch_size = batch_sz;

    hblock_v18::HBlockIndex idx(d, p);

    auto t0 = Clock::now();
    idx.train(base.data(), nb);
    auto t1 = Clock::now();
    idx.add(base.data(), nb);
    auto t2 = Clock::now();
    printf("Train: %.1f ms   Add: %.1f ms\n",
           Ms(t1-t0).count(), Ms(t2-t1).count());

    // Warm-up
    {
        std::vector<float> wd(nq * k);
        std::vector<int>   wi(nq * k);
        idx.search(query.data(), nq, k, wd.data(), wi.data());
    }

    // Timed run
    int RUNS = 3;
    double total_ms = 0.0;
    std::vector<float> dists(nq * k);
    std::vector<int>   ids(nq * k);
    for (int r = 0; r < RUNS; r++) {
        auto ta = Clock::now();
        idx.search(query.data(), nq, k, dists.data(), ids.data());
        auto tb = Clock::now();
        total_ms += Ms(tb - ta).count();
    }

    double avg_ms = total_ms / RUNS;
    double rec    = recall_at_k(ids.data(), gt.data(), nq, k, d_gt);
    double qps    = (double)nq / (avg_ms / 1000.0);

    printf("\nRecall@%d = %.4f   QPS = %.0f   Latency = %.2f ms  (avg over %d runs)\n",
           k, rec, qps, avg_ms, RUNS);
    printf("RESULT: %.4f %.0f\n", rec, qps);
    return 0;
}
