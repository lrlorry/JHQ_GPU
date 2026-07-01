#include "hblock_v2/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double recall_at_k(const int* labels, const int* gt,
                           int nq, int k, int gt_k) {
    int hits = 0;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < k; j++) {
            int found = labels[i * k + j];
            for (int g = 0; g < gt_k; g++)
                if (gt[i * gt_k + g] == found) { hits++; break; }
        }
    return (double)hits / (double)(nq * k);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> "
            "<M> <B> <Br> <alpha1> [alpha2=2.0] [k=10] [nlist=1024] [nprobe=8] "
            "[ivf_iters=8] [batch_size=256]\n", argv[0]);
        return 1;
    }

    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int   M          = atoi(argv[4]);
    int   B          = atoi(argv[5]);
    int   Br         = atoi(argv[6]);
    float alpha1     = (float)atof(argv[7]);
    float alpha2     = (argc > 8)  ? (float)atof(argv[8])  : 2.0f;
    int   k          = (argc > 9)  ? atoi(argv[9])          : 10;
    int   nlist      = (argc > 10) ? atoi(argv[10])         : 1024;
    int   nprobe     = (argc > 11) ? atoi(argv[11])         : 8;
    int   ivf_iters  = (argc > 12) ? atoi(argv[12])         : 8;
    int   batch_size = (argc > 13) ? atoi(argv[13])         : 256;

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;

    int nb = read_fvecs(base_path,  base,  d_base);
    int nq = read_fvecs(query_path, query, d_query);
    int ng = read_ivecs(gt_path,    gt,    d_gt);

    int d = d_base;
    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n", nb, d, nq, d_query, ng, d_gt);
    printf("M=%d  B=%d  Br=%d  alpha1=%.1f  alpha2=%.1f  k=%d  "
           "nlist=%d  nprobe=%d  ivf_iters=%d  batch_size=%d\n",
           M, B, Br, alpha1, alpha2, k, nlist, nprobe, ivf_iters, batch_size);

    jhq_gpu::JHQHBlockIndex::Params p;
    p.M = M; p.B = B; p.Br = Br;
    p.alpha1 = alpha1; p.alpha2 = alpha2;
    p.nlist = nlist; p.nprobe = nprobe;
    p.ivf_iters = ivf_iters;
    p.batch_size = batch_size;

    jhq_gpu::JHQHBlockIndex idx(d, p);

    int n_train = std::min(nb, 100000);
    printf("Training on %d vectors...\n", n_train);
    auto t0 = Clock::now();
    idx.train(base.data(), n_train);
    double train_ms = Ms(Clock::now() - t0).count();
    printf("  train: %.1f ms\n", train_ms);

    printf("Adding %d vectors...\n", nb);
    t0 = Clock::now();
    idx.add(base.data(), nb);
    double add_ms = Ms(Clock::now() - t0).count();
    printf("  add:   %.1f ms\n", add_ms);

    std::vector<float> out_dists((long long)nq * k);
    std::vector<int>   out_ids  ((long long)nq * k);

    // Warm-up (also captures the CUDA graph)
    idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());

    const int REPS = 5;
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
