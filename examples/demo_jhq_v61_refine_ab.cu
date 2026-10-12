// One index, every refinement kernel, measured against each other in one
// process (jhq_v61_warp_refine).
//
// Why one process. Training is not reproducible bit for bit, so two runs are
// two indexes unless the trained-state cache makes them one. Here the index is
// built once -- from the same parameters, training sample and cache key as
// demo_jhq_v36 and demo_jhq_budget_arms, so JHQ_INDEX_CACHE pointed at the
// arms6 cache loads the index the paper measured -- and every kernel searches
// it: the same lists, codes and queries, and the same timing scope as the demo
// the paper's QPS come from (wall clock around search(): H2D copy, graph
// launch, D2H copy, sync).
//
// For each alpha in JHQ_AB_ALPHAS (default: the alpha argument):
//   1. Results. Every kernel in JHQ_AB_LANES (default 0,1,8,16,32) answers the
//      queries once after an untimed warm-up. Ids and distances are written
//      out and compared query by query against lanes 0 (v58's kernel):
//      identical ordered ids, identical id sets, bit-identical distances, and
//      the largest distance error where the ids agree.
//   2. Timing. JHQ_AB_ROUNDS rounds (default 5). In each, every kernel -- the
//      order rotated each round, so none always runs first -- gets one untimed
//      search, which re-captures the CUDA graph for that kernel, then
//      JHQ_AB_REPS timed searches (default 3), whose mean is that round's time.
//      Speedups are paired within a round.
//
// JHQ_AB_NQ keeps only the first n queries (compute-sanitizer runs).
// Outputs: <prefix>_a<alpha>_L<lanes>.ivecs / .fvecs and <prefix>.json.
//
// Usage is demo_jhq_v36's; out_prefix is required.

#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v61_warp_refine/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"
#include "common/recall.cuh"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static std::vector<double> parse_list(const char* s) {
    std::vector<double> v;
    if (!s) return v;
    std::string t(s);
    size_t p = 0;
    while (p < t.size()) {
        size_t c = t.find(',', p);
        if (c == std::string::npos) c = t.size();
        if (c > p) v.push_back(std::atof(t.substr(p, c - p).c_str()));
        p = c + 1;
    }
    return v;
}

static void write_fvecs_local(const char* path, const float* data, int n, int dim) {
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return; }
    for (int i = 0; i < n; ++i) {
        std::fwrite(&dim, sizeof(int), 1, f);
        std::fwrite(data + (size_t)i * dim, sizeof(float), dim, f);
    }
    std::fclose(f);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> "
            "<M> <B> <Br> <alpha> <k> <nlist> <nprobe> <ivf_iters> <batch_size> "
            "<out_prefix> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int   M          = atoi(argv[4]);
    int   B          = atoi(argv[5]);
    int   Br         = atoi(argv[6]);
    float alpha      = (float)atof(argv[7]);
    int   k          = atoi(argv[8]);
    int   nlist      = atoi(argv[9]);
    int   nprobe     = atoi(argv[10]);
    int   ivf_iters  = atoi(argv[11]);
    int   batch_size = atoi(argv[12]);
    const char* out_prefix = argv[13];
    int   kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    std::vector<float> query;
    std::vector<int>   gt;
    int d_query, d_gt;
    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    int nb = base.n, d = base.d;
    int nq = read_fvecs(query_path, query, d_query);
    read_ivecs(gt_path, gt, d_gt);
    if (d_gt < k) { fprintf(stderr, "ground truth %d wide < k=%d\n", d_gt, k); return 1; }
    int nq_use = nq;
    if (const char* e = std::getenv("JHQ_AB_NQ"))
        if (std::atoi(e) > 0) nq_use = std::min(nq, std::atoi(e));
    printf("base=%d x %d  query=%d (using %d)  gt width %d\n", nb, d, nq, nq_use, d_gt);
    printf("M=%d B=%d Br=%d alpha=%.1f k=%d nlist=%d nprobe=%d ivf_iters=%d batch=%d kmeans_iters=%d\n",
           M, B, Br, alpha, k, nlist, nprobe, ivf_iters, batch_size, kmeans_iters);

    // ---- index: exactly demo_jhq_v36's construction, so the cache key matches
    jhq_gpu::JHQGpuIndex::Params p;
    p.M = M;
    if (const char* ab = std::getenv("JHQ_ADD_BATCH")) {
        const int v = std::atoi(ab);
        if (v > 0) p.add_batch = v;
    }
    p.B = B; p.Br = Br; p.alpha = alpha;
    p.nlist = nlist; p.nprobe = nprobe;
    p.ivf_iters = ivf_iters;
    p.batch_size = batch_size;
    p.kmeans_iters = kmeans_iters;
    jhq_gpu::JHQGpuIndex idx(d, p);

    int n_train = std::min(nb, 100000);
    if (const char* e = std::getenv("JHQ_N_TRAIN")) {
        const long long want = (std::strcmp(e, "all") == 0) ? nb : std::atoll(e);
        n_train = (want <= 0 || want > nb) ? nb : (int)want;
    }
    std::vector<float> sample;
    const float* train_src = base.data;
    if (n_train < nb) {
        sample.resize((size_t)n_train * d);
        const long long stride = (long long)nb / n_train;
        for (int i = 0; i < n_train; ++i)
            std::memcpy(sample.data() + (size_t)i * d,
                        base.data + (size_t)(i * stride) * d, (size_t)d * sizeof(float));
        train_src = sample.data();
    }
    std::vector<float> res_sample;
    const float* res_src = base.data;
    int n_res_train = nb;
    if (const char* rt = std::getenv("JHQ_RES_TRAIN_N")) {
        n_res_train = std::atoi(rt);
        if (n_res_train <= 0 || n_res_train > nb) n_res_train = nb;
        if (n_res_train == nb) {
            res_src = base.data;
        } else {
            res_sample.resize((size_t)n_res_train * d);
            const long long rstride = (long long)nb / n_res_train;
            for (int i = 0; i < n_res_train; ++i)
                std::memcpy(res_sample.data() + (size_t)i * d,
                            base.data + (size_t)(i * rstride) * d, (size_t)d * sizeof(float));
            res_src = res_sample.data();
        }
    }
    auto t0 = Clock::now();
    idx.train(train_src, n_train, res_src, n_res_train);
    printf("  train: %.1f ms\n", Ms(Clock::now() - t0).count());
    t0 = Clock::now();
    idx.add(base.data, nb);
    printf("  add:   %.1f ms\n", Ms(Clock::now() - t0).count());

    // ---- the A/B --------------------------------------------------------
    std::vector<double> alphas = parse_list(std::getenv("JHQ_AB_ALPHAS"));
    if (alphas.empty()) alphas.push_back(alpha);
    std::vector<int> lanes;
    for (double v : parse_list(std::getenv("JHQ_AB_LANES"))) lanes.push_back((int)v);
    if (lanes.empty()) lanes = {0, 1, 8, 16, 32};
    if (lanes[0] != 0) { fprintf(stderr, "JHQ_AB_LANES must start with 0, the reference\n"); return 1; }
    const int rounds = std::getenv("JHQ_AB_ROUNDS") ? std::atoi(std::getenv("JHQ_AB_ROUNDS")) : 5;
    const int reps   = std::getenv("JHQ_AB_REPS")   ? std::atoi(std::getenv("JHQ_AB_REPS"))   : 3;
    const int nL = (int)lanes.size();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string jpath = std::string(out_prefix) + ".json";
    FILE* js = std::fopen(jpath.c_str(), "w");
    if (!js) { fprintf(stderr, "cannot write %s\n", jpath.c_str()); return 1; }
    std::fprintf(js, "{\"gpu\":\"%s\",\"base\":\"%s\",\"n\":%d,\"d\":%d,\"nq\":%d,\"k\":%d,"
                 "\"M\":%d,\"B\":%d,\"Br\":%d,\"nlist\":%d,\"nprobe\":%d,\"batch\":%d,"
                 "\"rounds\":%d,\"reps\":%d,\"alphas\":[",
                 prop.name, base_path, nb, d, nq_use, k, M, B, Br, nlist, nprobe, batch_size,
                 rounds, reps);

    std::vector<float> dists((size_t)nq_use * k);
    std::vector<int>   ids((size_t)nq_use * k);
    for (size_t ai = 0; ai < alphas.size(); ++ai) {
        const float a = (float)alphas[ai];
        idx.set_alpha(a);
        printf("\n=== alpha=%g  nprobe=%d  (ck=%d) ===\n", a, nprobe,
               std::max(k, (int)std::ceil(a * (float)k)));

        // 1. results, and per-query comparison against lanes 0
        std::vector<std::vector<int>>   rid(nL);
        std::vector<std::vector<float>> rdist(nL);
        std::vector<double> rec(nL);
        for (int li = 0; li < nL; ++li) {
            jhq_gpu::set_refine_lanes(lanes[li]);
            idx.search(query.data(), nq_use, k, dists.data(), ids.data());   // warm-up, capture
            idx.search(query.data(), nq_use, k, dists.data(), ids.data());
            rid[li] = ids; rdist[li] = dists;
            rec[li] = evaluate_recall(ids.data(), gt.data(), nq_use, k, d_gt).recall;
            char path[1024];
            std::snprintf(path, sizeof(path), "%s_a%g_L%d.ivecs", out_prefix, a, lanes[li]);
            write_ivecs(path, ids.data(), nq_use, k);
            std::snprintf(path, sizeof(path), "%s_a%g_L%d.fvecs", out_prefix, a, lanes[li]);
            write_fvecs_local(path, dists.data(), nq_use, k);
        }
        struct Cmp { int same_order = 0, same_set = 0, bit_identical = 0;
                     double max_abs = 0, max_rel = 0, kth_rel_on_diff = 0; };
        std::vector<Cmp> cmp(nL);
        for (int li = 0; li < nL; ++li) {
            Cmp& c = cmp[li];
            for (int q = 0; q < nq_use; ++q) {
                const int*   a0 = rid[0].data()  + (size_t)q * k;
                const int*   a1 = rid[li].data() + (size_t)q * k;
                const float* d0 = rdist[0].data()  + (size_t)q * k;
                const float* d1 = rdist[li].data() + (size_t)q * k;
                bool order = true, bits = true;
                for (int i = 0; i < k; ++i) {
                    if (a0[i] != a1[i]) order = false;
                    if (std::memcmp(&d0[i], &d1[i], sizeof(float)) != 0) bits = false;
                }
                std::vector<int> s0(a0, a0 + k), s1(a1, a1 + k);
                std::sort(s0.begin(), s0.end()); std::sort(s1.begin(), s1.end());
                const bool set = (s0 == s1);
                c.same_order += order; c.same_set += set; c.bit_identical += (order && bits);
                for (int i = 0; i < k; ++i)
                    if (a0[i] == a1[i] && std::isfinite(d0[i]) && std::isfinite(d1[i])) {
                        const double e = std::fabs((double)d1[i] - (double)d0[i]);
                        c.max_abs = std::max(c.max_abs, e);
                        c.max_rel = std::max(c.max_rel, e / std::max(std::fabs((double)d0[i]), 1e-30));
                    }
                if (!set && std::isfinite(d0[k - 1]))
                    c.kth_rel_on_diff = std::max(c.kth_rel_on_diff,
                        std::fabs((double)d1[k - 1] - (double)d0[k - 1]) /
                        std::max(std::fabs((double)d0[k - 1]), 1e-30));
            }
        }

        // 2. timing, rotated order, paired within a round
        std::vector<std::vector<double>> ms(nL, std::vector<double>(rounds, 0.0));
        for (int r = 0; r < rounds; ++r)
            for (int t = 0; t < nL; ++t) {
                const int li = (r + t) % nL;
                jhq_gpu::set_refine_lanes(lanes[li]);
                idx.search(query.data(), nq_use, k, dists.data(), ids.data());   // capture
                double sum = 0;
                for (int rep = 0; rep < reps; ++rep) {
                    auto tr = Clock::now();
                    idx.search(query.data(), nq_use, k, dists.data(), ids.data());
                    sum += Ms(Clock::now() - tr).count();
                }
                ms[li][r] = sum / reps;
            }

        printf("%-6s %8s %10s %10s %10s %9s %9s %9s %7s %7s %6s %10s\n", "lanes", "recall",
               "QPS med", "QPS min", "QPS max", "x med", "x min", "x max",
               "order", "set", "bits", "max relerr");
        std::fprintf(js, "%s{\"alpha\":%g,\"variants\":[", ai ? "," : "", a);
        for (int li = 0; li < nL; ++li) {
            std::vector<double> qps(rounds), sp(rounds);
            for (int r = 0; r < rounds; ++r) {
                qps[r] = nq_use / (ms[li][r] / 1000.0);
                sp[r]  = ms[0][r] / ms[li][r];
            }
            const Cmp& c = cmp[li];
            printf("%-6d %8.4f %10.0f %10.0f %10.0f %9.3f %9.3f %9.3f %7d %7d %6d %10.2e\n",
                   lanes[li], rec[li], median(qps),
                   *std::min_element(qps.begin(), qps.end()), *std::max_element(qps.begin(), qps.end()),
                   median(sp), *std::min_element(sp.begin(), sp.end()), *std::max_element(sp.begin(), sp.end()),
                   c.same_order, c.same_set, c.bit_identical, c.max_rel);
            std::fprintf(js, "%s{\"lanes\":%d,\"recall\":%.6f,\"ms\":[", li ? "," : "", lanes[li], rec[li]);
            for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.6f", r ? "," : "", ms[li][r]);
            std::fprintf(js, "],\"same_order\":%d,\"same_set\":%d,\"bit_identical\":%d,"
                         "\"max_abs\":%.6e,\"max_rel\":%.6e,\"kth_rel_on_set_diff\":%.6e}",
                         c.same_order, c.same_set, c.bit_identical, c.max_abs, c.max_rel,
                         c.kth_rel_on_diff);
        }
        std::fprintf(js, "]}");
        printf("(order/set/bits: queries of %d whose ordered ids / id set / ids and distance bits "
               "equal lanes 0's)\n", nq_use);
    }
    std::fprintf(js, "]}\n");
    std::fclose(js);
    jhq_gpu::set_refine_lanes(0);
    printf("\nwrote %s\n=== AB_OK ===\n", jpath.c_str());
    return 0;
}
