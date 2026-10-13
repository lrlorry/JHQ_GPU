// Rank-optimal quantization study (experiments/rank_optimal_quantization), GPU prototype: v65's search on
// two representations of one index, measured side by side.
//
//   A  the paper index as built (zero-threshold sign bits, shared levels, its residual)
//   B  the same index with a primary/residual representation imported from an offline build
//      (jhq_v67_import::import_state): same rotation, IVF lists, positions, nprobe, kernels
//
// The variant directory (JHQ_RQ_VARDIR) holds raw files written by experiments/rank_optimal_quantization/
// export_variant.py: lo.f32, hi.f32 [d], bits.u8 [N][M], res_cb.f32 [M][Kr], res.u8 [N][bpv], corr.f32 [N].
//
// Configurations: (A, alpha) for alpha in JHQ_RQ_ALPHAS_A and (B, alpha) for alpha in JHQ_RQ_ALPHAS_B, at every
// batch size of JHQ_BP_BATCHES (8,32,256,1000). v65's execution throughout: refinement lanes 16, the scheduled
// list groups with the shared bound (calibrated once on base pseudo-queries, never on the evaluation queries).
// Timing is demo_jhq_batch_pipeline's: per batch size, floor(nq/b) consecutive batches of exactly b, each
// search() synchronous and timed on its own (wall clock around search(): H2D, graph launch, D2H, sync);
// throughput = queries / summed time over at least JHQ_BP_MIN_BATCHES batches; JHQ_BP_ROUNDS rounds (5) with
// the configuration order rotated; every configuration answers one untimed batch first (graph capture, JIT).
// Recall@10 of every configuration over all used queries. Writes <out_prefix>.json.

#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v67_import/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"
#include "common/recall.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }
static std::vector<double> parse_list(const char* s, const char* dflt) {
    std::vector<double> v; std::string t = s ? s : (dflt ? dflt : "");
    size_t p = 0;
    while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); if (q > p) v.push_back(std::atof(t.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}
template <class T> static std::vector<T> read_raw(const std::string& path, size_t n) {
    std::vector<T> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "cannot read %zu items of %s\n", n, path.c_str()); std::exit(1); }
    std::fclose(f);
    return v;
}

static void build(jhq_gpu::JHQGpuIndex& idx, const MmapFloatMatrix& base, int d) {
    const int nb = base.n;
    int n_train = std::min(nb, 100000);
    if (const char* e = std::getenv("JHQ_N_TRAIN")) {
        const long long want = (std::strcmp(e, "all") == 0) ? nb : std::atoll(e);
        n_train = (want <= 0 || want > nb) ? nb : (int)want;
    }
    std::vector<float> sample; const float* train_src = base.data;
    if (n_train < nb) {
        sample.resize((size_t)n_train * d);
        const long long stride = (long long)nb / n_train;
        for (int i = 0; i < n_train; ++i)
            std::memcpy(sample.data() + (size_t)i * d, base.data + (size_t)(i * stride) * d, (size_t)d * sizeof(float));
        train_src = sample.data();
    }
    std::vector<float> res_sample; const float* res_src = base.data; int n_res_train = nb;
    if (const char* rt = std::getenv("JHQ_RES_TRAIN_N")) {
        n_res_train = std::atoi(rt);
        if (n_res_train <= 0 || n_res_train > nb) n_res_train = nb;
        if (n_res_train != nb) {
            res_sample.resize((size_t)n_res_train * d);
            const long long rstride = (long long)nb / n_res_train;
            for (int i = 0; i < n_res_train; ++i)
                std::memcpy(res_sample.data() + (size_t)i * d, base.data + (size_t)(i * rstride) * d, (size_t)d * sizeof(float));
            res_src = res_sample.data();
        }
    }
    idx.train(train_src, n_train, res_src, n_res_train);
    idx.add(base.data, nb);
}

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr, "Usage: %s <base> <query> <gt> <M> <B> <Br> <alpha> <k> <nlist> <nprobe> <ivf_iters> "
                        "<batch_size> <out_prefix> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1]; const char* query_path = argv[2]; const char* gt_path = argv[3];
    int M = atoi(argv[4]), Bb = atoi(argv[5]), Br = atoi(argv[6]);
    float alpha = (float)atof(argv[7]);
    int k = atoi(argv[8]), nlist = atoi(argv[9]), nprobe = atoi(argv[10]);
    int ivf_iters = atoi(argv[11]), batch_size = atoi(argv[12]);
    const std::string prefix = argv[13];
    int kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;
    const char* vdir = std::getenv("JHQ_RQ_VARDIR");
    if (!vdir) { fprintf(stderr, "JHQ_RQ_VARDIR is required\n"); return 1; }

    std::vector<float> query; std::vector<int> gt; int d_query, d_gt;
    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    const int nb = base.n, d = base.d;
    int nq = read_fvecs(query_path, query, d_query);
    read_ivecs(gt_path, gt, d_gt);
    nq = std::min(nq, env_int("JHQ_BP_NQ", nq));

    jhq_gpu::JHQGpuIndex::Params p;
    p.M = M; p.B = Bb; p.Br = Br; p.alpha = alpha; p.nlist = nlist; p.nprobe = nprobe;
    p.ivf_iters = ivf_iters; p.batch_size = batch_size; p.kmeans_iters = kmeans_iters;
    if (const char* ab = std::getenv("JHQ_ADD_BATCH")) { const int v = std::atoi(ab); if (v > 0) p.add_batch = v; }
    jhq_gpu::JHQGpuIndex A(d, p), Bx(d, p);
    auto t0 = Clock::now();
    build(A, base, d);
    const double buildA = Ms(Clock::now() - t0).count();
    t0 = Clock::now();
    build(Bx, base, d);
    jhq_gpu::JHQGpuIndex::HostState S;
    A.export_host_state(S);
    const long long N = S.ntotal;
    const int bpv = S.bpv, Kr = S.Kr;
    jhq_gpu::JHQGpuIndex::ImportState I;
    const std::string vd = vdir;
    I.lo = read_raw<float>(vd + "/lo.f32", d);
    I.hi = read_raw<float>(vd + "/hi.f32", d);
    I.primary = read_raw<uint8_t>(vd + "/bits.u8", (size_t)N * M);
    I.res_c1d = read_raw<float>(vd + "/res_cb.f32", (size_t)M * Kr);
    I.res = read_raw<uint8_t>(vd + "/res.u8", (size_t)N * bpv);
    I.corr = read_raw<float>(vd + "/corr.f32", (size_t)N);
    Bx.import_state(I);
    const double buildB = Ms(Clock::now() - t0).count();
    printf("built A in %.0f ms, B (build + import) in %.0f ms\n", buildA, buildB);

    // v65's execution
    jhq_gpu::set_refine_lanes(16);
    {
        const int n_cal = std::max(1, env_int("JHQ_BP_SCHED_CAL", 512));
        std::vector<float> cal((size_t)n_cal * d);
        const long long st = std::max(1LL, (long long)nb / n_cal);
        for (int i = 0; i < n_cal; ++i) {
            const long long r = std::min((long long)nb - 1, i * st + st / 2);
            std::memcpy(cal.data() + (size_t)i * d, base.data + (size_t)r * d, (size_t)d * sizeof(float));
        }
        A.calibrate_schedule(cal.data(), n_cal);
        printf("schedule: per-query candidate count CV %.3f on %d pseudo-queries\n", jhq_gpu::schedule_skew(), n_cal);
    }
    jhq_gpu::set_primary_groups(-1);
    jhq_gpu::set_shared_bound(true);

    struct Cfg { char rep; double alpha; };
    std::vector<Cfg> cfgs;
    for (double a : parse_list(std::getenv("JHQ_RQ_ALPHAS_A"), "100")) cfgs.push_back({'A', a});
    for (double a : parse_list(std::getenv("JHQ_RQ_ALPHAS_B"), "100")) cfgs.push_back({'B', a});
    std::vector<double> batches = parse_list(std::getenv("JHQ_BP_BATCHES"), "8,32,256,1000");
    const int rounds = env_int("JHQ_BP_ROUNDS", 5);
    const int min_batches = env_int("JHQ_BP_MIN_BATCHES", 24);
    const int nC = (int)cfgs.size();

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    FILE* js = std::fopen((prefix + ".json").c_str(), "w");
    std::fprintf(js, "{\"gpu\":\"%s\",\"sms\":%d,\"n\":%d,\"d\":%d,\"nq\":%d,\"k\":%d,\"M\":%d,\"nlist\":%d,\"nprobe\":%d,"
                 "\"vardir\":\"%s\",\"build_ms_A\":%.1f,\"build_import_ms_B\":%.1f,\"sched_cv\":%.5f,\"rounds\":%d,\"runs\":[",
                 prop.name, prop.multiProcessorCount, nb, d, nq, k, M, nlist, nprobe, vdir, buildA, buildB,
                 jhq_gpu::schedule_skew(), rounds);
    bool first = true;
    for (double bd : batches) {
        const int b = (int)bd;
        if (b > batch_size || b > nq) { printf("skip b=%d\n", b); continue; }
        const int nbatch = std::max(1, nq / b), nused = nbatch * b;
        const int reps = std::max(1, (min_batches + nbatch - 1) / nbatch);
        std::vector<float> dists((size_t)b * k); std::vector<int> ids((size_t)b * k);
        std::vector<double> rec(nC);
        for (int ci = 0; ci < nC; ++ci) {
            jhq_gpu::JHQGpuIndex& X = (cfgs[ci].rep == 'A') ? A : Bx;
            X.set_alpha((float)cfgs[ci].alpha);
            X.search(query.data(), b, k, dists.data(), ids.data());              // capture + JIT
            std::vector<int> rid((size_t)nused * k); std::vector<float> rd((size_t)nused * k);
            for (int bi = 0; bi < nbatch; ++bi)
                X.search(query.data() + (size_t)bi * b * d, b, k, rd.data() + (size_t)bi * b * k, rid.data() + (size_t)bi * b * k);
            rec[ci] = evaluate_recall(rid.data(), gt.data(), nused, k, d_gt).recall;
        }
        std::vector<std::vector<double>> qps(nC, std::vector<double>(rounds)), lat(nC, std::vector<double>(rounds));
        for (int r = 0; r < rounds; ++r)
            for (int oi = 0; oi < nC; ++oi) {
                const int ci = (oi + r) % nC;
                jhq_gpu::JHQGpuIndex& X = (cfgs[ci].rep == 'A') ? A : Bx;
                X.set_alpha((float)cfgs[ci].alpha);
                X.search(query.data(), b, k, dists.data(), ids.data());          // the graph for this alpha
                std::vector<double> per; double tot = 0.0;
                for (int rp = 0; rp < reps; ++rp)
                    for (int bi = 0; bi < nbatch; ++bi) {
                        const auto s0 = Clock::now();
                        X.search(query.data() + (size_t)bi * b * d, b, k, dists.data(), ids.data());
                        const double ms = Ms(Clock::now() - s0).count();
                        per.push_back(ms); tot += ms;
                    }
                std::sort(per.begin(), per.end());
                lat[ci][r] = per[per.size() / 2];
                qps[ci][r] = (double)nused * reps / (tot / 1e3);
            }
        printf("\n=== b=%d ===\n", b);
        for (int ci = 0; ci < nC; ++ci) {
            std::vector<double> q = qps[ci]; std::sort(q.begin(), q.end());
            printf("  %c alpha=%-7g recall %.4f  QPS median %.0f [%.0f, %.0f]\n", cfgs[ci].rep, cfgs[ci].alpha, rec[ci],
                   q[q.size() / 2], q.front(), q.back());
            std::fprintf(js, "%s{\"batch\":%d,\"rep\":\"%c\",\"alpha\":%g,\"recall\":%.6f,\"nused\":%d,\"qps\":[", first ? "" : ",",
                         b, cfgs[ci].rep, cfgs[ci].alpha, rec[ci], nused);
            first = false;
            for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.1f", r ? "," : "", qps[ci][r]);
            std::fprintf(js, "],\"lat_med_ms\":[");
            for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.5f", r ? "," : "", lat[ci][r]);
            std::fprintf(js, "]}");
        }
    }
    std::fprintf(js, "]}\n");
    std::fclose(js);
    printf("=== RQ_BENCH_OK ===\n");
    return 0;
}
