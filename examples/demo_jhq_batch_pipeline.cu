// Batch-size sweep of the search pipeline on one index: per-batch latency and
// steady-state throughput at several batch sizes, with every timed block
// wrapped in an NVTX range so nsys can put each kernel of each search under
// its stage, batch size and variant.
//
// Built twice from this file:
//   demo_jhq_v61_batch  v61, refinement lanes 16. The Phase 0 baseline of
//                       experiments/pipeline_v62.
//   demo_jhq_v64_batch  v64 (JHQ_PIPELINE_V64): sweeps the routing mode
//                       instead (JHQ_BP_ROUTES, default 0,1,2: v61's, Pi^T c
//                       serial, Pi^T c overlapped) and checks every mode
//                       against 0: probe ids, centroid dot products, primary
//                       top-ck, final ids and distances, recall.
//   demo_jhq_v65_batch  v65 (JHQ_PIPELINE_V62, _V63 and _V65): variant -1 is
//                       the scheduled G (calibrated first on JHQ_BP_SCHED_CAL,
//                       default 512, base vectors as pseudo-queries), g >= 1000
//                       is refinement lanes 0 with g - 1000 (v61 L0 at 1000),
//                       and g in 100..108 the shared-bound groups of v63.
//   demo_jhq_v63_batch  v63 (JHQ_PIPELINE_V62 and _V63): as v62's, where a
//                       variant g >= 100 is G = g - 100 with the shared bound
//                       and g < 100 v62's independent groups, and each g >= 100
//                       is also checked for identical top-ck and final results
//                       against g - 100.
//   demo_jhq_v62_batch  v62 (JHQ_PIPELINE_V62): also sweeps the primary
//                       execution, G = 0 (v61's query-centric scan, unchanged)
//                       and G = 1, 2, 4, 8 list groups per query, and checks
//                       every G against G = 0 query by query: probe ids,
//                       primary top-ck sets and distances, final ids and
//                       distances, recall.
//
// Two throughput notions, kept apart. For batch size b the queries are cut
// into floor(nq / b) consecutive batches of exactly b (no ragged remainder, so
// one captured graph serves every batch); each search() call is synchronous
// and timed on its own:
//   latency     median and p95 over those calls -- what one batch costs;
//   throughput  queries / summed time over the whole sequence, repeated until
//               at least JHQ_BP_MIN_BATCHES batches -- a synchronous client
//               issuing batches back to back.
// Before timing, each variant answers one untimed batch, which captures its
// graph and JIT-compiles its kernels, so neither is in any number.
//
// Rounds (JHQ_BP_ROUNDS, default 5) rotate the variant order; each round is
// one NVTX range per (alpha, b, variant): "a<alpha>|b<b>|G<G>|r<round>".
//
// JHQ_BP_ROUNDS=0 skips the timing: a results-and-checks-only run.
// Environment: JHQ_BP_ALPHAS (default: the alpha argument), JHQ_BP_BATCHES
// (8,32,256,1000), JHQ_BP_GROUPS (v62; 0,1,2,4,8), JHQ_BP_LANES (16),
// JHQ_BP_ROUNDS (5), JHQ_BP_MIN_BATCHES (24), JHQ_BP_CHECK (1), JHQ_BP_NQ.
// Usage is demo_jhq_v36's with out_prefix required; writes <prefix>.json.
//
// JHQ_BP_DUMP=1 (v62, with checks on, G = 1 among the variants) also writes,
// for the first batch size of each alpha, <prefix>_a<alpha>.dump: the G = 1
// probe ids and exact primary top-ck (positions, distances, in key order) of
// every used query, and the IVF list sizes -- what an offline study needs to
// place each top-ck candidate in its group for any G. Layout, all int32 except
// the distances (float32): nused, nprobe, ck, nlist; sizes[nlist];
// probes[nused][nprobe]; pos[nused][ck]; val[nused][ck].

#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v61_warp_refine/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"
#include "common/recall.cuh"
#include "nvtx3/nvToolsExt.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <map>

#if defined(JHQ_PIPELINE_V62) || defined(JHQ_PIPELINE_V64)
#define JHQ_BP_CHECKS 1          // a reference variant 0 and per-query checks against it
#endif

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static std::vector<double> parse_list(const char* s, const char* dflt) {
    std::vector<double> v;
    std::string t(s ? s : (dflt ? dflt : ""));
    size_t p = 0;
    while (p < t.size()) {
        size_t c = t.find(',', p);
        if (c == std::string::npos) c = t.size();
        if (c > p) v.push_back(std::atof(t.substr(p, c - p).c_str()));
        p = c + 1;
    }
    return v;
}
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }
static double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = q * (v.size() - 1);
    const size_t i = (size_t)x;
    return i + 1 < v.size() ? v[i] + (x - i) * (v[i + 1] - v[i]) : v[i];
}

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr, "Usage: %s <base> <query> <gt> <M> <B> <Br> <alpha> <k> <nlist> <nprobe> "
                        "<ivf_iters> <batch_size> <out_prefix> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1]; const char* query_path = argv[2]; const char* gt_path = argv[3];
    int M = atoi(argv[4]), B = atoi(argv[5]), Br = atoi(argv[6]);
    float alpha = (float)atof(argv[7]);
    int k = atoi(argv[8]), nlist = atoi(argv[9]), nprobe = atoi(argv[10]);
    int ivf_iters = atoi(argv[11]), batch_size = atoi(argv[12]);
    const char* out_prefix = argv[13];
    int kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    std::vector<float> query; std::vector<int> gt; int d_query, d_gt;
    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    int nb = base.n, d = base.d;
    int nq = read_fvecs(query_path, query, d_query);
    read_ivecs(gt_path, gt, d_gt);
    if (d_gt < k) { fprintf(stderr, "ground truth %d wide < k=%d\n", d_gt, k); return 1; }
    if (env_int("JHQ_BP_NQ", 0) > 0) nq = std::min(nq, env_int("JHQ_BP_NQ", 0));

    // ---- index: demo_jhq_v36's construction exactly, so the cache key matches
    jhq_gpu::JHQGpuIndex::Params p;
    p.M = M;
    if (const char* ab = std::getenv("JHQ_ADD_BATCH")) { const int v = std::atoi(ab); if (v > 0) p.add_batch = v; }
    p.B = B; p.Br = Br; p.alpha = alpha; p.nlist = nlist; p.nprobe = nprobe;
    p.ivf_iters = ivf_iters; p.batch_size = batch_size; p.kmeans_iters = kmeans_iters;
    jhq_gpu::JHQGpuIndex idx(d, p);
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
    auto t0 = Clock::now();
    idx.train(train_src, n_train, res_src, n_res_train);
    printf("train %.1f ms\n", Ms(Clock::now() - t0).count());
    t0 = Clock::now();
    idx.add(base.data, nb);
    printf("add %.1f ms\n", Ms(Clock::now() - t0).count());

    // ---- sweep -------------------------------------------------------------
    const int lanes = env_int("JHQ_BP_LANES", 16);
    jhq_gpu::set_refine_lanes(lanes);
    std::vector<double> alphas = parse_list(std::getenv("JHQ_BP_ALPHAS"), nullptr);
    if (alphas.empty()) alphas.push_back(alpha);
    std::vector<double> batches = parse_list(std::getenv("JHQ_BP_BATCHES"), "8,32,256,1000");
#ifdef JHQ_PIPELINE_V62
    std::vector<int> groups;
    for (double g : parse_list(std::getenv("JHQ_BP_GROUPS"), "0,1,2,4,8")) groups.push_back((int)g);
    if (groups[0] != 0) { fprintf(stderr, "JHQ_BP_GROUPS must start with 0, the v61 reference\n"); return 1; }
#if defined(JHQ_PIPELINE_V65)
    auto set_variant = [&](int g) {
        const int base = (g >= 1000) ? g - 1000 : g;
        jhq_gpu::set_refine_lanes(g >= 1000 ? 0 : lanes);
        jhq_gpu::set_primary_groups(base < 0 ? -1 : base % 100);
        jhq_gpu::set_shared_bound(base < 0 || base >= 100);
    };
    {
        // the schedule's one statistic, on base vectors offset half a stride from
        // the training sample's -- never on the evaluation queries
        const int n_cal = std::max(1, env_int("JHQ_BP_SCHED_CAL", 512));
        std::vector<float> cal((size_t)n_cal * d);
        const long long st = std::max(1LL, (long long)nb / n_cal);
        for (int i = 0; i < n_cal; ++i) {
            const long long r = std::min((long long)nb - 1, i * st + st / 2);
            std::memcpy(cal.data() + (size_t)i * d, base.data + (size_t)r * d, (size_t)d * sizeof(float));
        }
        idx.calibrate_schedule(cal.data(), n_cal);
        printf("schedule: per-query candidate count CV %.3f on %d pseudo-queries\n", jhq_gpu::schedule_skew(), n_cal);
    }
#elif defined(JHQ_PIPELINE_V63)
    auto set_variant = [&](int g) { jhq_gpu::set_primary_groups(g % 100); jhq_gpu::set_shared_bound(g >= 100); };
#else
    auto set_variant = [&](int g) { jhq_gpu::set_primary_groups(g); };
#endif
    // IVF list sizes for JHQ_BP_DUMP, from the id -> list map the index builds on demand
    const bool dump = env_int("JHQ_BP_DUMP", 0) != 0;
    std::vector<int> list_sizes;
    if (dump) {
        std::vector<int> vl; idx.vector_lists(vl);
        list_sizes.assign((size_t)nlist, 0);
        for (int l : vl) if (l >= 0 && l < nlist) ++list_sizes[l];
    }
#elif defined(JHQ_PIPELINE_V64)
    std::vector<int> groups;                // here: the routing modes
    for (double g : parse_list(std::getenv("JHQ_BP_ROUTES"), "0,1,2")) groups.push_back((int)g);
    if (groups[0] != 0) { fprintf(stderr, "JHQ_BP_ROUTES must start with 0, v61's routing\n"); return 1; }
    auto set_variant = [&](int g) { jhq_gpu::set_route_mode(g); };
    double orth = 0.0, nrel = 0.0;
    idx.route_check(orth, nrel);
    printf("rotation: max |Pi^T Pi - I| = %.3e; max rel | ||Pi^T c||^2 - ||c||^2 | = %.3e\n", orth, nrel);
#else
    std::vector<int> groups = {-1};       // the library's only primary execution
    auto set_variant = [&](int) {};
#endif
    const int nV = (int)groups.size();
    const int rounds = env_int("JHQ_BP_ROUNDS", 5);
    const int min_batches = env_int("JHQ_BP_MIN_BATCHES", 24);
    const bool check = env_int("JHQ_BP_CHECK", 1) != 0;

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    std::string jpath = std::string(out_prefix) + ".json";
    FILE* js = std::fopen(jpath.c_str(), "w");
    if (!js) { fprintf(stderr, "cannot write %s\n", jpath.c_str()); return 1; }
    std::fprintf(js, "{\"gpu\":\"%s\",\"sms\":%d,\"base\":\"%s\",\"n\":%d,\"d\":%d,\"nq\":%d,\"k\":%d,\"M\":%d,"
                 "\"nlist\":%d,\"nprobe\":%d,\"batch_cap\":%d,\"lanes\":%d,\"rounds\":%d,",
                 prop.name, prop.multiProcessorCount, base_path, nb, d, nq, k, M, nlist, nprobe,
                 batch_size, lanes, rounds);
#ifdef JHQ_PIPELINE_V64
    std::fprintf(js, "\"orth_max_abs\":%.6e,\"norm_max_rel\":%.6e,", orth, nrel);
#endif
#ifdef JHQ_PIPELINE_V65
    std::fprintf(js, "\"sched_cv\":%.5f,", jhq_gpu::schedule_skew());
#endif
    std::fprintf(js, "\"runs\":[");
    bool first_run = true;

    for (double a : alphas) {
        idx.set_alpha((float)a);
        const int ck = std::max(k, (int)std::ceil((float)a * (float)k));
#ifdef JHQ_PIPELINE_V62
        bool dumped = false;
#endif
        for (double bd : batches) {
            const int b = (int)bd;
            if (b > batch_size) { printf("skip b=%d > batch_size %d\n", b, batch_size); continue; }
            if (b > nq) { printf("skip b=%d > %d queries\n", b, nq); continue; }
            const int nbatch = std::max(1, nq / b);
            const int nused  = nbatch * b;
            const int reps   = std::max(1, (min_batches + nbatch - 1) / nbatch);
            printf("\n=== alpha=%g ck=%d  b=%d: %d batches x %d reps = %d queries per measurement ===\n",
                   a, ck, b, nbatch, reps, nused * reps);
            std::vector<float> dists((size_t)b * k);
            std::vector<int>   ids((size_t)b * k);

            // 1. results of every variant over the used queries
            std::vector<std::vector<int>>   rid(nV, std::vector<int>((size_t)nused * k));
            std::vector<std::vector<float>> rdist(nV, std::vector<float>((size_t)nused * k));
            std::vector<double> rec(nV, 0.0);
#ifdef JHQ_BP_CHECKS
            std::vector<std::vector<int>>   rprobe(nV), rtpos(nV);
            std::vector<std::vector<float>> rtval(nV);
#endif
#ifdef JHQ_PIPELINE_V62
            std::vector<double> scratch_mb(nV, 0.0);
#endif
#ifdef JHQ_PIPELINE_V64
            std::vector<std::vector<float>> rdots(nV);
#endif
            std::vector<double> used_mb(nV, 0.0);
            for (int vi = 0; vi < nV; ++vi) {
                set_variant(groups[vi]);
                idx.search(query.data(), b, k, dists.data(), ids.data());          // capture + JIT
                for (int bi = 0; bi < nbatch; ++bi) {
                    const float* qb = query.data() + (size_t)bi * b * d;
                    idx.search(qb, b, k, rdist[vi].data() + (size_t)bi * b * k, rid[vi].data() + (size_t)bi * b * k);
#ifdef JHQ_PIPELINE_V62
                    if (check) {
                        std::vector<int> pr, tp; std::vector<float> tv; int lB, lnp, lck;
                        idx.debug_last_batch(pr, tp, tv, lB, lnp, lck);
                        rprobe[vi].insert(rprobe[vi].end(), pr.begin(), pr.end());
                        rtpos[vi].insert(rtpos[vi].end(), tp.begin(), tp.end());
                        rtval[vi].insert(rtval[vi].end(), tv.begin(), tv.end());
                    }
#elif defined(JHQ_PIPELINE_V64)
                    if (check) {
                        std::vector<int> pr, tp; std::vector<float> tv, dt; int lB, lnp, lck, lnl;
                        idx.debug_last_batch(pr, tp, tv, dt, lB, lnp, lck, lnl);
                        rprobe[vi].insert(rprobe[vi].end(), pr.begin(), pr.end());
                        rtpos[vi].insert(rtpos[vi].end(), tp.begin(), tp.end());
                        rtval[vi].insert(rtval[vi].end(), tv.begin(), tv.end());
                        rdots[vi].insert(rdots[vi].end(), dt.begin(), dt.end());
                    }
#endif
                }
                rec[vi] = evaluate_recall(rid[vi].data(), gt.data(), nused, k, d_gt).recall;
                size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
                used_mb[vi] = (double)(tot - fr) / (1024.0 * 1024.0);
#ifdef JHQ_PIPELINE_V62
                scratch_mb[vi] = jhq_gpu::primary_group_scratch_bytes() / (1024.0 * 1024.0);
#endif
            }

            // 2. per-query comparison against variant 0 (v62 only)
            std::fprintf(js, "%s{\"alpha\":%g,\"ck\":%d,\"batch\":%d,\"nbatch\":%d,\"reps\":%d,\"nused\":%d,",
                         first_run ? "" : ",", a, ck, b, nbatch, reps, nused);
#ifdef JHQ_PIPELINE_V65
            std::fprintf(js, "\"sched_G\":%d,", jhq_gpu::scheduled_groups(b));
            printf("  scheduled G for b=%d: %d\n", b, jhq_gpu::scheduled_groups(b));
#endif
            std::fprintf(js, "\"variants\":[");
            first_run = false;
            std::vector<std::string> cmpjson(nV, "");
#ifdef JHQ_BP_CHECKS
            if (check) {
                for (int vi = 1; vi < nV; ++vi) {
                    int probe_same = 0, top_same = 0, top_tie_only = 0, top_nontie = 0;
                    double prim_abs = 0.0;
                    int ord = 0, set = 0, bits = 0; double fin_rel = 0.0;
                    int nontie_psame = 0;
                    for (int q = 0; q < nused; ++q) {
                        const bool psame = std::equal(rprobe[0].begin() + (size_t)q * nprobe, rprobe[0].begin() + (size_t)(q + 1) * nprobe,
                                                      rprobe[vi].begin() + (size_t)q * nprobe);
                        probe_same += psame;
                        // primary top-ck: sets of positions, and where they differ, whether only at ties
                        std::map<int, float> R, V;
                        float rmax = -1.f, vmax = -1.f;
                        for (int c = 0; c < ck; ++c) {
                            const int pr0 = rtpos[0][(size_t)q * ck + c], pr1 = rtpos[vi][(size_t)q * ck + c];
                            if (pr0 >= 0) { R[pr0] = rtval[0][(size_t)q * ck + c];  rmax = std::max(rmax, R[pr0]); }
                            if (pr1 >= 0) { V[pr1] = rtval[vi][(size_t)q * ck + c]; vmax = std::max(vmax, V[pr1]); }
                        }
                        bool same = (R.size() == V.size()), tie_only = true;
                        for (auto& kv : R) {
                            auto it = V.find(kv.first);
                            if (it == V.end()) { same = false; if (kv.second != rmax) tie_only = false; }
                            else prim_abs = std::max(prim_abs, (double)std::fabs(it->second - kv.second));
                        }
                        for (auto& kv : V) if (!R.count(kv.first)) { same = false; if (kv.second != vmax) tie_only = false; }
                        if (rmax != vmax) tie_only = false;
                        if (same) ++top_same; else if (tie_only) ++top_tie_only; else { ++top_nontie; nontie_psame += psame; }
                        // final top-k
                        const int* a0 = rid[0].data() + (size_t)q * k; const int* a1 = rid[vi].data() + (size_t)q * k;
                        const float* d0 = rdist[0].data() + (size_t)q * k; const float* d1 = rdist[vi].data() + (size_t)q * k;
                        bool o = true, bt = true;
                        for (int i = 0; i < k; ++i) {
                            if (a0[i] != a1[i]) o = false;
                            if (std::memcmp(&d0[i], &d1[i], sizeof(float))) bt = false;
                            if (a0[i] == a1[i] && std::isfinite(d0[i]))
                                fin_rel = std::max(fin_rel, std::fabs((double)d1[i] - d0[i]) / std::max(std::fabs((double)d0[i]), 1e-30));
                        }
                        std::vector<int> s0(a0, a0 + k), s1(a1, a1 + k);
                        std::sort(s0.begin(), s0.end()); std::sort(s1.begin(), s1.end());
                        ord += o; set += (s0 == s1); bits += (o && bt);
                    }
                    char buf[512];
                    std::snprintf(buf, sizeof(buf),
                        ",\"probe_same\":%d,\"topck_same\":%d,\"topck_tie_only\":%d,\"topck_nontie\":%d,"
                        "\"primary_max_abs\":%.6e,\"final_same_order\":%d,\"final_same_set\":%d,"
                        "\"final_bit_identical\":%d,\"final_max_rel\":%.6e",
                        probe_same, top_same, top_tie_only, top_nontie, prim_abs, ord, set, bits, fin_rel);
                    cmpjson[vi] = buf;
                    printf("  G=%d vs G=0: probes %d/%d  primary top-ck same %d, tie-only %d, NON-TIE %d, "
                           "max |dprimary| %.2e  final order/set/bits %d/%d/%d  max rel %.2e\n",
                           groups[vi], probe_same, nused, top_same, top_tie_only, top_nontie, prim_abs,
                           ord, set, bits, fin_rel);
#ifdef JHQ_PIPELINE_V64
                    // the routing distance is ||c||^2 - 2<q', c>, so its error is twice the dot's
                    double dist_abs = 0.0, dot_rel = 0.0;
                    for (int q = 0; q < nused; ++q) {
                        const float* a0 = rdots[0].data() + (size_t)q * nlist;
                        const float* a1 = rdots[vi].data() + (size_t)q * nlist;
                        double scale = 0.0, mx = 0.0;
                        for (int l = 0; l < nlist; ++l) {
                            scale = std::max(scale, (double)std::fabs(a0[l]));
                            mx = std::max(mx, std::fabs((double)a1[l] - (double)a0[l]));
                        }
                        dist_abs = std::max(dist_abs, 2.0 * mx);
                        dot_rel  = std::max(dot_rel, mx / std::max(scale, 1e-30));
                    }
                    char b2[256];
                    std::snprintf(b2, sizeof(b2), ",\"route_dist_max_abs\":%.6e,\"route_dot_max_rel\":%.6e,"
                                  "\"nontie_same_probes\":%d", dist_abs, dot_rel, nontie_psame);
                    cmpjson[vi] += b2;
                    printf("  route %d vs 0: max |d routing distance| %.3e, max |d dot| / max |dot| %.3e; "
                           "NON-TIE among queries with identical probes %d\n", groups[vi], dist_abs, dot_rel, nontie_psame);
#endif
                }
#ifdef JHQ_PIPELINE_V63
                // the shared bound must not change the result: same top-ck, in the same
                // order, with the same distances, and the same final answer as v62's G
                for (int vi = 1; vi < nV; ++vi) {
                    if (groups[vi] < 100) continue;
                    const int pj = (int)(std::find(groups.begin(), groups.end(), groups[vi] - 100) - groups.begin());
                    if (pj >= nV) continue;
                    int same_topck = 0, same_final = 0;
                    for (int q = 0; q < nused; ++q) {
                        same_topck += std::equal(rtpos[vi].begin() + (size_t)q * ck, rtpos[vi].begin() + (size_t)(q + 1) * ck,
                                                 rtpos[pj].begin() + (size_t)q * ck) &&
                                      !std::memcmp(rtval[vi].data() + (size_t)q * ck, rtval[pj].data() + (size_t)q * ck,
                                                   (size_t)ck * sizeof(float));
                        same_final += std::equal(rid[vi].begin() + (size_t)q * k, rid[vi].begin() + (size_t)(q + 1) * k,
                                                 rid[pj].begin() + (size_t)q * k) &&
                                      !std::memcmp(rdist[vi].data() + (size_t)q * k, rdist[pj].data() + (size_t)q * k,
                                                   (size_t)k * sizeof(float));
                    }
                    char b3[192];
                    std::snprintf(b3, sizeof(b3), ",\"vs_independent_topck_identical\":%d,\"vs_independent_final_identical\":%d",
                                  same_topck, same_final);
                    cmpjson[vi] += b3;
                    printf("  G=%d shared bound vs independent: top-ck identical %d/%d, final identical %d/%d\n",
                           groups[vi] - 100, same_topck, nused, same_final, nused);
                }
#endif
#ifdef JHQ_PIPELINE_V62
                const int v1 = (int)(std::find(groups.begin(), groups.end(), 1) - groups.begin());
                if (dump && !dumped && v1 < nV) {
                    char dp[512];
                    std::snprintf(dp, sizeof(dp), "%s_a%g.dump", out_prefix, a);
                    if (FILE* f = std::fopen(dp, "wb")) {
                        const int hdr[4] = {nused, nprobe, ck, nlist};
                        std::fwrite(hdr, sizeof(int), 4, f);
                        std::fwrite(list_sizes.data(), sizeof(int), list_sizes.size(), f);
                        std::fwrite(rprobe[v1].data(), sizeof(int), (size_t)nused * nprobe, f);
                        std::fwrite(rtpos[v1].data(), sizeof(int), (size_t)nused * ck, f);
                        std::fwrite(rtval[v1].data(), sizeof(float), (size_t)nused * ck, f);
                        std::fclose(f);
                        printf("  dumped G=1 probes and top-ck to %s\n", dp);
                    }
                    dumped = true;
                }
#endif
            }
#endif
            // 3. timing: rounds, variant order rotated, one NVTX range per (variant, round)
            std::vector<std::vector<double>> lat_med(nV), lat_p95(nV), qps(nV);
            for (int r = 0; r < rounds; ++r)
                for (int t = 0; t < nV; ++t) {
                    const int vi = (r + t) % nV;
                    set_variant(groups[vi]);
                    idx.search(query.data(), b, k, dists.data(), ids.data());      // capture for this variant
                    char rn[128];
                    std::snprintf(rn, sizeof(rn), "a%g|b%d|G%d|r%d", a, b, groups[vi], r);
                    std::vector<double> lat; lat.reserve((size_t)reps * nbatch);
                    double sum = 0.0;
                    nvtxRangePushA(rn);
                    for (int rep = 0; rep < reps; ++rep)
                        for (int bi = 0; bi < nbatch; ++bi) {
                            auto tb = Clock::now();
                            idx.search(query.data() + (size_t)bi * b * d, b, k, dists.data(), ids.data());
                            const double ms = Ms(Clock::now() - tb).count();
                            lat.push_back(ms); sum += ms;
                        }
                    nvtxRangePop();
                    lat_med[vi].push_back(quantile(lat, 0.5));
                    lat_p95[vi].push_back(quantile(lat, 0.95));
                    qps[vi].push_back((double)reps * nused / (sum / 1000.0));
                }
            printf("  %-4s %8s %10s %10s %11s %11s %11s %9s\n", "G", "recall", "lat med", "lat p95",
                   "QPS med", "QPS min", "QPS max", "used MiB");
            for (int vi = 0; vi < nV; ++vi) {
                // JHQ_BP_ROUNDS=0 is a results-only run: no timing to print
                const double qmin = qps[vi].empty() ? 0.0 : *std::min_element(qps[vi].begin(), qps[vi].end());
                const double qmax = qps[vi].empty() ? 0.0 : *std::max_element(qps[vi].begin(), qps[vi].end());
                printf("  %-4d %8.4f %8.3f ms %8.3f ms %11.0f %11.0f %11.0f %9.0f\n", groups[vi], rec[vi],
                       quantile(lat_med[vi], 0.5), quantile(lat_p95[vi], 0.5), quantile(qps[vi], 0.5),
                       qmin, qmax, used_mb[vi]);
                std::fprintf(js, "%s{\"G\":%d,\"recall\":%.6f,\"used_mib\":%.1f", vi ? "," : "", groups[vi], rec[vi], used_mb[vi]);
#ifdef JHQ_PIPELINE_V62
                std::fprintf(js, ",\"group_scratch_mib\":%.2f", scratch_mb[vi]);
#endif
                std::fprintf(js, ",\"lat_med_ms\":[");
                for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.5f", r ? "," : "", lat_med[vi][r]);
                std::fprintf(js, "],\"lat_p95_ms\":[");
                for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.5f", r ? "," : "", lat_p95[vi][r]);
                std::fprintf(js, "],\"qps\":[");
                for (int r = 0; r < rounds; ++r) std::fprintf(js, "%s%.1f", r ? "," : "", qps[vi][r]);
                std::fprintf(js, "]%s}", cmpjson[vi].c_str());
            }
            std::fprintf(js, "]}");
        }
    }
    std::fprintf(js, "]}\n");
    std::fclose(js);
    printf("\nwrote %s\n=== BP_OK ===\n", jpath.c_str());
    return 0;
}
