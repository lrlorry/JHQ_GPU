// Progressive primary screening pilot (experiments/progressive_primary): the
// primary codes and rotated queries the offline study needs, nothing else.
//
// Same index (arms6 cache), same seeded 1000-query sample as
// experiments/local_first_refinement, whose per-candidate files (probe order,
// D_prim, D_H, ground truth, v65's results) the study reuses. Writes into
// <out_dir>:
//   meta.json          shape, levels, and the checks below
//   primary.u8         [N][M] code bytes, decoded from the word layout the scan
//                      reads: byte i of word (m/4)*N + pos is subspace m
//   list_offsets.i32   [nlist + 1]
//   qrot.f64           [nS][d] q' = Pi q in FP64, in the sample's order
//   qids.i32           [nS] the sample's query ids
// Checks: the two levels are exactly symmetric (+c, -c bit for bit); every
// codebook entry is the level of its digit; and the word <-> dimension map:
// word w holds subspaces 4w..4w+3 = dimensions 32w..32w+31, byte i of the
// word = subspace 4w+i, and within a byte the most significant bit is the
// subspace's first dimension (digit 1 = +c).
//
// Environment: JHQ_RB_NQ (1000), JHQ_RB_SEED (20261013), index variables as
// demo_jhq_batch_pipeline.

#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v66_boundpilot/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr, "Usage: %s <base> <query> <gt> <M> <B> <Br> <alpha> <k> <nlist> <nprobe> "
                        "<ivf_iters> <batch_size> <out_dir> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1]; const char* query_path = argv[2];
    int M = atoi(argv[4]), Bb = atoi(argv[5]), Br = atoi(argv[6]);
    float alpha = (float)atof(argv[7]);
    int nlist = atoi(argv[9]), nprobe = atoi(argv[10]);
    int ivf_iters = atoi(argv[11]), batch_size = atoi(argv[12]);
    const std::string out = argv[13];
    int kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    std::vector<float> query; int d_query;
    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    const int nb = base.n, d = base.d;
    const int nq = read_fvecs(query_path, query, d_query);

    jhq_gpu::JHQGpuIndex::Params p;
    p.M = M; p.B = Bb; p.Br = Br; p.alpha = alpha; p.nlist = nlist; p.nprobe = nprobe;
    p.ivf_iters = ivf_iters; p.batch_size = batch_size; p.kmeans_iters = kmeans_iters;
    if (const char* ab = std::getenv("JHQ_ADD_BATCH")) { const int v = std::atoi(ab); if (v > 0) p.add_batch = v; }
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
    idx.train(train_src, n_train, res_src, n_res_train);
    idx.add(base.data, nb);

    // the sample of experiments/local_first_refinement: same seed, same order
    const int nS = std::min(env_int("JHQ_RB_NQ", 1000), std::min(nq, batch_size));
    const unsigned seed = (unsigned)env_int("JHQ_RB_SEED", 20261013);
    std::vector<int> order(nq);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int> qsel(order.begin(), order.begin() + nS);

    jhq_gpu::JHQGpuIndex::HostState S;
    idx.export_host_state(S);
    const long long N = S.ntotal;
    const int Ds = S.Ds, K = S.K, MS = S.M;

    // checks
    bool sym = (S.L == 2) && (S.levels[0] == -S.levels[1]);
    long long lvl_bad = 0;
    for (int m = 0; m < MS; ++m)
        for (int c = 0; c < K; ++c)
            for (int j = 0; j < Ds; ++j) {
                const int digit = (c >> (Ds - 1 - j)) & 1;
                if (S.cent[((size_t)m * K + c) * Ds + j] != S.levels[digit]) ++lvl_bad;
            }
    printf("levels %.9g %.9g symmetric=%d; codebook entries not equal to their digit's level: %lld\n",
           S.levels[0], S.levels[1], (int)sym, lvl_bad);

    const std::string mk = "mkdir -p " + out;
    if (std::system(mk.c_str()) != 0) return 1;
    FILE* f = std::fopen((out + "/primary.u8").c_str(), "wb");
    std::fwrite(S.primary.data(), 1, S.primary.size(), f); std::fclose(f);
    f = std::fopen((out + "/list_offsets.i32").c_str(), "wb");
    std::fwrite(S.list_offsets.data(), sizeof(int), S.list_offsets.size(), f); std::fclose(f);
    // the same FP64 product, in the same order, as demo_jhq_v66_bound_pilot
    std::vector<double> qr((size_t)nS * d, 0.0);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < nS; ++i) {
        const float* q = query.data() + (size_t)qsel[i] * d;
        double* o = qr.data() + (size_t)i * d;
        for (int kk = 0; kk < d; ++kk) {
            const double qk = q[kk];
            const float* col = S.Pi.data() + (size_t)kk * d;
            for (int r = 0; r < d; ++r) o[r] += (double)col[r] * qk;
        }
    }
    f = std::fopen((out + "/qrot.f64").c_str(), "wb");
    std::fwrite(qr.data(), sizeof(double), qr.size(), f); std::fclose(f);
    f = std::fopen((out + "/qids.i32").c_str(), "wb");
    std::fwrite(qsel.data(), sizeof(int), qsel.size(), f); std::fclose(f);
    f = std::fopen((out + "/meta.json").c_str(), "w");
    std::fprintf(f, "{\"N\":%lld,\"d\":%d,\"M\":%d,\"Ds\":%d,\"K\":%d,\"nlist\":%d,\"nprobe\":%d,\"alpha\":%g,\"seed\":%u,\"n_queries\":%d,"
                 "\"levels\":[%.9g,%.9g],\"levels_symmetric\":%s,\"codebook_level_mismatches\":%lld,"
                 "\"words_per_vector\":%d,\"word_map\":\"word w = subspaces 4w..4w+3 = dims 32w..32w+31; byte i of the word (bits 8i..8i+7) = subspace 4w+i; MSB of a byte = the subspace's first dim; bit 1 = +c\"}\n",
                 N, d, MS, Ds, K, nlist, nprobe, alpha, seed, nS, S.levels[0], S.levels[1], sym ? "true" : "false", lvl_bad, d / 32);
    std::fclose(f);
    printf("wrote %s: N=%lld d=%d M=%d, %d queries\n=== EP_OK ===\n", out.c_str(), N, d, MS, nS);
    return 0;
}
