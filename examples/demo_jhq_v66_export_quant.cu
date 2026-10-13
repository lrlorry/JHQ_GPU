// Rank-optimal quantization study (experiments/rank_optimal_quantization): the
// rest of the representation an offline re-encoding needs, read-only. The
// primary codes, list offsets and rotated queries are
// experiments/progressive_primary's export (demo_jhq_v66_export_primary) of the
// same index; this adds what re-deriving every code from the base vectors needs.
// Writes into <out_dir>:
//   meta.json      shape, levels, residual parameters
//   Pi.f32         [d*d] the rotation as stored: q'_i = sum_k Pi[k*d+i] q_k
//   res_c1d.f32    [M][Kr] residual scalar codebooks (sorted)
//   res_codes.u8   [N][bpv] residual codes, position order
//   corr.f32       [N] stored correction 2<y_hat, r_hat>, position order
//   list_ids.i32   [N] vector id at each position
//
// Environment: index variables as demo_jhq_batch_pipeline.

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
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr, "Usage: %s <base> <query> <gt> <M> <B> <Br> <alpha> <k> <nlist> <nprobe> "
                        "<ivf_iters> <batch_size> <out_dir> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1];
    int M = atoi(argv[4]), Bb = atoi(argv[5]), Br = atoi(argv[6]);
    float alpha = (float)atof(argv[7]);
    int nlist = atoi(argv[9]), nprobe = atoi(argv[10]);
    int ivf_iters = atoi(argv[11]), batch_size = atoi(argv[12]);
    const std::string out = argv[13];
    int kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    const int nb = base.n, d = base.d;

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

    jhq_gpu::JHQGpuIndex::HostState S;
    idx.export_host_state(S);
    const long long N = S.ntotal;
    const std::string mk = "mkdir -p " + out;
    if (std::system(mk.c_str()) != 0) return 1;
    auto dump = [&](const char* name, const void* p, size_t bytes) {
        FILE* f = std::fopen((out + "/" + name).c_str(), "wb");
        if (!f || std::fwrite(p, 1, bytes, f) != bytes) { fprintf(stderr, "write %s failed\n", name); std::exit(1); }
        std::fclose(f);
    };
    dump("Pi.f32", S.Pi.data(), S.Pi.size() * sizeof(float));
    dump("res_c1d.f32", S.res_c1d.data(), S.res_c1d.size() * sizeof(float));
    dump("res_codes.u8", S.res.data(), S.res.size());
    dump("corr.f32", S.corr.data(), S.corr.size() * sizeof(float));
    dump("list_ids.i32", S.list_ids.data(), S.list_ids.size() * sizeof(int));
    long long unsorted = 0;
    for (int m = 0; m < S.M; ++m)
        for (int c = 1; c < S.Kr; ++c)
            if (S.res_c1d[(size_t)m * S.Kr + c] < S.res_c1d[(size_t)m * S.Kr + c - 1]) ++unsorted;
    FILE* f = std::fopen((out + "/meta.json").c_str(), "w");
    std::fprintf(f, "{\"N\":%lld,\"d\":%d,\"M\":%d,\"Ds\":%d,\"K\":%d,\"Kr\":%d,\"Br\":%d,\"bpv\":%d,\"nlist\":%d,"
                 "\"levels\":[%.9g,%.9g],\"n_train\":%d,\"res_train_n\":%d,\"res_unsorted_pairs\":%lld}\n",
                 N, S.d, S.M, S.Ds, S.K, S.Kr, S.Br, S.bpv, S.nlist,
                 S.levels.empty() ? 0.0 : S.levels[0], S.levels.size() < 2 ? 0.0 : S.levels[1],
                 n_train, n_res_train, unsorted);
    std::fclose(f);
    printf("wrote %s: N=%lld d=%d M=%d Kr=%d bpv=%d, residual codebook unsorted pairs %lld\n=== EQ_OK ===\n",
           out.c_str(), N, S.d, S.M, S.Kr, S.bpv, unsorted);
    return 0;
}
