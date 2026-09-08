// IVF-RaBitQ against JHQ, on one machine and one protocol.
//
// cuVS 26.08.01 ships ivf_rabitq in libcuvs.so and in the installed headers,
// but not in the Python package -- `from cuvs.neighbors import ivf_rabitq`
// fails while `nm -D libcuvs.so | grep rabitq` returns seventeen defined
// symbols. So this links the C++ API directly rather than building the fork
// from source.
//
// The timed region matches examples/demo_jhq_v36.cu exactly: host queries in,
// host neighbours and distances out, the whole query set per call. cuVS's own
// harness times the kernel with the queries already resident, which on a
// 1.6 ms call is worth about 10%; that asymmetry is the reason scripts/
// bench_all.py now records both.
//
// bits_per_dim is the total including the 1-bit code, so the budgets that
// match JHQ are:
//     JHQ Br=4  ->  1 + 4 = 5 bits/dim
//     JHQ Br=8  ->  1 + 8 = 9 bits/dim
// Passing 8 against JHQ's Br=8 would give RaBitQ 11% less memory.
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static std::vector<float> read_fvecs(const char* path, int& n, int& d) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long bytes = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    int dim = 0;
    if (std::fread(&dim, 4, 1, f) != 1) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    d = dim;
    n = (int)(bytes / (4L + 4L * dim));
    std::vector<float> v((size_t)n * d);
    std::fseek(f, 0, SEEK_SET);
    for (int i = 0; i < n; ++i) {
        int dd = 0;
        if (std::fread(&dd, 4, 1, f) != 1 ||
            std::fread(v.data() + (size_t)i * d, 4, d, f) != (size_t)d) {
            std::fprintf(stderr, "short read %s at row %d\n", path, i); std::exit(1);
        }
    }
    std::fclose(f);
    return v;
}

static std::vector<int> read_ivecs(const char* path, int& n, int& d) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long bytes = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    int dim = 0;
    if (std::fread(&dim, 4, 1, f) != 1) { std::exit(1); }
    d = dim;
    n = (int)(bytes / (4L + 4L * dim));
    std::vector<int> v((size_t)n * d);
    std::fseek(f, 0, SEEK_SET);
    for (int i = 0; i < n; ++i) {
        int dd = 0;
        if (std::fread(&dd, 4, 1, f) != 1 ||
            std::fread(v.data() + (size_t)i * d, 4, d, f) != (size_t)d) std::exit(1);
    }
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> <n_lists> "
            "<bits_per_dim> <n_probes> <k> [mode] [reps]\n"
            "  mode: 0=LUT16 1=LUT32 2=QUANT4 3=QUANT8 (default 2)\n", argv[0]);
        return 1;
    }
    const char* base_p = argv[1];
    const char* qry_p  = argv[2];
    const char* gt_p   = argv[3];
    const uint32_t n_lists  = (uint32_t)std::atoi(argv[4]);
    const uint32_t bits     = (uint32_t)std::atoi(argv[5]);
    const uint32_t n_probes = (uint32_t)std::atoi(argv[6]);
    const int      k        = std::atoi(argv[7]);
    const int      mode_i   = (argc > 8) ? std::atoi(argv[8]) : 2;
    const int      reps     = (argc > 9) ? std::atoi(argv[9]) : 5;

    int nb = 0, d = 0, nq = 0, dq = 0, ng = 0, dgt = 0;
    auto xb = read_fvecs(base_p, nb, d);
    auto xq = read_fvecs(qry_p,  nq, dq);
    auto gt = read_ivecs(gt_p,   ng, dgt);
    if (dq != d) { std::fprintf(stderr, "query dim %d != base dim %d\n", dq, d); return 1; }
    std::printf("base=%dx%d  query=%dx%d  gt=%dx%d\n", nb, d, nq, dq, ng, dgt);
    std::printf("n_lists=%u  bits_per_dim=%u  n_probes=%u  k=%d  mode=%d\n",
                n_lists, bits, n_probes, k, mode_i);

    raft::device_resources res;

    size_t free0 = 0, total0 = 0;
    cudaMemGetInfo(&free0, &total0);

    cuvs::neighbors::ivf_rabitq::index_params ip;
    ip.n_lists      = n_lists;
    ip.bits_per_dim = bits;
    // The dataset stays on the host; cuVS streams it in, which is what a
    // 12 GiB base at d=3072 needs on a 32 GiB card.
    auto xb_host = raft::make_host_matrix_view<const float, int64_t>(xb.data(), nb, d);

    auto t0 = clk::now();
    auto idx = cuvs::neighbors::ivf_rabitq::build(res, ip, xb_host);
    raft::resource::sync_stream(res);
    const double build_ms = ms_since(t0);

    size_t free1 = 0, total1 = 0;
    cudaMemGetInfo(&free1, &total1);

    cuvs::neighbors::ivf_rabitq::search_params sp;
    sp.n_probes = n_probes;
    sp.mode = static_cast<cuvs::neighbors::ivf_rabitq::search_mode>(mode_i);

    auto d_q  = raft::make_device_matrix<float,   int64_t>(res, nq, d);
    auto d_id = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto d_di = raft::make_device_matrix<float,   int64_t>(res, nq, k);
    std::vector<int64_t> h_id((size_t)nq * k);
    std::vector<float>   h_di((size_t)nq * k);

    // One call to match JHQ's timed region: queries up, search, results back.
    auto one_pass = [&]() {
        raft::copy(d_q.data_handle(), xq.data(), (size_t)nq * d,
                   raft::resource::get_cuda_stream(res));
        cuvs::neighbors::ivf_rabitq::search(res, sp, idx, d_q.view(),
                                            d_id.view(), d_di.view());
        raft::copy(h_id.data(), d_id.data_handle(), (size_t)nq * k,
                   raft::resource::get_cuda_stream(res));
        raft::copy(h_di.data(), d_di.data_handle(), (size_t)nq * k,
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
    };

    one_pass();                       // warm up
    std::vector<double> rep_ms;
    for (int r = 0; r < reps; ++r) {
        auto t = clk::now();
        one_pass();
        rep_ms.push_back(ms_since(t));
    }
    double ms = 0.0;
    for (double v : rep_ms) ms += v;
    ms /= reps;

    // Recall@k against the first k of the ground truth, the same definition
    // common/recall.cuh uses.
    long long hit = 0;
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            const int want = gt[(size_t)q * dgt + j];
            for (int i = 0; i < k; ++i)
                if ((long long)h_id[(size_t)q * k + i] == (long long)want) { ++hit; break; }
        }
    }
    const double recall = (double)hit / ((double)nq * k);

    std::printf("\nRecall@%d : %.4f\n", k, recall);
    std::printf("Latency   : %.2f ms  (%d queries)\n", ms, nq);
    std::printf("QPS       : %.0f\n", nq / (ms / 1000.0));
    std::printf("  train: %.1f ms\n", build_ms);
    std::printf("VRAM used : %.1f MiB\n", (double)(free0 - free1) / (1024.0 * 1024.0));
    std::printf("rep_ms    :");
    for (double v : rep_ms) std::printf(" %.3f", v);
    std::printf("\n");
    return 0;
}
