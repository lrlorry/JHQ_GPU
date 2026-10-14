// COMP8800 poster, experiment E1: cuVS IVF-RaBitQ on the protocol JHQ v69's harness measures.
//
//   bench_rabitq_sweep <base.fvecs> <query.fvecs> <gt.ivecs> <n_lists> <bits_per_dim> <out.jsonl>
//   env RQ_NPROBES  comma list of n_probes            (default 16,32,64,96,128,192,256,384,512,768,1024)
//       RQ_MODES    0=LUT16 1=LUT32 2=QUANT4 3=QUANT8  (default 0,1,2,3)
//       RQ_WARMUP 3, RQ_ROUNDS 5, RQ_CALLS 24, RQ_SLOW_MS 40
//       RQ_IDX      the serialize/deserialize round-trip file (keep it on the data disk)
//
// Taken from examples/bench_ivf_rabitq.cu, whose measured facts it keeps -- the dataset goes up
// as a device matrix (the host path returned ids that do not index the dataset), the metric is
// L2Expanded, and the index is round-tripped through serialize/deserialize before any search
// (build() does not leave it in the layout search reads; self-recall 0/20 without, 20/20 with).
// What changed, each in RaBitQ's favour or neutral:
//   * one build and one round-trip per process, then every (mode, n_probes) is searched on the
//     same index -- the old benchmark rebuilt it for every point. Build, serialize and
//     deserialize are each timed and reported; none of them is in a search number;
//   * the queries go up from a pinned staging buffer, as JHQ's search() does (the old one copied
//     from pageable memory);
//   * every timed call is kept, protocol B (queries resident, search only) is timed beside A,
//     and the ids are checked: range, a repeated id within a row, the same ids on every call.
//
// Protocol A (host to host, the poster's): memcpy of the host fp32 queries into the pinned
// buffer -> H2D -> search -> D2H of ids and distances -> stream synchronise. One call answers
// the whole query set. Output: one JSON object per line ("meta", "index", "point", "failure").
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static std::vector<float> read_fvecs(const char* path, int& n, int& d) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long bytes = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    int dim = 0;
    if (std::fread(&dim, 4, 1, f) != 1) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    d = dim; n = (int)(bytes / (4L + 4L * dim));
    std::vector<float> v((size_t)n * d);
    std::fseek(f, 0, SEEK_SET);
    for (int i = 0; i < n; ++i) {
        int dd = 0;
        if (std::fread(&dd, 4, 1, f) != 1 || std::fread(v.data() + (size_t)i * d, 4, d, f) != (size_t)d) {
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
    if (std::fread(&dim, 4, 1, f) != 1) std::exit(1);
    d = dim; n = (int)(bytes / (4L + 4L * dim));
    std::vector<int> v((size_t)n * d);
    std::fseek(f, 0, SEEK_SET);
    for (int i = 0; i < n; ++i) {
        int dd = 0;
        if (std::fread(&dd, 4, 1, f) != 1 || std::fread(v.data() + (size_t)i * d, 4, d, f) != (size_t)d) std::exit(1);
    }
    std::fclose(f);
    return v;
}

static std::vector<int> parse_ints(const char* s, const char* dflt) {
    std::vector<int> v; std::string t(s ? s : dflt);
    size_t p = 0;
    while (p < t.size()) {
        size_t c = t.find(',', p); if (c == std::string::npos) c = t.size();
        if (c > p) v.push_back(std::atoi(t.substr(p, c - p).c_str()));
        p = c + 1;
    }
    return v;
}
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }
// rows of b that hold a different id set from a, and rows equal as sets but in another order
static void row_diffs(const std::vector<int64_t>& a, const int64_t* b, int nq, int k, long long& set_diff,
                      long long& order_diff) {
    set_diff = order_diff = 0;
    for (int q = 0; q < nq; ++q) {
        std::vector<int64_t> x(a.begin() + (size_t)q * k, a.begin() + (size_t)(q + 1) * k), y(b + (size_t)q * k, b + (size_t)(q + 1) * k);
        if (x == y) continue;
        std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
        if (x == y) ++order_diff; else ++set_diff;
    }
}
static double recall_of(const int64_t* ids, const std::vector<int>& gt, int nq, int k, int dgt) {
    long long hit = 0;
    for (int q = 0; q < nq; ++q)
        for (int j = 0; j < k; ++j) {
            const long long want = gt[(size_t)q * dgt + j];
            for (int i = 0; i < k; ++i) if (ids[(size_t)q * k + i] == want) { ++hit; break; }
        }
    return (double)hit / ((double)nq * k);
}
static double used_mib() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return (double)(t - f) / 1048576.0; }
static std::string utc() {
    char b[32]; std::time_t t = std::time(nullptr); std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return b;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}
static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end()); if (v.empty()) return 0.0;
    const double x = q * (v.size() - 1); const size_t i = (size_t)x;
    return i + 1 < v.size() ? v[i] + (x - i) * (v[i + 1] - v[i]) : v[i];
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> <n_lists> <bits_per_dim> <out.jsonl>\n",
                     argv[0]);
        return 1;
    }
    const uint32_t n_lists = (uint32_t)std::atoi(argv[4]);
    const uint32_t bits    = (uint32_t)std::atoi(argv[5]);
    const int k = 10;
    FILE* js = std::fopen(argv[6], "a");
    if (!js) { std::fprintf(stderr, "cannot write %s\n", argv[6]); return 1; }
    const std::vector<int> nprobes = parse_ints(std::getenv("RQ_NPROBES"), "16,32,64,96,128,192,256,384,512,768,1024");
    const std::vector<int> modes   = parse_ints(std::getenv("RQ_MODES"), "0,1,2,3");
    const int warmup = std::max(1, env_int("RQ_WARMUP", 3));
    const int rounds = env_int("RQ_ROUNDS", 5), calls0 = env_int("RQ_CALLS", 24);
    const double slow_ms = env_int("RQ_SLOW_MS", 40);
    const char* rt_file = std::getenv("RQ_IDX") ? std::getenv("RQ_IDX") : "/root/autodl-tmp/rq_e1.idx";

    raft::device_resources res;
    cudaFree(0);
    const double ctx_mib = used_mib();

    int nb = 0, d = 0, nq = 0, dq = 0, ng = 0, dgt = 0;
    auto xb = read_fvecs(argv[1], nb, d);
    auto xq = read_fvecs(argv[2], nq, dq);
    auto gt = read_ivecs(argv[3], ng, dgt);
    if (dq != d || dgt < k || ng < nq) {
        std::fprintf(stderr, "shapes: base %dx%d query %dx%d gt %dx%d\n", nb, d, nq, dq, ng, dgt); return 1;
    }
    std::fprintf(js, "{\"type\":\"meta\",\"method\":\"ivf-rabitq\",\"base\":\"%s\",\"n\":%d,\"d\":%d,\"nq\":%d,"
                 "\"gt_width\":%d,\"k\":%d,\"n_lists\":%u,\"bits_per_dim\":%u,\"ctx_mib\":%.1f,\"utc\":\"%s\","
                 "\"protocol_A\":\"pinned staging + H2D + search + D2H ids and distances + sync\","
                 "\"protocol_B\":\"resident queries, search only + sync\"}\n",
                 argv[1], nb, d, nq, dgt, k, n_lists, bits, ctx_mib, utc().c_str());
    std::fflush(js);

    // ---- build once, round-trip once --------------------------------------------------------
    cuvs::neighbors::ivf_rabitq::index_params ip;
    ip.n_lists = n_lists; ip.bits_per_dim = bits;
    ip.metric = cuvs::distance::DistanceType::L2Expanded;
    cuvs::neighbors::ivf_rabitq::index<int64_t> idx(res);
    double t_upload = 0, t_build = 0, t_ser = 0, t_deser = 0, m_before = 0, m_index = 0;
    try {
        m_before = used_mib();
        {
            auto t0 = clk::now();
            auto d_base = raft::make_device_matrix<float, int64_t>(res, nb, d);
            cudaMemcpy(d_base.data_handle(), xb.data(), (size_t)nb * d * sizeof(float), cudaMemcpyHostToDevice);
            t_upload = ms_since(t0);
            t0 = clk::now();
            auto built = cuvs::neighbors::ivf_rabitq::build(
                res, ip, raft::make_device_matrix_view<const float, int64_t>(d_base.data_handle(), nb, d));
            raft::resource::sync_stream(res);
            t_build = ms_since(t0);
            t0 = clk::now();
            cuvs::neighbors::ivf_rabitq::serialize(res, rt_file, built);
            raft::resource::sync_stream(res);
            t_ser = ms_since(t0);
        }   // the device copy of the dataset and the unreorganised index are released here
        auto t0 = clk::now();
        cuvs::neighbors::ivf_rabitq::deserialize(res, rt_file, &idx);
        raft::resource::sync_stream(res);
        t_deser = ms_since(t0);
        cudaDeviceSynchronize();
        m_index = used_mib() - m_before;
    } catch (const std::exception& ex) {
        std::fprintf(js, "{\"type\":\"failure\",\"method\":\"ivf-rabitq\",\"stage\":\"build\",\"n_lists\":%u,"
                     "\"bits_per_dim\":%u,\"error\":\"%.300s\",\"used_mib\":%.1f,\"utc\":\"%s\"}\n",
                     n_lists, bits, ex.what(), used_mib(), utc().c_str());
        std::fclose(js); std::remove(rt_file);
        std::fprintf(stderr, "build failed: %s\n", ex.what());
        return 2;
    }
    std::remove(rt_file);
    std::fprintf(js, "{\"type\":\"index\",\"method\":\"ivf-rabitq\",\"n_lists\":%u,\"bits_per_dim\":%u,"
                 "\"size\":%lld,\"dim\":%u,\"upload_ms\":%.1f,\"build_ms\":%.1f,\"serialize_ms\":%.1f,"
                 "\"deserialize_ms\":%.1f,\"index_mib\":%.1f,\"ctx_mib\":%.1f,\"utc\":\"%s\"}\n",
                 n_lists, bits, (long long)idx.size(), idx.dim(), t_upload, t_build, t_ser, t_deser, m_index,
                 ctx_mib, utc().c_str());
    std::fflush(js);
    std::fprintf(stderr, "index: size %lld (base %d), build %.0f ms, round trip %.0f ms, %.0f MiB\n",
                 (long long)idx.size(), nb, t_build, t_ser + t_deser, m_index);
    if ((long long)idx.size() != (long long)nb) { std::fprintf(stderr, "index size != base rows\n"); return 3; }

    // ---- buffers, allocated once ------------------------------------------------------------
    auto st = raft::resource::get_cuda_stream(res);
    float* pin_q = nullptr; int64_t* pin_id = nullptr; float* pin_di = nullptr;
    cudaHostAlloc(&pin_q, (size_t)nq * d * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&pin_id, (size_t)nq * k * sizeof(int64_t), cudaHostAllocDefault);
    cudaHostAlloc(&pin_di, (size_t)nq * k * sizeof(float), cudaHostAllocDefault);
    auto d_q  = raft::make_device_matrix<float, int64_t>(res, nq, d);
    auto d_qr = raft::make_device_matrix<float, int64_t>(res, nq, d);     // protocol B's resident copy
    auto d_id = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto d_di = raft::make_device_matrix<float, int64_t>(res, nq, k);
    auto d_idB = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto d_diB = raft::make_device_matrix<float, int64_t>(res, nq, k);
    cudaMemcpy(d_qr.data_handle(), xq.data(), (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<int64_t> ref((size_t)nq * k), hb((size_t)nq * k);
    const char* mode_name[4] = {"LUT16", "LUT32", "QUANT4", "QUANT8"};

    for (int mode : modes) {
        for (int np : nprobes) {
            if (np > (int)n_lists) continue;
            cuvs::neighbors::ivf_rabitq::search_params sp;
            sp.n_probes = (uint32_t)np;
            sp.mode = static_cast<cuvs::neighbors::ivf_rabitq::search_mode>(mode);
            auto qv  = raft::make_device_matrix_view<const float, int64_t>(d_q.data_handle(), nq, d);
            auto qvB = raft::make_device_matrix_view<const float, int64_t>(d_qr.data_handle(), nq, d);
            auto iv  = raft::make_device_matrix_view<int64_t, int64_t>(d_id.data_handle(), nq, k);
            auto dv  = raft::make_device_matrix_view<float, int64_t>(d_di.data_handle(), nq, k);
            auto ivB = raft::make_device_matrix_view<int64_t, int64_t>(d_idB.data_handle(), nq, k);
            auto dvB = raft::make_device_matrix_view<float, int64_t>(d_diB.data_handle(), nq, k);
            auto call_a = [&]() {
                std::memcpy(pin_q, xq.data(), (size_t)nq * d * sizeof(float));
                cudaMemcpyAsync(d_q.data_handle(), pin_q, (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice, st);
                cuvs::neighbors::ivf_rabitq::search(res, sp, idx, qv, iv, dv);
                cudaMemcpyAsync(pin_id, d_id.data_handle(), (size_t)nq * k * sizeof(int64_t), cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync(pin_di, d_di.data_handle(), (size_t)nq * k * sizeof(float), cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
            };
            auto call_b = [&]() {
                cuvs::neighbors::ivf_rabitq::search(res, sp, idx, qvB, ivB, dvB);
                cudaStreamSynchronize(st);
            };
            if (std::getenv("RQ_DIAG")) {
                // where the time of a host-to-host call goes: each phase synchronised on its own
                const int n = env_int("RQ_DIAG", 200);
                std::vector<double> tm, th, ts, td, tall;
                for (int i = 0; i < n; ++i) {
                    auto t = clk::now();
                    std::memcpy(pin_q, xq.data(), (size_t)nq * d * sizeof(float)); tm.push_back(ms_since(t)); t = clk::now();
                    cudaMemcpyAsync(d_q.data_handle(), pin_q, (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st); th.push_back(ms_since(t)); t = clk::now();
                    cuvs::neighbors::ivf_rabitq::search(res, sp, idx, qv, iv, dv);
                    cudaStreamSynchronize(st); ts.push_back(ms_since(t)); t = clk::now();
                    cudaMemcpyAsync(pin_id, d_id.data_handle(), (size_t)nq * k * sizeof(int64_t), cudaMemcpyDeviceToHost, st);
                    cudaMemcpyAsync(pin_di, d_di.data_handle(), (size_t)nq * k * sizeof(float), cudaMemcpyDeviceToHost, st);
                    cudaStreamSynchronize(st); td.push_back(ms_since(t));
                    tall.push_back(tm.back() + th.back() + ts.back() + td.back());
                }
                auto q3 = [](std::vector<double> v) {
                    char b[96]; std::snprintf(b, sizeof b, "p50 %.3f p95 %.3f p99 %.3f max %.3f", pct(v, .5), pct(v, .95), pct(v, .99), pct(v, 1.0));
                    return std::string(b);
                };
                std::fprintf(stderr, "DIAG mode=%d np=%d n=%d\n  memcpy %s\n  h2d    %s\n  search %s\n  d2h    %s\n  total  %s\n",
                             mode, np, n, q3(tm).c_str(), q3(th).c_str(), q3(ts).c_str(), q3(td).c_str(), q3(tall).c_str());
            }
            try {
                auto t0 = clk::now();
                call_a();
                const double first_ms = ms_since(t0);
                for (int w = 1; w < warmup; ++w) call_a();
                const double per_call = ms_since(t0) / warmup;
                const cudaError_t ce = cudaGetLastError();
                if (ce != cudaSuccess) throw std::runtime_error(cudaGetErrorString(ce));
                std::memcpy(ref.data(), pin_id, ref.size() * sizeof(int64_t));
                long long hit = 0, bad = 0, dup_rows = 0;
                for (int q = 0; q < nq; ++q) {
                    std::set<long long> seen;
                    for (int i = 0; i < k; ++i) {
                        const long long id = ref[(size_t)q * k + i];
                        if (id < 0 || id >= nb) ++bad;
                        seen.insert(id);
                    }
                    if ((int)seen.size() != k) ++dup_rows;
                    for (int j = 0; j < k; ++j) {
                        const long long want = gt[(size_t)q * dgt + j];
                        if (seen.count(want)) ++hit;
                    }
                }
                const double recall = (double)hit / ((double)nq * k);
                const int calls = per_call <= slow_ms ? calls0 : std::max(3, (int)(calls0 * slow_ms / per_call));
                auto run_rounds = [&](auto&& fn) {
                    std::vector<std::vector<double>> rr;
                    for (int r = 0; r < rounds; ++r) {
                        std::vector<double> ts;
                        for (int c = 0; c < calls; ++c) { auto t = clk::now(); fn(); ts.push_back(ms_since(t)); }
                        rr.push_back(ts);
                    }
                    return rr;
                };
                auto ta = run_rounds(call_a);
                const bool det = std::memcmp(ref.data(), pin_id, ref.size() * sizeof(int64_t)) == 0;
                long long a_set = 0, a_order = 0, b_set = 0, b_order = 0;
                row_diffs(ref, pin_id, nq, k, a_set, a_order);
                const double recall_last = recall_of(pin_id, gt, nq, k, dgt);
                call_b();
                auto tb = run_rounds(call_b);
                cudaMemcpy(hb.data(), d_idB.data_handle(), hb.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
                const bool b_same = hb == ref;
                row_diffs(ref, hb.data(), nq, k, b_set, b_order);
                const double recall_b = recall_of(hb.data(), gt, nq, k, dgt);
                const double mem_now = used_mib();
                auto emit_proto = [&](const char* name, const std::vector<std::vector<double>>& tt) {
                    std::vector<double> q, lm, lp;
                    for (auto& r : tt) {
                        double s = 0; for (double v : r) s += v;
                        q.push_back(calls * (double)nq / (s / 1000.0)); lm.push_back(median(r)); lp.push_back(pct(r, 0.95));
                    }
                    std::fprintf(js, ",\"%s\":{\"qps_median\":%.1f,\"qps_min\":%.1f,\"qps_max\":%.1f,\"qps_rounds\":[",
                                 name, median(q), *std::min_element(q.begin(), q.end()), *std::max_element(q.begin(), q.end()));
                    for (size_t i = 0; i < q.size(); ++i) std::fprintf(js, "%s%.1f", i ? "," : "", q[i]);
                    std::fprintf(js, "],\"lat_med_ms_rounds\":[");
                    for (size_t i = 0; i < lm.size(); ++i) std::fprintf(js, "%s%.4f", i ? "," : "", lm[i]);
                    std::fprintf(js, "],\"lat_p95_ms_rounds\":[");
                    for (size_t i = 0; i < lp.size(); ++i) std::fprintf(js, "%s%.4f", i ? "," : "", lp[i]);
                    std::fprintf(js, "],\"call_ms\":[");
                    for (size_t r = 0; r < tt.size(); ++r) {
                        std::fprintf(js, "%s[", r ? "," : "");
                        for (size_t i = 0; i < tt[r].size(); ++i) std::fprintf(js, "%s%.4f", i ? "," : "", tt[r][i]);
                        std::fprintf(js, "]");
                    }
                    std::fprintf(js, "]}");
                    return median(q);
                };
                std::fprintf(js, "{\"type\":\"point\",\"method\":\"ivf-rabitq\",\"label\":\"ivf-rabitq-b%u-%s\","
                             "\"index\":{\"n_lists\":%u,\"bits_per_dim\":%u,\"index_mib\":%.1f,\"build_ms\":%.1f,"
                             "\"serialize_ms\":%.1f,\"deserialize_ms\":%.1f},\"search\":{\"n_probes\":%d,\"mode\":%d,"
                             "\"mode_name\":\"%s\"},\"recall\":%.6f,\"invalid_ids\":%lld,\"dup_rows\":%lld,"
                             "\"deterministic\":%s,\"b_matches_a\":%s,\"first_call_ms\":%.3f,\"warmup\":%d,"
                             "\"rounds\":%d,\"calls_per_round\":%d,\"mem_used_mib\":%.1f,\"recall_last_a\":%.6f,"
                             "\"recall_b\":%.6f,\"rows_set_diff_a\":%lld,\"rows_order_diff_a\":%lld,"
                             "\"rows_set_diff_b\":%lld,\"rows_order_diff_b\":%lld",
                             bits, mode_name[mode & 3], n_lists, bits, m_index, t_build, t_ser, t_deser, np, mode,
                             mode_name[mode & 3], recall, bad, dup_rows, det ? "true" : "false",
                             b_same ? "true" : "false", first_ms, warmup, rounds, calls, mem_now, recall_last,
                             recall_b, a_set, a_order, b_set, b_order);
                const double qa = emit_proto("A", ta);
                const double qb = emit_proto("B", tb);
                std::fprintf(js, ",\"utc\":\"%s\"}\n", utc().c_str());
                std::fflush(js);
                std::fprintf(stderr, "  bits=%u %s np=%d R=%.4f A=%.0f B=%.0f calls=%d bad=%lld dup=%lld det=%d\n",
                             bits, mode_name[mode & 3], np, recall, qa, qb, calls, bad, dup_rows, (int)det);
            } catch (const std::exception& ex) {
                std::fprintf(js, "{\"type\":\"failure\",\"method\":\"ivf-rabitq\",\"stage\":\"search\",\"n_lists\":%u,"
                             "\"bits_per_dim\":%u,\"n_probes\":%d,\"mode\":%d,\"error\":\"%.300s\",\"used_mib\":%.1f,"
                             "\"utc\":\"%s\"}\n", n_lists, bits, np, mode, ex.what(), used_mib(), utc().c_str());
                std::fflush(js);
                std::fprintf(stderr, "  FAILED bits=%u mode=%d np=%d: %s\n", bits, mode, np, ex.what());
                cudaGetLastError();
            }
        }
    }
    cudaFreeHost(pin_q); cudaFreeHost(pin_id); cudaFreeHost(pin_di);
    std::fclose(js);
    std::fprintf(stderr, "RABITQ_SWEEP_DONE\n");
    return 0;
}
