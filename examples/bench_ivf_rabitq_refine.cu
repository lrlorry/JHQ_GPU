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
// IVF-RaBitQ WITH the re-ranking stage cuVS does not build into the index.
//
// cuvs::neighbors::ivf_rabitq::search_params carries n_probes and mode and
// nothing else: the library implements RaBitQ's quantiser and not the exact
// re-ranking its published results depend on. Run without it, bits_per_dim=1
// tops out at Recall@10 0.78 on vogue-768 and 0.79 on arxiv-768 however deep
// the probe, so reaching the recalls this paper compares at forces
// bits_per_dim=8 and an eight-times more expensive scan. That is why the
// baseline lands 2.3x to 4.9x behind cuVS CAGRA-int8 at R=0.90-0.95, which
// inverts RaBitQ's own published ordering.
//
// cuvs::neighbors::refine is shipped separately and refine.hpp's own example
// is exactly this pipeline: search m = alpha*k candidates, then refine to k
// against the fp32 dataset. JHQ_RQ_ALPHA is that multiplier, and it is the
// same quantity as JHQ's own alpha -- both systems then scan cheaply and
// spend exact work on a survivor set.
//
// The fp32 dataset stays resident for refine: 6.6 GiB at arxiv-768 and
// 11.4 GiB at openai3-3072 beside the index, on a 32 GB card.
//
// bits_per_dim is the total including the 1-bit code, so the budgets that
// match JHQ are:
//     JHQ Br=4  ->  1 + 4 = 5 bits/dim
//     JHQ Br=8  ->  1 + 8 = 9 bits/dim
// Passing 8 against JHQ's Br=8 would give RaBitQ 11% less memory.
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_resources.hpp>
#include <memory>
#include <raft/core/resource/device_memory_resource.hpp>
// RMM 26.8 flattened this layout: the header is rmm/mr/..., not
// rmm/mr/device/... . The same move is why an earlier attempt at a pool
// resource looked like it had been removed.
#include <rmm/mr/managed_memory_resource.hpp>
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
            "<bits_per_dim> <n_probes> <k> [mode] [reps] [metric]\n"
            "  mode:   0=LUT16 1=LUT32 2=QUANT4 3=QUANT8 (default 2)\n"
            "  metric: cuvs::distance::DistanceType, 0=L2Expanded (default)\n",
            argv[0]);
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
    // cuvs::neighbors::index_params carries a metric and this benchmark left
    // it at whatever the default is. The first run returned ids that were
    // plausible -- in range, distinct, distances ascending from 0.65 -- and
    // matched the ground truth exactly zero times out of ten thousand, which
    // is what a different metric looks like rather than what low recall looks
    // like. 0 = L2Expanded, 1 = L2SqrtExpanded, 2 = CosineExpanded,
    // 3 = InnerProduct, following cuvs::distance::DistanceType.
    const int      metric_i = (argc > 10) ? std::atoi(argv[10]) : 0;
    // Candidate multiplier for the refine stage: search alpha*k, refine to k.
    const int      alpha    = [&]{
        const char* e = std::getenv("JHQ_RQ_ALPHA");
        const int v = e ? std::atoi(e) : 8;
        return v > 0 ? v : 8;
    }();

    int nb = 0, d = 0, nq = 0, dq = 0, ng = 0, dgt = 0;
    auto xb = read_fvecs(base_p, nb, d);
    auto xq = read_fvecs(qry_p,  nq, dq);
    auto gt = read_ivecs(gt_p,   ng, dgt);
    if (dq != d) { std::fprintf(stderr, "query dim %d != base dim %d\n", dq, d); return 1; }
    std::printf("base=%dx%d  query=%dx%d  gt=%dx%d\n", nb, d, nq, dq, ng, dgt);
    std::printf("n_lists=%u  bits_per_dim=%u  n_probes=%u  k=%d  mode=%d  metric=%d  alpha=%d\n",
                n_lists, bits, n_probes, k, mode_i, metric_i, alpha);

    // A pool allocator was tried here and the headers it needs are gone: RMM
    // 26.8 replaced rmm/mr/device/pool_memory_resource.hpp with the CCCL-style
    // cuda/memory_pool API, so cuVS's own benchmark setup cannot be
    // reproduced against this wheel without following that migration too.
    raft::device_resources res;

    // JHQ_RQ_MANAGED=1 backs the large-workspace allocations with managed
    // memory.
    //
    // On stella and bge-m3 the build throws rmm::out_of_memory on
    // n_rows * dim * 4 -- the whole dataset as float -- even with
    // force_streaming set and the training set cut to a thirty-second of its
    // default. Streaming skips the dataset upload for the encoding pass, but
    // the k-means sampler is handed the original host mdspan and materialises
    // it. cuVS's own error handler names this as the other way out: "set
    // large_workspace_resource appropriately". This box has 754 GB of host
    // RAM behind managed memory, so the question is whether it completes at
    // all and at what cost, not whether it fits.
    // raft 26.8's setter takes a type-erased raft::mr::device_resource by
    // value, not a shared_ptr to the old device_memory_resource base -- that
    // base is gone from this RMM.
    if (std::getenv("JHQ_RQ_MANAGED")) {
        raft::resource::set_large_workspace_resource(
            res, raft::mr::device_resource{rmm::mr::managed_memory_resource{}});
        std::printf("large workspace: managed memory\n");
    }

    size_t free0 = 0, total0 = 0;
    cudaMemGetInfo(&free0, &total0);

    cuvs::neighbors::ivf_rabitq::index_params ip;
    ip.n_lists      = n_lists;
    ip.bits_per_dim = bits;
    ip.metric       = static_cast<cuvs::distance::DistanceType>(metric_i);
    // The k-means training set is materialised on the device as
    // min(n_rows, max_train_points_per_cluster * n_lists) x dim floats. At
    // nlist=32768 and the default 256 that is 8,388,608 x 1024 x 4 =
    // 34,359,738,368 bytes, which is what stella and bge-m3 threw
    // rmm::out_of_memory on -- the dataset streams past fine; the trainset
    // does not. The library says as much in its own error handler.
    if (const char* t = std::getenv("JHQ_RQ_TRAIN_PER_LIST"))
        ip.max_train_points_per_cluster = (uint32_t)std::atoi(t);
    if (std::getenv("JHQ_RQ_FORCE_STREAM")) ip.force_streaming = true;
    std::printf("index_params: n_lists=%u bits_per_dim=%u train_per_list=%u\n",
                ip.n_lists, ip.bits_per_dim, ip.max_train_points_per_cluster);

    // Build from device memory, not host.
    //
    // A host matrix sends cuVS down its streaming path -- it logs "Using
    // streaming construction: dataset size (2.67 GB) exceeds comfortable GPU
    // memory limit" even for a 2.7 GB set on a 32 GiB card -- and that path
    // returns neighbour ids that do not index the dataset: 65% of them come
    // back as 0, the distances beside them look like genuine near-neighbour
    // distances, and recomputing the distance to the id that was returned
    // gives 1.03 where the ground truth's own first neighbour is at 0.63. The
    // header says streaming is not applicable once the data is resident, so
    // it is uploaded first. 11.4 GiB at d=3072 fits the card.
    // 3.2 GB at 1 M x 3072 and 8 bits a dimension. /tmp is the 30 GB system
    // overlay on this box; the data disk is not. JHQ_RQ_IDX moves it.
    const char* rt_file = std::getenv("JHQ_RQ_IDX")
                        ? std::getenv("JHQ_RQ_IDX") : "/tmp/rq_bench.idx";
    // stella is 17.8 M x 1024 and bge-m3 10.1 M x 1024 -- 72 GB and 41 GB of
    // raw float -- so neither can be uploaded to a 32 GB card at all, and
    // JHQ_RQ_HOST=1 hands cuVS a host matrix instead. That path logs "Using
    // streaming construction" and used to return nonsense, but so did every
    // other path for want of the round-trip below, so it is worth asking
    // again.
    const bool host_in = std::getenv("JHQ_RQ_HOST") != nullptr;

    // build() alone, and the serialize/deserialize round-trip separately.
    // The round-trip is required -- cuVS's own test does it because build()
    // does not leave the index in the layout search reads -- but it goes
    // through a file, so what it costs is disk bandwidth against index size,
    // not index construction. Reporting only their sum compares JHQ's
    // in-memory build against RaBitQ's build plus a few GB written and read
    // back, which is not the same quantity.
    // refine() reads the fp32 dataset, so unlike bench_ivf_rabitq.cu this copy
    // outlives the build scope. Its allocation is outside the build timer
    // because it is the baseline's search-side memory, not its build cost.
    auto d_base = raft::make_device_matrix<float, int64_t>(res, nb, d);

    double t_built = 0.0, t_ser = 0.0;
    auto t0 = clk::now();
    cuvs::neighbors::ivf_rabitq::index<int64_t> loaded(res);
    {
        // build() does not leave the index in the layout search reads. cuVS's
        // own test round-trips it through serialize/deserialize with the
        // comment "Serialize and deserialize to reorganize data for efficient
        // search" (cpp/tests/neighbors/ann_ivf_rabitq.cuh); nothing in the
        // public header says so. Without it the search returns ids that are
        // essentially random beside distances below the true minimum --
        // examples/rabitq_selftest.cu puts it exactly: querying with a row of
        // the index and probing every list, self-recall is 0/20 built and
        // 20/20 round-tripped. The round-trip is inside the timed build, so
        // its cost is recorded rather than hidden.
        //
        // The device copy lives only in this scope: at d=3072 it is 11.4 GiB
        // that the search has no use for.
        if (host_in) {
            auto h_view = raft::make_host_matrix_view<const float, int64_t>(
                xb.data(), nb, d);
            auto built = cuvs::neighbors::ivf_rabitq::build(res, ip, h_view);
            raft::resource::sync_stream(res);
            t_built = ms_since(t0);
            cuvs::neighbors::ivf_rabitq::serialize(res, rt_file, built);
        } else {
            raft::copy(d_base.data_handle(), xb.data(), (size_t)nb * d,
                       raft::resource::get_cuda_stream(res));
            raft::resource::sync_stream(res);
            auto built = cuvs::neighbors::ivf_rabitq::build(
                res, ip, raft::make_device_matrix_view<const float, int64_t>(
                             d_base.data_handle(), nb, d));
            raft::resource::sync_stream(res);
            t_built = ms_since(t0);
            cuvs::neighbors::ivf_rabitq::serialize(res, rt_file, built);
        }
        raft::resource::sync_stream(res);
        t_ser = ms_since(t0);
        cuvs::neighbors::ivf_rabitq::deserialize(res, rt_file, &loaded);
        raft::resource::sync_stream(res);
    }
    auto& idx = loaded;
    const double build_ms = ms_since(t0);

    // Ask the index what it actually holds. There is no extend() in this API,
    // so build() is meant to have added every row; if size() disagrees with
    // nb the vectors never went in, which is what 65% zero ids would look
    // like from the search side.
    std::printf("index: size=%lld dim=%u  (base rows %d, dim %d)\n",
                (long long)idx.size(), idx.dim(), nb, d);

    size_t free1 = 0, total1 = 0;
    cudaMemGetInfo(&free1, &total1);

    cuvs::neighbors::ivf_rabitq::search_params sp;
    sp.n_probes = n_probes;
    sp.mode = static_cast<cuvs::neighbors::ivf_rabitq::search_mode>(mode_i);

    const int kc = k * alpha;          // candidates the quantised scan returns
    auto d_q   = raft::make_device_matrix<float,   int64_t>(res, nq, d);
    auto d_cid = raft::make_device_matrix<int64_t, int64_t>(res, nq, kc);
    auto d_cdi = raft::make_device_matrix<float,   int64_t>(res, nq, kc);
    auto d_id  = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto d_di  = raft::make_device_matrix<float,   int64_t>(res, nq, k);
    std::vector<int64_t> h_id((size_t)nq * k);
    std::vector<float>   h_di((size_t)nq * k);

    // Batching, to match demo_jhq_v36's argv[12] rather than always answering
    // the whole query set in one call.
    //
    // Everything in this evaluation runs at one batch size, and IVF-RaBitQ's
    // own paper runs at 10^4 on a different card -- while results/front6/v54.log
    // shows the LUT-versus-bitwise verdict flips with a card's bandwidth. A
    // single operating point is the weakest part of the comparison, so the
    // batch is a knob here too.
    const int qbatch = [&]{
        const char* e = std::getenv("JHQ_RQ_BATCH");
        const int v = e ? std::atoi(e) : nq;
        return (v > 0 && v < nq) ? v : nq;
    }();

    // One call per batch to match JHQ's timed region: queries up, search,
    // results back.
    auto one_pass = [&]() {
        auto st = raft::resource::get_cuda_stream(res);
        for (int off = 0; off < nq; off += qbatch) {
            const int B = std::min(qbatch, nq - off);
            auto qv = raft::make_device_matrix_view<const float, int64_t>(
                d_q.data_handle(), B, d);
            auto civ = raft::make_device_matrix_view<int64_t, int64_t>(
                d_cid.data_handle(), B, kc);
            auto cdv = raft::make_device_matrix_view<float, int64_t>(
                d_cdi.data_handle(), B, kc);
            auto iv = raft::make_device_matrix_view<int64_t, int64_t>(
                d_id.data_handle(), B, k);
            auto dv = raft::make_device_matrix_view<float, int64_t>(
                d_di.data_handle(), B, k);
            raft::copy(d_q.data_handle(), xq.data() + (size_t)off * d,
                       (size_t)B * d, st);
            // The two stages refine.hpp's own example prescribes: the
            // quantised scan proposes alpha*k, exact distance picks k.
            cuvs::neighbors::ivf_rabitq::search(res, sp, idx, qv, civ, cdv);
            cuvs::neighbors::refine(
                res,
                raft::make_device_matrix_view<const float, int64_t>(
                    d_base.data_handle(), nb, d),
                qv,
                raft::make_device_matrix_view<const int64_t, int64_t>(
                    d_cid.data_handle(), B, kc),
                iv, dv, static_cast<cuvs::distance::DistanceType>(metric_i));
            raft::copy(h_id.data() + (size_t)off * k, d_id.data_handle(),
                       (size_t)B * k, st);
            raft::copy(h_di.data() + (size_t)off * k, d_di.data_handle(),
                       (size_t)B * k, st);
            raft::resource::sync_stream(res);
        }
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

    // Recall came back 0.0000 on the first run at every configuration while
    // the search itself ran and timed. Either the ids are not what the ground
    // truth indexes, or they never left the device. Print enough of query 0 to
    // tell those apart rather than guessing.
    if (std::getenv("JHQ_RQ_DEBUG")) {
        std::printf("dbg gt[0][0..4] :");
        for (int j = 0; j < 5 && j < dgt; ++j) std::printf(" %d", gt[j]);
        std::printf("\ndbg id[0][0..4] :");
        for (int i = 0; i < 5 && i < k; ++i) std::printf(" %lld", (long long)h_id[i]);
        std::printf("\ndbg di[0][0..4] :");
        for (int i = 0; i < 5 && i < k; ++i) std::printf(" %.4f", h_di[i]);
        // Which is actually closer: the ground truth's first neighbour or the
        // one that came back? Computed here on the host, from the same arrays
        // the benchmark read, so it tests the reader and the search together.
        auto l2 = [&](long long id) {
            double a = 0.0;
            for (int j = 0; j < d; ++j) {
                const double t = (double)xq[j] - (double)xb[(size_t)id * d + j];
                a += t * t;
            }
            return a;
        };
        if (dgt > 0 && k > 0) {
            const long long g0 = gt[0], r0 = h_id[0];
            std::printf("dbg |q0-gt0|^2 = %.6f  (id %lld)\n", l2(g0), g0);
            std::printf("dbg |q0-r0|^2  = %.6f  (id %lld)\n", l2(r0), r0);
        }
        // cuVS reported a first distance of 0.0223 on vogue while the ground
        // truth's own first neighbour measures 0.63 here. A distance below the
        // true minimum is impossible, so one of the two datasets is not what
        // the other thinks. Brute-force query 0 over the whole base on the
        // host -- a second at 1M x 768 -- and see which one it agrees with.
        {
            double best = 1e30; long long besti = -1;
            for (long long i = 0; i < nb; ++i) {
                double a = 0.0;
                const float* r = xb.data() + (size_t)i * d;
                for (int j = 0; j < d; ++j) { const double t = (double)xq[j] - (double)r[j]; a += t * t; }
                if (a < best) { best = a; besti = i; }
            }
            std::printf("dbg host brute force NN of q0: id %lld at %.6f\n", besti, best);
        }
        long long nz = 0, neg = 0;
        for (size_t i = 0; i < h_id.size(); ++i) { if (h_id[i] != 0) ++nz; if (h_id[i] < 0) ++neg; }
        std::printf("\ndbg nonzero ids %lld / %zu, negative %lld\n", nz, h_id.size(), neg);
    }

    std::printf("\nRecall@%d : %.4f\n", k, recall);
    std::printf("Latency   : %.2f ms  (%d queries, batch %d)\n", ms, nq, qbatch);
    std::printf("QPS       : %.0f\n", nq / (ms / 1000.0));
    std::printf("  train: %.1f ms\n", build_ms);
    std::printf("  build_only: %.1f ms\n", t_built);
    std::printf("  serialize: %.1f ms\n", t_ser - t_built);
    std::printf("  deserialize: %.1f ms\n", build_ms - t_ser);
    std::printf("VRAM used : %.1f MiB\n", (double)(free0 - free1) / (1024.0 * 1024.0));
    std::printf("rep_ms    :");
    for (double v : rep_ms) std::printf(" %.3f", v);
    std::printf("\n");
    return 0;
}
