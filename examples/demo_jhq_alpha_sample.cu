// Choosing alpha by sampling, instead of by sweeping it in advance.
//
// alpha fixes the refinement budget ck = alpha*k, and the paper leaves its
// value unspecified. results/front6/alpha_ds.log measures what that costs:
// the point where ranking loss stops improving spans 25x across datasets --
// openai3-3072 is flat from alpha=4, arxiv-768 is still improving at 200 --
// so one swept constant is either wasteful or lossy, and which it is cannot
// be known without the sweep.
//
// The rule measured here needs no sweep and no ground truth. Take S queries
// out of the batch, answer them once at a generous alpha_max, and treat that
// answer as the reference. Then answer the same S at successively smaller
// alpha and keep the smallest whose top-k still agrees with the reference to
// within eps. Run the rest of the batch at that alpha.
//
// It is asking the only question that matters at search time -- "would a
// smaller budget have changed what I return?" -- rather than "would it have
// changed recall", which needs labels the system does not have.
//
// What this prints, and what each part is for:
//   * the alpha the rule picks, and the agreement curve it picked from
//   * what the calibration cost, in ms and as a share of the batch
//   * recall and QPS at the picked alpha, and at alpha_max, on the full set
//
// Compare the picked alpha against the saturation point in
// results/front6/alpha_ds.log: agreeing with a sweep it never ran is the
// whole claim.
//
// JHQ_AS_SAMPLE   how many queries to calibrate on      (default 32)
// JHQ_AS_EPS      allowed disagreement, as a fraction   (default 0.001)
// JHQ_AS_GRID     descending alpha grid                 (default 100,64,32,16,8,4,2)

// v16 = v15's harness + the primary quantiser the paper's own implementation
// uses. v15's primary was a Cartesian product of per-dimension scalar
// quantisers, which needs B % Ds == 0 with B <= 8 and so cannot represent a
// code below one bit per dimension. The official IndexJHQ's primary level is a
// product quantiser (primary_pq_, primary_ksub() = 1 << level_bits[0]) with Ds
// = d/M unconstrained, which is how the paper reaches a 128-bit primary code
// on 3072-d data. See cpu/pq_codebook.h.
//
// Usage note: M no longer has to keep Ds near 8. M=16 on 768-d gives Ds=48 and
// a 16-byte primary code -- the regime the paper's JHQ actually operates in,
// which v15 could not express.
//
// What changed versus every earlier demo:
//   * Recall@k comes from common/recall.cuh -- standard set-intersection
//     against the true top-k, not against the whole ground-truth row. See that
//     header for what the old loop actually measured.
//   * The old score is still printed, as "Pre-v15 score", so a re-run lines up
//     against the CSVs already in results/ instead of silently replacing them.
//   * Returned ids are dumped to <prefix>.ivecs and the run to <prefix>.json,
//     so the next metric change costs a re-parse instead of a re-search.

// One demo serves nine targets, each linking a different version's library.
// The header it saw was hard-coded to v21 regardless, so a target compiled its
// caller against v21's declarations and linked v22/v23/v24's definitions --
// which only worked while the signatures happened to agree, and stopped the
// moment one changed. add_jhq_dir passes the matching header.
#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v21_cascade/jhq_gpu_index.cuh"
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
#include <string>
#include <utility>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> "
            "<M> <B> <Br> <alpha> [k=10] [nlist=1024] [nprobe=8] "
            "[ivf_iters=8] [batch_size=256] [out_prefix] [kmeans_iters=5]\n"
            "\n"
            "  out_prefix  optional; writes <prefix>.ivecs (returned ids) and\n"
            "              <prefix>.json (full run record incl. gt_width).\n", argv[0]);
        return 1;
    }

    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int   M          = atoi(argv[4]);
    int   B          = atoi(argv[5]);
    int   Br         = atoi(argv[6]);
    float alpha      = (float)atof(argv[7]);
    int   k          = (argc > 8)  ? atoi(argv[8])  : 10;
    int   nlist      = (argc > 9)  ? atoi(argv[9])  : 1024;
    int   nprobe     = (argc > 10) ? atoi(argv[10]) : 8;
    int   ivf_iters  = (argc > 11) ? atoi(argv[11]) : 8;
    int   batch_size = (argc > 12) ? atoi(argv[12]) : 256;
    const char* out_prefix = (argc > 13) ? argv[13] : nullptr;
    int   kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    std::vector<float> query;
    std::vector<int>   gt;
    int d_query, d_gt;

    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    int nb     = base.n;
    int d_base = base.d;
    int nq = read_fvecs(query_path, query, d_query);
    int ng = read_ivecs(gt_path,    gt,    d_gt);

    int d = d_base;
    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n", nb, d, nq, d_query, ng, d_gt);
    printf("M=%d  B=%d  Br=%d  alpha=%.1f  k=%d  nlist=%d  nprobe=%d  "
           "ivf_iters=%d  batch_size=%d  kmeans_iters=%d\n",
           M, B, Br, alpha, k, nlist, nprobe, ivf_iters, batch_size, kmeans_iters);
    printf("primary: Ds=%d  K=%d  code=%d B/vec (%.3f bit/dim)\n",
           d / M, 1 << B, M, (double)M * 8.0 / d);

    // Fail before spending minutes on train+add if the ground truth is too
    // shallow to define Recall@k at all -- GT_K is 20 for the datasets built
    // by scripts/download_jhq_datasets.py, so k=100 needs regeneration first.
    if (d_gt < k) {
        fprintf(stderr,
            "\nERROR: ground truth is %d wide but k=%d. Recall@%d is undefined.\n"
            "Regenerate at least k wide: scripts/download_jhq_datasets.py (GT_K)\n"
            "or scripts/preprocess.py --k.\n", d_gt, k, k);
        return 1;
    }
    if (d_gt > k)
        printf("note: ground truth is %d wide; Recall@%d compares against its "
               "first %d entries.\n", d_gt, k, k);

    jhq_gpu::JHQGpuIndex::Params p;
    p.M = M;
    // add_batch and the assignment batch have never been swept and never
    // recorded (results/parameter_coverage/). add() is 95% of the build, and
    // 65536 was reasoned from footprint rather than throughput, so make it
    // reachable. Unset keeps the header's default exactly.
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

    // This sample trains sigma for eq. 3, the primary codebook of eq. 4, and
    // the IVF centroids -- not eq. 5, which has its own set below and now
    // defaults to all of Y. 100k by default; JHQ_N_TRAIN moves it (0 or "all"
    // takes every vector) so the sensitivity can be measured rather than
    // assumed.
    int n_train = std::min(nb, 100000);
    if (const char* e = std::getenv("JHQ_N_TRAIN")) {
        const long long want = (std::strcmp(e, "all") == 0) ? nb : std::atoll(e);
        n_train = (want <= 0 || want > nb) ? nb : (int)want;
    }
    // The claim that this sample size does not have to grow with N rests on
    // the sample being representative. A contiguous prefix is not: these files
    // arrive in whatever order they were built in, and arXiv is ordered by
    // date. Take every (N/S)-th vector instead, which costs one gather and is
    // uniform whatever the file's order.
    printf("Training on %d vectors...\n", n_train);
    std::vector<float> sample;
    const float* train_src = base.data;
    if (n_train < nb) {
        sample.resize((size_t)n_train * d);
        const long long stride = (long long)nb / n_train;
        for (int i = 0; i < n_train; ++i)
            std::memcpy(sample.data() + (size_t)i * d,
                        base.data + (size_t)(i * stride) * d,
                        (size_t)d * sizeof(float));
        train_src = sample.data();
    }
    // Equation 5 collects the residual of every y in Y, and O(n*K_r) is the
    // cost the paper quotes for it, so all of Y is the default here. It is
    // reachable at every scale -- one pass over the base, residuals spilled to
    // disk, the estimator reading them back a chunk of subspaces at a time --
    // and at stella's 17.8M vectors costs 172 s and 67.8 GB of scratch.
    //
    // JHQ_RES_TRAIN_N still sizes it down, because that path is measured and
    // reported: results/v30_disk_stage/ puts the difference between all of Y
    // and a 100K sample at +2e-4, +8e-4 and -3e-4 against build noise of
    // 2e-4. The reference implementation samples 20,000 scalar values a
    // subspace (IndexJHQ.h, sample_residual), which is what that measurement
    // explains; it is not what the paper specifies.
    std::vector<float> res_sample;
    const float* res_src = base.data;
    int n_res_train = nb;
    if (const char* rt = std::getenv("JHQ_RES_TRAIN_N")) {
        n_res_train = std::atoi(rt);
        if (n_res_train <= 0 || n_res_train > nb) n_res_train = nb;
        if (n_res_train == nb) {
            res_src = base.data;                 // every vector; no gather
        } else {
            res_sample.resize((size_t)n_res_train * d);
            const long long rstride = (long long)nb / n_res_train;
            for (int i = 0; i < n_res_train; ++i)
                std::memcpy(res_sample.data() + (size_t)i * d,
                            base.data + (size_t)(i * rstride) * d,
                            (size_t)d * sizeof(float));
            res_src = res_sample.data();
        }
    }
    printf("residual codebook trains on %d of %d vectors%s\n",
           n_res_train, nb, n_res_train == nb ? "  (eq. 5, all of Y)" : "");
    auto t0 = Clock::now();
    idx.train(train_src, n_train, res_src, n_res_train);
    double train_ms = Ms(Clock::now() - t0).count();
    printf("  train: %.1f ms\n", train_ms);

    printf("Adding %d vectors...\n", nb);
    t0 = Clock::now();
    idx.add(base.data, nb);
    double add_ms = Ms(Clock::now() - t0).count();
    printf("  add:   %.1f ms\n", add_ms);

    std::vector<float> out_dists((long long)nq * k);
    std::vector<int>   out_ids  ((long long)nq * k);

    // Warm-up (also captures the CUDA graph on first call)
    idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());

    // ── calibration ────────────────────────────────────────────────────────
    const int S = [](){ const char* e = std::getenv("JHQ_AS_SAMPLE");
                        return e ? atoi(e) : 32; }();
    const double EPS = [](){ const char* e = std::getenv("JHQ_AS_EPS");
                             return e ? atof(e) : 0.001; }();
    std::vector<float> grid;
    {
        const char* g = std::getenv("JHQ_AS_GRID");
        std::string t = g ? g : "100,64,32,16,8,4,2";
        size_t p = 0;
        while (p < t.size()) {
            size_t c = t.find(',', p);
            if (c == std::string::npos) c = t.size();
            grid.push_back((float)atof(t.substr(p, c - p).c_str()));
            p = c + 1;
        }
    }
    const int Sn = (S < nq) ? S : nq;
    // Strided, not the first Sn: query files often arrive grouped, and a
    // contiguous head would calibrate on one corner of the distribution.
    const int stride = nq / Sn;
    std::vector<float> sq((size_t)Sn * d);
    for (int i = 0; i < Sn; ++i)
        std::copy(query.begin() + (size_t)(i * stride) * d,
                  query.begin() + (size_t)(i * stride + 1) * d,
                  sq.begin() + (size_t)i * d);

    std::vector<int>   ref_ids((size_t)Sn * k), s_ids((size_t)Sn * k);
    std::vector<float> s_dst((size_t)Sn * k);
    auto tc0 = Clock::now();
    idx.set_alpha(grid[0]);
    idx.search(sq.data(), Sn, k, s_dst.data(), ref_ids.data());

    printf("\ncalibration on %d of %d queries, eps=%.4f\n", Sn, nq, EPS);
    float picked = grid[0];
    printf("  %-8s %-10s %s\n", "alpha", "agreement", "");
    printf("  %-8.0f %-10.4f (reference)\n", grid[0], 1.0);
    for (size_t gi = 1; gi < grid.size(); ++gi) {
        idx.set_alpha(grid[gi]);
        idx.search(sq.data(), Sn, k, s_dst.data(), s_ids.data());
        // Agreement is set intersection per query, not position equality:
        // two budgets that return the same neighbours in a different order
        // have not changed the answer.
        long long hit = 0;
        for (int q = 0; q < Sn; ++q) {
            const int* a = ref_ids.data() + (size_t)q * k;
            const int* b = s_ids.data()   + (size_t)q * k;
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    if (a[i] == b[j]) { ++hit; break; }
        }
        const double agree = (double)hit / ((double)Sn * k);
        const bool ok = (1.0 - agree) <= EPS;
        printf("  %-8.0f %-10.4f %s\n", grid[gi], agree, ok ? "accept" : "reject -> stop");
        if (!ok) break;
        picked = grid[gi];
    }
    const double cal_ms = Ms(Clock::now() - tc0).count();
    printf("picked alpha = %.0f   (calibration %.1f ms)\n", picked, cal_ms);

    // Both budgets on the full batch, so the rule is priced against what it
    // replaced rather than against nothing.
    auto full = [&](float a, const char* tag) {
        idx.set_alpha(a);
        idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());
        const int REPS_L = 3;
        double t = 0;
        for (int r = 0; r < REPS_L; ++r) {
            auto tr = Clock::now();
            idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());
            t += Ms(Clock::now() - tr).count();
        }
        t /= REPS_L;
        RecallResult rr = evaluate_recall(out_ids.data(), gt.data(), nq, k, d_gt);
        printf("%-10s alpha=%-6.0f recall=%.4f  qps=%.0f  (%.2f ms)\n",
               tag, a, rr.recall, nq / (t / 1000.0), t);
        return std::make_pair(rr.recall, nq / (t / 1000.0));
    };
    auto rp = full(picked,  "picked");
    auto rm = full(grid[0], "alpha_max");
    printf("AS_RESULT picked=%.0f recall_picked=%.4f qps_picked=%.0f "
           "recall_max=%.4f qps_max=%.0f gain=%.3f cal_ms=%.1f cal_share=%.4f\n",
           picked, rp.first, rp.second, rm.first, rm.second,
           rp.second / rm.second, cal_ms, cal_ms / (1000.0 * (double)nq / rm.second));
    idx.set_alpha(picked);
    // ── end calibration ────────────────────────────────────────────────────

    // Resident device memory with the index built and the search workspace
    // allocated, measured rather than derived from bytes-per-vector.
    size_t mem_free = 0, mem_total = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&mem_free, &mem_total);

    // Per-repetition timings, so the harness can report a spread rather than a
    // single number. Reported alongside the mean, not instead of it.
    const int REPS = 5;
    std::vector<double> rep_ms;
    rep_ms.reserve(REPS);
    for (int r = 0; r < REPS; r++) {
        auto tr = Clock::now();
        idx.search(query.data(), nq, k, out_dists.data(), out_ids.data());
        rep_ms.push_back(Ms(Clock::now() - tr).count());
    }
    double ms = 0.0;
    for (double v : rep_ms) ms += v;
    ms /= REPS;

    RecallResult rec = evaluate_recall(out_ids.data(), gt.data(), nq, k, d_gt);
    double qps = nq / (ms / 1000.0);

    printf("\nRecall@%d : %.4f     (standard: vs true top-%d)\n", k, rec.recall, k);
    printf("Pre-v15 score : %.4f  (old metric: vs true top-%d -- for lining up "
           "against existing results/*.csv only)\n", rec.legacy, rec.gt_width);
    if (rec.dup_queries)
        printf("WARNING   : %lld / %d queries returned a duplicate id\n",
               rec.dup_queries, nq);
    printf("Latency   : %.2f ms  (%d queries)\n", ms, nq);
    printf("QPS       : %.0f\n", qps);
    printf("VRAM used : %.1f MiB  (of %.0f MiB total, measured after build)\n",
           (double)(mem_total - mem_free) / (1024.0 * 1024.0),
           (double)mem_total / (1024.0 * 1024.0));
    printf("rep_ms    :");
    for (double v : rep_ms) printf(" %.3f", v);
    printf("\n");

#ifdef JHQ_HAS_DIAG
    // JHQ_DIAG=1: the two numbers that say whether a low recall is a routing
    // failure or a ranking one. Off by default -- the readbacks sit inside
    // the timed region, so a diag run's QPS is not comparable.
    {
        const jhq_gpu::SearchDiag& dg = jhq_gpu::search_diag();
        if (dg.on && !dg.cand.empty()) {
            double sum = 0; long long mn = dg.cand[0], mx = dg.cand[0];
            for (int v : dg.cand) { sum += v; mn = std::min<long long>(mn, v); mx = std::max<long long>(mx, v); }
            const double mean = sum / (double)nq;
            const double uniform = (double)nb * p.nprobe / (double)p.nlist;
            printf("cand_mean : %.0f   (min %lld, max %lld; "
                   "N*nprobe/nlist estimate = %.0f, ratio %.2fx)\n",
                   mean, mn, mx, uniform, mean / uniform);

            std::vector<int> vlist;
            idx.vector_lists(vlist);
            std::vector<char> opened((size_t)p.nlist, 0);
            long long hit = 0, tot = 0;
            for (int q = 0; q < nq; ++q) {
                for (int j = 0; j < dg.nprobe; ++j) {
                    int l = dg.probes[(size_t)q * dg.nprobe + j];
                    if (l >= 0 && l < p.nlist) opened[l] = 1;
                }
                for (int j = 0; j < k; ++j) {
                    int id = gt[(size_t)q * d_gt + j];
                    if (id < 0 || id >= nb) continue;
                    ++tot;
                    int l = vlist[id];
                    if (l >= 0 && opened[l]) ++hit;
                }
                for (int j = 0; j < dg.nprobe; ++j) {
                    int l = dg.probes[(size_t)q * dg.nprobe + j];
                    if (l >= 0 && l < p.nlist) opened[l] = 0;
                }
            }
            const double ivf_rec = tot ? (double)hit / (double)tot : 0.0;
            printf("ivf_recall: %.4f  (of the true top-%d, the share coarse "
                   "routing brought in at all)\n", ivf_rec, k);
            printf("lost_route: %.4f  lost_rank: %.4f  (the two ways the "
                   "%.4f that is missing was lost)\n",
                   1.0 - ivf_rec, ivf_rec - rec.recall, 1.0 - rec.recall);
        }
    }
#endif

    if (out_prefix) {
        char path[4096];
        snprintf(path, sizeof(path), "%s.ivecs", out_prefix);
        write_ivecs(path, out_ids.data(), nq, k);

        snprintf(path, sizeof(path), "%s.json", out_prefix);
        FILE* jf = fopen(path, "w");
        if (!jf) { fprintf(stderr, "Cannot write %s\n", path); return 1; }
        fprintf(jf,
            "{\"version\":\"jhq_v19_tiled_scan\","
            "\"dataset_base\":\"%s\",\"n\":%d,\"d\":%d,\"nq\":%d,"
            "\"params\":{\"M\":%d,\"B\":%d,\"Br\":%d,\"alpha\":%.4f,"
            "\"nlist\":%d,\"nprobe\":%d,\"ivf_iters\":%d,\"batch_size\":%d,"
            "\"n_train\":%d,\"k\":%d},"
            "\"eval\":{\"metric\":\"recall@k standard set-intersection\","
            "\"k\":%d,\"gt_width\":%d,\"eval_gt_k\":%d,"
            "\"recall_at_k\":%.6f,\"pre_v15_score\":%.6f,\"dup_queries\":%lld},"
            "\"perf\":{\"qps\":%.2f,\"latency_ms\":%.4f,\"reps\":%d,"
            "\"train_ms\":%.2f,\"add_ms\":%.2f},"
            "\"neighbors_file\":\"%s.ivecs\"}\n",
            base_path, nb, d, nq,
            M, B, Br, alpha, nlist, nprobe, ivf_iters, batch_size, n_train, k,
            rec.k, rec.gt_width, rec.eval_depth,
            rec.recall, rec.legacy, rec.dup_queries,
            qps, ms, REPS, train_ms, add_ms,
            out_prefix);
        fclose(jf);
        printf("wrote     : %s.ivecs  %s.json\n", out_prefix, out_prefix);
    }

    return 0;
}
