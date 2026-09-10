// Is the adaptive budget worth having, or would one small fixed alpha do?
//
// The paper reports the rule's gain against alpha=100, the budget JHQ leaves as
// a default. That is the wrong denominator to stop at: a reviewer will ask why
// not pin alpha=8 and skip calibration. This puts four arms on one index and
// one timing harness so the question has an answer.
//
//   FIXED   every alpha on the grid: held-out recall and QPS
//   RULE    the sampled bisection rule, repeated over random samples at each S
//   FULL    the same criterion on the whole calibration pool, no sampling
//   ORACLE  the fastest alpha whose held-out recall is within JHQ_ORACLE_TAU of
//           the best any alpha reaches -- chosen with ground truth, so it is a
//           bound on what any selector could do here, not a method
//
// Queries are split before anything runs: even indices calibrate, odd indices
// evaluate. The earlier version calibrated on a stride sample of all queries
// and then scored recall on all of them, which let the rule see its own test
// set. Every recall below is on the held-out half.
//
// Repeating the rule over random samples is what turns "S=32 worked once" into
// a risk statement. The searches are on S queries, so the repeats are cheap
// next to one full-batch timing run.
//
// The algorithm is untouched: this only measures arms of the policy that
// already exists, plus two references it can be judged against.
//
// JHQ_AS_GRID     descending alpha grid          (default 200,100,64,32,16,8,4,2)
// JHQ_ARM_S       sample sizes to try            (default 32,64,128)
// JHQ_ARM_REPS    random samples per S           (default 64)
// JHQ_ORACLE_TAU  recall slack for the oracle    (default 0.001)

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
#include <tuple>
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
    const int S      = [](){ const char* e=std::getenv("JHQ_AS_SAMPLE"); return e?atoi(e):32; }();
    const int SLOTS  = [](){ const char* e=std::getenv("JHQ_AS_SLOTS");  return e?atoi(e):1;  }();
    const bool LIN   = [](){ const char* e=std::getenv("JHQ_AS_LINEAR"); return e && e[0]=='1'; }();
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
    // ---- split before anything else: even calibrates, odd evaluates -----
    std::vector<int> ci, ti;
    for (int q = 0; q < nq; ++q) (q % 2 ? ti : ci).push_back(q);
    const int nc = (int)ci.size(), nt = (int)ti.size();
    auto gather = [&](const std::vector<int>& ix) {
        std::vector<float> v((size_t)ix.size() * d);
        for (size_t i = 0; i < ix.size(); ++i)
            std::copy(query.begin() + (size_t)ix[i] * d,
                      query.begin() + (size_t)(ix[i] + 1) * d,
                      v.begin() + i * d);
        return v;
    };
    const std::vector<float> cq = gather(ci), tq = gather(ti);
    printf("\nsplit: %d calibration queries, %d held-out\n", nc, nt);

    // ---- per-query recall on the held-out half, for one alpha -----------
    std::vector<int>   t_ids((size_t)nt * k);
    std::vector<float> t_dst((size_t)nt * k);
    auto test_arm = [&](float a) {
        idx.set_alpha(a);
        idx.search(tq.data(), nt, k, t_dst.data(), t_ids.data());
        const int R = 3; double t = 0;
        for (int r = 0; r < R; ++r) {
            auto t0 = Clock::now();
            idx.search(tq.data(), nt, k, t_dst.data(), t_ids.data());
            t += Ms(Clock::now() - t0).count();
        }
        t /= R;
        std::vector<double> per(nt);
        long long hit = 0;
        for (int q = 0; q < nt; ++q) {
            const int* got = t_ids.data() + (size_t)q * k;
            const int* tru = gt.data() + (size_t)ti[q] * d_gt;
            int h = 0;
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k && j < d_gt; ++j)
                    if (got[i] == tru[j]) { ++h; break; }
            per[q] = (double)h / k; hit += h;
        }
        return std::make_tuple((double)hit / ((double)nt * k),
                               nt / (t / 1000.0), per);
    };

    std::vector<float> A(grid.begin(), grid.end());
    std::vector<double> Arec(A.size()), Aqps(A.size());
    std::vector<std::vector<double>> Aper(A.size());
    printf("\n");
    for (size_t i = 0; i < A.size(); ++i) {
        auto r = test_arm(A[i]);
        Arec[i] = std::get<0>(r); Aqps[i] = std::get<1>(r); Aper[i] = std::get<2>(r);
        printf("ARM FIXED alpha=%-6.0f recall=%.4f qps=%.0f\n", A[i], Arec[i], Aqps[i]);
    }

    // ---- what ground truth would have picked ----------------------------
    const double TAU = std::getenv("JHQ_ORACLE_TAU")
                     ? atof(std::getenv("JHQ_ORACLE_TAU")) : 1e-3;
    double ceil_rec = 0;
    for (double r : Arec) ceil_rec = std::max(ceil_rec, r);
    size_t orc = 0; double orq = -1;
    for (size_t i = 0; i < A.size(); ++i)
        if (Arec[i] >= ceil_rec - TAU && Aqps[i] > orq) { orq = Aqps[i]; orc = i; }
    printf("ARM ORACLE alpha=%.0f tau=%.4f ceiling=%.4f recall=%.4f qps=%.0f\n",
           A[orc], TAU, ceil_rec, Arec[orc], Aqps[orc]);

    // ---- the criterion, on a given set of calibration queries -----------
    std::vector<int>   r_ref((size_t)nc * k), r_ids((size_t)nc * k);
    std::vector<float> r_dst((size_t)nc * k);
    auto run_rule = [&](const std::vector<int>& sample, double* cal_ms, int* probes,
                        size_t* enum_out = nullptr, double* enum_ms = nullptr) {
        const int S2 = (int)sample.size();
        std::vector<float> sq((size_t)S2 * d);
        for (int i = 0; i < S2; ++i)
            std::copy(cq.begin() + (size_t)sample[i] * d,
                      cq.begin() + (size_t)(sample[i] + 1) * d,
                      sq.begin() + (size_t)i * d);
        idx.set_calibrating(true);
        auto t0 = Clock::now();
        idx.set_alpha(A[0]);
        idx.search(sq.data(), S2, k, r_dst.data(), r_ref.data());
        int pr = 0;
        auto miss_at = [&](size_t gi) {
            idx.set_alpha(A[gi]);
            idx.search(sq.data(), S2, k, r_dst.data(), r_ids.data());
            ++pr;
            long long hit = 0;
            for (int q = 0; q < S2; ++q) {
                const int* a = r_ref.data() + (size_t)q * k;
                const int* b = r_ids.data() + (size_t)q * k;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        if (a[i] == b[j]) { ++hit; break; }
            }
            return (long long)S2 * k - hit;
        };
        size_t lo = 0, hi = A.size() - 1;
        while (lo < hi) {
            size_t mid = (lo + hi + 1) / 2;
            if (miss_at(mid) <= SLOTS) lo = mid; else hi = mid - 1;
        }
        *cal_ms = Ms(Clock::now() - t0).count();
        *probes = pr;
        idx.set_calibrating(false);
        if (enum_out) {
            // The comparison the FULL arm did not make: on *this* sample, with
            // this epsilon and this reference, walk the whole grid and take the
            // smallest acceptable alpha. Bisection assumes the accepted set is
            // a prefix; if it is not, this finds a budget bisection walked past.
            //
            // Both timings must cover the same work. cal_ms above includes the
            // reference search at alpha_max, so this repeats it rather than
            // reusing the one already in ref_ids -- otherwise enum_ms would be
            // the cheaper of two differently defined quantities and the
            // comparison of calibration cost would be meaningless.
            idx.set_calibrating(true);
            auto e0 = Clock::now();
            idx.set_alpha(A[0]);
            idx.search(sq.data(), S2, k, r_dst.data(), r_ref.data());
            size_t best = 0;
            for (size_t gi = 1; gi < A.size(); ++gi)
                if (miss_at(gi) <= SLOTS) best = gi;
            *enum_ms = Ms(Clock::now() - e0).count();
            idx.set_calibrating(false);
            *enum_out = best;
        }
        return lo;
    };

    // ---- the same criterion with no sampling at all ---------------------
    {
        std::vector<int> all(nc); for (int i = 0; i < nc; ++i) all[i] = i;
        double cm; int pr;
        size_t b = run_rule(all, &cm, &pr);
        printf("ARM FULL alpha=%.0f recall=%.4f qps=%.0f cal_ms=%.1f probes=%d\n",
               A[b], Arec[b], Aqps[b], cm, pr);
    }

    // ---- the sampled rule, repeated, at several S -----------------------
    const char* SS = std::getenv("JHQ_ARM_S");
    std::vector<int> Slist;
    { std::string t = SS ? SS : "32,64,128"; size_t p = 0;
      while (p < t.size()) { size_t c = t.find(',', p); if (c == std::string::npos) c = t.size();
        Slist.push_back(atoi(t.substr(p, c - p).c_str())); p = c + 1; } }
    const int REPS = std::getenv("JHQ_ARM_REPS") ? atoi(std::getenv("JHQ_ARM_REPS")) : 64;
    std::srand(12345);
    for (int S2 : Slist) {
        if (S2 > nc) continue;
        std::vector<double> loss; std::vector<int> picks;
        double cm_sum = 0, em_sum = 0; int disagree = 0;
        std::vector<double> qps_pick;
        for (int r = 0; r < REPS; ++r) {
            std::vector<int> pool(nc); for (int i = 0; i < nc; ++i) pool[i] = i;
            for (int i = 0; i < S2; ++i) std::swap(pool[i], pool[i + std::rand() % (nc - i)]);
            std::vector<int> sample(pool.begin(), pool.begin() + S2);
            double cm, em = 0; int pr; size_t eb = 0;
            size_t b = run_rule(sample, &cm, &pr, &eb, &em);
            cm_sum += cm; em_sum += em; picks.push_back((int)A[b]);
            if (A[eb] != A[b]) ++disagree;
            loss.push_back(ceil_rec - Arec[b]); qps_pick.push_back(Aqps[b]);
        }
        std::sort(loss.begin(), loss.end());
        double mean = 0, over = 0, mq = 0;
        for (size_t i = 0; i < loss.size(); ++i) { mean += loss[i]; if (loss[i] > 1e-3) ++over; }
        for (double q : qps_pick) mq += q;
        std::sort(picks.begin(), picks.end());
        printf("ARM RULE S=%-4d reps=%d cal_ms=%.1f mean_loss=%.4f p95_loss=%.4f "
               "frac_over_1e-3=%.3f median_alpha=%d mean_qps=%.0f\n",
               S2, REPS, cm_sum / REPS, mean / loss.size(),
               loss[(size_t)(0.95 * (loss.size() - 1))], over / loss.size(),
               picks[picks.size() / 2], mq / qps_pick.size());
        printf("ARM ENUM S=%-4d reps=%d disagree=%d/%d enum_ms=%.1f bisect_ms=%.1f\n",
               S2, REPS, disagree, REPS, em_sum / REPS, cm_sum / REPS);
    }
    printf("\n=== ARMS_OK ===\n");
    return 0;
}
