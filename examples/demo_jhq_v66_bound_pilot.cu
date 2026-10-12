// Residual-aware factorized screening, offline pilot (experiments/residual_aware_bound).
//
// Builds the paper's index (trained-state cache), searches a seeded random
// sample of queries with v65 unchanged, copies the representation to the host
// (jhq_v66_boundpilot's export_host_state) and, in FP64:
//
//   audit     - the primary codebook is Cartesian: the first Ds/2 coordinates of
//               a code's centroid depend only on its high nibble, the rest only
//               on its low nibble (and every coordinate equals the level of its
//               digit, when the level construction is in use);
//             - the residual scalar codebooks are sorted;
//             - the stored correction is 2<yhat, rhat>;
//             - the composite distance v65 returns equals D_H + ||q'||^2;
//   envelope  for every subspace m, half s, half-code h and coordinate of that
//             half: min and max of the decoded residual over EVERY indexed
//             vector carrying that half-code (streamed over all N, no sampling);
//             plus the global envelope [min, max] of each residual codebook;
//   per query for every candidate of the probed lists: D_prim = ||q'-yhat||^2,
//             D_H = ||q'-yhat-rhat||^2, LB_cond and LB_glob, each bound read off a
//             16+16-entry table per subspace from the primary code alone.
//
// Writes <prefix>_summary.json and one <prefix>_q<idx>.bin per sampled query
// (layout in write_query below); analyze.py does the rest.
//
// Environment: JHQ_RB_NQ (sampled queries, default 128), JHQ_RB_SEED (20261013),
// plus the index variables of demo_jhq_batch_pipeline. Usage is its usage.

#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v66_boundpilot/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }

// squared distance from a to the interval [lo, hi]
static inline double gap2(double a, double lo, double hi) {
    const double g = (a < lo) ? (lo - a) : (a > hi ? a - hi : 0.0);
    return g * g;
}

int main(int argc, char** argv) {
    if (argc < 14) {
        fprintf(stderr, "Usage: %s <base> <query> <gt> <M> <B> <Br> <alpha> <k> <nlist> <nprobe> "
                        "<ivf_iters> <batch_size> <out_prefix> [kmeans_iters=5]\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1]; const char* query_path = argv[2]; const char* gt_path = argv[3];
    int M = atoi(argv[4]), Bb = atoi(argv[5]), Br = atoi(argv[6]);
    float alpha = (float)atof(argv[7]);
    int k = atoi(argv[8]), nlist = atoi(argv[9]), nprobe = atoi(argv[10]);
    int ivf_iters = atoi(argv[11]), batch_size = atoi(argv[12]);
    const std::string prefix = argv[13];
    int kmeans_iters = (argc > 14) ? atoi(argv[14]) : 5;

    std::vector<float> query; std::vector<int> gt; int d_query, d_gt;
    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    const int nb = base.n, d = base.d;
    const int nq = read_fvecs(query_path, query, d_query);
    read_ivecs(gt_path, gt, d_gt);

    // ---- the index, exactly as demo_jhq_batch_pipeline builds it (cache key matches)
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
    auto t0 = Clock::now();
    idx.train(train_src, n_train, res_src, n_res_train);
    printf("train %.1f ms\n", Ms(Clock::now() - t0).count());
    t0 = Clock::now();
    idx.add(base.data, nb);
    printf("add %.1f ms\n", Ms(Clock::now() - t0).count());

    // ---- seeded query sample
    const int nS = std::min(env_int("JHQ_RB_NQ", 128), std::min(nq, batch_size));
    const unsigned seed = (unsigned)env_int("JHQ_RB_SEED", 20261013);
    std::vector<int> order(nq);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int> qsel(order.begin(), order.begin() + nS);
    std::vector<float> qs((size_t)nS * d);
    for (int i = 0; i < nS; ++i)
        std::memcpy(qs.data() + (size_t)i * d, query.data() + (size_t)qsel[i] * d, (size_t)d * sizeof(float));

    // ---- v65, unchanged: scheduled groups, shared bound, refinement lanes 16, one batch
    jhq_gpu::set_refine_lanes(16);
    {
        const int n_cal = 512;
        std::vector<float> cal((size_t)n_cal * d);
        const long long st = std::max(1LL, (long long)nb / n_cal);
        for (int i = 0; i < n_cal; ++i) {
            const long long r = std::min((long long)nb - 1, i * st + st / 2);
            std::memcpy(cal.data() + (size_t)i * d, base.data + (size_t)r * d, (size_t)d * sizeof(float));
        }
        idx.calibrate_schedule(cal.data(), n_cal);
    }
    jhq_gpu::set_primary_groups(-1);
    jhq_gpu::set_shared_bound(true);
    idx.set_alpha(alpha);
    std::vector<float> vd((size_t)nS * k); std::vector<int> vi((size_t)nS * k);
    idx.search(qs.data(), nS, k, vd.data(), vi.data());     // capture
    idx.search(qs.data(), nS, k, vd.data(), vi.data());
    std::vector<int> probes, topck_pos; std::vector<float> topck_val; int lB, lnp, lck;
    idx.debug_last_batch(probes, topck_pos, topck_val, lB, lnp, lck);
    if (lB != nS || lnp != nprobe) { fprintf(stderr, "debug batch mismatch\n"); return 1; }
    const int ck = lck;
    printf("v65: %d queries (seed %u), ck=%d, scheduled G=%d\n", nS, seed, ck, jhq_gpu::scheduled_groups(nS));

    // ---- the representation, on the host
    jhq_gpu::JHQGpuIndex::HostState S;
    t0 = Clock::now();
    idx.export_host_state(S);
    printf("export %.1f ms: N=%d d=%d M=%d Ds=%d K=%d Kr=%d Br=%d bpv=%d L=%d\n", Ms(Clock::now() - t0).count(),
           S.ntotal, S.d, S.M, S.Ds, S.K, S.Kr, S.Br, S.bpv, S.L);
    const long long N = S.ntotal;
    const int Ds = S.Ds, H = Ds / 2, K = S.K, Kr = S.Kr;
    if (K != 256 || (Ds & 1) || S.M * Ds != d) { fprintf(stderr, "unexpected shape: K=%d Ds=%d\n", K, Ds); return 1; }
    auto rcode = [&](long long pos, int j) -> int {
        return (S.Br == 8) ? S.res[(size_t)pos * S.bpv + j]
                           : ((j & 1) ? (S.res[(size_t)pos * S.bpv + (j >> 1)] >> 4) : (S.res[(size_t)pos * S.bpv + (j >> 1)] & 0x0F));
    };
    // yhat of coordinate jj of subspace m for half-code h: the representative centroid's value
    auto yhalf = [&](int m, int s, int h, int jj) -> double {
        const int code = s == 0 ? (h << 4) : h;
        const int j = s == 0 ? jj : H + jj;
        return (double)S.cent[((size_t)m * K + code) * Ds + j];
    };

    // ---- audit 1: Cartesian (separable) primary codebook, and the level construction
    long long sep_bad = 0, lvl_bad = 0, lvl_checked = 0;
    for (int m = 0; m < S.M; ++m)
        for (int c = 0; c < K; ++c)
            for (int j = 0; j < Ds; ++j) {
                const float v = S.cent[((size_t)m * K + c) * Ds + j];
                const int rep = (j < H) ? (c & 0xF0) : (c & 0x0F);
                if (v != S.cent[((size_t)m * K + rep) * Ds + j]) ++sep_bad;
                if (S.L > 0) {
                    // digits base L, first coordinate most significant (encode.cu)
                    int dig = c;
                    for (int t = Ds - 1; t > j; --t) dig /= S.L;
                    dig %= S.L;
                    ++lvl_checked;
                    if (v != S.levels[dig]) ++lvl_bad;
                }
            }
    // ---- audit 2: residual codebooks sorted
    long long unsorted = 0;
    for (int m = 0; m < S.M; ++m)
        for (int r = 1; r < Kr; ++r) if (S.res_c1d[(size_t)m * Kr + r] < S.res_c1d[(size_t)m * Kr + r - 1]) ++unsorted;
    // ---- audit 3: stored correction = 2 <yhat, rhat> (FP64 reference), over every vector
    double corr_abs = 0.0, corr_rel = 0.0;
    #pragma omp parallel for reduction(max:corr_abs, corr_rel) schedule(static)
    for (long long pos = 0; pos < N; ++pos) {
        double dot = 0.0, scale = 0.0;
        for (int m = 0; m < S.M; ++m) {
            const int c = S.primary[(size_t)pos * S.M + m];
            for (int j = 0; j < Ds; ++j) {
                const double y = S.cent[((size_t)m * K + c) * Ds + j];
                const double r = S.res_c1d[(size_t)m * Kr + rcode(pos, m * Ds + j)];
                dot += y * r; scale += std::fabs(y * r);
            }
        }
        const double e = std::fabs((double)S.corr[pos] - 2.0 * dot);
        corr_abs = std::max(corr_abs, e);
        corr_rel = std::max(corr_rel, e / std::max(2.0 * scale, 1e-30));
    }
    printf("audit: separability mismatches %lld; level mismatches %lld of %lld; unsorted residual pairs %lld; "
           "corr vs 2<y,r> max abs %.3e, rel to sum|y r| %.3e\n", sep_bad, lvl_bad, lvl_checked, unsorted, corr_abs, corr_rel);

    // ---- residual envelopes over every indexed vector (streamed, per-thread partials)
    const size_t ENV = (size_t)S.M * 2 * 16 * H;
    std::vector<float> emin(ENV, std::numeric_limits<float>::infinity()), emax(ENV, -std::numeric_limits<float>::infinity());
    std::vector<long long> ecount((size_t)S.M * 2 * 16, 0);
    t0 = Clock::now();
    #pragma omp parallel
    {
        std::vector<float> tmin(ENV, std::numeric_limits<float>::infinity()), tmax(ENV, -std::numeric_limits<float>::infinity());
        std::vector<long long> tcnt((size_t)S.M * 2 * 16, 0);
        #pragma omp for schedule(static)
        for (long long pos = 0; pos < N; ++pos)
            for (int m = 0; m < S.M; ++m) {
                const int c = S.primary[(size_t)pos * S.M + m];
                for (int s = 0; s < 2; ++s) {
                    const int h = s == 0 ? (c >> 4) : (c & 15);
                    ++tcnt[((size_t)m * 2 + s) * 16 + h];
                    for (int jj = 0; jj < H; ++jj) {
                        const float r = S.res_c1d[(size_t)m * Kr + rcode(pos, m * Ds + s * H + jj)];
                        const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + jj;
                        tmin[e] = std::min(tmin[e], r); tmax[e] = std::max(tmax[e], r);
                    }
                }
            }
        #pragma omp critical
        {
            for (size_t e = 0; e < ENV; ++e) { emin[e] = std::min(emin[e], tmin[e]); emax[e] = std::max(emax[e], tmax[e]); }
            for (size_t e = 0; e < tcnt.size(); ++e) ecount[e] += tcnt[e];
        }
    }
    const double env_ms = Ms(Clock::now() - t0).count();
    long long unseen = 0;
    for (long long c : ecount) unseen += (c == 0);
    std::vector<float> gmin(S.M), gmax(S.M);
    double wcond = 0.0, wglob = 0.0; long long nw = 0;
    for (int m = 0; m < S.M; ++m) {
        gmin[m] = *std::min_element(S.res_c1d.begin() + (size_t)m * Kr, S.res_c1d.begin() + (size_t)(m + 1) * Kr);
        gmax[m] = *std::max_element(S.res_c1d.begin() + (size_t)m * Kr, S.res_c1d.begin() + (size_t)(m + 1) * Kr);
        for (int s = 0; s < 2; ++s)
            for (int h = 0; h < 16; ++h)
                if (ecount[((size_t)m * 2 + s) * 16 + h])
                    for (int jj = 0; jj < H; ++jj) {
                        const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + jj;
                        wcond += emax[e] - emin[e]; wglob += gmax[m] - gmin[m]; ++nw;
                    }
    }
    printf("envelope: %.1f ms over %lld vectors, %zu floats (%.1f KB), unseen half-code states %lld of %zu; "
           "mean width conditioned %.4g vs global %.4g\n", env_ms, N, 2 * ENV, 2.0 * ENV * 4 / 1024.0,
           unseen, ecount.size(), wcond / nw, wglob / nw);

    // ---- per query, FP64
    const int MS = S.M;
    std::vector<double> sumLBm((size_t)MS, 0.0), sumDHm((size_t)MS, 0.0), sumLBgm((size_t)MS, 0.0), zeroLBm((size_t)MS, 0.0);
    long long tot_cand = 0, viol_c = 0, viol_g = 0, nan_lb = 0;
    double maxviol_c = -1e300, maxviol_g = -1e300, comp_dev = 0.0, prim_dev = 0.0, ret_dev = 0.0;
    t0 = Clock::now();
    #pragma omp parallel
    {
        std::vector<double> lSLB((size_t)MS, 0.0), lSDH((size_t)MS, 0.0), lSLBg((size_t)MS, 0.0), lZ((size_t)MS, 0.0);
        long long l_tot = 0, l_vc = 0, l_vg = 0, l_nan = 0;
        double l_mc = -1e300, l_mg = -1e300, l_comp = 0.0, l_prim = 0.0, l_ret = 0.0;
        #pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < nS; ++qi) {
            const float* q = qs.data() + (size_t)qi * d;
            std::vector<double> qr(d, 0.0);
            for (int kk = 0; kk < d; ++kk) {
                const double qk = q[kk];
                const float* col = S.Pi.data() + (size_t)kk * d;
                for (int i = 0; i < d; ++i) qr[i] += (double)col[i] * qk;
            }
            double qn2 = 0.0;
            for (int i = 0; i < d; ++i) qn2 += qr[i] * qr[i];
            // 16+16 tables per subspace: primary, conditioned bound, global bound
            std::vector<double> tP((size_t)MS * 32), tC((size_t)MS * 32), tG((size_t)MS * 32);
            for (int m = 0; m < MS; ++m)
                for (int s = 0; s < 2; ++s)
                    for (int h = 0; h < 16; ++h) {
                        double sp = 0.0, sc = 0.0, sg = 0.0;
                        const bool seen = ecount[((size_t)m * 2 + s) * 16 + h] > 0;
                        for (int jj = 0; jj < H; ++jj) {
                            const int j = m * Ds + s * H + jj;
                            const double a = qr[j] - yhalf(m, s, h, jj);
                            const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + jj;
                            sp += a * a;
                            sc += seen ? gap2(a, emin[e], emax[e]) : std::numeric_limits<double>::quiet_NaN();
                            sg += gap2(a, gmin[m], gmax[m]);
                        }
                        tP[(size_t)m * 32 + s * 16 + h] = sp;
                        tC[(size_t)m * 32 + s * 16 + h] = sc;
                        tG[(size_t)m * 32 + s * 16 + h] = sg;
                    }
            // candidates of the probed lists, in probe order
            std::vector<int> cpos, cid, crank;
            std::vector<double> cP, cH, cLc, cLg;
            std::unordered_map<int, size_t> at;    // position -> candidate index
            for (int r = 0; r < nprobe; ++r) {
                const int l = probes[(size_t)qi * nprobe + r];
                for (int pos = S.list_offsets[l]; pos < S.list_offsets[l + 1]; ++pos) {
                    double P = 0.0, LC = 0.0, LG = 0.0, DH = 0.0, comp_r = 0.0;
                    for (int m = 0; m < MS; ++m) {
                        const int c = S.primary[(size_t)pos * MS + m];
                        const int hh = c >> 4, ll = c & 15;
                        const double lc = tC[(size_t)m * 32 + hh] + tC[(size_t)m * 32 + 16 + ll];
                        const double lg = tG[(size_t)m * 32 + hh] + tG[(size_t)m * 32 + 16 + ll];
                        P  += tP[(size_t)m * 32 + hh] + tP[(size_t)m * 32 + 16 + ll];
                        LC += lc; LG += lg;
                        double dh = 0.0;
                        for (int j = m * Ds; j < (m + 1) * Ds; ++j) {
                            const double y = S.cent[((size_t)m * K + c) * Ds + (j - m * Ds)];
                            const double rh = S.res_c1d[(size_t)m * Kr + rcode(pos, j)];
                            const double a = qr[j] - y;
                            dh += (a - rh) * (a - rh);
                            comp_r += (qr[j] - rh) * (qr[j] - rh);
                        }
                        DH += dh;
                        lSLB[m] += lc; lSDH[m] += dh; lSLBg[m] += lg; lZ[m] += (lc == 0.0);
                    }
                    // v65's composite from the same codes, in FP64: ||q'-y||^2 + ||q'-r||^2 + corr
                    const double comp = P + comp_r + (double)S.corr[pos];
                    l_comp = std::max(l_comp, std::fabs(comp - DH - qn2) / std::max(qn2, 1e-30));
                    if (std::isnan(LC) || std::isnan(LG)) ++l_nan;   // an unseen half-code state reached: impossible by construction
                    const double tol = 1e-12 * std::max(1.0, DH);
                    if (LC - DH > tol) { ++l_vc; } l_mc = std::max(l_mc, LC - DH);
                    if (LG - DH > tol) { ++l_vg; } l_mg = std::max(l_mg, LG - DH);
                    at[pos] = cpos.size();
                    cpos.push_back(pos); cid.push_back(S.list_ids[pos]); crank.push_back(r);
                    cP.push_back(P); cH.push_back(DH); cLc.push_back(LC); cLg.push_back(LG);
                }
            }
            l_tot += (long long)cpos.size();
            // v65's own primary distances of its top-ck, against FP64
            for (int c = 0; c < ck; ++c) {
                const int pos = topck_pos[(size_t)qi * ck + c];
                if (pos < 0) continue;
                auto it = at.find(pos);
                if (it == at.end()) { l_prim = 1e300; continue; }
                const double ref = cP[it->second];
                l_prim = std::max(l_prim, std::fabs((double)topck_val[(size_t)qi * ck + c] - ref) / std::max(ref, 1e-30));
            }
            // v65's returned distances against D_H + ||q'||^2, and the D_H of what it returned
            std::vector<double> v65_dh(k, std::numeric_limits<double>::quiet_NaN());
            std::unordered_map<int, size_t> byid;
            for (size_t i = 0; i < cid.size(); ++i) byid[cid[i]] = i;
            for (int t = 0; t < k; ++t) {
                auto it = byid.find(vi[(size_t)qi * k + t]);
                if (it == byid.end()) continue;
                v65_dh[t] = cH[it->second];
                const double ref = cH[it->second] + qn2;
                l_ret = std::max(l_ret, std::fabs((double)vd[(size_t)qi * k + t] - ref) / ref);
            }
            // one file per query:
            //   int32 qid, n_cand, nprobe, ck, k
            //   int32 probes[nprobe]
            //   int32 pos[n], id[n], rank[n]; float64 D_prim[n], D_H[n], LB_cond[n], LB_glob[n]
            //   float64 qnorm2; int32 v65_ids[k]; float32 v65_dists[k]; float64 v65_DH[k]
            //   int32 topck_pos[ck]; int32 gt[k]
            char path[1024];
            std::snprintf(path, sizeof(path), "%s_q%d.bin", prefix.c_str(), qsel[qi]);
            if (FILE* f = std::fopen(path, "wb")) {
                const int n = (int)cpos.size();
                const int hdr[5] = {qsel[qi], n, nprobe, ck, k};
                std::fwrite(hdr, sizeof(int), 5, f);
                std::fwrite(probes.data() + (size_t)qi * nprobe, sizeof(int), nprobe, f);
                std::fwrite(cpos.data(), sizeof(int), n, f); std::fwrite(cid.data(), sizeof(int), n, f);
                std::fwrite(crank.data(), sizeof(int), n, f);
                std::fwrite(cP.data(), sizeof(double), n, f); std::fwrite(cH.data(), sizeof(double), n, f);
                std::fwrite(cLc.data(), sizeof(double), n, f); std::fwrite(cLg.data(), sizeof(double), n, f);
                std::fwrite(&qn2, sizeof(double), 1, f);
                std::fwrite(vi.data() + (size_t)qi * k, sizeof(int), k, f);
                std::fwrite(vd.data() + (size_t)qi * k, sizeof(float), k, f);
                std::fwrite(v65_dh.data(), sizeof(double), k, f);
                std::fwrite(topck_pos.data() + (size_t)qi * ck, sizeof(int), ck, f);
                std::vector<int> g(k);
                for (int t = 0; t < k; ++t) g[t] = gt[(size_t)qsel[qi] * d_gt + t];
                std::fwrite(g.data(), sizeof(int), k, f);
                std::fclose(f);
            }
        }
        #pragma omp critical
        {
            for (int m = 0; m < MS; ++m) { sumLBm[m] += lSLB[m]; sumDHm[m] += lSDH[m]; sumLBgm[m] += lSLBg[m]; zeroLBm[m] += lZ[m]; }
            tot_cand += l_tot; viol_c += l_vc; viol_g += l_vg; nan_lb += l_nan;
            maxviol_c = std::max(maxviol_c, l_mc); maxviol_g = std::max(maxviol_g, l_mg);
            comp_dev = std::max(comp_dev, l_comp); prim_dev = std::max(prim_dev, l_prim); ret_dev = std::max(ret_dev, l_ret);
        }
    }
    const double q_ms = Ms(Clock::now() - t0).count();
    printf("queries: %lld candidates in %.1f ms; violations cond %lld (max LB-D_H %.3e), glob %lld (max %.3e); NaN bounds %lld\n",
           tot_cand, q_ms, viol_c, maxviol_c, viol_g, maxviol_g, nan_lb);
    printf("relation: max |comp_fp64 - D_H - ||q'||^2| / ||q'||^2 = %.3e; v65 returned vs D_H+||q'||^2 max rel %.3e; "
           "v65 primary (fp32 table) vs FP64 max rel %.3e\n", comp_dev, ret_dev, prim_dev);

    // ---- summary
    std::string js = prefix + "_summary.json";
    FILE* f = std::fopen(js.c_str(), "w");
    if (!f) return 1;
    std::fprintf(f, "{\"dataset_base\":\"%s\",\"N\":%lld,\"d\":%d,\"M\":%d,\"Ds\":%d,\"K\":%d,\"Kr\":%d,\"Br\":%d,\"L\":%d,"
                 "\"nlist\":%d,\"nprobe\":%d,\"alpha\":%g,\"ck\":%d,\"k\":%d,\"seed\":%u,\"n_queries\":%d,",
                 base_path, N, d, S.M, Ds, K, Kr, S.Br, S.L, nlist, nprobe, alpha, ck, k, seed, nS);
    std::fprintf(f, "\"levels\":[");
    for (int i = 0; i < S.L; ++i) std::fprintf(f, "%s%.9g", i ? "," : "", S.levels[i]);
    std::fprintf(f, "],\"audit\":{\"separability_mismatches\":%lld,\"level_mismatches\":%lld,\"level_checked\":%lld,"
                 "\"residual_unsorted_pairs\":%lld,\"corr_max_abs\":%.6e,\"corr_max_rel\":%.6e,"
                 "\"comp_minus_DH_minus_qnorm2_max_rel\":%.6e,\"v65_returned_vs_DH_plus_qnorm2_max_rel\":%.6e,"
                 "\"v65_primary_vs_fp64_max_rel\":%.6e},",
                 sep_bad, lvl_bad, lvl_checked, unsorted, corr_abs, corr_rel, comp_dev, ret_dev, prim_dev);
    std::fprintf(f, "\"envelope\":{\"build_ms\":%.3f,\"floats\":%zu,\"bytes\":%zu,\"unseen_states\":%lld,\"states\":%zu,"
                 "\"mean_width_cond\":%.6e,\"mean_width_glob\":%.6e,\"incremental\":\"min/max are monotone: an insert widens in O(d); a delete needs a rebuild of its states\"},",
                 env_ms, 2 * ENV, 2 * ENV * sizeof(float), unseen, ecount.size(), wcond / nw, wglob / nw);
    std::fprintf(f, "\"validation\":{\"candidates\":%lld,\"violations_cond\":%lld,\"max_violation_cond\":%.6e,"
                 "\"violations_glob\":%lld,\"max_violation_glob\":%.6e,\"nan_bounds\":%lld,\"tolerance\":\"1e-12*max(1,D_H)\",\"ms\":%.1f},",
                 tot_cand, viol_c, maxviol_c, viol_g, maxviol_g, nan_lb, q_ms);
    std::fprintf(f, "\"per_subspace\":{\"sum_LB_cond\":[");
    for (int m = 0; m < MS; ++m) std::fprintf(f, "%s%.6e", m ? "," : "", sumLBm[m]);
    std::fprintf(f, "],\"sum_LB_glob\":[");
    for (int m = 0; m < MS; ++m) std::fprintf(f, "%s%.6e", m ? "," : "", sumLBgm[m]);
    std::fprintf(f, "],\"sum_DH\":[");
    for (int m = 0; m < MS; ++m) std::fprintf(f, "%s%.6e", m ? "," : "", sumDHm[m]);
    std::fprintf(f, "],\"zero_LB_cond_fraction\":[");
    for (int m = 0; m < MS; ++m) std::fprintf(f, "%s%.6f", m ? "," : "", zeroLBm[m] / std::max<long long>(tot_cand, 1));
    std::fprintf(f, "]},\"queries\":[");
    for (int i = 0; i < nS; ++i) std::fprintf(f, "%s%d", i ? "," : "", qsel[i]);
    std::fprintf(f, "]}\n");
    std::fclose(f);
    printf("wrote %s\n=== RB_OK ===\n", js.c_str());
    return 0;
}
