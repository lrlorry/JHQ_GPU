// Bound-aware residual quantization, offline pilot (experiments/bound_aware_quant).
//
// Same index (arms6 cache), same seeded 128 queries and the same v65 search as
// experiments/residual_aware_bound; the representation comes from
// jhq_v66_boundpilot's export_host_state. In FP64, for every candidate of the
// probed lists, lower bounds on D_H = ||q' - yhat - rhat||^2 from:
//
//   A   the top p bits of each existing 8-bit residual code (p = 0..8; 0 is the
//       whole codebook range). The codebooks are sorted, so the codes sharing a
//       prefix are a contiguous run and their reconstructions lie in
//       [R_m[first], R_m[last]] -- exact, no training. p = 8 is D_H itself.
//   Ac  the same intervals intersected with last round's half-code envelope
//       (Ac p=0 is that envelope's bound, the primary-only one).
//   B   a coarse code of b = 2, 4, 8 bits per 4-dimensional half-subspace:
//       k-means on the decoded residual half-vectors (sample), every indexed
//       vector labelled, and per label the exact box and ball enclosing every
//       member's decoded residual; bounds box, ball, max(box, ball), and
//       max(box intersected with the half-code envelope, ball).
//
// Every envelope covers every indexed vector (the candidates are indexed
// vectors). Per query and bound: violations, survivors at the oracle tau and at
// two seed taus, progressive refinements (probe order, 64 seeds from the
// nearest 8 lists), and a fixed sample of LB / D_H.
//
// Writes <prefix>_rows.csv, <prefix>_ratios.bin and <prefix>_summary.json.
// Environment: JHQ_RB_NQ (128), JHQ_RB_SEED (20261013), JHQ_BQ_KM_SAMPLE
// (65536), JHQ_BQ_KM_ITERS (10); index variables as demo_jhq_batch_pipeline.

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
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }
static inline double gap2(double a, double lo, double hi) {
    const double g = (a < lo) ? (lo - a) : (a > hi ? a - hi : 0.0);
    return g * g;
}

static const int PSET[]  = {0, 1, 2, 4, 6, 8};           // family A
static const int PCSET[] = {0, 1, 2, 4, 6};              // family Ac
static const int BSET[]  = {2, 4, 8};                    // family B
static const char* BKIND[] = {"box", "ball", "max", "maxcond"};
constexpr int NA = 6, NAC = 5, NB = 3, NBK = 4, NBOUND = NA + NAC + NB * NBK;

struct Coarse {
    int b = 0, K = 0;
    std::vector<float> cen, blo, bhi, bc, brad;   // [2M][K][4] / [2M][K]
    std::vector<long long> cnt;                   // [2M][K]
    std::vector<uint8_t> label;                   // [2M][N], half-major: one writer per half
    double train_ms = 0, assign_ms = 0;
    long long empty = 0;
};

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

    // ---- the index, built exactly as in experiments/residual_aware_bound
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

    // ---- the same seeded sample and the same v65 search
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
    idx.search(qs.data(), nS, k, vd.data(), vi.data());
    idx.search(qs.data(), nS, k, vd.data(), vi.data());
    std::vector<int> probes, topck_pos; std::vector<float> topck_val; int lB, lnp, lck;
    idx.debug_last_batch(probes, topck_pos, topck_val, lB, lnp, lck);
    const int ck = lck;

    jhq_gpu::JHQGpuIndex::HostState S;
    idx.export_host_state(S);
    const long long N = S.ntotal;
    const int Ds = S.Ds, H = Ds / 2, K = S.K, Kr = S.Kr, MS = S.M, NH = 2 * MS;
    if (K != 256 || Kr != 256 || H != 4 || S.Br != 8) { fprintf(stderr, "this pilot assumes K=Kr=256, Ds=8, Br=8\n"); return 1; }
    printf("index: N=%lld d=%d M=%d ck=%d, %d queries (seed %u)\n", N, d, MS, ck, nS, seed);
    auto zc = [&](long long pos, int j) -> int { return S.res[(size_t)pos * S.bpv + j]; };
    auto rh = [&](int m, int z) -> float { return S.res_c1d[(size_t)m * Kr + z]; };

    // ---- family A: prefix intervals of the sorted residual codebooks
    std::vector<float> plo((size_t)NA * MS * 256), phi((size_t)NA * MS * 256);
    for (int ip = 0; ip < NA; ++ip) {
        const int pb = PSET[ip], sh = 8 - pb;
        for (int m = 0; m < MS; ++m)
            for (int z = 0; z < 256; ++z) {
                const int first = (z >> sh) << sh, last = first + (1 << sh) - 1;
                plo[((size_t)ip * MS + m) * 256 + z] = rh(m, first);
                phi[((size_t)ip * MS + m) * 256 + z] = rh(m, last);
            }
    }
    // ---- last round's half-code envelope, over every indexed vector
    const size_t ENV = (size_t)NH * 16 * H;
    std::vector<float> emin(ENV, std::numeric_limits<float>::infinity()), emax(ENV, -std::numeric_limits<float>::infinity());
    #pragma omp parallel
    {
        std::vector<float> tmin(ENV, std::numeric_limits<float>::infinity()), tmax(ENV, -std::numeric_limits<float>::infinity());
        #pragma omp for schedule(static)
        for (long long pos = 0; pos < N; ++pos)
            for (int m = 0; m < MS; ++m) {
                const int c = S.primary[(size_t)pos * MS + m];
                for (int s = 0; s < 2; ++s) {
                    const int h = s == 0 ? (c >> 4) : (c & 15);
                    for (int jj = 0; jj < H; ++jj) {
                        const float r = rh(m, zc(pos, m * Ds + s * H + jj));
                        const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + jj;
                        tmin[e] = std::min(tmin[e], r); tmax[e] = std::max(tmax[e], r);
                    }
                }
            }
        #pragma omp critical
        for (size_t e = 0; e < ENV; ++e) { emin[e] = std::min(emin[e], tmin[e]); emax[e] = std::max(emax[e], tmax[e]); }
    }

    // ---- family B: coarse codes per 4-d half-subspace, envelopes over every indexed vector
    const int km_sample = (int)std::min<long long>(N, env_int("JHQ_BQ_KM_SAMPLE", 65536));
    const int km_iters = env_int("JHQ_BQ_KM_ITERS", 10);
    std::vector<Coarse> CB(NB);
    for (int ib = 0; ib < NB; ++ib) {
        Coarse& C = CB[ib];
        C.b = BSET[ib]; C.K = 1 << C.b;
        const int KC = C.K;
        C.cen.assign((size_t)NH * KC * 4, 0.f);
        C.blo.assign((size_t)NH * KC * 4, std::numeric_limits<float>::infinity());
        C.bhi.assign((size_t)NH * KC * 4, -std::numeric_limits<float>::infinity());
        C.bc.assign((size_t)NH * KC * 4, 0.f);
        C.brad.assign((size_t)NH * KC, 0.f);
        C.cnt.assign((size_t)NH * KC, 0);
        C.label.assign((size_t)N * NH, 0);
        auto t0 = Clock::now();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int hf = 0; hf < NH; ++hf) {
            const int m = hf / 2, j0 = m * Ds + (hf % 2) * H;
            const long long stride = std::max(1LL, N / km_sample);
            std::vector<double> pts((size_t)km_sample * 4);
            for (int i = 0; i < km_sample; ++i)
                for (int t = 0; t < 4; ++t) pts[(size_t)i * 4 + t] = rh(m, zc((long long)i * stride, j0 + t));
            std::vector<double> cen((size_t)KC * 4);
            std::mt19937 r2(seed ^ (unsigned)(hf * 2654435761u) ^ (unsigned)C.b);
            for (int c = 0; c < KC; ++c) {
                const int i = (int)(r2() % (unsigned)km_sample);
                for (int t = 0; t < 4; ++t) cen[(size_t)c * 4 + t] = pts[(size_t)i * 4 + t];
            }
            std::vector<double> acc((size_t)KC * 4); std::vector<long long> nc(KC);
            std::vector<int> asg(km_sample);
            for (int it = 0; it < km_iters; ++it) {
                std::fill(acc.begin(), acc.end(), 0.0); std::fill(nc.begin(), nc.end(), 0);
                for (int i = 0; i < km_sample; ++i) {
                    int best = 0; double bd = 1e300;
                    for (int c = 0; c < KC; ++c) {
                        double dd = 0.0;
                        for (int t = 0; t < 4; ++t) { const double x = pts[(size_t)i * 4 + t] - cen[(size_t)c * 4 + t]; dd += x * x; }
                        if (dd < bd) { bd = dd; best = c; }
                    }
                    asg[i] = best; ++nc[best];
                    for (int t = 0; t < 4; ++t) acc[(size_t)best * 4 + t] += pts[(size_t)i * 4 + t];
                }
                for (int c = 0; c < KC; ++c)
                    if (nc[c]) for (int t = 0; t < 4; ++t) cen[(size_t)c * 4 + t] = acc[(size_t)c * 4 + t] / nc[c];
                    else { const int i = (int)(r2() % (unsigned)km_sample); for (int t = 0; t < 4; ++t) cen[(size_t)c * 4 + t] = pts[(size_t)i * 4 + t]; }
            }
            for (size_t i = 0; i < cen.size(); ++i) C.cen[(size_t)hf * KC * 4 + i] = (float)cen[i];
        }
        C.train_ms = Ms(Clock::now() - t0).count();
        t0 = Clock::now();
        // every indexed vector: label, then the exact box and ball of each label's members
        #pragma omp parallel for schedule(dynamic, 1)
        for (int hf = 0; hf < NH; ++hf) {
            const int m = hf / 2, j0 = m * Ds + (hf % 2) * H;
            const float* cen = C.cen.data() + (size_t)hf * KC * 4;
            float* blo = C.blo.data() + (size_t)hf * KC * 4; float* bhi = C.bhi.data() + (size_t)hf * KC * 4;
            long long* cnt = C.cnt.data() + (size_t)hf * KC;
            std::vector<double> rc((size_t)KC, 0.0), rb((size_t)KC, 0.0);
            for (long long pos = 0; pos < N; ++pos) {
                float r[4];
                for (int t = 0; t < 4; ++t) r[t] = rh(m, zc(pos, j0 + t));
                int best = 0; double bd = 1e300;
                for (int c = 0; c < KC; ++c) {
                    double dd = 0.0;
                    for (int t = 0; t < 4; ++t) { const double x = (double)r[t] - cen[c * 4 + t]; dd += x * x; }
                    if (dd < bd) { bd = dd; best = c; }
                }
                C.label[(size_t)hf * N + pos] = (uint8_t)best;
                ++cnt[best];
                rc[best] = std::max(rc[best], bd);
                for (int t = 0; t < 4; ++t) { blo[best * 4 + t] = std::min(blo[best * 4 + t], r[t]); bhi[best * 4 + t] = std::max(bhi[best * 4 + t], r[t]); }
            }
            for (long long pos = 0; pos < N; ++pos) {      // radius about the box centre, too
                const int best = C.label[(size_t)hf * N + pos];
                double dd = 0.0;
                for (int t = 0; t < 4; ++t) {
                    // the centre as it will be stored (float), so the radius is measured from it
                    const double ctr = (float)(0.5 * ((double)blo[best * 4 + t] + bhi[best * 4 + t]));
                    const double x = rh(m, zc(pos, j0 + t)) - ctr; dd += x * x;
                }
                rb[best] = std::max(rb[best], dd);
            }
            for (int c = 0; c < KC; ++c) {
                const bool use_box_ctr = rb[c] < rc[c];
                for (int t = 0; t < 4; ++t)
                    C.bc[((size_t)hf * KC + c) * 4 + t] = use_box_ctr ? (float)(0.5 * ((double)blo[c * 4 + t] + bhi[c * 4 + t])) : cen[c * 4 + t];   // the same float as above
                // the stored radius is rounded up, so the ball still holds every member
                C.brad[(size_t)hf * KC + c] = std::nextafter((float)std::sqrt(std::min(rb[c], rc[c])), std::numeric_limits<float>::infinity());
            }
        }
        C.assign_ms = Ms(Clock::now() - t0).count();
        for (long long c : C.cnt) C.empty += (c == 0);
        printf("coarse b=%d: k-means %.0f ms (sample %d, %d iters), labels+envelopes over all %lld vectors %.0f ms, empty groups %lld of %zu\n",
               C.b, C.train_ms, km_sample, km_iters, N, C.assign_ms, C.empty, C.cnt.size());
    }

    // ---- per query
    std::vector<std::string> names, family; std::vector<int> param; std::vector<double> extra_bits;
    for (int ip = 0; ip < NA; ++ip) { names.push_back("A_p" + std::to_string(PSET[ip])); family.push_back("A"); param.push_back(PSET[ip]); extra_bits.push_back((double)d * PSET[ip]); }
    for (int ip = 0; ip < NAC; ++ip) { names.push_back("Ac_p" + std::to_string(PCSET[ip])); family.push_back("Ac"); param.push_back(PCSET[ip]); extra_bits.push_back((double)d * PCSET[ip]); }
    for (int ib = 0; ib < NB; ++ib)
        for (int kk = 0; kk < NBK; ++kk) {
            names.push_back("B_b" + std::to_string(BSET[ib]) + "_" + BKIND[kk]); family.push_back("B"); param.push_back(BSET[ib]);
            extra_bits.push_back((double)NH * BSET[ib]);
        }
    const int NSAMP = 256;
    int ac_of[NAC];                       // family Ac's prefix lengths, as indices into family A's tables
    for (int ip = 0; ip < NAC; ++ip) ac_of[ip] = (int)(std::find(PSET, PSET + NA, PCSET[ip]) - PSET);
    std::vector<std::vector<std::string>> rows_by_q(nS);
    std::vector<float> ratios((size_t)nS * NBOUND * NSAMP, std::numeric_limits<float>::quiet_NaN());
    std::vector<long long> viol(NBOUND, 0), mono_bad(2, 0);
    std::vector<double> maxex(NBOUND, -1e300);
    auto t0 = Clock::now();
    #pragma omp parallel
    {
        std::vector<long long> lviol(NBOUND, 0), lmono(2, 0);
        std::vector<double> lmax(NBOUND, -1e300);
        #pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < nS; ++qi) {
            const float* q = qs.data() + (size_t)qi * d;
            std::vector<double> qr(d, 0.0);
            for (int kk = 0; kk < d; ++kk) {
                const double qk = q[kk];
                const float* col = S.Pi.data() + (size_t)kk * d;
                for (int i = 0; i < d; ++i) qr[i] += (double)col[i] * qk;
            }
            std::vector<int> cpos, crank;
            for (int r = 0; r < nprobe; ++r) {
                const int l = probes[(size_t)qi * nprobe + r];
                for (int pos = S.list_offsets[l]; pos < S.list_offsets[l + 1]; ++pos) { cpos.push_back(pos); crank.push_back(r); }
            }
            const int n = (int)cpos.size();
            std::vector<double> DH(n), DP(n), LB((size_t)NBOUND * n);
            for (int ci = 0; ci < n; ++ci) {
                const long long pos = cpos[ci];
                double dh = 0.0, dp = 0.0;
                double lb[NBOUND];
                for (int t = 0; t < NBOUND; ++t) lb[t] = 0.0;
                for (int m = 0; m < MS; ++m) {
                    const int c = S.primary[(size_t)pos * MS + m];
                    for (int s = 0; s < 2; ++s) {
                        const int h = s == 0 ? (c >> 4) : (c & 15);
                        const int hf = m * 2 + s;
                        double a4[4];
                        for (int jj = 0; jj < H; ++jj) {
                            const int j = m * Ds + s * H + jj;
                            const double y = S.cent[((size_t)m * K + c) * Ds + s * H + jj];
                            const double a = qr[j] - y;
                            a4[jj] = a;
                            const int z = zc(pos, j);
                            const double r = rh(m, z);
                            dh += (a - r) * (a - r);
                            dp += a * a;
                            const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + jj;
                            for (int ip = 0; ip < NA; ++ip) {
                                const size_t o = ((size_t)ip * MS + m) * 256 + z;
                                lb[ip] += gap2(a, plo[o], phi[o]);
                            }
                            for (int ip = 0; ip < NAC; ++ip) {
                                const size_t o = ((size_t)ac_of[ip] * MS + m) * 256 + z;
                                lb[NA + ip] += gap2(a, std::max(plo[o], emin[e]), std::min(phi[o], emax[e]));
                            }
                        }
                        for (int ib = 0; ib < NB; ++ib) {
                            const Coarse& C = CB[ib];
                            const int lab = C.label[(size_t)hf * N + pos];
                            const size_t g = (size_t)hf * C.K + lab;
                            double box = 0.0, boxc = 0.0, dc2 = 0.0;
                            for (int t = 0; t < 4; ++t) {
                                const double lo = C.blo[g * 4 + t], hi = C.bhi[g * 4 + t];
                                const size_t e = (((size_t)m * 2 + s) * 16 + h) * H + t;
                                box  += gap2(a4[t], lo, hi);
                                boxc += gap2(a4[t], std::max(lo, (double)emin[e]), std::min(hi, (double)emax[e]));
                                const double x = a4[t] - C.bc[g * 4 + t]; dc2 += x * x;
                            }
                            const double gb = std::max(0.0, std::sqrt(dc2) - (double)C.brad[g]);
                            const double ball = gb * gb;
                            const int o = NA + NAC + ib * NBK;
                            lb[o + 0] += box; lb[o + 1] += ball; lb[o + 2] += std::max(box, ball); lb[o + 3] += std::max(boxc, ball);
                        }
                    }
                }
                DH[ci] = dh; DP[ci] = dp;
                for (int t = 0; t < NBOUND; ++t) LB[(size_t)t * n + ci] = lb[t];
                const double tol = 1e-12 * std::max(1.0, dh);
                for (int t = 0; t < NBOUND; ++t) {
                    if (lb[t] - dh > tol) ++lviol[t];
                    lmax[t] = std::max(lmax[t], lb[t] - dh);
                }
                for (int ip = 0; ip + 1 < NA; ++ip) if (lb[ip] > lb[ip + 1] + tol) ++lmono[0];
                for (int ip = 0; ip + 1 < NAC; ++ip) if (lb[NA + ip] > lb[NA + ip + 1] + tol) ++lmono[1];
            }
            // thresholds
            std::vector<double> sorted_dh(DH);
            std::nth_element(sorted_dh.begin(), sorted_dh.begin() + (k - 1), sorted_dh.end());
            const double tau_full = sorted_dh[k - 1];
            auto seeds_of = [&](int L, int s_cnt) {
                std::vector<int> pool;
                for (int i = 0; i < n; ++i) if (crank[i] < L) pool.push_back(i);
                std::sort(pool.begin(), pool.end(), [&](int x, int y) { return DP[x] < DP[y] || (DP[x] == DP[y] && x < y); });
                if ((int)pool.size() > s_cnt) pool.resize(s_cnt);
                return pool;
            };
            auto kth = [&](const std::vector<int>& sd) {
                if ((int)sd.size() < k) return std::numeric_limits<double>::infinity();
                std::vector<double> v; for (int i : sd) v.push_back(DH[i]);
                std::nth_element(v.begin(), v.begin() + (k - 1), v.end());
                return v[k - 1];
            };
            const std::vector<int> s8 = seeds_of(8, 64), s1 = seeds_of(1, 32);
            const double tau8 = kth(s8), tau1 = kth(s1);
            std::vector<char> is_seed(n, 0);
            for (int i : s8) is_seed[i] = 1;
            // a fixed sample of candidates for LB / D_H, the same for every bound
            std::mt19937 r3(seed ^ (unsigned)qsel[qi]);
            std::vector<int> samp(NSAMP);
            for (int t = 0; t < NSAMP; ++t) samp[t] = (int)(r3() % (unsigned)n);
            for (int b = 0; b < NBOUND; ++b) {
                const double* L = LB.data() + (size_t)b * n;
                long long sf = 0, s8c = 0, s1c = 0;
                for (int i = 0; i < n; ++i) { sf += (L[i] <= tau_full); s8c += (L[i] <= tau8); s1c += (L[i] <= tau1); }
                // progressive: the 64 seeds, then the probe order (candidates are stored in it)
                std::priority_queue<double> heap;
                long long refined = 0;
                for (int i : s8) { ++refined; heap.push(DH[i]); if ((int)heap.size() > k) heap.pop(); }
                for (int i = 0; i < n; ++i) {
                    if (is_seed[i]) continue;
                    const double tau = ((int)heap.size() >= k) ? heap.top() : std::numeric_limits<double>::infinity();
                    if (L[i] <= tau) { ++refined; if ((int)heap.size() < k || DH[i] < heap.top()) { heap.push(DH[i]); if ((int)heap.size() > k) heap.pop(); } }
                }
                char line[512];
                std::snprintf(line, sizeof(line), "%d,%d,%d,%s,%s,%d,%.0f,%lld,%lld,%lld,%lld,%.9g,%.9g,%.9g",
                              qsel[qi], n, ck, names[b].c_str(), family[b].c_str(), param[b], extra_bits[b],
                              sf, s8c, s1c, refined, tau_full, tau8, tau1);
                rows_by_q[qi].push_back(line);
                for (int t = 0; t < NSAMP; ++t) {
                    const int i = samp[t];
                    ratios[((size_t)qi * NBOUND + b) * NSAMP + t] = DH[i] > 1e-12 ? (float)(L[i] / DH[i]) : std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        #pragma omp critical
        {
            for (int t = 0; t < NBOUND; ++t) { viol[t] += lviol[t]; maxex[t] = std::max(maxex[t], lmax[t]); }
            mono_bad[0] += lmono[0]; mono_bad[1] += lmono[1];
        }
    }
    const double q_ms = Ms(Clock::now() - t0).count();
    printf("queries: %.0f ms; violations:", q_ms);
    for (int t = 0; t < NBOUND; ++t) printf(" %s=%lld", names[t].c_str(), viol[t]);
    printf("; monotonicity breaks A %lld, Ac %lld\n", mono_bad[0], mono_bad[1]);

    FILE* f = std::fopen((prefix + "_rows.csv").c_str(), "w");
    std::fprintf(f, "qid,n_cand,ck,bound,family,param,extra_bits,survivors_oracle,survivors_seed_L8_s64,survivors_seed_L1_s32,prog_refined_L8_s64,tau_full,tau_seed_L8_s64,tau_seed_L1_s32\n");
    for (auto& v : rows_by_q) for (auto& s : v) std::fprintf(f, "%s\n", s.c_str());
    std::fclose(f);
    f = std::fopen((prefix + "_ratios.bin").c_str(), "wb");
    const int hdr[3] = {nS, NBOUND, NSAMP};
    std::fwrite(hdr, sizeof(int), 3, f);
    std::fwrite(ratios.data(), sizeof(float), ratios.size(), f);
    std::fclose(f);
    f = std::fopen((prefix + "_summary.json").c_str(), "w");
    std::fprintf(f, "{\"N\":%lld,\"d\":%d,\"M\":%d,\"ck\":%d,\"k\":%d,\"alpha\":%g,\"nprobe\":%d,\"seed\":%u,\"n_queries\":%d,\"km_sample\":%d,\"km_iters\":%d,\"bounds\":[",
                 N, d, MS, ck, k, alpha, nprobe, seed, nS, km_sample, km_iters);
    for (int t = 0; t < NBOUND; ++t)
        std::fprintf(f, "%s{\"name\":\"%s\",\"family\":\"%s\",\"param\":%d,\"extra_bits\":%.0f,\"violations\":%lld,\"max_excess\":%.6e}",
                     t ? "," : "", names[t].c_str(), family[t].c_str(), param[t], extra_bits[t], viol[t], maxex[t]);
    std::fprintf(f, "],\"monotonicity_breaks_A\":%lld,\"monotonicity_breaks_Ac\":%lld,\"coarse\":[", mono_bad[0], mono_bad[1]);
    for (int ib = 0; ib < NB; ++ib) {
        const Coarse& C = CB[ib];
        double wsum = 0.0, rsum = 0.0; long long ng = 0;
        for (size_t g = 0; g < C.cnt.size(); ++g)
            if (C.cnt[g]) { for (int t = 0; t < 4; ++t) wsum += C.bhi[g * 4 + t] - C.blo[g * 4 + t]; rsum += C.brad[g]; ++ng; }
        std::fprintf(f, "%s{\"b\":%d,\"K\":%d,\"train_ms\":%.1f,\"assign_ms\":%.1f,\"empty_groups\":%lld,\"groups\":%zu,"
                     "\"mean_box_width\":%.6e,\"mean_ball_radius\":%.6e,\"metadata_floats\":%zu}",
                     ib ? "," : "", C.b, C.K, C.train_ms, C.assign_ms, C.empty, C.cnt.size(), wsum / (4.0 * std::max(ng, 1LL)),
                     rsum / std::max(ng, 1LL), (size_t)NH * C.K * (4 + 4 + 4 + 4 + 1));
    }
    std::fprintf(f, "],\"query_ms\":%.1f}\n", q_ms);
    std::fclose(f);
    printf("=== BQ_OK ===\n");
    return 0;
}
