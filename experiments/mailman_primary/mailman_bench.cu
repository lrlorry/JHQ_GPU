// mailman_bench.cu -- scoring the real JHQ primary codes: the factorised 16+16 table of the v65/v69
// scan (B0) against Mailman-style shared summation over r candidates of one list (experiments/
// mailman_primary). Isolated scoring: every implementation scores the same candidates (the probed
// lists of real queries, in probe order) and writes the primary distance of each, so the outputs
// are compared one to one and with the exported FP64 primary distance.
//
//   B0      one thread per candidate; per 32-bit word of the [W][N] layout, two table reads per
//           byte (high nibble, then low nibble) from the shared-memory table -- the scan's loop
//   H1 r=8  one warp per group of 8 candidates; patterns [group][d] (byte = 8 candidates' bits at a
//           dimension); per warp a 256-bucket FP32 histogram in shared memory, atomicAdd of q'_j
//           into bucket p_j; then z_k = sum of the buckets with bit k set (warp reductions)
//   H1 r=4  the same with 4-bit patterns and 16 buckets
//   H2 r=8  one lane per group; patterns [d/4][group] (coalesced across lanes); a private
//           256-bucket column per lane, [bucket][lane] in shared memory (no atomics, no bank
//           conflicts: 32 KB a warp); the O(2^r) fold recovers the 8 sums
//   H2 r=4  the same with 16 buckets (2 KB a warp)
// D_P = ||q'||^2 + d c^2 + 2c sum(q') - 4c z, z = sum_j q'_j b_j (b = 1 for +c).
//
// Inputs, read only: <pp>/meta.json, primary.u8 [N][M], list_offsets.i32, qrot.f64, qids.i32
// (experiments/progressive_primary export) and <lf>/rb_q<qid>.bin (probes in probe order and the
// FP64 primary distance of every candidate; experiments/residual_aware_bound export).
//
//   mailman_bench <pp dir> <lf dir> <out.json> [batches]      env: MB_ROUNDS (7), MB_REPS (10),
//   MB_BLOCK_B0 (512 at d=768, 1024 at d=3072), MB_PROFILE (0: per-phase clock64 of H1/H2 instead)

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

using u32 = uint32_t;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
static int env_int(const char* n, int d) { const char* e = std::getenv(n); return e ? std::atoi(e) : d; }

// ── phase profile (MB_PROFILE=1): cycles per warp, summed ─────────────────────────────────────
__device__ unsigned long long g_phase[4];   // zero, accumulate, recover, write

// ── kernels ──────────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void k_build_lut(const float* __restrict__ qf, int d, int M, float c, float* __restrict__ lut)
{
    // lut[q][m*32 + h]: dims 8m..8m+3 for the high nibble h (bit 3 of h = dim 8m), lut[..16 + l]:
    // dims 8m+4..8m+7 for the low nibble l; level +c for bit 1, -c for bit 0
    const int q = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * 32) return;
    const int m = i >> 5, e = i & 31, half = e >> 4, nib = e & 15;
    const float* qq = qf + (long)q * d + 8 * m + 4 * half;
    float s = 0.f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float lev = ((nib >> (3 - t)) & 1) ? c : -c;
        const float df = qq[t] - lev;
        s += df * df;
    }
    lut[(long)q * M * 32 + i] = s;
}

// Mailman query preparation: C0 = ||q'||^2 + d c^2 + 2c sum(q') (q' itself is already on the device)
__global__ void k_prep_mm(const float* __restrict__ qf, int d, float c, float* __restrict__ C0)
{
    __shared__ float s1[32], s2[32];
    const int q = blockIdx.x;
    float a = 0.f, b = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) { const float v = qf[(long)q * d + j]; a += v * v; b += v; }
    a = warp_sum(a); b = warp_sum(b);
    if ((threadIdx.x & 31) == 0) { s1[threadIdx.x >> 5] = a; s2[threadIdx.x >> 5] = b; }
    __syncthreads();
    if (threadIdx.x < 32) {
        a = threadIdx.x < (blockDim.x >> 5) ? s1[threadIdx.x] : 0.f;
        b = threadIdx.x < (blockDim.x >> 5) ? s2[threadIdx.x] : 0.f;
        a = warp_sum(a); b = warp_sum(b);
        if (threadIdx.x == 0) C0[q] = a + d * c * c + 2.f * c * b;
    }
}

__global__ void k_b0(const u32* __restrict__ words, long N, int M, const float* __restrict__ lut,
                     const int* __restrict__ cand_pos, const long* __restrict__ cand_off, float* __restrict__ out)
{
    extern __shared__ float s_lut[];
    const int q = blockIdx.y;
    const float* g = lut + (long)q * M * 32;
    for (int i = threadIdx.x; i < M * 32; i += blockDim.x) s_lut[i] = g[i];
    __syncthreads();
    const long base = cand_off[q], n = cand_off[q + 1] - base;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long)gridDim.x * blockDim.x) {
        const int pos = cand_pos[base + idx];
        float a = 0.f;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const u32 word = __ldg(&words[(long)(m0 >> 2) * N + pos]);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const u32 cm = (word >> (8 * i)) & 0xFFu;
                const int m = m0 + i;
                a += s_lut[m * 32 + (cm >> 4)] + s_lut[m * 32 + 16 + (cm & 15)];
            }
        }
        out[base + idx] = a;
    }
}

struct GroupArgs {
    const int* grp_id;            // global group of each entry
    const int* grp_out;           // output index of its first member, within the query's candidates
    const unsigned char* grp_cnt; // members (r, or fewer for a list's last group)
    const long* grp_off;          // [Q + 1]
    const long* cand_off;         // [Q + 1]
    const float* qf;              // [Q][d]
    const float* C0;              // [Q] ||q'||^2 + d c^2 + 2c sum(q')
    float coef;                   // -4c
};

template <bool PROF>
__global__ void k_h1_r8(const u32* __restrict__ P8g, int d, GroupArgs a, float* __restrict__ out)
{
    extern __shared__ float sm[];
    float* s_q = sm;
    const int q = blockIdx.y, nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* h = sm + d + warp * 256;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s_q[i] = a.qf[(long)q * d + i];
    __syncthreads();
    const int d4 = d >> 2;
    const long gb = a.grp_off[q], ng = a.grp_off[q + 1] - gb, ob = a.cand_off[q];
    const float c0 = a.C0[q];
    unsigned long long t0 = 0, ph[4] = {0, 0, 0, 0};
    for (long e = (long)blockIdx.x * nw + warp; e < ng; e += (long)gridDim.x * nw) {
        if (PROF) t0 = clock64();
        #pragma unroll
        for (int i = 0; i < 8; ++i) h[lane + 32 * i] = 0.f;
        __syncwarp();
        if (PROF) { ph[0] += clock64() - t0; t0 = clock64(); }
        const u32* pp = P8g + (long)a.grp_id[gb + e] * d4;
        for (int j4 = lane; j4 < d4; j4 += 32) {
            const u32 w = __ldg(&pp[j4]);
            const float4 qv = reinterpret_cast<const float4*>(s_q)[j4];
            atomicAdd(&h[w & 255u], qv.x);
            atomicAdd(&h[(w >> 8) & 255u], qv.y);
            atomicAdd(&h[(w >> 16) & 255u], qv.z);
            atomicAdd(&h[w >> 24], qv.w);
        }
        __syncwarp();
        if (PROF) { ph[1] += clock64() - t0; t0 = clock64(); }
        float v[8], tot = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) { v[u] = h[lane * 8 + u]; tot += v[u]; }
        float z[8];
        z[0] = v[1] + v[3] + v[5] + v[7];
        z[1] = v[2] + v[3] + v[6] + v[7];
        z[2] = v[4] + v[5] + v[6] + v[7];
        #pragma unroll
        for (int k = 3; k < 8; ++k) z[k] = ((lane >> (k - 3)) & 1) ? tot : 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) z[k] = warp_sum(z[k]);
        float my = z[0];
        #pragma unroll
        for (int k = 1; k < 8; ++k) my = (lane == k) ? z[k] : my;
        if (PROF) { ph[2] += clock64() - t0; t0 = clock64(); }
        const int cnt = a.grp_cnt[gb + e];
        if (lane < cnt) out[ob + a.grp_out[gb + e] + lane] = fmaf(a.coef, my, c0);
        __syncwarp();
        if (PROF) ph[3] += clock64() - t0;
    }
    if (PROF && lane == 0) for (int i = 0; i < 4; ++i) atomicAdd(&g_phase[i], ph[i]);
}

template <bool PROF>
__global__ void k_h1_r4(const u32* __restrict__ P4g, int d, GroupArgs a, float* __restrict__ out)
{
    extern __shared__ float sm[];
    float* s_q = sm;
    const int q = blockIdx.y, nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* h = sm + d + warp * 16;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s_q[i] = a.qf[(long)q * d + i];
    __syncthreads();
    const int d8 = d >> 3;
    const long gb = a.grp_off[q], ng = a.grp_off[q + 1] - gb, ob = a.cand_off[q];
    const float c0 = a.C0[q];
    unsigned long long t0 = 0, ph[4] = {0, 0, 0, 0};
    for (long e = (long)blockIdx.x * nw + warp; e < ng; e += (long)gridDim.x * nw) {
        if (PROF) t0 = clock64();
        if (lane < 16) h[lane] = 0.f;
        __syncwarp();
        if (PROF) { ph[0] += clock64() - t0; t0 = clock64(); }
        const u32* pp = P4g + (long)a.grp_id[gb + e] * d8;
        for (int j8 = lane; j8 < d8; j8 += 32) {
            const u32 w = __ldg(&pp[j8]);
            const float4 qa = reinterpret_cast<const float4*>(s_q)[2 * j8];
            const float4 qb = reinterpret_cast<const float4*>(s_q)[2 * j8 + 1];
            atomicAdd(&h[w & 15u], qa.x);         atomicAdd(&h[(w >> 4) & 15u], qa.y);
            atomicAdd(&h[(w >> 8) & 15u], qa.z);  atomicAdd(&h[(w >> 12) & 15u], qa.w);
            atomicAdd(&h[(w >> 16) & 15u], qb.x); atomicAdd(&h[(w >> 20) & 15u], qb.y);
            atomicAdd(&h[(w >> 24) & 15u], qb.z); atomicAdd(&h[w >> 28], qb.w);
        }
        __syncwarp();
        if (PROF) { ph[1] += clock64() - t0; t0 = clock64(); }
        const float v = lane < 16 ? h[lane] : 0.f;
        float z[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) z[k] = warp_sum(((lane >> k) & 1) ? v : 0.f);
        float my = z[0];
        #pragma unroll
        for (int k = 1; k < 4; ++k) my = (lane == k) ? z[k] : my;
        if (PROF) { ph[2] += clock64() - t0; t0 = clock64(); }
        const int cnt = a.grp_cnt[gb + e];
        if (lane < cnt) out[ob + a.grp_out[gb + e] + lane] = fmaf(a.coef, my, c0);
        __syncwarp();
        if (PROF) ph[3] += clock64() - t0;
    }
    if (PROF && lane == 0) for (int i = 0; i < 4; ++i) atomicAdd(&g_phase[i], ph[i]);
}

template <bool PROF>
__global__ void k_h2_r8(const u32* __restrict__ P8d, long G, int d, GroupArgs a, float* __restrict__ out)
{
    extern __shared__ float sm[];
    float* s_q = sm;
    const int q = blockIdx.y, nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* h = sm + d + (long)warp * 256 * 32 + lane;            // my column: h[p * 32]
    for (int i = threadIdx.x; i < d; i += blockDim.x) s_q[i] = a.qf[(long)q * d + i];
    __syncthreads();
    const int d4 = d >> 2;
    const long gb = a.grp_off[q], ng = a.grp_off[q + 1] - gb, ob = a.cand_off[q];
    const float c0 = a.C0[q];
    unsigned long long t0 = 0, ph[4] = {0, 0, 0, 0};
    for (long e0 = ((long)blockIdx.x * nw + warp) * 32; e0 < ng; e0 += (long)gridDim.x * nw * 32) {
        const long e = e0 + lane;
        const bool valid = e < ng;
        const long g = valid ? a.grp_id[gb + e] : 0;
        if (PROF) t0 = clock64();
        #pragma unroll 8
        for (int p = 0; p < 256; ++p) h[p * 32] = 0.f;
        if (PROF) { ph[0] += clock64() - t0; t0 = clock64(); }
        for (int j4 = 0; j4 < d4; ++j4) {
            const u32 w = valid ? __ldg(&P8d[(long)j4 * G + g]) : 0u;
            const float4 qv = reinterpret_cast<const float4*>(s_q)[j4];
            h[(w & 255u) * 32] += qv.x;
            h[((w >> 8) & 255u) * 32] += qv.y;
            h[((w >> 16) & 255u) * 32] += qv.z;
            h[(w >> 24) * 32] += qv.w;
        }
        if (PROF) { ph[1] += clock64() - t0; t0 = clock64(); }
        float z[8];
        #pragma unroll
        for (int k = 7; k >= 0; --k) {
            const int half = 1 << k;
            float s = 0.f;
            for (int t = 0; t < half; ++t) s += h[(half + t) * 32];
            z[k] = s;
            if (k > 0)
                for (int t = 0; t < half; ++t) h[t * 32] += h[(half + t) * 32];
        }
        if (PROF) { ph[2] += clock64() - t0; t0 = clock64(); }
        if (valid) {
            const int cnt = a.grp_cnt[gb + e];
            float* o = out + ob + a.grp_out[gb + e];
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i < cnt) o[i] = fmaf(a.coef, z[i], c0);
        }
        if (PROF) ph[3] += clock64() - t0;
    }
    if (PROF && lane == 0) for (int i = 0; i < 4; ++i) atomicAdd(&g_phase[i], ph[i]);
}

template <bool PROF>
__global__ void k_h2_r4(const u32* __restrict__ P4d, long G, int d, GroupArgs a, float* __restrict__ out)
{
    extern __shared__ float sm[];
    float* s_q = sm;
    const int q = blockIdx.y, nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* h = sm + d + (long)warp * 16 * 32 + lane;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s_q[i] = a.qf[(long)q * d + i];
    __syncthreads();
    const int d8 = d >> 3;
    const long gb = a.grp_off[q], ng = a.grp_off[q + 1] - gb, ob = a.cand_off[q];
    const float c0 = a.C0[q];
    unsigned long long t0 = 0, ph[4] = {0, 0, 0, 0};
    for (long e0 = ((long)blockIdx.x * nw + warp) * 32; e0 < ng; e0 += (long)gridDim.x * nw * 32) {
        const long e = e0 + lane;
        const bool valid = e < ng;
        const long g = valid ? a.grp_id[gb + e] : 0;
        if (PROF) t0 = clock64();
        #pragma unroll
        for (int p = 0; p < 16; ++p) h[p * 32] = 0.f;
        if (PROF) { ph[0] += clock64() - t0; t0 = clock64(); }
        for (int j8 = 0; j8 < d8; ++j8) {
            const u32 w = valid ? __ldg(&P4d[(long)j8 * G + g]) : 0u;
            const float4 qa = reinterpret_cast<const float4*>(s_q)[2 * j8];
            const float4 qb = reinterpret_cast<const float4*>(s_q)[2 * j8 + 1];
            h[(w & 15u) * 32] += qa.x;          h[((w >> 4) & 15u) * 32] += qa.y;
            h[((w >> 8) & 15u) * 32] += qa.z;   h[((w >> 12) & 15u) * 32] += qa.w;
            h[((w >> 16) & 15u) * 32] += qb.x;  h[((w >> 20) & 15u) * 32] += qb.y;
            h[((w >> 24) & 15u) * 32] += qb.z;  h[(w >> 28) * 32] += qb.w;
        }
        if (PROF) { ph[1] += clock64() - t0; t0 = clock64(); }
        float z[4];
        #pragma unroll
        for (int k = 3; k >= 0; --k) {
            const int half = 1 << k;
            float s = 0.f;
            #pragma unroll
            for (int t = 0; t < half; ++t) s += h[(half + t) * 32];
            z[k] = s;
            if (k > 0) {
                #pragma unroll
                for (int t = 0; t < half; ++t) h[t * 32] += h[(half + t) * 32];
            }
        }
        if (PROF) { ph[2] += clock64() - t0; t0 = clock64(); }
        if (valid) {
            const int cnt = a.grp_cnt[gb + e];
            float* o = out + ob + a.grp_out[gb + e];
            #pragma unroll
            for (int i = 0; i < 4; ++i) if (i < cnt) o[i] = fmaf(a.coef, z[i], c0);
        }
        if (PROF) ph[3] += clock64() - t0;
    }
    if (PROF && lane == 0) for (int i = 0; i < 4; ++i) atomicAdd(&g_phase[i], ph[i]);
}

// ── host ─────────────────────────────────────────────────────────────────────────────────────

template <class T> static std::vector<T> read_all(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    const size_t n = f.tellg();
    f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

static double json_num(const std::string& s, const std::string& key)
{
    const size_t p = s.find("\"" + key + "\":");
    if (p == std::string::npos) { std::fprintf(stderr, "no %s in meta\n", key.c_str()); std::exit(1); }
    return std::atof(s.c_str() + p + key.size() + 3);
}

struct QueryData {
    int qid, n, nprobe, ck;
    std::vector<int> probes, cpos, topck;
    std::vector<double> dprim;
};

static QueryData read_rb(const std::string& lf, int qid)
{
    QueryData r;
    const std::string p = lf + "/rb_q" + std::to_string(qid) + ".bin";
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    int hdr[5];
    if (std::fread(hdr, 4, 5, f) != 5) std::exit(1);
    r.qid = hdr[0]; r.n = hdr[1]; r.nprobe = hdr[2]; r.ck = hdr[3];
    const int k = hdr[4];
    r.probes.resize(r.nprobe); r.cpos.resize(r.n); r.dprim.resize(r.n); r.topck.resize(r.ck);
    size_t ok = std::fread(r.probes.data(), 4, r.nprobe, f);
    ok += std::fread(r.cpos.data(), 4, r.n, f);
    std::fseek(f, (long)r.n * 8, SEEK_CUR);                   // id, rank
    ok += std::fread(r.dprim.data(), 8, r.n, f);
    std::fseek(f, (long)r.n * 24 + 8 + (long)k * (4 + 4 + 8), SEEK_CUR);   // D_H, LB_cond, LB_glob, qnorm2, v65 ids/dists/DH
    ok += std::fread(r.topck.data(), 4, r.ck, f);
    std::fclose(f);
    (void)ok;
    return r;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start() { CK(cudaEventRecord(a)); }
    float stop() { CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: %s <pp dir> <lf dir> <out.json> [batches 8,1000]\n", argv[0]); return 1; }
    const std::string pp = argv[1], lf = argv[2], outp = argv[3];
    std::vector<int> batches = {8, 1000};
    if (argc > 4) { batches.clear(); for (char* s = argv[4]; *s;) { batches.push_back(std::atoi(s)); while (*s && *s != ',') ++s; if (*s) ++s; } }
    const int rounds = env_int("MB_ROUNDS", 7), reps = env_int("MB_REPS", 10);
    const bool profile = env_int("MB_PROFILE", 0) != 0;

    std::ifstream mf(pp + "/meta.json");
    std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const long N = (long)json_num(meta, "N");
    const int d = (int)json_num(meta, "d"), M = (int)json_num(meta, "M"), nlist = (int)json_num(meta, "nlist");
    const size_t lp = meta.find("\"levels\":[");
    const double c = std::fabs(std::atof(meta.c_str() + meta.find(',', lp) + 1));
    const int W = M / 4;
    const int block_b0 = env_int("MB_BLOCK_B0", d >= 3072 ? 1024 : 512);
    std::printf("N=%ld d=%d M=%d nlist=%d c=%.10f block_b0=%d\n", N, d, M, nlist, c, block_b0);

    auto t0 = clk::now();
    std::vector<uint8_t> codes = read_all<uint8_t>(pp + "/primary.u8");
    std::vector<int> offs = read_all<int>(pp + "/list_offsets.i32");
    std::vector<double> qrot = read_all<double>(pp + "/qrot.f64");
    std::vector<int> qids = read_all<int>(pp + "/qids.i32");
    const int nS = (int)qids.size();
    if ((long)codes.size() != N * M || (int)offs.size() != nlist + 1) { std::fprintf(stderr, "shape mismatch\n"); return 1; }
    std::printf("loaded codes %.1f MB in %.1f s\n", codes.size() / 1e6, secs(t0));

    // ── layouts (build costs measured) ───────────────────────────────────────────────────────
    t0 = clk::now();
    std::vector<u32> words((size_t)W * N);
    #pragma omp parallel for schedule(static)
    for (long pos = 0; pos < N; ++pos)
        for (int w = 0; w < W; ++w) {
            const uint8_t* b = &codes[(size_t)pos * M + 4 * w];
            words[(size_t)w * N + pos] = (u32)b[0] | ((u32)b[1] << 8) | ((u32)b[2] << 16) | ((u32)b[3] << 24);
        }
    const double t_words = secs(t0);
    // groups: per list, r consecutive positions
    auto make_groups = [&](int r, std::vector<long>& gbase, std::vector<int>& gstart, std::vector<unsigned char>& gcnt) {
        gbase.assign(nlist + 1, 0);
        for (int l = 0; l < nlist; ++l) gbase[l + 1] = gbase[l] + (offs[l + 1] - offs[l] + r - 1) / r;
        gstart.resize(gbase[nlist]); gcnt.resize(gbase[nlist]);
        for (int l = 0; l < nlist; ++l)
            for (long g = gbase[l]; g < gbase[l + 1]; ++g) {
                const long s = offs[l] + (g - gbase[l]) * r;
                gstart[g] = (int)s;
                gcnt[g] = (unsigned char)std::min<long>(r, offs[l + 1] - s);
            }
    };
    std::vector<long> gb8, gb4; std::vector<int> gs8, gs4; std::vector<unsigned char> gc8, gc4;
    make_groups(8, gb8, gs8, gc8);
    make_groups(4, gb4, gs4, gc4);
    const long G8 = gb8[nlist], G4 = gb4[nlist];
    auto bit = [&](long pos, int j) -> u32 { return (codes[(size_t)pos * M + (j >> 3)] >> (7 - (j & 7))) & 1u; };
    t0 = clk::now();
    std::vector<u32> P8g((size_t)G8 * (d / 4)), P8d((size_t)(d / 4) * G8);
    #pragma omp parallel for schedule(dynamic, 256)
    for (long g = 0; g < G8; ++g)
        for (int j4 = 0; j4 < d / 4; ++j4) {
            u32 w = 0;
            for (int t = 0; t < 4; ++t) {
                u32 p = 0;
                for (int i = 0; i < gc8[g]; ++i) p |= bit(gs8[g] + i, 4 * j4 + t) << i;
                w |= p << (8 * t);
            }
            P8g[(size_t)g * (d / 4) + j4] = w;
            P8d[(size_t)j4 * G8 + g] = w;
        }
    const double t_p8 = secs(t0);
    t0 = clk::now();
    std::vector<u32> P4g((size_t)G4 * (d / 8)), P4d((size_t)(d / 8) * G4);
    #pragma omp parallel for schedule(dynamic, 256)
    for (long g = 0; g < G4; ++g)
        for (int j8 = 0; j8 < d / 8; ++j8) {
            u32 w = 0;
            for (int t = 0; t < 8; ++t) {
                u32 p = 0;
                for (int i = 0; i < gc4[g]; ++i) p |= bit(gs4[g] + i, 8 * j8 + t) << i;
                w |= p << (4 * t);
            }
            P4g[(size_t)g * (d / 8) + j8] = w;
            P4d[(size_t)j8 * G4 + g] = w;
        }
    const double t_p4 = secs(t0);
    std::printf("layouts: words %.2f s, r=8 patterns %.2f s (G=%ld), r=4 patterns %.2f s (G=%ld), threads %d\n",
                t_words, t_p8, G8, t_p4, G4, omp_get_max_threads());

    // ── queries ──────────────────────────────────────────────────────────────────────────────
    t0 = clk::now();
    std::vector<QueryData> qd(nS);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < nS; ++i) qd[i] = read_rb(lf, qids[i]);
    std::printf("read %d query files in %.1f s\n", nS, secs(t0));
    // candidates: probed lists in probe order, each in list order; check against the export's order
    std::vector<std::vector<int>> cand(nS);
    std::vector<std::vector<double>> dref(nS);
    long order_mismatch = 0;
    for (int i = 0; i < nS; ++i) {
        for (int l : qd[i].probes) for (int p = offs[l]; p < offs[l + 1]; ++p) cand[i].push_back(p);
        if ((int)cand[i].size() != qd[i].n || !std::equal(cand[i].begin(), cand[i].end(), qd[i].cpos.begin())) {
            ++order_mismatch;
            // map the FP64 reference by position
            std::vector<std::pair<int, double>> m(qd[i].n);
            for (int t = 0; t < qd[i].n; ++t) m[t] = {qd[i].cpos[t], qd[i].dprim[t]};
            std::sort(m.begin(), m.end());
            dref[i].resize(cand[i].size());
            for (size_t t = 0; t < cand[i].size(); ++t)
                dref[i][t] = std::lower_bound(m.begin(), m.end(), std::make_pair(cand[i][t], -1e300))->second;
        } else {
            dref[i] = qd[i].dprim;
        }
    }
    std::printf("candidate order differs from the export's in %ld queries (mapped by position)\n", order_mismatch);

    // ── device data ──────────────────────────────────────────────────────────────────────────
    u32 *d_words, *d_P8g, *d_P8d, *d_P4g, *d_P4d;
    CK(cudaMalloc(&d_words, words.size() * 4)); CK(cudaMemcpy(d_words, words.data(), words.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_P8g, P8g.size() * 4)); CK(cudaMemcpy(d_P8g, P8g.data(), P8g.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_P8d, P8d.size() * 4)); CK(cudaMemcpy(d_P8d, P8d.data(), P8d.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_P4g, P4g.size() * 4)); CK(cudaMemcpy(d_P4g, P4g.data(), P4g.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_P4d, P4d.size() * 4)); CK(cudaMemcpy(d_P4d, P4d.data(), P4d.size() * 4, cudaMemcpyHostToDevice));
    int dev_sms = 0, smem_optin = 0;
    CK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    FILE* jf = std::fopen(outp.c_str(), "w");
    std::fprintf(jf, "{\"gpu\":\"%s\",\"sms\":%d,\"N\":%ld,\"d\":%d,\"M\":%d,\"nlist\":%d,\"c\":%.10g,\"block_b0\":%d,"
                     "\"layout_bytes\":{\"words\":%zu,\"P8\":%zu,\"P4\":%zu,\"raw_bits_bytes\":%ld},"
                     "\"build_s\":{\"words\":%.3f,\"P8_both\":%.3f,\"P4_both\":%.3f},\"groups\":{\"G8\":%ld,\"G4\":%ld},"
                     "\"profile\":%d,\"runs\":[",
                 prop.name, dev_sms, N, d, M, nlist, c, block_b0, words.size() * 4, P8g.size() * 4, P4g.size() * 4,
                 N * d / 8, t_words, t_p8, t_p4, G8, G4, (int)profile);
    bool first = true;

    for (int B : batches) {
        const int nb = std::max(1, std::min(nS / B, B >= 256 ? 1 : 16));     // batches timed
        // per-batch device arrays, built for the first batch of size B and reused shape-wise
        constexpr int NS = 5, NI = 8;                     // 5 scorers (checked), then 3 timed-only items
        std::vector<std::vector<double>> st(NI);         // ms per launch per item
        const char* names[NI] = {"B0", "H1_r8", "H1_r4", "H2_r8", "H2_r4", "B0_scangrid", "B0_prep_lut", "MM_prep_C0"};
        double err_max[5] = {0, 0, 0, 0, 0}, err_sum[5] = {0, 0, 0, 0, 0}; long err_n = 0;
        std::vector<double> errs[5];
        double vs_b0_max[5] = {0, 0, 0, 0, 0};
        int topT_diff_fp64[5] = {0, 0, 0, 0, 0}, topT_diff_b0[5] = {0, 0, 0, 0, 0};
        double topT_gap_max[5] = {0, 0, 0, 0, 0};
        long ncand_tot = 0;
        unsigned long long phase[5][4] = {};
        double c0_dev_rel = 0.0;
        for (int bi = 0; bi < nb; ++bi) {
            const int q0 = bi * B;
            // candidates, groups
            std::vector<long> cand_off(B + 1, 0), g8off(B + 1, 0), g4off(B + 1, 0);
            std::vector<int> cpos, g8id, g8out, g4id, g4out; std::vector<unsigned char> g8cnt, g4cnt;
            std::vector<float> qf((size_t)B * d), C0(B);
            for (int i = 0; i < B; ++i) {
                const QueryData& Q = qd[q0 + i];
                long o = 0;
                for (int l : Q.probes) {
                    for (int p = offs[l]; p < offs[l + 1]; ++p) cpos.push_back(p);
                    for (long g = gb8[l]; g < gb8[l + 1]; ++g) { g8id.push_back((int)g); g8out.push_back((int)(o + (g - gb8[l]) * 8)); g8cnt.push_back(gc8[g]); }
                    for (long g = gb4[l]; g < gb4[l + 1]; ++g) { g4id.push_back((int)g); g4out.push_back((int)(o + (g - gb4[l]) * 4)); g4cnt.push_back(gc4[g]); }
                    o += offs[l + 1] - offs[l];
                }
                cand_off[i + 1] = cand_off[i] + o;
                g8off[i + 1] = (long)g8id.size(); g4off[i + 1] = (long)g4id.size();
                double qn2 = 0, sq = 0;
                for (int j = 0; j < d; ++j) {
                    const double v = qrot[(size_t)(q0 + i) * d + j];
                    qf[(size_t)i * d + j] = (float)v; qn2 += v * v; sq += v;
                }
                C0[i] = (float)(qn2 + d * c * c + 2 * c * sq);
            }
            const long nc = cand_off[B];
            ncand_tot += nc;
            int *d_cpos, *d_g8id, *d_g8out, *d_g4id, *d_g4out; unsigned char *d_g8cnt, *d_g4cnt;
            long *d_coff, *d_g8off, *d_g4off; float *d_qf, *d_C0, *d_C0x, *d_lut, *d_out[5];
            CK(cudaMalloc(&d_cpos, nc * 4)); CK(cudaMemcpy(d_cpos, cpos.data(), nc * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d_coff, (B + 1) * 8)); CK(cudaMemcpy(d_coff, cand_off.data(), (B + 1) * 8, cudaMemcpyHostToDevice));
            auto up = [](auto& v, auto*& d) { CK(cudaMalloc(&d, v.size() * sizeof(v[0]))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice)); };
            up(g8id, d_g8id); up(g8out, d_g8out); up(g8cnt, d_g8cnt); up(g4id, d_g4id); up(g4out, d_g4out); up(g4cnt, d_g4cnt);
            up(g8off, d_g8off); up(g4off, d_g4off); up(qf, d_qf); up(C0, d_C0);
            CK(cudaMalloc(&d_lut, (size_t)B * M * 32 * 4));
            CK(cudaMalloc(&d_C0x, (size_t)B * 4));
            for (int k = 0; k < 5; ++k) { CK(cudaMalloc(&d_out[k], nc * 4)); CK(cudaMemset(d_out[k], 0xff, nc * 4)); }
            GroupArgs a8{d_g8id, d_g8out, d_g8cnt, d_g8off, d_coff, d_qf, d_C0, (float)(-4 * c)};
            GroupArgs a4{d_g4id, d_g4out, d_g4cnt, d_g4off, d_coff, d_qf, d_C0, (float)(-4 * c)};
            long maxn = 0, maxg8 = 0, maxg4 = 0;
            for (int i = 0; i < B; ++i) { maxn = std::max(maxn, cand_off[i + 1] - cand_off[i]); maxg8 = std::max(maxg8, g8off[i + 1] - g8off[i]); maxg4 = std::max(maxg4, g4off[i + 1] - g4off[i]); }
            const int target = std::max(1, 4 * dev_sms / B);     // blocks a query, before the work bound
            const int h1w = 8, h2w8 = 2, h2w4 = 8;
            const int gx_b0 = (int)std::min<long>(target, (maxn + block_b0 - 1) / block_b0);
            const int gx_h18 = (int)std::min<long>(target, (maxg8 + h1w - 1) / h1w);
            const int gx_h14 = (int)std::min<long>(target, (maxg4 + h1w - 1) / h1w);
            const int gx_h28 = (int)std::max<long>(1, std::min<long>(4 * target, (maxg8 + 32 * h2w8 - 1) / (32 * h2w8)));
            const int gx_h24 = (int)std::max<long>(1, std::min<long>(target, (maxg4 + 32 * h2w4 - 1) / (32 * h2w4)));
            const int sm_b0 = M * 32 * 4, sm_h18 = d * 4 + h1w * 256 * 4, sm_h14 = d * 4 + h1w * 16 * 4;
            const int sm_h28 = d * 4 + h2w8 * 256 * 32 * 4, sm_h24 = d * 4 + h2w4 * 16 * 32 * 4;
            CK(cudaFuncSetAttribute(k_b0, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_b0));
            CK(cudaFuncSetAttribute(k_h1_r8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h18));
            CK(cudaFuncSetAttribute(k_h1_r4<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h14));
            CK(cudaFuncSetAttribute(k_h2_r8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h28));
            CK(cudaFuncSetAttribute(k_h2_r4<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h24));
            CK(cudaFuncSetAttribute(k_h1_r8<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h18));
            CK(cudaFuncSetAttribute(k_h1_r4<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h14));
            CK(cudaFuncSetAttribute(k_h2_r8<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h28));
            CK(cudaFuncSetAttribute(k_h2_r4<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_h24));
            auto launch = [&](int k, bool prof) {
                switch (k) {
                case 0: k_b0<<<dim3(gx_b0, B), block_b0, sm_b0>>>(d_words, N, M, d_lut, d_cpos, d_coff, d_out[0]); break;
                case 5: k_b0<<<dim3(B <= 32 ? 8 : 1, B), block_b0, sm_b0>>>(d_words, N, M, d_lut, d_cpos, d_coff, d_out[0]); break;
                case 6: k_build_lut<<<dim3((M * 32 + 255) / 256, B), 256>>>(d_qf, d, M, (float)c, d_lut); break;
                case 7: k_prep_mm<<<B, 256>>>(d_qf, d, (float)c, d_C0x); break;
                case 1: if (prof) k_h1_r8<true><<<dim3(gx_h18, B), 32 * h1w, sm_h18>>>(d_P8g, d, a8, d_out[1]);
                        else k_h1_r8<false><<<dim3(gx_h18, B), 32 * h1w, sm_h18>>>(d_P8g, d, a8, d_out[1]); break;
                case 2: if (prof) k_h1_r4<true><<<dim3(gx_h14, B), 32 * h1w, sm_h14>>>(d_P4g, d, a4, d_out[2]);
                        else k_h1_r4<false><<<dim3(gx_h14, B), 32 * h1w, sm_h14>>>(d_P4g, d, a4, d_out[2]); break;
                case 3: if (prof) k_h2_r8<true><<<dim3(gx_h28, B), 32 * h2w8, sm_h28>>>(d_P8d, G8, d, a8, d_out[3]);
                        else k_h2_r8<false><<<dim3(gx_h28, B), 32 * h2w8, sm_h28>>>(d_P8d, G8, d, a8, d_out[3]); break;
                case 4: if (prof) k_h2_r4<true><<<dim3(gx_h24, B), 32 * h2w4, sm_h24>>>(d_P4d, G4, d, a4, d_out[4]);
                        else k_h2_r4<false><<<dim3(gx_h24, B), 32 * h2w4, sm_h24>>>(d_P4d, G4, d, a4, d_out[4]); break;
                }
            };
            // correctness: one run each, against FP64 and B0 (the LUT first; the C0 the scorers use is the
            // host's FP64 value, and the device preparation is checked against it)
            launch(6, false);
            for (int k = 0; k < NS; ++k) launch(k, false);
            launch(7, false);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<std::vector<float>> h(5, std::vector<float>(nc));
            for (int k = 0; k < 5; ++k) CK(cudaMemcpy(h[k].data(), d_out[k], nc * 4, cudaMemcpyDeviceToHost));
            {
                std::vector<float> cx(B);
                CK(cudaMemcpy(cx.data(), d_C0x, B * 4, cudaMemcpyDeviceToHost));
                for (int i = 0; i < B; ++i) c0_dev_rel = std::max(c0_dev_rel, (double)std::fabs(cx[i] - C0[i]) / std::fabs(C0[i]));
            }
            for (int i = 0; i < B; ++i) {
                const QueryData& Q = qd[q0 + i];
                const long o = cand_off[i], n = cand_off[i + 1] - o;
                const std::vector<double>& ref = dref[q0 + i];
                const int T = Q.ck;
                auto topT = [&](auto val) {
                    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
                    std::nth_element(idx.begin(), idx.begin() + (T - 1), idx.end(), [&](int x, int y) {
                        const double vx = val(x), vy = val(y);
                        return vx < vy || (vx == vy && cand[q0 + i][x] < cand[q0 + i][y]); });
                    std::vector<int> s(idx.begin(), idx.begin() + T); std::sort(s.begin(), s.end()); return s; };
                const std::vector<int> t64 = topT([&](int x) { return ref[x]; });
                const std::vector<int> tb0 = topT([&](int x) { return (double)h[0][o + x]; });
                double kth64 = -1e300;
                for (int x : t64) kth64 = std::max(kth64, ref[x]);
                for (int k = 0; k < 5; ++k) {
                    for (long t = 0; t < n; ++t) {
                        const double e = std::fabs((double)h[k][o + t] - ref[t]);
                        err_max[k] = std::max(err_max[k], e); err_sum[k] += e;
                        if ((t & 63) == 0) errs[k].push_back(e);
                        vs_b0_max[k] = std::max(vs_b0_max[k], std::fabs((double)h[k][o + t] - (double)h[0][o + t]));
                    }
                    const std::vector<int> tk = topT([&](int x) { return (double)h[k][o + x]; });
                    if (tk != t64) {
                        ++topT_diff_fp64[k];
                        for (int x : tk) if (!std::binary_search(t64.begin(), t64.end(), x)) topT_gap_max[k] = std::max(topT_gap_max[k], ref[x] - kth64);
                    }
                    if (tk != tb0) ++topT_diff_b0[k];
                }
                err_n += n;
            }
            // phase profile
            if (profile) {
                for (int k = 1; k < 5; ++k) {
                    unsigned long long z[4] = {0, 0, 0, 0};
                    CK(cudaMemcpyToSymbol(g_phase, z, sizeof(z)));
                    launch(k, true); CK(cudaDeviceSynchronize());
                    unsigned long long r[4]; CK(cudaMemcpyFromSymbol(r, g_phase, sizeof(r)));
                    for (int t = 0; t < 4; ++t) phase[k][t] += r[t];
                }
            }
            // timing: rounds rotate the order; reps launches each, one event pair a launch
            Timer tm;
            for (int k = 0; k < NI; ++k) launch(k, false);
            CK(cudaDeviceSynchronize());
            for (int r = 0; r < rounds; ++r)
                for (int t = 0; t < NI; ++t) {
                    const int k = (t + r) % NI;
                    for (int rep = 0; rep < reps; ++rep) { tm.start(); launch(k, false); st[k].push_back(tm.stop()); }
                }
            CK(cudaGetLastError());
            CK(cudaFree(d_cpos)); CK(cudaFree(d_coff)); CK(cudaFree(d_g8id)); CK(cudaFree(d_g8out)); CK(cudaFree(d_g8cnt));
            CK(cudaFree(d_g4id)); CK(cudaFree(d_g4out)); CK(cudaFree(d_g4cnt)); CK(cudaFree(d_g8off)); CK(cudaFree(d_g4off));
            CK(cudaFree(d_qf)); CK(cudaFree(d_C0)); CK(cudaFree(d_C0x)); CK(cudaFree(d_lut));
            for (int k = 0; k < 5; ++k) CK(cudaFree(d_out[k]));
            if (bi == 0) {
                cudaFuncAttributes fa[5];
                CK(cudaFuncGetAttributes(&fa[0], k_b0)); CK(cudaFuncGetAttributes(&fa[1], k_h1_r8<false>));
                CK(cudaFuncGetAttributes(&fa[2], k_h1_r4<false>)); CK(cudaFuncGetAttributes(&fa[3], k_h2_r8<false>));
                CK(cudaFuncGetAttributes(&fa[4], k_h2_r4<false>));
                const int bs[5] = {block_b0, 32 * h1w, 32 * h1w, 32 * h2w8, 32 * h2w4};
                const int sms[5] = {sm_b0, sm_h18, sm_h14, sm_h28, sm_h24};
                const int gxs[5] = {gx_b0, gx_h18, gx_h14, gx_h28, gx_h24};
                int occ[5];
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[0], k_b0, bs[0], sms[0]));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[1], k_h1_r8<false>, bs[1], sms[1]));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[2], k_h1_r4<false>, bs[2], sms[2]));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[3], k_h2_r8<false>, bs[3], sms[3]));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[4], k_h2_r4<false>, bs[4], sms[4]));
                for (int k = 0; k < 5; ++k)
                    std::printf("  %-6s block %4d smem %6d regs %3d grid %5d x %4d  resident blocks/SM %d (warps %d)\n",
                                names[k], bs[k], sms[k], fa[k].numRegs, gxs[k], B, occ[k], occ[k] * bs[k] / 32);
                std::fprintf(jf, "%s{\"batch\":%d,\"config\":[", first ? "" : ",", B);
                for (int k = 0; k < 5; ++k)
                    std::fprintf(jf, "%s{\"impl\":\"%s\",\"block\":%d,\"smem\":%d,\"regs\":%d,\"grid_x\":%d,\"blocks_per_sm\":%d}",
                                 k ? "," : "", names[k], bs[k], sms[k], fa[k].numRegs, gxs[k], occ[k]);
                std::fprintf(jf, "],");
                first = false;
            }
        }
        std::printf("batch %d: %d batches, %ld candidates scored per implementation per pass; device C0 vs FP64 max rel %.2e\n",
                    B, nb, ncand_tot, c0_dev_rel);
        std::fprintf(jf, "\"batches_timed\":%d,\"candidates\":%ld,\"c0_device_rel_err\":%.3e,\"impls\":[", nb, ncand_tot, c0_dev_rel);
        double med0 = 0;
        { std::vector<double> s = st[0]; std::sort(s.begin(), s.end()); med0 = s[s.size() / 2]; }
        for (int k = 0; k < NI; ++k) {
            std::vector<double> s = st[k];
            std::sort(s.begin(), s.end());
            const double med = s[s.size() / 2], p10 = s[s.size() / 10], p90 = s[s.size() * 9 / 10];
            if (k >= NS) {
                std::printf("  %-11s median %8.1f us/launch [p10 %8.1f, p90 %8.1f]%s\n", names[k], med * 1e3, p10 * 1e3,
                            p90 * 1e3, k == 5 ? "  (B0 at the scan's own grid: G blocks a query)" : "");
                std::fprintf(jf, ",{\"impl\":\"%s\",\"launches\":%zu,\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f}",
                             names[k], s.size(), med * 1e3, p10 * 1e3, p90 * 1e3);
                continue;
            }
            std::vector<double> e = errs[k]; std::sort(e.begin(), e.end());
            const double e50 = e.empty() ? 0 : e[e.size() / 2], e99 = e.empty() ? 0 : e[std::min(e.size() - 1, e.size() * 99 / 100)];
            std::printf("  %-6s median %8.1f us/launch [p10 %8.1f, p90 %8.1f]  B0/this %.3f | err vs FP64 max %.3e p50 %.3e p99 %.3e | vs B0 max %.3e | top-T != FP64 in %d queries (max FP64 gap past the T-th %.3e), != B0 in %d\n",
                        names[k], med * 1e3, p10 * 1e3, p90 * 1e3, med0 / med, err_max[k], e50, e99, vs_b0_max[k], topT_diff_fp64[k], topT_gap_max[k], topT_diff_b0[k]);
            std::fprintf(jf, "%s{\"impl\":\"%s\",\"launches\":%zu,\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,"
                             "\"err_fp64_max\":%.4e,\"err_fp64_p50\":%.4e,\"err_fp64_p99\":%.4e,\"err_fp64_mean\":%.4e,\"vs_b0_max\":%.4e,"
                             "\"topT_diff_fp64_queries\":%d,\"topT_fp64_gap_max\":%.4e,\"topT_diff_b0_queries\":%d,"
                             "\"phase_cycles\":[%llu,%llu,%llu,%llu]}",
                         k ? "," : "", names[k], s.size(), med * 1e3, p10 * 1e3, p90 * 1e3, err_max[k], e50, e99,
                         err_n ? err_sum[k] / err_n : 0.0, vs_b0_max[k], topT_diff_fp64[k], topT_gap_max[k], topT_diff_b0[k],
                         phase[k][0], phase[k][1], phase[k][2], phase[k][3]);
        }
        std::fprintf(jf, "]}");
        std::fflush(jf);
    }
    std::fprintf(jf, "]}\n");
    std::fclose(jf);
    std::printf("MB_DONE\n");
    return 0;
}
