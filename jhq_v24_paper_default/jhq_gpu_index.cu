#include "jhq_v24_paper_default/jhq_gpu_index.cuh"

// Ablation switch for the residual codebook layout. The official
// get_scalar_codebook_ptr(subspace_idx, level) keeps one scalar codebook per
// subspace; v15 and earlier kept a single global one. Setting this to 1 trains
// the global codebook and replicates it into all M slots, so the device
// buffer, the kernels and the indexing stay byte-identical and the only thing
// that differs is the codebook contents -- which is what makes the comparison
// clean.  -DJHQ_GLOBAL_RESIDUAL_CB=1
#ifndef JHQ_GLOBAL_RESIDUAL_CB
#define JHQ_GLOBAL_RESIDUAL_CB 0
#endif
#include "jhq_v24_paper_default/encode.cuh"
#include "jhq_v24_paper_default/train_pq_gpu.cuh"
#include "jhq_v24_paper_default/train_res_gpu.cuh"
#include "jhq_v24_paper_default/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <numeric>
#include <chrono>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace jhq_gpu {

namespace {

// ── Assign kernel ─────────────────────────────────────────────────────────────
// One warp per row. The dots for a row are contiguous (the GEMM writes the
// nlist x nb product column-major with ld = nlist), so a thread per row had
// the 32 lanes of a warp reading addresses 4*nlist bytes apart -- 64 KB at
// nlist=16384 -- and every load instruction touched 32 lines. With the lanes
// spread along the row each load is 512 contiguous bytes, and the row's
// minimum is a shuffle reduction at the end. Ties resolve to the lowest
// index, as the sequential scan did.
__global__ void assign_from_dots_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    int*                  assigns,
    int nlist, int nb)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nb) return;
    const float* col = dots + (long long)row * nlist;
    float best    = INFINITY;
    int   best_id = 0x7fffffff;
    int c = lane * 4;
    if ((nlist & 3) == 0) {
        for (; c < nlist; c += 128) {
            const float4 v = *reinterpret_cast<const float4*>(col + c);
            const float4 q = *reinterpret_cast<const float4*>(cent_norms + c);
            const float d0 = q.x - 2.f * v.x, d1 = q.y - 2.f * v.y,
                        d2 = q.z - 2.f * v.z, d3 = q.w - 2.f * v.w;
            if (d0 < best) { best = d0; best_id = c;     }
            if (d1 < best) { best = d1; best_id = c + 1; }
            if (d2 < best) { best = d2; best_id = c + 2; }
            if (d3 < best) { best = d3; best_id = c + 3; }
        }
    } else {
        for (c = lane; c < nlist; c += 32) {
            const float dd = cent_norms[c] - 2.f * col[c];
            if (dd < best) { best = dd; best_id = c; }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, off);
        const int   oi = __shfl_xor_sync(0xffffffffu, best_id, off);
        if (ob < best || (ob == best && oi < best_id)) { best = ob; best_id = oi; }
    }
    if (lane == 0) assigns[row] = best_id;
}

// The same argmin over dots the GEMM wrote as fp16. Reading the fp32 product
// back was 0.7 s of stella-trec24's 3.4 s assignment, at the memory roofline,
// and this halves it. The rounding is bounded -- half an fp16 ulp, so the
// distance moves by at most |v|/1024 -- and the bound is used, not assumed:
// a centroid whose distance cannot reach the best's even with both errors
// against it is out exactly, and only when some other centroid stays within
// the bound are the sums in question recomputed in fp32 from the same fp16
// inputs the GEMM read. The assignment is then the one the fp32 product
// gave, up to summation order.
__global__ void assign_from_dots16_kernel(
    const __half* __restrict__ dots,        // [nb][nlist]
    const float*  __restrict__ cent_norms,
    const __half* __restrict__ cent16,      // [nlist][d]
    const __half* __restrict__ y16,         // rows with ld = ldy, or dimension-major
    int y_transposed, long long ldy,
    int*                  assigns,
    int nlist, int nb, int d, float rel, int* amb_count)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nb) return;
    const __half* col = dots + (long long)row * nlist;
    // Half an fp16 ulp is 2^-11 of the value; the distance carries twice the
    // dot, so 2^-10, with a little over it for the fp32 arithmetic on top and
    // an absolute floor for values that are subnormal in fp16.
    const float REL = rel, ABS = 1e-6f;

    // best_e is the best's error bound; second_lo the smallest lower bound of
    // any other centroid seen. Ties fall through to the exact path.
    float best = INFINITY, best_e = 0.f, second_lo = INFINITY;
    int   best_id = 0x7fffffff;
    auto consider = [&](float v, float q, int c) {
        const float dd = q - 2.f * v;
        const float e  = fabsf(v) * REL + ABS;
        if (dd < best) {
            second_lo = fminf(second_lo, best - best_e);   // the old best, at its lower bound
            best = dd; best_e = e; best_id = c;
        } else {
            second_lo = fminf(second_lo, dd - e);
        }
    };
    if ((nlist & 7) == 0) {
        for (int c = lane * 8; c < nlist; c += 256) {
            const uint4 raw = *reinterpret_cast<const uint4*>(col + c);
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            const float4 q0 = *reinterpret_cast<const float4*>(cent_norms + c);
            const float4 q1 = *reinterpret_cast<const float4*>(cent_norms + c + 4);
            const float2 a = __half22float2(h[0]), b = __half22float2(h[1]),
                         cc = __half22float2(h[2]), dd = __half22float2(h[3]);
            consider(a.x,  q0.x, c);     consider(a.y,  q0.y, c + 1);
            consider(b.x,  q0.z, c + 2); consider(b.y,  q0.w, c + 3);
            consider(cc.x, q1.x, c + 4); consider(cc.y, q1.y, c + 5);
            consider(dd.x, q1.z, c + 6); consider(dd.y, q1.w, c + 7);
        }
    } else {
        for (int c = lane; c < nlist; c += 32)
            consider(__half2float(col[c]), cent_norms[c], c);
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob  = __shfl_xor_sync(0xffffffffu, best, off);
        const float oe  = __shfl_xor_sync(0xffffffffu, best_e, off);
        const float osl = __shfl_xor_sync(0xffffffffu, second_lo, off);
        const int   oi  = __shfl_xor_sync(0xffffffffu, best_id, off);
        // Whichever best loses becomes a runner-up, at its lower bound.
        if (ob < best || (ob == best && oi < best_id)) {
            second_lo = fminf(fminf(second_lo, osl), best - best_e);
            best = ob; best_e = oe; best_id = oi;
        } else {
            second_lo = fminf(fminf(second_lo, osl), ob - oe);
        }
    }
    const float best_hi = best + best_e;
    if (second_lo > best_hi) { if (lane == 0) assigns[row] = best_id; return; }

    // Some other centroid could be the true nearest: recompute in fp32 for
    // every centroid within the bound and settle it exactly. Eight centroids a
    // lane again, their flags in a byte, and the warp visits each flagged
    // centroid together.
    const float T = best_hi;
    if (amb_count && lane == 0) atomicAdd(amb_count, 1);
    float ex_best = INFINITY; int ex_id = 0x7fffffff;
    const __half* yrow = y_transposed ? y16 + row : y16 + (long long)row * ldy;
    const long long ystride = y_transposed ? ldy : 1;
    auto exact_dist = [&](int cid) {
        const __half* cr = cent16 + (long long)cid * d;
        float dot = 0.f;
        if ((d & 7) == 0) {
            for (int j = lane * 8; j < d; j += 256) {
                const uint4 raw = *reinterpret_cast<const uint4*>(cr + j);
                const __half2* h = reinterpret_cast<const __half2*>(&raw);
                float yv[8];
                if (y_transposed) {
#pragma unroll
                    for (int k = 0; k < 8; ++k) yv[k] = __half2float(yrow[(j + k) * ystride]);
                } else {
                    const uint4 yraw = *reinterpret_cast<const uint4*>(yrow + j);
                    const __half2* yh = reinterpret_cast<const __half2*>(&yraw);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float2 f = __half22float2(yh[k]);
                        yv[2 * k] = f.x; yv[2 * k + 1] = f.y;
                    }
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 f = __half22float2(h[k]);
                    dot = fmaf(f.x, yv[2 * k], dot);
                    dot = fmaf(f.y, yv[2 * k + 1], dot);
                }
            }
        } else {
            for (int j = lane; j < d; j += 32)
                dot = fmaf(__half2float(cr[j]), __half2float(yrow[j * ystride]), dot);
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        return cent_norms[cid] - 2.f * dot;
    };
    for (int base = 0; base < nlist; base += 256) {
        unsigned mask = 0;
        const int c0 = base + lane * 8;
        if ((nlist & 7) == 0) {
            if (c0 < nlist) {
                const uint4 raw = *reinterpret_cast<const uint4*>(col + c0);
                const __half2* h = reinterpret_cast<const __half2*>(&raw);
                const float4 q0 = *reinterpret_cast<const float4*>(cent_norms + c0);
                const float4 q1 = *reinterpret_cast<const float4*>(cent_norms + c0 + 4);
                const float qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 f = __half22float2(h[k]);
                    const float v0 = f.x, v1 = f.y;
                    if (qs[2 * k]     - 2.f * v0 - (fabsf(v0) * REL + ABS) <= T) mask |= 1u << (2 * k);
                    if (qs[2 * k + 1] - 2.f * v1 - (fabsf(v1) * REL + ABS) <= T) mask |= 1u << (2 * k + 1);
                }
            }
        } else {
            for (int k = 0; k < 8; ++k) {
                const int c = c0 + k;
                if (c < nlist) {
                    const float v = __half2float(col[c]);
                    if (cent_norms[c] - 2.f * v - (fabsf(v) * REL + ABS) <= T) mask |= 1u << k;
                }
            }
        }
        unsigned lm = __ballot_sync(0xffffffffu, mask != 0);
        while (lm) {
            const int src = __ffs(lm) - 1; lm &= lm - 1;
            unsigned m = __shfl_sync(0xffffffffu, mask, src);
            while (m) {
                const int k = __ffs(m) - 1; m &= m - 1;
                const int cid = base + src * 8 + k;
                const float dex = exact_dist(cid);
                if (dex < ex_best || (dex == ex_best && cid < ex_id)) { ex_best = dex; ex_id = cid; }
            }
        }
    }
    if (lane == 0) assigns[row] = ex_id;
}

// ── int8 assignment ───────────────────────────────────────────────────────────
// The tensor cores run the int8 product at 2.4x the fp16 rate on this card
// (552 against 235 TOPS on the 32768 x 16384 x 1024 batch) and its int32 sums
// are exact. What is not exact is the rounding of the inputs to eight bits,
// and that is bounded rather than assumed. With y = s_y*y_q + e for a row and
// c = s*c_q + f for a centroid, Cauchy-Schwarz gives
//     |y.c - s_y*s*(y_q.c_q)| <= ||y||*||f|| + ||e||*||s*c_q||,
// and ||s*c_q|| <= ||c|| + ||f||. The row's ||y|| and ||e|| come out of its
// quantisation; the centroids share one scale s and the largest ||f|| among
// them, so the argmin still reads one float per centroid, ||c||^2, as before.
// A centroid whose distance cannot reach the best's within the two bounds is
// out; otherwise every centroid within them is recomputed in fp32 from the
// fp32 inputs and the row settled exactly. The assignment is then the fp32
// one up to summation order -- nearer to it than the fp16 product's, whose
// inputs were rounded to eleven bits with no settlement against that.
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
// Non-negative floats order as their bit patterns, so their max is an
// integer atomicMax and needs no host round trip.
__global__ void absmax_kernel(const float* __restrict__ x, long long n4, unsigned* out) {
    __shared__ float red[8];
    float m = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (long long)gridDim.x * blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fmaxf(fabsf(v.z), fabsf(v.w)), m));
    }
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        m = warp_max(m);
        if (threadIdx.x == 0) atomicMax(out, __float_as_uint(m));
    }
}
__device__ __forceinline__ float scale_of(unsigned bits) {
    const float m = __uint_as_float(bits);
    return m > 0.f ? m * (1.f / 127.f) : 1.f;
}
// Four values to their codes; the error is taken from the code actually
// written, one rounding, so it is what the bound needs.
__device__ __forceinline__ char4 quant4(float4 v, float s, float inv, float& sq, float& err2) {
    const float q0 = fminf(fmaxf(rintf(v.x * inv), -127.f), 127.f);
    const float q1 = fminf(fmaxf(rintf(v.y * inv), -127.f), 127.f);
    const float q2 = fminf(fmaxf(rintf(v.z * inv), -127.f), 127.f);
    const float q3 = fminf(fmaxf(rintf(v.w * inv), -127.f), 127.f);
    const float e0 = fmaf(-s, q0, v.x), e1 = fmaf(-s, q1, v.y);
    const float e2 = fmaf(-s, q2, v.z), e3 = fmaf(-s, q3, v.w);
    sq   = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, sq))));
    err2 = fmaf(e0, e0, fmaf(e1, e1, fmaf(e2, e2, fmaf(e3, e3, err2))));
    return make_char4((signed char)q0, (signed char)q1, (signed char)q2, (signed char)q3);
}
// Warp per centroid, all on the shared scale; ||f|| goes into the max.
__global__ void quantize_centroids_kernel(const float* __restrict__ c, int nlist, int d,
                                          const unsigned* __restrict__ absmax_bits,
                                          int8_t* __restrict__ c8, unsigned* nf_max_bits)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nlist) return;
    const float s = scale_of(*absmax_bits), inv = 1.f / s;
    const float* cr = c + (long long)row * d;
    int8_t* qr = c8 + (long long)row * d;
    float sq = 0.f, f2 = 0.f;
    for (int j = lane * 4; j < d; j += 128)
        *reinterpret_cast<char4*>(qr + j) =
            quant4(*reinterpret_cast<const float4*>(cr + j), s, inv, sq, f2);
    f2 = warp_sum(f2);
    if (lane == 0) atomicMax(nf_max_bits, __float_as_uint(sqrtf(f2)));
}
// Warp per row of a row-major y: its own scale, its int8 image, and
// {s_y, ||y||, ||e||, 0}.
__global__ void quantize_rows_kernel(const float* __restrict__ y, int n, int d,
                                     int8_t* __restrict__ y8, float4* __restrict__ stats)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    const float* yr = y + (long long)row * d;
    float m = 0.f;
    for (int j = lane * 4; j < d; j += 128) {
        const float4 v = *reinterpret_cast<const float4*>(yr + j);
        m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fmaxf(fabsf(v.z), fabsf(v.w)), m));
    }
    m = warp_max(m);
    const float s = m > 0.f ? m * (1.f / 127.f) : 1.f, inv = 1.f / s;
    int8_t* qr = y8 + (long long)row * d;
    float y2 = 0.f, e2 = 0.f;
    for (int j = lane * 4; j < d; j += 128)
        *reinterpret_cast<char4*>(qr + j) =
            quant4(*reinterpret_cast<const float4*>(yr + j), s, inv, y2, e2);
    y2 = warp_sum(y2); e2 = warp_sum(e2);
    if (lane == 0) stats[row] = make_float4(s, sqrtf(y2), sqrtf(e2), 0.f);
}
// The same for a dimension-major y (ld = n), where a row is a strided column:
// the lanes take consecutive rows so every load is one line, and a 32-row
// tile in shared memory turns the int8 image back into contiguous rows. The
// product then reads y8 the fast way in either layout.
__global__ void __launch_bounds__(256)
quantize_cols_kernel(const float* __restrict__ yt, int n, int d,
                     int8_t* __restrict__ y8, float4* __restrict__ stats)
{
    __shared__ float part[8][32], part2[8][32], sc[2][32];
    __shared__ __align__(16) int8_t tile[32][264];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32, row = row0 + lane;
    const bool ok = row < n;
    float m = 0.f;
    if (ok) for (int j = w; j < d; j += 8) m = fmaxf(m, fabsf(yt[(long long)j * n + row]));
    part[w][lane] = m;
    __syncthreads();
    if (w == 0) {
        float mm = 0.f;
        for (int k = 0; k < 8; ++k) mm = fmaxf(mm, part[k][lane]);
        const float s = mm > 0.f ? mm * (1.f / 127.f) : 1.f;
        sc[0][lane] = s; sc[1][lane] = 1.f / s;
    }
    __syncthreads();
    const float s = sc[0][lane], inv = sc[1][lane];
    float y2 = 0.f, e2 = 0.f;
    for (int j0 = 0; j0 < d; j0 += 256) {
        const int jn = min(256, d - j0);
        for (int jj = w; jj < jn; jj += 8) {
            const float v = ok ? yt[(long long)(j0 + jj) * n + row] : 0.f;
            const float q = fminf(fmaxf(rintf(v * inv), -127.f), 127.f);
            const float e = fmaf(-s, q, v);
            y2 = fmaf(v, v, y2); e2 = fmaf(e, e, e2);
            tile[lane][jj] = (int8_t)q;
        }
        __syncthreads();
        for (int r = w; r < 32; r += 8) {
            if (row0 + r < n) {
                int8_t* dst = y8 + (long long)(row0 + r) * d + j0;
                for (int b = lane * 8; b < jn; b += 256)
                    *reinterpret_cast<uint2*>(dst + b) =
                        *reinterpret_cast<const uint2*>(&tile[r][b]);
            }
        }
        __syncthreads();
    }
    part[w][lane] = y2; part2[w][lane] = e2;
    __syncthreads();
    if (w == 0 && ok) {
        float Y = 0.f, E = 0.f;
        for (int k = 0; k < 8; ++k) { Y += part[k][lane]; E += part2[k][lane]; }
        stats[row] = make_float4(s, sqrtf(Y), sqrtf(E), 0.f);
    }
}
// The argmin over the int32 product, with the bound above, and the exact
// settlement in fp32 from the fp32 inputs. Four rows a block, and the row
// that settles is copied once into shared memory: dimension-major y has it
// strided by ldy, and reading that per candidate would touch a line per
// element.
__global__ void __launch_bounds__(128)
assign_from_dots8_kernel(
    const int*    __restrict__ dots,        // [nb][nlist]
    const float*  __restrict__ cent_norms,  // ||c||^2 of the fp32 centroids
    const float*  __restrict__ cent,        // fp32 centroids [nlist][d]
    const float*  __restrict__ y,           // fp32 rows with ld = ldy, or dimension-major
    int y_transposed, long long ldy,
    const float4* __restrict__ row_stats,   // {s_y, ||y||, ||e||, -}
    const unsigned* __restrict__ cstats,    // [0] max |c| bits, [1] max ||f|| bits
    int* assigns, int nlist, int nb, int d, int* amb_count)
{
    extern __shared__ float yrow_s[];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nb) return;
    const int* col = dots + (long long)row * nlist;
    const float4 rs = row_stats[row];
    const float s_c = scale_of(cstats[0]);
    const float nf  = __uint_as_float(cstats[1]) * (1.f + 1e-5f);   // its sqrt was approximate
    const float K   = rs.x * s_c;                                    // dequantises the dot
    // Half-width of the distance bound: twice the dot's, so
    // 2*(||y||*||f|| + ||e||*(||c|| + ||f||)), then the fp32 arithmetic on top.
    const float A = 2.f * nf * (rs.y + rs.z);
    const float B = 2.f * rs.z * (1.f + 1e-5f);
    const float REL = 4e-6f, ABS = 1e-7f;
    auto bound = [&](float v, float q) {
        return A + B * sqrtf(q) + (q + 2.f * fabsf(v)) * REL + ABS;
    };
    float best = INFINITY, best_e = 0.f, second_lo = INFINITY;
    int   best_id = 0x7fffffff;
    auto consider = [&](int iv, float q, int c) {
        const float v  = K * (float)iv;
        const float dd = q - 2.f * v;
        const float e  = bound(v, q);
        if (dd < best) {
            second_lo = fminf(second_lo, best - best_e);
            best = dd; best_e = e; best_id = c;
        } else {
            second_lo = fminf(second_lo, dd - e);
        }
    };
    if ((nlist & 7) == 0) {
        for (int c = lane * 8; c < nlist; c += 256) {
            const int4 r0 = *reinterpret_cast<const int4*>(col + c);
            const int4 r1 = *reinterpret_cast<const int4*>(col + c + 4);
            const float4 q0 = *reinterpret_cast<const float4*>(cent_norms + c);
            const float4 q1 = *reinterpret_cast<const float4*>(cent_norms + c + 4);
            consider(r0.x, q0.x, c);     consider(r0.y, q0.y, c + 1);
            consider(r0.z, q0.z, c + 2); consider(r0.w, q0.w, c + 3);
            consider(r1.x, q1.x, c + 4); consider(r1.y, q1.y, c + 5);
            consider(r1.z, q1.z, c + 6); consider(r1.w, q1.w, c + 7);
        }
    } else {
        for (int c = lane; c < nlist; c += 32) consider(col[c], cent_norms[c], c);
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob  = __shfl_xor_sync(0xffffffffu, best, off);
        const float oe  = __shfl_xor_sync(0xffffffffu, best_e, off);
        const float osl = __shfl_xor_sync(0xffffffffu, second_lo, off);
        const int   oi  = __shfl_xor_sync(0xffffffffu, best_id, off);
        if (ob < best || (ob == best && oi < best_id)) {
            second_lo = fminf(fminf(second_lo, osl), best - best_e);
            best = ob; best_e = oe; best_id = oi;
        } else {
            second_lo = fminf(fminf(second_lo, osl), ob - oe);
        }
    }
    const float best_hi = best + best_e;
    if (second_lo > best_hi) { if (lane == 0) assigns[row] = best_id; return; }

    const float T = best_hi;
    if (amb_count && lane == 0) atomicAdd(amb_count, 1);
    float* ys = yrow_s + w * d;
    if (y_transposed) {
        const float* yr = y + row;
        for (int j = lane; j < d; j += 32) ys[j] = yr[(long long)j * ldy];
    } else {
        const float* yr = y + (long long)row * ldy;
        if ((d & 3) == 0)
            for (int j = lane * 4; j < d; j += 128)
                *reinterpret_cast<float4*>(ys + j) = *reinterpret_cast<const float4*>(yr + j);
        else
            for (int j = lane; j < d; j += 32) ys[j] = yr[j];
    }
    __syncwarp();
    float ex_best = INFINITY; int ex_id = 0x7fffffff;
    auto exact_dist = [&](int cid) {
        const float* cr = cent + (long long)cid * d;
        float dot = 0.f;
        if ((d & 7) == 0) {
            for (int j = lane * 8; j < d; j += 256) {
                const float4 c0 = *reinterpret_cast<const float4*>(cr + j);
                const float4 c1 = *reinterpret_cast<const float4*>(cr + j + 4);
                const float4 y0 = *reinterpret_cast<const float4*>(ys + j);
                const float4 y1 = *reinterpret_cast<const float4*>(ys + j + 4);
                dot = fmaf(c0.x, y0.x, dot); dot = fmaf(c0.y, y0.y, dot);
                dot = fmaf(c0.z, y0.z, dot); dot = fmaf(c0.w, y0.w, dot);
                dot = fmaf(c1.x, y1.x, dot); dot = fmaf(c1.y, y1.y, dot);
                dot = fmaf(c1.z, y1.z, dot); dot = fmaf(c1.w, y1.w, dot);
            }
        } else {
            for (int j = lane; j < d; j += 32) dot = fmaf(cr[j], ys[j], dot);
        }
        dot = warp_sum(dot);
        return cent_norms[cid] - 2.f * dot;
    };
    for (int base = 0; base < nlist; base += 256) {
        unsigned mask = 0;
        const int c0 = base + lane * 8;
        if ((nlist & 7) == 0) {
            if (c0 < nlist) {
                const int4 r0 = *reinterpret_cast<const int4*>(col + c0);
                const int4 r1 = *reinterpret_cast<const int4*>(col + c0 + 4);
                const float4 q0 = *reinterpret_cast<const float4*>(cent_norms + c0);
                const float4 q1 = *reinterpret_cast<const float4*>(cent_norms + c0 + 4);
                const int   iv[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
                const float qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float v = K * (float)iv[k];
                    if (qs[k] - 2.f * v - bound(v, qs[k]) <= T) mask |= 1u << k;
                }
            }
        } else {
            for (int k = 0; k < 8; ++k) {
                const int c = c0 + k;
                if (c < nlist) {
                    const float v = K * (float)col[c], q = cent_norms[c];
                    if (q - 2.f * v - bound(v, q) <= T) mask |= 1u << k;
                }
            }
        }
        unsigned lm = __ballot_sync(0xffffffffu, mask != 0);
        while (lm) {
            const int src = __ffs(lm) - 1; lm &= lm - 1;
            unsigned m = __shfl_sync(0xffffffffu, mask, src);
            while (m) {
                const int k = __ffs(m) - 1; m &= m - 1;
                const int cid = base + src * 8 + k;
                const float dex = exact_dist(cid);
                if (dex < ex_best || (dex == ex_best && cid < ex_id)) { ex_best = dex; ex_id = cid; }
            }
        }
    }
    if (lane == 0) assigns[row] = ex_id;
}

__global__ void count_mismatch_kernel(const int* a, const int* b, int n, int* cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(cnt, 1);
}

namespace {
__global__ void f32_to_f16_kernel(const float* __restrict__ src,
                                  __half* __restrict__ dst, long long n)
{
    long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(src + i);
        __half2* o = reinterpret_cast<__half2*>(dst + i);
        o[0] = __floats2half2_rn(v.x, v.y);
        o[1] = __floats2half2_rn(v.z, v.w);
    } else {
        for (; i < n; ++i) dst[i] = __float2half_rn(src[i]);
    }
}

// Centroid c starts as training row c*n/nlist, the same evenly spaced seeds
// the host path takes, gathered on the device instead of nlist single-row
// copies back to the host.
__global__ void seed_centroids_kernel(const float* __restrict__ y, int n, int d,
                                      int nlist, float* __restrict__ cent)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nlist * d) return;
    const int c = (int)(i / d), j = (int)(i - (long long)c * d);
    const int src = (int)((long long)c * n / nlist);
    cent[i] = y[(long long)src * d + j];
}

// One warp per row, summed in double so the norms do not depend on lane order.
__global__ void row_sqnorms_kernel(const float* __restrict__ rows, int nrows, int d,
                                   float* __restrict__ out)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nrows) return;
    const float* r = rows + (long long)row * d;
    double s = 0.0;
    for (int j = lane; j < d; j += 32) s += (double)r[j] * r[j];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) out[row] = (float)s;
}
} // namespace

// ── Gather kernel ─────────────────────────────────────────────────────────────
// One warp per sorted position. The primary and residual rows are copied as
// 16-byte words when both widths allow it (M and bpv multiples of 16, which
// every configuration measured here satisfies), a lane per word; otherwise
// a lane per byte. The thread-per-vector version copied M + bpv bytes one at a
// time from rows 4 KB apart across the warp -- 228 ms of the stella-trec24
// add tail for 1.9 GB moved.
__global__ void gather_list_storage_kernel(
    const int*     __restrict__ sorted_ids,
    const uint8_t* __restrict__ primary,
    const uint8_t* __restrict__ residual,
    const float*   __restrict__ corrections,
    int*                        list_ids,
    uint8_t*                    list_primary,
    uint8_t*                    list_res,
    float*                      list_corr,
    int n, int M, int bpv)
{
    const int lane = threadIdx.x & 31;
    const int pos  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (pos >= n) return;
    const int id = sorted_ids[pos];
    if (lane == 0) { list_ids[pos] = id; list_corr[pos] = corrections[id]; }
    const uint8_t* pc = primary + (long long)id * M;
    uint8_t* out_pc = list_primary + (long long)pos * M;
    const uint8_t* rc = residual + (long long)id * bpv;
    uint8_t* out_rc = list_res + (long long)pos * bpv;
    if (((M | bpv) & 15) == 0) {
        const uint4* pc4 = reinterpret_cast<const uint4*>(pc);
        uint4* out_pc4 = reinterpret_cast<uint4*>(out_pc);
        for (int w = lane; w < (M >> 4); w += 32) out_pc4[w] = pc4[w];
        const uint4* rc4 = reinterpret_cast<const uint4*>(rc);
        uint4* out_rc4 = reinterpret_cast<uint4*>(out_rc);
        for (int w = lane; w < (bpv >> 4); w += 32) out_rc4[w] = rc4[w];
    } else {
        for (int m = lane; m < M; m += 32) out_pc[m] = pc[m];
        for (int b = lane; b < bpv; b += 32) out_rc[b] = rc[b];
    }
}

// The two-pass build's counterpart: a batch of freshly encoded rows goes
// straight to its list positions, the primary codes into the transposed
// [M, N] layout the scan reads, so no list-ordered copy of the whole code
// array ever exists beside the original.
__global__ void scatter_list_storage_kernel(
    const int*     __restrict__ list_pos,     // [nb] sorted position of each row
    const uint8_t* __restrict__ primary,      // [nb, M]
    const uint8_t* __restrict__ residual,     // [nb, bpv]
    const float*   __restrict__ corrections,  // [nb]
    uint8_t*                    list_primary_t,  // [M, n]
    uint8_t*                    list_res,        // [n, bpv]
    float*                      list_corr,       // [n]
    int nb, long long n, int M, int bpv)
{
    const int lane = threadIdx.x & 31;
    const int r    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (r >= nb) return;
    const long long pos = list_pos[r];
    if (lane == 0) list_corr[pos] = corrections[r];
    const uint8_t* pc = primary + (long long)r * M;
    for (int m = lane; m < M; m += 32) list_primary_t[(long long)m * n + pos] = pc[m];
    const uint8_t* rc = residual + (long long)r * bpv;
    uint8_t* out_rc = list_res + pos * bpv;
    if ((bpv & 15) == 0) {
        const uint4* rc4 = reinterpret_cast<const uint4*>(rc);
        uint4* out_rc4 = reinterpret_cast<uint4*>(out_rc);
        for (int w = lane; w < (bpv >> 4); w += 32) out_rc4[w] = rc4[w];
    } else {
        for (int b = lane; b < bpv; b += 32) out_rc[b] = rc[b];
    }
}

__global__ void invert_permutation_kernel(const int* __restrict__ order, int* inv, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inv[order[i]] = i;
}

// ── Transpose kernel: [N, M] → [M, N] with shared-memory tiling ──────────────
// TILE=32 avoids bank conflicts (padding +1 on shared dim).
// Grid: (ceil(N/TILE), ceil(M/TILE)) -- N (millions of vectors) MUST be the
// grid.x axis, not grid.y: CUDA caps grid.y/grid.z at 65535 regardless of
// compute capability, while grid.x goes up to 2^31-1. The original version
// put N on grid.y (ceil(N/TILE) with TILE=32 exceeds 65535 once N exceeds
// ~2.1M), which failed with "invalid argument" at the kernel launch on
// arxiv-abstracts-768 (2,253,000 vectors) -- never caught before since every
// dataset this was tested against until now (Vogue-768, openai3-1536) had
// fewer than ~2.1M vectors. M (subspace count, order 100s) safely stays
// under 65535 either way, so it's the one that belongs on grid.y.
// Each block transposes a TILE×TILE sub-block.
template <int TILE>
__global__ void transpose_uint8_kernel(
    const uint8_t* __restrict__ src,   // [N, M]
    uint8_t*                    dst,   // [M, N]
    long long N, int M)
{
    __shared__ uint8_t tile[TILE][TILE + 1];  // +1 avoids bank conflicts

    // Read: block reads tile[threadIdx.y][threadIdx.x] from src[row_src, col_src]
    long long col_src = (long long)blockIdx.y * TILE + threadIdx.x;  // m-axis
    long long row_src = (long long)blockIdx.x * TILE + threadIdx.y;  // n-axis
    if (col_src < M && row_src < N)
        tile[threadIdx.y][threadIdx.x] = src[row_src * M + col_src];
    __syncthreads();

    // Write: transposed — dst[row_dst, col_dst] where row is m-axis, col is n-axis
    long long col_dst = (long long)blockIdx.x * TILE + threadIdx.x;  // n-axis
    long long row_dst = (long long)blockIdx.y * TILE + threadIdx.y;  // m-axis
    if (col_dst < N && row_dst < M)
        dst[row_dst * N + col_dst] = tile[threadIdx.x][threadIdx.y];
}

} // namespace

// ── Constructor / Destructor ──────────────────────────────────────────────────
JHQGpuIndex::JHQGpuIndex(int d, Params p)
    : d_(d), M_(p.M), B_(p.B), Br_(p.Br),
      nlist_(p.nlist), nprobe_(p.nprobe), ivf_iters_(p.ivf_iters),
      batch_size_(p.batch_size), add_batch_(p.add_batch),
      kmeans_iters_(p.kmeans_iters), seed_(p.seed),
      alpha_(p.alpha),
      jl_(d, p.seed)
{
    if (d <= 0)          throw std::invalid_argument("d must be positive");
    if (p.M <= 0)        throw std::invalid_argument("M must be positive");
    if (d % p.M != 0)   throw std::invalid_argument("d must be divisible by M");
    // No B % Ds constraint: a subspace codeword is one of K = 2^B free
    // centroids in Ds dims, not a packed tuple of per-dimension indices, so
    // Ds may exceed B. That is what allows a primary code below one bit per
    // dimension -- see cpu/pq_codebook.h.
    if (p.B > 8)         throw std::invalid_argument("B must be <= 8");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (p.nlist <= 0)    throw std::invalid_argument("nlist must be positive");
    if (p.nprobe <= 0)   throw std::invalid_argument("nprobe must be positive");
    if (p.ivf_iters <= 0) throw std::invalid_argument("ivf_iters must be positive");
    if (p.batch_size <= 0) throw std::invalid_argument("batch_size must be positive");
    if (p.add_batch <= 0) throw std::invalid_argument("add_batch must be positive");
    if (p.kmeans_iters < 0) throw std::invalid_argument("kmeans_iters must be >= 0");
    if (p.alpha <= 0.0f) throw std::invalid_argument("alpha must be positive");

    Ds_           = d_ / M_;
    K_            = 1 << B_;
    Kr_           = 1 << Br_;
    bpv_          = (d_ * Br_ + 7) / 8;
    nprobe_       = std::min(nprobe_, nlist_);

    CUBLAS_CHECK(cublasCreate(&cublas_));
    {
        // The first GEMM of a process also loads cuBLAS's kernels; that is
        // resource setup, so it happens here with the handle rather than
        // inside train()'s rotation.
        // The fp16 tensor-core products the coarse assignment uses are
        // separate kernels, loaded on their own first call; that call sits in
        // train(), so they are warmed here too, in both output widths.
        float* d_w = nullptr;
        CUDA_CHECK(cudaMalloc(&d_w, (size_t)3 * 256 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_w, 0, (size_t)3 * 256 * 256 * sizeof(float)));
        const float one = 1.f, zero = 0.f;
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, 256, 256, 256,
                                 &one, d_w, 256, d_w + 256 * 256, 256,
                                 &zero, d_w + 2 * 256 * 256, 256));
        CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 256, 256, 256, &one,
                                  d_w, CUDA_R_16F, 256, d_w + 256 * 128, CUDA_R_16F, 256,
                                  &zero, d_w + 2 * 256 * 256, CUDA_R_32F, 256,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 256, 256, 256, &one,
                                  d_w, CUDA_R_16F, 256, d_w + 256 * 128, CUDA_R_16F, 256,
                                  &zero, d_w + 2 * 256 * 256, CUDA_R_16F, 256,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256, &one,
                                  d_w, CUDA_R_16F, 256, d_w + 256 * 128, CUDA_R_16F, 256,
                                  &zero, d_w + 2 * 256 * 256, CUDA_R_16F, 256,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        {
            const int ione = 1, izero = 0;
            CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 256, 256, 256, &ione,
                                      d_w, CUDA_R_8I, 256, d_w + 256 * 64, CUDA_R_8I, 256,
                                      &izero, d_w + 2 * 256 * 256, CUDA_R_32I, 256,
                                      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_w);
    }
    // The JL rotation is a d x d product per vector -- 9.4 MFLOP at d=3072, and
    // 9.4 TFLOP over a million of them. Measured at 35 TFLOP/s it is already
    // near the card's fp32 peak, so the only headroom left is the tensor cores.
    // TF32 keeps 10 mantissa bits where fp32 keeps 23, and what consumes the
    // rotated vector is a quantiser that rounds it to an 8-bit code, so the
    // margin is wide -- but that is an argument for measuring it, not for
    // assuming it. Off unless JHQ_TF32 is set.
    // Whether it pays turns on the dimension, and the measurement is not
    // one-sided. Rotation time and recall, fp32 against TF32:
    //     d=768   18.5 -> 11.5 ms,  0.9444 -> 0.9410   (-0.0034)
    //     d=3072   271 -> 179  ms,  0.9538 -> 0.9534   (-0.0004)
    // At 768 the rotation is 5% of add and the loss shows through; the
    // quantiser's cells are fine enough there that TF32's ten mantissa bits sit
    // above its noise. At 3072 the rotation is 17% and the loss is absorbed.
    // JHQ_TF32=0 or 1 overrides.
    {
        const char* tf = std::getenv("JHQ_TF32");
        const bool use_tf32 = tf ? (tf[0] == '1') : (d >= 1536);
        if (use_tf32)
            CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
}

JHQGpuIndex::~JHQGpuIndex() {
    if (ws_.graph_exec) cudaGraphExecDestroy(ws_.graph_exec);
    if (ws_.graph)      cudaGraphDestroy(ws_.graph);
    if (ws_.stream)     cudaStreamDestroy(ws_.stream);

    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_cent_);
    cudaFree(d_res_c1d_);
    cudaFree(d_centroids_);
    drop_assign_centroids();
    cudaFree(d_cent_norms_);
    cudaFree(d_list_offsets_);
    cudaFree(d_list_ids_);
    cudaFree(d_list_primary_t_);
    cudaFree(d_list_res_);
    cudaFree(d_list_corr_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);
    cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_dots);
    cudaFree(ws_.d_byte_lut);
    cudaFree(ws_.d_probe_ids);
    cudaFree(ws_.d_probe_offsets);
    cudaFree(ws_.d_query_total);
    cudaFree(ws_.d_topck_pos);
    cudaFree(ws_.d_topck_primary);
    cudaFree(ws_.d_lut_r);
    cudaFree(ws_.d_comp_dists);
    cudaFree(ws_.d_final_ids);
    cudaFree(ws_.d_final_dists);
}

// ── Workspace allocation ──────────────────────────────────────────────────────
void JHQGpuIndex::alloc_workspace(int batch_size) {
    if (ws_.graph_exec) { cudaGraphExecDestroy(ws_.graph_exec); ws_.graph_exec = nullptr; }
    if (ws_.graph)      { cudaGraphDestroy(ws_.graph);          ws_.graph      = nullptr; }
    ws_.graph_ck     = 0;
    ws_.graph_nprobe = 0;

    cudaFree(ws_.d_q_batch);       ws_.d_q_batch        = nullptr;
    cudaFree(ws_.d_q_rot);         ws_.d_q_rot          = nullptr;
    cudaFree(ws_.d_dots);          ws_.d_dots           = nullptr;
    cudaFree(ws_.d_probe_ids);     ws_.d_probe_ids      = nullptr;
    cudaFree(ws_.d_probe_offsets); ws_.d_probe_offsets  = nullptr;
    cudaFree(ws_.d_query_total);   ws_.d_query_total    = nullptr;
    cudaFree(ws_.d_byte_lut);      ws_.d_byte_lut       = nullptr;
    cudaFree(ws_.d_topck_pos);     ws_.d_topck_pos      = nullptr;
    cudaFree(ws_.d_topck_primary); ws_.d_topck_primary  = nullptr;
    cudaFree(ws_.d_lut_r);         ws_.d_lut_r          = nullptr;
    cudaFree(ws_.d_comp_dists);    ws_.d_comp_dists     = nullptr;
    cudaFree(ws_.d_final_ids);     ws_.d_final_ids      = nullptr;
    cudaFree(ws_.d_final_dists);   ws_.d_final_dists    = nullptr;
    ws_.ck_cap = 0;
    ws_.k_cap  = 0;

    long long B = batch_size;
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,        B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_rot,          B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots,           B * nlist_           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_probe_ids,      B * nprobe_          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_probe_offsets,  B * (nprobe_ + 1)   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_query_total,    B                    * sizeof(int)));
    // jhq_lut_t is the header's, so this cannot disagree with what search.cu
    // stores into the table.
    CUDA_CHECK(cudaMalloc(&ws_.d_byte_lut, B * M_ * 256 * sizeof(jhq_gpu::jhq_lut_t)));
    // Only materialised when JHQ_RESID_LUT=1 asks for the paper's per-query
    // construction; the fused refine kernel recomputes the entries. The buffer
    // is B*d*Kr floats -- 3.0 GB at B=1024, d=3072, Kr=256 -- and it counts
    // against the index memory the benchmark reports.
    {
        const char* rl = std::getenv("JHQ_RESID_LUT");
        if (rl && rl[0] == '1')
            CUDA_CHECK(cudaMalloc(&ws_.d_lut_r, B * d_ * Kr_ * sizeof(float)));
        else
            ws_.d_lut_r = nullptr;
    }
    ws_.batch_cap = batch_size;

    if (ws_.h_q_pinned) { cudaFreeHost(ws_.h_q_pinned); ws_.h_q_pinned = nullptr; }
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned, B * d_ * sizeof(float)));

    if (!ws_.stream) {
        CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    }
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
}

// ── GPU rotation helper ───────────────────────────────────────────────────────
// Host-to-device through a pair of pinned 32 MB buffers, each filled by a
// parallel memcpy while the other is in flight. The source is the mmap of the
// base file, and a pageable cudaMemcpy from it runs at the speed of one core
// faulting the pages in -- 1.3 GB/s measured, 300 ms for the training sample
// on stella-trec24 -- against 56 GB/s on the link.
namespace {
void staged_h2d(void* d_dst, const void* h_src, size_t bytes) {
    constexpr size_t CHUNK = 32u << 20;
    if (bytes <= CHUNK) {
        CUDA_CHECK(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice));
        return;
    }
    char* h_buf[2] = {nullptr, nullptr};
    cudaStream_t st[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaHostAlloc(&h_buf[b], CHUNK, cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[b], cudaStreamNonBlocking));
    }
    const char* se = std::getenv("JHQ_STAGE_THREADS");
    const int want = se ? std::atoi(se) : 32;
    const int nthr = std::max(1, std::min(omp_get_max_threads(), want));
    int k = 0;
    for (size_t off = 0; off < bytes; off += CHUNK, ++k) {
        const int b = k & 1;
        const size_t len = std::min(CHUNK, bytes - off);
        CUDA_CHECK(cudaStreamSynchronize(st[b]));
        const char* src = (const char*)h_src + off;
#pragma omp parallel num_threads(nthr)
        {
            const int nt = omp_get_num_threads(), ti = omp_get_thread_num();
            const size_t chunk = (len + nt - 1) / nt;
            const size_t lo = (size_t)ti * chunk;
            const size_t hi = lo + chunk < len ? lo + chunk : len;
            if (lo < len) std::memcpy(h_buf[b] + lo, src + lo, hi - lo);
        }
        CUDA_CHECK(cudaMemcpyAsync((char*)d_dst + off, h_buf[b], len,
                                   cudaMemcpyHostToDevice, st[b]));
    }
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(st[b]));
        cudaStreamDestroy(st[b]);
        cudaFreeHost(h_buf[b]);
    }
}
} // namespace

float* JHQGpuIndex::rotate_on_gpu(const float* h_x, int n, double* sum_sq) const {
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (long long)n * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (long long)n * d_ * sizeof(float)));
    staged_h2d(d_x, h_x, (size_t)n * d_ * sizeof(float));
    // Lemma 2's sigma^2 = E[||x||^2]/d wants one reduction over the sample,
    // and the sample is here already; on the host it was a pass over 0.4 GB.
    if (sum_sq) {
        float ss = 0.f;
        CUBLAS_CHECK(cublasSdot(cublas_, n * d_, d_x, 1, d_x, 1, &ss));
        *sum_sq = ss;
    }
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d_, n, d_,
                             &one, d_Pi_, d_,
                                   d_x,  d_,
                             &zero, d_y, d_));
    cudaFree(d_x);
    return d_y;
}

// ── IVF centroid training ─────────────────────────────────────────────────────
void JHQGpuIndex::train_ivf_centroids(
    const float* h_y_train, const float* d_y_train, int n_train)
{
    if (n_train < nlist_)
        throw std::invalid_argument("n_train must be >= nlist for v12_transposed");

    // The device path keeps everything resident across the iterations: the
    // seeds are gathered there, the norms are a kernel, and the centroids come
    // back once at the end for the trained-state cache. Before this, each
    // iteration fetched the nlist seed rows one memcpy at a time, freed and
    // re-allocated the centroid buffers, and took the norms on the host from a
    // 64 MB round trip: 739 ms of a 951 ms train on stella-trec24, for eight
    // iterations whose GEMM is 10 ms each.
    const bool gpu_ivf = std::getenv("JHQ_GPU_CODEBOOK") != nullptr;
    if (gpu_ivf) {
        cudaFree(d_centroids_);  d_centroids_  = nullptr;
        cudaFree(d_cent_norms_); d_cent_norms_ = nullptr;
        drop_assign_centroids();
        CUDA_CHECK(cudaMalloc(&d_centroids_,  (long long)nlist_ * d_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cent_norms_, (long long)nlist_ * sizeof(float)));
        {
            const long long tot = (long long)nlist_ * d_;
            seed_centroids_kernel<<<(int)((tot + 255) / 256), 256>>>(
                d_y_train, n_train, d_, nlist_, d_centroids_);
            CUDA_CHECK(cudaGetLastError());
        }

        const int prec = assign_precision();
        const char* sb = std::getenv("JHQ_ASSIGN_BATCH");
        const int batch = std::min(n_train, sb ? std::atoi(sb)
                                               : (nlist_ >= 8192 ? 32768 : 8192));
        int* d_assign = nullptr; float* d_dots = nullptr; __half* d_y16 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_assign, (long long)n_train * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dots, (long long)nlist_ * batch * sizeof(float)));
        if (prec == 16) {
            CUDA_CHECK(cudaMalloc(&d_y16, (long long)n_train * d_ * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_centroids16_,
                                  (long long)nlist_ * d_ * sizeof(__half)));
        }
        auto refresh = [&]() {
            row_sqnorms_kernel<<<(nlist_ * 32 + 255) / 256, 256>>>(
                d_centroids_, nlist_, d_, d_cent_norms_);
            CUDA_CHECK(cudaGetLastError());
            if (prec == 16) {
                const long long nc = (long long)nlist_ * d_;
                f32_to_f16_kernel<<<(int)((nc / 4 + 255) / 256), 256>>>(
                    d_centroids_, d_centroids16_, nc);
                CUDA_CHECK(cudaGetLastError());
            }
            if (assign_int8()) requantize_centroids8(0);
        };
        for (int iter = 0; iter < ivf_iters_; ++iter) {
            refresh();
            assign_into(d_y_train, n_train, d_assign, d_dots, 0, 0, d_y16);
            launch_ivf_accumulate(d_y_train, d_assign, d_centroids_,
                                  n_train, d_, nlist_, iter);
        }
        refresh();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUBLAS_CHECK(cublasSetStream(cublas_, 0));
        centroids_.resize((long long)nlist_ * d_);
        CUDA_CHECK(cudaMemcpy(centroids_.data(), d_centroids_,
                              (long long)nlist_ * d_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_assign); cudaFree(d_dots); if (d_y16) cudaFree(d_y16);
        return;
    }

    // Seed every list from an evenly spaced training vector. With the rotated
    // set no longer copied to the host, those rows are fetched one at a time
    // from the device -- nlist of them, against the whole set.
    centroids_.assign((long long)nlist_ * d_, 0.0f);
    for (int c = 0; c < nlist_; ++c) {
        const int src = (int)((long long)c * n_train / nlist_);
        if (h_y_train) {
            std::memcpy(centroids_.data() + (long long)c * d_,
                        h_y_train + (long long)src * d_,
                        (size_t)d_ * sizeof(float));
        } else {
            CUDA_CHECK(cudaMemcpy(centroids_.data() + (long long)c * d_,
                                  d_y_train + (long long)src * d_,
                                  (size_t)d_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }

    auto upload_centroids = [&]() {
        std::vector<float> cent_norms(nlist_, 0.0f);
        for (int c = 0; c < nlist_; ++c) {
            double s = 0.0;
            const float* cc = centroids_.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) s += (double)cc[j] * cc[j];
            cent_norms[c] = (float)s;
        }
        cudaFree(d_centroids_);
        drop_assign_centroids();
        cudaFree(d_cent_norms_);
        d_centroids_ = nullptr;
        d_cent_norms_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_centroids_,  (long long)nlist_ * d_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_.data(),
                              (long long)nlist_ * d_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cent_norms_, (long long)nlist_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_norms_, cent_norms.data(),
                              (long long)nlist_ * sizeof(float),
                              cudaMemcpyHostToDevice));
    };

    std::vector<int>    h_assign(n_train);
    std::vector<double> sums((long long)nlist_ * d_);
    std::vector<int>    counts(nlist_);

    for (int iter = 0; iter < ivf_iters_; ++iter) {
        upload_centroids();
        int* d_assign = assign_on_gpu(d_y_train, n_train);
        CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign,
                              (long long)n_train * sizeof(int),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_assign);

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n_train; ++i) {
            int c = h_assign[i];
            counts[c]++;
            const float* yi = h_y_train + (long long)i * d_;
            double* sc = sums.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) sc[j] += yi[j];
        }
        for (int c = 0; c < nlist_; ++c) {
            float* cc = centroids_.data() + (long long)c * d_;
            if (counts[c] == 0) {
                int src = (int)(((long long)c * 1103515245 + iter * 12345) % n_train);
                std::memcpy(cc, h_y_train + (long long)src * d_,
                            (size_t)d_ * sizeof(float));
                continue;
            }
            const double inv = 1.0 / (double)counts[c];
            const double* sc = sums.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) cc[j] = (float)(sc[j] * inv);
        }
    }
    upload_centroids();
}

// ── Residual codebook training ────────────────────────────────────────────────
void JHQGpuIndex::train_residual_codebook(
    const float* d_y_train, const uint8_t* d_codes_train, int n_train)
{
    std::vector<float>   h_y    ((long long)n_train * d_);
    std::vector<uint8_t> h_codes((long long)n_train * M_);

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_train,
                          (long long)n_train * d_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_codes.data(), d_codes_train,
                          (long long)n_train * M_ * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    // One codebook per subspace, not one shared by all d dimensions. The
    // official get_scalar_codebook_ptr() is indexed by (subspace, level) for a
    // reason: the primary quantiser leaves a different residual scale in each
    // subspace, and a single global codebook has to straddle all of them.
    res_c1d_.assign((size_t)M_ * Kr_, 0.f);

    std::vector<float> yhat(d_);

    // Reconstruct once per vector, then slice per subspace, rather than
    // reconstructing M times.
    std::vector<float> all_resid((size_t)n_train * d_);
    for (int i = 0; i < n_train; i++) {
        cb_->reconstruct(h_codes.data() + (long long)i * M_, yhat.data());
        const float* yi = h_y.data() + (long long)i * d_;
        float* ri = all_resid.data() + (size_t)i * d_;
        for (int j = 0; j < d_; j++) ri[j] = yi[j] - yhat[j];
    }

#if JHQ_GLOBAL_RESIDUAL_CB
    {
        // One codebook over every dimension of every subspace, then replicated
        // so the layout the kernels see is unchanged.
        std::vector<float> cb =
            train_1d_kmeans(all_resid.data(), (int)all_resid.size(), Kr_);
        for (int m = 0; m < M_; ++m)
            std::copy(cb.begin(), cb.end(), res_c1d_.begin() + (size_t)m * Kr_);
    }
#else
    // This loop, not the primary codebook, is the index build: phase timing put
    // it at 23.5 s of a 28.5 s train against the PQ k-means's 4.2 s. Each
    // subspace reads its own stride of all_resid and writes its own slice of
    // res_c1d_, so the axis parallelises the same way -- resid moves inside so
    // each thread has its own scratch.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < M_; ++m) {
        std::vector<float> resid_m;
        resid_m.reserve((size_t)n_train * Ds_);
        for (int i = 0; i < n_train; i++) {
            const float* ri = all_resid.data() + (size_t)i * d_ + (size_t)m * Ds_;
            resid_m.insert(resid_m.end(), ri, ri + Ds_);
        }
        std::vector<float> cb = train_1d_kmeans(resid_m.data(), (int)resid_m.size(), Kr_);
        std::copy(cb.begin(), cb.end(), res_c1d_.begin() + (size_t)m * Kr_);
    }
#endif

    CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_res_c1d_, res_c1d_.data(),
                          (size_t)M_ * Kr_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// ── Train ─────────────────────────────────────────────────────────────────────
// ── Trained-state cache ───────────────────────────────────────────────────────
//
// Training reads only the sample and the seed, so every point in a parameter
// sweep rebuilds the same codebooks: ~29 s on Vogue against ~1 s of search,
// which was three quarters of the wall time of a sweep. Keyed on everything
// train() consumes -- the parameters, the codebook path, and a checksum of the
// sample -- so a changed dataset, parameter or construction misses rather than
// silently reusing the wrong codebooks.

// The fused encoder locates a residual by the cell it falls in, which needs
// each subspace's Kr codewords in increasing order. Lloyd's iterations keep
// the order they start in and the initialisations are quantiles, so this is
// normally a no-op -- but it is the encoder's precondition, so it is enforced
// here rather than assumed. Relabelling codewords changes nothing else: the
// codes are produced against, and the search LUT built from, the same array.
void JHQGpuIndex::sort_residual_codebook() {
    if (res_c1d_.empty()) return;
    for (int m = 0; m < M_; ++m)
        std::sort(res_c1d_.begin() + (size_t)m * Kr_,
                  res_c1d_.begin() + (size_t)(m + 1) * Kr_);
    if (!d_res_c1d_)
        CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_res_c1d_, res_c1d_.data(),
                          (size_t)M_ * Kr_ * sizeof(float), cudaMemcpyHostToDevice));
}

void JHQGpuIndex::upload_trained() {
    if (!d_Pi_) CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    if (!d_cent_) CUDA_CHECK(cudaMalloc(&d_cent_, cb_->size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cent_, cb_->data(), cb_->size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    sort_residual_codebook();

    // Same work as the upload_centroids lambda inside train_ivf_centroids;
    // that one is local to the function, so it cannot be reused from here.
    std::vector<float> cent_norms(nlist_, 0.0f);
    for (int c = 0; c < nlist_; ++c) {
        double sn = 0.0;
        const float* cc = centroids_.data() + (long long)c * d_;
        for (int j = 0; j < d_; ++j) sn += (double)cc[j] * cc[j];
        cent_norms[c] = (float)sn;
    }
    cudaFree(d_centroids_);
    d_centroids_  = nullptr;
    drop_assign_centroids();
    cudaFree(d_cent_norms_); d_cent_norms_ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_centroids_, (long long)nlist_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_.data(),
                          (long long)nlist_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cent_norms_, (long long)nlist_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cent_norms_, cent_norms.data(),
                          (long long)nlist_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

std::string JHQGpuIndex::cache_path(const char* dir, const float* h_x,
                                    int n_train) const {
    // FNV-1a over the parameters and a strided sample of the training data.
    // Striding keeps the hash O(1) in n_train while still covering the file.
    unsigned long long h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    // The primary codebook path is something train() consumes, so it belongs in
    // the key: without it a state trained under the paper's eq. 4 construction
    // and one trained by Lloyd hash the same, and a sweep that sets neither
    // variable silently reuses whichever was written first. That is how a
    // frontier can end up reporting recall for a quantiser its own environment
    // never selected.
    const int codebook_path = (std::getenv("JHQ_PAPER_CODEBOOK") ? 2 : 0)
                            + (std::getenv("JHQ_GPU_CODEBOOK")   ? 1 : 0);
    const int params[] = { d_, M_, B_, Br_, K_, Kr_, nlist_,
                           ivf_iters_, kmeans_iters_, seed_, n_train,
                           codebook_path };
    mix(params, sizeof params);
    const long long total = (long long)n_train * d_;
    const long long step  = total > 4096 ? total / 4096 : 1;
    for (long long i = 0; i < total; i += step) mix(&h_x[i], sizeof(float));

    char buf[64];
    std::snprintf(buf, sizeof buf, "/jhq_trained_%016llx.bin", h);
    return std::string(dir) + buf;
}

bool JHQGpuIndex::load_trained(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    if (!jl_.read_state(f)) return false;
    cb_ = std::make_unique<PQCodebook>(d_, M_, B_);
    if (!cb_->read_state(f)) return false;

    long long nr = 0, nc = 0;
    f.read(reinterpret_cast<char*>(&nr), sizeof nr);
    f.read(reinterpret_cast<char*>(&nc), sizeof nc);
    if (!f || nr != (long long)M_ * Kr_ || nc != (long long)nlist_ * d_) return false;
    res_c1d_.resize((size_t)nr);
    centroids_.resize((size_t)nc);
    f.read(reinterpret_cast<char*>(res_c1d_.data()),   (std::streamsize)nr * sizeof(float));
    f.read(reinterpret_cast<char*>(centroids_.data()), (std::streamsize)nc * sizeof(float));
    return (bool)f;
}

void JHQGpuIndex::save_trained(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    { std::ofstream f(tmp, std::ios::binary);
      if (!f) return;
      jl_.write_state(f);
      cb_->write_state(f);
      const long long nr = (long long)res_c1d_.size(), nc = (long long)centroids_.size();
      f.write(reinterpret_cast<const char*>(&nr), sizeof nr);
      f.write(reinterpret_cast<const char*>(&nc), sizeof nc);
      f.write(reinterpret_cast<const char*>(res_c1d_.data()),   (std::streamsize)nr * sizeof(float));
      f.write(reinterpret_cast<const char*>(centroids_.data()), (std::streamsize)nc * sizeof(float));
      if (!f) { std::remove(tmp.c_str()); return; }
    }
    // Rename last so a run killed mid-write never leaves a half file behind
    // that the next run would load as if it were complete.
    if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}

// Phase timing for train(). The 29 s build against cuVS's 4.5 s was attributed
// to the PQ codebook k-means on the strength of an nsys trace showing only
// ~700 ms of GPU work, but that only says the time is on the host, not which
// host loop owns it -- parallelising the codebook changed the total by nothing.
#define JHQ_TRAIN_PHASE(label)                                                \
    do { if (phase_timing) {                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
        const double now = std::chrono::duration<double, std::milli>(         \
            std::chrono::high_resolution_clock::now() - t_phase).count();     \
        std::printf("  [train] %-22s %8.1f ms\n", label, now);                \
        t_phase = std::chrono::high_resolution_clock::now();                   \
    } } while (0)

// Which primary construction is in force. The paper's analytical Cartesian
// product (§3.2, equation 4) needs a whole number of levels per dimension --
// K^(1/Ds) = 2^(B/Ds), so Ds must divide B -- and is the default whenever that
// holds. Where it does not (Ds > B, the short-code configurations), the only
// construction available is the Lloyd-refined product quantiser, which the
// caller has to ask for: it leaves the per-dimension levels unset, and the
// general encoder that then runs has not been validated against the reference.
bool JHQGpuIndex::paper_codebook_selected() const {
    // Equation 4 is the algorithm this port implements, and the uneven bit
    // split makes it constructible at every Ds, so it is the default
    // everywhere. JHQ_PAPER_CODEBOOK=0 selects the Lloyd-refined product
    // quantiser the reference implementation uses, which is the control this
    // construction is measured against. Where the split is uneven the
    // separable and fused encoders cannot run and the general one does; that
    // costs throughput at build time, not accuracy.
    const char* e = std::getenv("JHQ_PAPER_CODEBOOK");
    return !(e && e[0] == '0');
}

void JHQGpuIndex::train(const float* h_x, int n_train) {
    const bool phase_timing = std::getenv("JHQ_TRAIN_PHASES") != nullptr;
    auto t_phase = std::chrono::high_resolution_clock::now();

    // sigma comes back from the device with the rotation below unless the
    // trained state is cached, which stores it.
    const bool device_sigma = std::getenv("JHQ_HOST_SIGMA") == nullptr;
    if (!device_sigma) {
        jl_.estimate_sigma(h_x, n_train);
        JHQ_TRAIN_PHASE("estimate_sigma");
    }

    // The paper's primary is the analytical Cartesian product of equation 4,
    // so that is the default wherever it is admissible (K^(1/Ds) whole, i.e.
    // Ds divides B). JHQ_PAPER_CODEBOOK=0 selects the Lloyd-refined product
    // quantiser the reference implementation uses instead; that path does not
    // set the per-dimension levels and so falls back to the general encoder,
    // which is not validated -- it is opt-in and says so.
    const bool paper_codebook = paper_codebook_selected();

    const char* cache_dir = std::getenv("JHQ_INDEX_CACHE");
    std::string cpath;
    if (cache_dir) {
        cpath = cache_path(cache_dir, h_x, n_train);
        if (load_trained(cpath)) {
            // The cache holds the trained state, not the derived tables. The
            // levels the separable encoder needs are a closed form in sigma,
            // which the state does restore, so rebuild them here rather than
            // storing them: without this a cache hit silently drops to the
            // general encoder and every code it writes is wrong.
            if (paper_codebook) {
                cb_->build_analytical_cartesian(jl_.sigma());
                n_levels_ = (int)cb_->levels().size();
                if (n_levels_ > 0) {
                    if (!d_levels_)
                        CUDA_CHECK(cudaMalloc(&d_levels_, (size_t)n_levels_ * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_levels_, cb_->levels().data(),
                                          (size_t)n_levels_ * sizeof(float),
                                          cudaMemcpyHostToDevice));
                }
            }
            upload_trained();
            JHQ_TRAIN_PHASE("loaded from cache");
            return;
        }
    }
    cb_ = std::make_unique<PQCodebook>(d_, M_, B_);

    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    // The PQ centroids need the rotated training data, so rotate first and
    // pull it back once -- train() already copies it to the host below for
    // the IVF, so this costs one extra D2H at build time, not per query.
    const bool gpu_codebook = std::getenv("JHQ_GPU_CODEBOOK") != nullptr;

    double sum_sq = 0.0;
    float* d_y_train = rotate_on_gpu(h_x, n_train, device_sigma ? &sum_sq : nullptr);
    if (device_sigma)
        jl_.set_sigma((float)std::sqrt(sum_sq / ((double)n_train * d_)));
    // The rotated training set is 2.9 GB on Vogue and every host-side consumer
    // of it has moved to the device, except the analytical initialisation --
    // and that reads only the per-subspace mean and variance. Computing those
    // here brings back M*Ds*2 floats instead.
    std::vector<float> h_y_train;
    if (!gpu_codebook) {
        h_y_train.resize((long long)n_train * d_);
        CUDA_CHECK(cudaMemcpy(h_y_train.data(), d_y_train,
                              (long long)n_train * d_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    JHQ_TRAIN_PHASE(gpu_codebook ? "rotate" : "rotate + D2H");

    // The primary codebook is 96 independent subspaces of 100k points against
    // 256 centroids in 8 dimensions -- about 98 GFLOP over five iterations, and
    // 3.5 s of the 8.3 s build even with every core busy. Setting
    // JHQ_GPU_CODEBOOK runs the Lloyd iterations on the device instead; the
    // analytical placement stays on the host so both paths start identically.
    // The paper builds the primary codebook analytically, as the Cartesian
    // product of per-dimension Lloyd-Max codewords (equation 4), reading no
    // data and costing O(MK): section 3.2 constructs it "directly without
    // leveraging the k-means method". The reference implementation instead
    // runs five Lloyd iterations on top of an analytical seed. Both are here;
    // JHQ_PAPER_CODEBOOK selects the paper's.
    if (paper_codebook) {
        cb_->build_analytical_cartesian(jl_.sigma());
        CUDA_CHECK(cudaMalloc(&d_cent_, cb_->size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_, cb_->data(), cb_->size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        // The level table the product is built from. Keeping it lets the
        // encoder use the separable argmin, which never visits the K codewords.
        // An uneven split has no single shared table; the expanded centroids
        // above are the whole codebook then, and the general encoder reads
        // them. Leaving d_levels_ null is what selects it.
        n_levels_ = (int)cb_->levels().size();
        if (n_levels_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_levels_, (size_t)n_levels_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_levels_, cb_->levels().data(),
                                  (size_t)n_levels_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        JHQ_TRAIN_PHASE("pq codebook (analytical, paper eq. 4)");
    } else if (gpu_codebook) {
        const int cols = M_ * Ds_;
        float *d_mean = nullptr, *d_var = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mean, (size_t)cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_var,  (size_t)cols * sizeof(float)));
        launch_subspace_stats(d_y_train, n_train, d_, M_, Ds_, d_mean, d_var);
        std::vector<float> h_mean(cols), h_var(cols);
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, (size_t)cols * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_var.data(), d_var, (size_t)cols * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_mean); cudaFree(d_var);
        cb_->init_from_stats(h_mean.data(), h_var.data(), seed_);
    } else {
        cb_->train(h_y_train.data(), n_train, kmeans_iters_, seed_);
    }
    JHQ_TRAIN_PHASE(gpu_codebook ? "pq codebook init (moments on device)"
                                 : "pq codebook kmeans");

    if (!paper_codebook) {
        CUDA_CHECK(cudaMalloc(&d_cent_, cb_->size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_, cb_->data(), cb_->size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    if (!paper_codebook && gpu_codebook && kmeans_iters_ > 0) {
        launch_pq_kmeans(d_y_train, n_train, d_, M_, Ds_, K_,
                         d_cent_, kmeans_iters_);
        // hand the result back: the residual level trains against these, and
        // the trained-state cache stores them
        CUDA_CHECK(cudaMemcpy(cb_->mutable_data(), d_cent_,
                              cb_->size() * sizeof(float), cudaMemcpyDeviceToHost));
        JHQ_TRAIN_PHASE("pq codebook kmeans (device)");
    }

    uint8_t* d_codes_train = nullptr;
    CUDA_CHECK(cudaMalloc(&d_codes_train, (long long)n_train * M_));

    launch_primary_encode(d_y_train, d_codes_train, d_cent_,
                          n_train, d_, M_, Ds_, K_);
    CUDA_CHECK(cudaDeviceSynchronize());
    JHQ_TRAIN_PHASE("primary encode");

    train_ivf_centroids(h_y_train.empty() ? nullptr : h_y_train.data(),
                        d_y_train, n_train);
    JHQ_TRAIN_PHASE("ivf centroids");

    if (gpu_codebook) {
        // The largest phase of the build, and almost all of it a sort: one per
        // subspace over n_train*Ds values. On the device that is a single
        // segmented radix sort, and the residual is written straight into the
        // segment-major layout the sort wants, so the host's gather disappears.
        res_c1d_.assign((size_t)M_ * Kr_, 0.f);
        if (!d_res_c1d_)
            CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
        // Two ways to the same codebook. The sorted path is exact; the
        // histogram path drops the O(n log n) sort for one O(n) pass and places
        // values by bucket instead of by magnitude. JHQ_RES_HIST picks it.
        const char* rh = std::getenv("JHQ_RES_HIST");
        if (rh && rh[0] == '1') {
            const char* nbe = std::getenv("JHQ_RES_BUCKETS");
            const int nbuckets = nbe ? std::atoi(nbe) : 2048;
            launch_residual_codebook_hist(d_y_train, d_codes_train, d_cent_,
                                          n_train, d_, M_, Ds_, K_, Kr_, 25,
                                          nbuckets, d_res_c1d_);
        } else {
            launch_residual_codebook(d_y_train, d_codes_train, d_cent_,
                                     n_train, d_, M_, Ds_, K_, Kr_, 25, d_res_c1d_);
        }
        CUDA_CHECK(cudaMemcpy(res_c1d_.data(), d_res_c1d_,
                              (size_t)M_ * Kr_ * sizeof(float), cudaMemcpyDeviceToHost));
        // The correction term per vector still comes from the host encode path
        // in add(); only the codebook moved.
        JHQ_TRAIN_PHASE("residual codebook (device)");
    } else {
        train_residual_codebook(d_y_train, d_codes_train, n_train);
        JHQ_TRAIN_PHASE("residual codebook");
    }

    sort_residual_codebook();

    if (cache_dir) { save_trained(cpath); JHQ_TRAIN_PHASE("cache write"); }

    cudaFree(d_y_train);
    cudaFree(d_codes_train);
}

// ── GPU assignment helper ─────────────────────────────────────────────────────
int* JHQGpuIndex::assign_on_gpu(const float* d_y, int n) const {
    int* d_assign = nullptr;
    CUDA_CHECK(cudaMalloc(&d_assign, (long long)n * sizeof(int)));
    const char* sb = std::getenv("JHQ_ASSIGN_BATCH");
    const int   batch = std::min(n, sb ? std::atoi(sb)
                                       : (nlist_ >= 8192 ? 32768 : 8192));
    float* d_dots = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dots, (long long)nlist_ * batch * sizeof(float)));
    __half* d_y16 = nullptr;
    if (assign_precision() == 16)
        CUDA_CHECK(cudaMalloc(&d_y16, (long long)n * d_ * sizeof(__half)));
    assign_into(d_y, n, d_assign, d_dots, 0, 0, d_y16);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasSetStream(cublas_, 0));
    cudaFree(d_dots);
    if (d_y16) cudaFree(d_y16);
    return d_assign;
}

// The assignment's precision is a separate decision from the rotation's. The
// rotation's output is what gets quantised, so its error shows up in every
// code; the assignment only picks a list, and a lower-precision product can
// only flip a vector between two lists it is nearly equidistant from -- which
// the probe order covers either way. Measured, fp32 -> fp16 inputs with fp32
// accumulation (assign time, recall):
//     vogue 1M/768        25.8 ->   12.9 ms   0.9357 -> 0.9348
//     openai 1M/3072      77.7 ->   47.3      0.9581 -> 0.9584
//     bge-m3 10.1M/1024   2403 ->    977      0.9220 -> 0.9220
//     stella 17.8M/1024   9118 ->   3370      0.9580 -> 0.9580
// fp32 on the CUDA cores was already at 77 TFLOPS of the card's ~105, so the
// tensor cores were the only headroom. JHQ_ASSIGN_PREC: 16 (default), tf32, 32.
int JHQGpuIndex::assign_precision() const {
    const char* p = std::getenv("JHQ_ASSIGN_PREC");
    if (!p) return 16;
    if (p[0] == 't' || p[0] == 'T') return 19;   // tf32
    return std::atoi(p) == 16 ? 16 : 32;
}

// The int8 tensor-core product with the bounded, exactly settled argmin (see
// assign_from_dots8_kernel). Measured against the fp16 product, add time on
// the card alone, warm page cache, the better of two runs each:
//     vogue 0.93M/768,  nlist 1024      224 ->  224 ms
//     bge-m3 10.1M/1024, nlist 8192    1679 -> 1574
//     stella 17.8M/1024, nlist 16384   4271 -> 3705
// The product itself halves; what it gives back is the settlement. The
// bound is Cauchy-Schwarz on rounding errors that are in fact uncorrelated
// with the data, so it is some thirty times wider than the error, and at
// nlist 8192 and up most rows (85-88%) fall inside it and are recomputed
// against their candidates from L2. The rows that end up differing from the
// fp16 path are the ties: 0 of 14838 on vogue's last batch, 6 of 31748 on
// bge-m3, 3 of 16359 on stella, recall the same to the third decimal.
// JHQ_ASSIGN_INT8=0 keeps the fp16 product.
bool JHQGpuIndex::assign_int8() const {
    if (assign_precision() != 16 || std::getenv("JHQ_ASSIGN_OUT32")) return false;
    if (d_ % 8 != 0) return false;
    const char* e = std::getenv("JHQ_ASSIGN_INT8");
    return e ? e[0] == '1' : true;
}

void JHQGpuIndex::ensure_assign_centroids(cudaStream_t stream, int x_domain) const {
    const int prec = assign_precision();
    const long long nc = (long long)nlist_ * d_;
    if (x_domain && !d_centroids_x_) {
        // R^T c for every centroid, as one GEMM against the rotation. The
        // rotation's own product is y^T = x^T R^T (see add()), so the matrix
        // taken transposed there is taken as stored here.
        CUDA_CHECK(cudaMalloc(&d_centroids_x_, nc * sizeof(float)));
        const float one = 1.f, zero = 0.f;
        CUBLAS_CHECK(cublasSetStream(cublas_, stream));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_, nlist_, d_, &one,
                                 d_Pi_, d_, d_centroids_, d_,
                                 &zero, d_centroids_x_, d_));
        CUBLAS_CHECK(cublasSetStream(cublas_, 0));
    }
    const float* cf  = x_domain ? d_centroids_x_ : d_centroids_;
    __half*&     c16 = x_domain ? d_centroids16_x_ : d_centroids16_;
    if (prec == 16 && !c16) {
        CUDA_CHECK(cudaMalloc(&c16, nc * sizeof(__half)));
        f32_to_f16_kernel<<<(int)((nc / 4 + 255) / 256), 256, 0, stream>>>(cf, c16, nc);
        CUDA_CHECK(cudaGetLastError());
    }
    if (assign_int8() && !(x_domain ? d_centroids8_x_ : d_centroids8_))
        requantize_centroids8(stream, x_domain);
}

// Every derived copy of the coarse centroids, dropped when they change.
void JHQGpuIndex::drop_assign_centroids() const {
    if (d_centroids16_)   { cudaFree(d_centroids16_);   d_centroids16_   = nullptr; }
    if (d_centroids8_)    { cudaFree(d_centroids8_);    d_centroids8_    = nullptr; }
    if (d_cstats8_)       { cudaFree(d_cstats8_);       d_cstats8_       = nullptr; }
    if (d_centroids_x_)   { cudaFree(d_centroids_x_);   d_centroids_x_   = nullptr; }
    if (d_centroids16_x_) { cudaFree(d_centroids16_x_); d_centroids16_x_ = nullptr; }
    if (d_centroids8_x_)  { cudaFree(d_centroids8_x_);  d_centroids8_x_  = nullptr; }
    if (d_cstats8_x_)     { cudaFree(d_cstats8_x_);     d_cstats8_x_     = nullptr; }
}

// The centroids' int8 image on one shared scale, with that scale and the
// largest per-centroid error kept on the device as float bits, so k-means
// can refresh it every iteration without a host round trip.
void JHQGpuIndex::requantize_centroids8(cudaStream_t stream, int x_domain) const {
    if (d_ % 8 != 0) throw std::runtime_error("assign: the int8 product needs d % 8 == 0");
    const float* cf = x_domain ? d_centroids_x_ : d_centroids_;
    int8_t*&   c8 = x_domain ? d_centroids8_x_ : d_centroids8_;
    unsigned*& st = x_domain ? d_cstats8_x_ : d_cstats8_;
    if (!st) CUDA_CHECK(cudaMalloc(&st, 2 * sizeof(unsigned)));
    if (!c8) CUDA_CHECK(cudaMalloc(&c8, (long long)nlist_ * d_));
    CUDA_CHECK(cudaMemsetAsync(st, 0, 2 * sizeof(unsigned), stream));
    const long long n4 = (long long)nlist_ * d_ / 4;
    absmax_kernel<<<(int)std::min<long long>((n4 + 255) / 256, 2048), 256, 0, stream>>>(
        cf, n4, st);
    quantize_centroids_kernel<<<(int)(((long long)nlist_ * 32 + 255) / 256), 256, 0, stream>>>(
        cf, nlist_, d_, st, c8, st + 1);
    CUDA_CHECK(cudaGetLastError());
}

void JHQGpuIndex::assign_into(const float* d_y, int n, int* d_out,
                              float* d_dots, int y_transposed,
                              cudaStream_t stream, __half* d_y16,
                              int x_domain) const {
    // At scale this is the largest thing add() does: every vector against every
    // one of nlist centroids is 17.8M * 16384 * 1024 FLOP on stella-trec24,
    // measured at 11.2 s of a 38 s add, two orders above the encode it feeds.
    // The sub-batch is what decides how much of the machine the GEMM can use --
    // 8192 rows against nlist=16384 leaves the product too narrow -- so it
    // scales with nlist instead of staying fixed.
    const char* sb = std::getenv("JHQ_ASSIGN_BATCH");
    const int   batch = sb ? std::atoi(sb)
                           : (nlist_ >= 8192 ? 32768 : 8192);
    const int   prec  = assign_precision();
    // The fp16-stored product with exact settlement (see the kernel), unless
    // JHQ_ASSIGN_OUT32 asks for the fp32 product; JHQ_ASSIGN_CHECK runs both
    // and counts the rows on which they disagree.
    const bool  out16 = (prec == 16) && !std::getenv("JHQ_ASSIGN_OUT32");
    const bool  int8  = out16 && assign_int8();
    const bool  check = out16 && std::getenv("JHQ_ASSIGN_CHECK");
    // JHQ_ASSIGN_CHECK=32 holds the int8 path against the fp32 product itself
    // rather than the fp16 path, which has rounding of its own.
    const bool  check32 = check && int8 && std::atoi(std::getenv("JHQ_ASSIGN_CHECK")) == 32;
    // The bound's relative term: half an fp16 ulp of the dot is 2^-11, the
    // distance carries twice the dot, so 2^-10 = 0.000977, with a little over.
    // JHQ_ASSIGN_REL widens it, to show the disagreements that remain are
    // summation order and not the bound.
    const char* re = std::getenv("JHQ_ASSIGN_REL");
    const float rel = re ? (float)std::atof(re) : 0.00099f;
    float* d_chk_dots = nullptr; int* d_chk_out = nullptr; int* d_chk_cnt = nullptr;
    if (check) {
        CUDA_CHECK(cudaMalloc(&d_chk_dots, (long long)nlist_ * batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_chk_out, (long long)batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chk_cnt, 2 * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_chk_cnt, 0, 2 * sizeof(int), stream));
    }
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // Which side of the rotation the input is on decides the centroid set;
    // the norms are the same on both.
    if (x_domain && prec != 16) ensure_assign_centroids(stream, 1);
    const float*    cf  = x_domain ? d_centroids_x_   : d_centroids_;
    int8_t* y8 = nullptr; float4* y8_stats = nullptr; size_t smem8 = 0;
    if (prec == 16) {
        if (!d_y16) throw std::runtime_error("assign_into: fp16 needs a half scratch");
        ensure_assign_centroids(stream, x_domain);
        const long long ny = (long long)n * d_;
        if (int8 && !check) {
            // The int8 image of y, row-major in either layout, and the row
            // statistics behind it -- both inside the half-sized scratch,
            // which is twice what the bytes need.
            y8 = reinterpret_cast<int8_t*>(d_y16);
            y8_stats = reinterpret_cast<float4*>(y8 + ((ny + 15) / 16) * 16);
            if ((ny + 15) / 16 * 16 + (long long)n * sizeof(float4) > ny * (long long)sizeof(__half))
                throw std::runtime_error("assign_into: the int8 scratch does not fit");
        } else if (int8) {
            // Checking against the fp16 path needs the fp16 y as well.
            CUDA_CHECK(cudaMalloc(&y8, ny + 16 + (long long)n * sizeof(float4)));
            y8_stats = reinterpret_cast<float4*>(y8 + ((ny + 15) / 16) * 16);
        }
        if (int8) {
            if (y_transposed)
                quantize_cols_kernel<<<(n + 31) / 32, 256, 0, stream>>>(d_y, n, d_, y8, y8_stats);
            else
                quantize_rows_kernel<<<(int)(((long long)n * 32 + 255) / 256), 256, 0, stream>>>(
                    d_y, n, d_, y8, y8_stats);
            CUDA_CHECK(cudaGetLastError());
            smem8 = 4 * (size_t)d_ * sizeof(float);
            static size_t smem8_set = 0;
            if (smem8 > 48 * 1024 && smem8 > smem8_set) {
                CUDA_CHECK(cudaFuncSetAttribute(assign_from_dots8_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
                smem8_set = smem8;
            }
        }
        if (!int8 || check) {
            // The whole batch converts once, so the sub-batch slicing below is
            // the same in either layout.
            f32_to_f16_kernel<<<(int)((ny / 4 + 255) / 256), 256, 0, stream>>>(
                d_y, d_y16, ny);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const __half*   c16 = x_domain ? d_centroids16_x_ : d_centroids16_;
    const int8_t*   c8  = x_domain ? d_centroids8_x_  : d_centroids8_;
    const unsigned* cs8 = x_domain ? d_cstats8_x_     : d_cstats8_;

    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);
        // y is (n x d) row-major, or with y_transposed (n x d) column-major
        // with ld = n: then it enters transposed and this sub-batch's slice
        // starts at column 0, row `start`.
        const cublasOperation_t opB = y_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const long long yoff = y_transposed ? start : (long long)start * d_;
        const int       ldb  = y_transposed ? n : d_;
        if (prec == 32) {
            CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, opB,
                                     nlist_, nb, d_, &one,
                                     cf, d_,
                                     d_y + yoff, ldb,
                                     &zero, d_dots, nlist_));
        } else if (prec == 19) {
            CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, opB,
                                      nlist_, nb, d_, &one,
                                      cf, CUDA_R_32F, d_,
                                      d_y + yoff,   CUDA_R_32F, ldb,
                                      &zero, d_dots, CUDA_R_32F, nlist_,
                                      CUBLAS_COMPUTE_32F_FAST_TF32,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (int8) {
            const int ione = 1, izero = 0;
            int* d_dots32 = reinterpret_cast<int*>(d_dots);
            CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                      nlist_, nb, d_, &ione,
                                      c8, CUDA_R_8I, d_,
                                      y8 + (long long)start * d_, CUDA_R_8I, d_,
                                      &izero, d_dots32, CUDA_R_32I, nlist_,
                                      CUBLAS_COMPUTE_32I,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            assign_from_dots8_kernel<<<(int)(((long long)nb * 32 + 127) / 128), 128, smem8, stream>>>(
                d_dots32, d_cent_norms_, cf, d_y + yoff, y_transposed, ldb,
                y8_stats + start, cs8, d_out + start, nlist_, nb, d_,
                check ? d_chk_cnt + 1 : nullptr);
            CUDA_CHECK(cudaGetLastError());
            if (check32) {
                CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, opB,
                                         nlist_, nb, d_, &one,
                                         cf, d_, d_y + yoff, ldb,
                                         &zero, d_chk_dots, nlist_));
                assign_from_dots_kernel<<<(nb * 32 + 255) / 256, 256, 0, stream>>>(
                    d_chk_dots, d_cent_norms_, d_chk_out, nlist_, nb);
                count_mismatch_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(
                    d_out + start, d_chk_out, nb, d_chk_cnt);
                CUDA_CHECK(cudaGetLastError());
            } else if (check) {
                // The fp16 product's assignment beside it, settled its own
                // way, and a count of the rows that differ.
                __half* d_dots16 = reinterpret_cast<__half*>(d_chk_dots);
                CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, opB,
                                          nlist_, nb, d_, &one,
                                          c16, CUDA_R_16F, d_,
                                          d_y16 + yoff,   CUDA_R_16F, ldb,
                                          &zero, d_dots16, CUDA_R_16F, nlist_,
                                          CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                assign_from_dots16_kernel<<<(nb * 32 + 255) / 256, 256, 0, stream>>>(
                    d_dots16, d_cent_norms_, c16, d_y16 + yoff,
                    y_transposed, ldb, d_chk_out, nlist_, nb, d_, rel, nullptr);
                count_mismatch_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(
                    d_out + start, d_chk_out, nb, d_chk_cnt);
                CUDA_CHECK(cudaGetLastError());
            }
            continue;
        } else if (out16) {
            // fp32 sums, stored as fp16: the scratch is sized for floats, so
            // the half product fits in its first half.
            __half* d_dots16 = reinterpret_cast<__half*>(d_dots);
            CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, opB,
                                      nlist_, nb, d_, &one,
                                      c16, CUDA_R_16F, d_,
                                      d_y16 + yoff,   CUDA_R_16F, ldb,
                                      &zero, d_dots16, CUDA_R_16F, nlist_,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            assign_from_dots16_kernel<<<(nb * 32 + 255) / 256, 256, 0, stream>>>(
                d_dots16, d_cent_norms_, c16, d_y16 + yoff,
                y_transposed, ldb, d_out + start, nlist_, nb, d_, rel,
                check ? d_chk_cnt + 1 : nullptr);
            CUDA_CHECK(cudaGetLastError());
            if (check) {
                // The fp32 product beside it, and a count of rows that differ.
                CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, opB,
                                          nlist_, nb, d_, &one,
                                          c16, CUDA_R_16F, d_,
                                          d_y16 + yoff,   CUDA_R_16F, ldb,
                                          &zero, d_chk_dots, CUDA_R_32F, nlist_,
                                          CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                assign_from_dots_kernel<<<(nb * 32 + 255) / 256, 256, 0, stream>>>(
                    d_chk_dots, d_cent_norms_, d_chk_out, nlist_, nb);
                count_mismatch_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(
                    d_out + start, d_chk_out, nb, d_chk_cnt);
                CUDA_CHECK(cudaGetLastError());
            }
            continue;
        } else {
            CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, opB,
                                      nlist_, nb, d_, &one,
                                      c16, CUDA_R_16F, d_,
                                      d_y16 + yoff,   CUDA_R_16F, ldb,
                                      &zero, d_dots, CUDA_R_32F, nlist_,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        assign_from_dots_kernel<<<(nb * 32 + 255) / 256, 256, 0, stream>>>(
            d_dots, d_cent_norms_, d_out + start, nlist_, nb);
        CUDA_CHECK(cudaGetLastError());
    }
    if (check) {
        int cnt[2] = {0, 0};
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(cnt, d_chk_cnt, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, check32 ? "  [assign check] %d of %d rows differ between the int8 and fp32 products; %d settled exactly\n"
                           : int8    ? "  [assign check] %d of %d rows differ between the int8 and fp16 products; %d settled exactly\n"
                                     : "  [assign check] %d of %d rows differ between the fp16 and fp32 products; %d settled exactly\n",
                     cnt[0], n, cnt[1]);
        cudaFree(d_chk_dots); cudaFree(d_chk_out); cudaFree(d_chk_cnt);
        if (int8) cudaFree(y8);
    }
}

// ── Add ───────────────────────────────────────────────────────────────────────
// v12_transposed's add() called rotate_on_gpu(h_x, n) on the WHOLE n at
// once -- two full n*d float device buffers (raw + JL-rotated). That's
// fine up to a few million rows, but at bge-m3 (10.09M x 1024) or
// stella-trec24 (17.8M x 1024) scale it needs ~83GB / ~145GB of VRAM
// before a single kernel runs, far past any single GPU.
//
// Every stage AFTER rotation, though, already only touches compressed
// per-vector data -- uint8 codes (d_pc, d_rc), scalars (d_co), or ints
// (d_assign) -- which for the whole n is one to two orders of magnitude
// smaller than the raw float vectors (e.g. stella-trec24: d_pc+d_rc+d_co+
// d_assign is ~10.7GB total vs. ~145GB for two full float buffers). So
// the fix isn't a full two-pass IVF-offset rewrite (unlike JHQ_official's
// IndexIVFJHQ::add_core(), which needs one because FAISS's InvertedLists
// storage is filled incrementally list-by-list) -- it's simpler: only the
// rotate+encode+assign step needs to stream, exactly mirroring how
// JHQ_repro's CPU add() chunks its own expensive stage (batch=32768)
// while accumulating into a single set of outputs. Everything from the
// thrust::sort_by_key onward is untouched from v12_transposed: it already
// only allocates n-sized compressed arrays, which comfortably fit (see
// jhq_v24_paper_default/jhq_gpu_index.cuh's Params::add_batch comment for
// the full VRAM accounting).
void JHQGpuIndex::add(const float* h_x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0)
        throw std::runtime_error("v14_streaming_add currently supports one add() call");

    // The phase sums leave a few hundred milliseconds of add() unaccounted for
    // and the tail is only 40 ms of it, so the rest should be here: these are
    // gigabyte-scale allocations and cudaMalloc synchronises.
    auto t_alloc0 = std::chrono::steady_clock::now();
    double t_alloc = 0;

    uint8_t* d_pc = nullptr;      // [n, M]   primary codes
    uint8_t* d_rc = nullptr;      // [n, bpv] residual codes
    float*   d_co = nullptr;      // [n]      residual corrections
    int*     d_assign = nullptr;  // [n]      IVF cluster assignment
    CUDA_CHECK(cudaMalloc(&d_assign, (long long)n * sizeof(int)));

    const long long AB = add_batch_;

    // add() is the part of the build that scales with N: train is capped at
    // 100k sample vectors whatever the dataset, so at 17.8M rows add is 96% of
    // the build. Timing the phases rather than guessing which one costs,
    // reported when JHQ_ADD_PHASES is set.
    const bool add_phases = std::getenv("JHQ_ADD_PHASES") != nullptr;
    double t_h2d = 0, t_rot = 0, t_penc = 0, t_renc = 0, t_asg = 0, t_scatter = 0;
    auto tick = [&]() {
        if (add_phases) CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::steady_clock::now();
    };
    auto lap = [&](std::chrono::steady_clock::time_point& t0, double& acc) {
        if (!add_phases) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };

    // Two-deep pipeline. The loop it replaces did a pageable copy, a device-wide
    // synchronise and a pair of mallocs per batch, so the link sat idle while the
    // GPU worked and vice versa. Two pinned staging buffers, two streams and one
    // hoisted scratch let the host copy for batch i+1 run against the GPU work for
    // batch i.
    //
    // Phase timing forces a synchronise after every step, so JHQ_ADD_PHASES
    // degenerates this to the serial order on purpose: attribution needs the
    // steps separated, throughput needs them overlapped.
    // Two buffers keep one batch of GPU work queued while the host stages the
    // next; a third keeps two, for when the host is slow to fill -- the
    // machine is shared, and a stage that takes longer than the GPU's batch
    // leaves the device idle. Three measured 3% faster than two on the 17.8M
    // set and the same elsewhere (JHQ_ADD_NBUF overrides).
    const char* nbe = std::getenv("JHQ_ADD_NBUF");
    const int NBUF = add_phases ? 1 : std::max(1, std::min(4, nbe ? std::atoi(nbe) : 3));
    float*       h_stage[4] = {nullptr, nullptr, nullptr, nullptr};
    float*       d_xb[4]    = {nullptr, nullptr, nullptr, nullptr};
    float*       d_yb[4]    = {nullptr, nullptr, nullptr, nullptr};
    cudaStream_t st[4]      = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t  ev[4]      = {nullptr, nullptr, nullptr, nullptr};
    // Size the staging by bytes, not by rows. add_batch is a row count, so at
    // d=3072 each buffer would be 805 MB and the pair 1.6 GB of pinned memory;
    // pinning that much costs a few hundred milliseconds, which is most of what
    // the phase timings could not account for. 128 MB a buffer is more than
    // enough to keep the link busy.
    const char* smb = std::getenv("JHQ_STAGE_MB");
    const size_t stage_budget = (size_t)(smb ? std::atoi(smb) : 128) << 20;
    const long long AB_rows = std::max<long long>(
        1, std::min<long long>(AB, (long long)(stage_budget / (sizeof(float) * d_))));
    size_t pipeline_bytes = 0;   // device memory the pipeline holds and frees before the tail
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaHostAlloc(&h_stage[b], (size_t)AB_rows * d_ * sizeof(float),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_xb[b], (long long)AB_rows * d_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_yb[b], (long long)AB_rows * d_ * sizeof(float)));
        pipeline_bytes += 2 * (size_t)AB_rows * d_ * sizeof(float);
        CUDA_CHECK(cudaStreamCreate(&st[b]));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
    }
    // The GEMM form of the primary encode needs somewhere to put y.c for a chunk
    // of subspaces: chunk*AB*K floats. 512 MB of it covers 8 subspaces at
    // AB=65536, K=256, so the batched call runs twelve times per batch rather
    // than ninety-six.
    float* d_cent_sqnorm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cent_sqnorm, (size_t)M_ * K_ * sizeof(float)));
    launch_centroid_sqnorms(d_cent_, d_cent_sqnorm, M_, K_, Ds_);
    const size_t dots_bytes = (size_t)512 << 20;
    const int dots_rows = (int)std::min<size_t>(dots_bytes / (sizeof(float) * K_),
                                                (size_t)M_ * AB_rows);
    float* d_dots_enc[4] = {nullptr, nullptr, nullptr, nullptr};

    // One assignment scratch per stream, not one for the loop. Hoisting it out
    // of the batch loop removes a malloc and a free per batch, but a single
    // buffer is written by both streams at once: the coarse assignment then
    // lands in the wrong list for whichever batch loses, and recall falls from
    // 0.9452 to 0.80. Single-threaded staging hid it -- the copy was slow enough
    // that the previous batch's assignment had finished -- so it only appeared
    // once the staging copy got fast.
    float* d_dots[4] = {nullptr, nullptr, nullptr, nullptr};
    {
        const char* sb = std::getenv("JHQ_ASSIGN_BATCH");
        const int abatch = sb ? std::atoi(sb) : (nlist_ >= 8192 ? 32768 : 8192);
        for (int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaMalloc(&d_dots[b],
                                  (long long)nlist_ * abatch * sizeof(float)));
            pipeline_bytes += (size_t)nlist_ * abatch * sizeof(float);
        }
    }
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaMalloc(&d_dots_enc[b], (size_t)dots_rows * K_ * sizeof(float)));
        pipeline_bytes += (size_t)dots_rows * K_ * sizeof(float);
    }
    __half* d_y16[4] = {nullptr, nullptr, nullptr, nullptr};
    if (assign_precision() == 16)
        for (int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaMalloc(&d_y16[b], (long long)AB_rows * d_ * sizeof(__half)));
            pipeline_bytes += (size_t)AB_rows * d_ * sizeof(__half);
        }
    // The reduced-precision centroid copies, before any slot's stream can ask
    // for them: built lazily on one slot's stream, another slot's product
    // could read them before they were written.
    ensure_assign_centroids(0);
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Dimension-major y, for the encoders' sake. Only the fused Cartesian
    // encoder and the coarse assignment read y in add(), and both are taught
    // the layout below, so this is off unless they are the ones running.
    // Which encoder runs is decided further down from the same conditions
    // repeated here: the generic and GEMM primary encoders take no layout
    // argument and read y row-major, so honouring the request while one of
    // them is running writes a garbage code for every vector -- silently, at
    // full speed, with recall landing at 1e-4. The request is therefore
    // granted only when the encoder that understands the layout is the one
    // that will run.
    const char* ytr = std::getenv("JHQ_Y_TRANSPOSED");
    const char* fu_env = std::getenv("JHQ_ENCODE_FUSED");
    const char* sep_env = std::getenv("JHQ_ENCODE_SEPARABLE");
    const bool sep_ok = d_levels_ && (!sep_env || sep_env[0] == '1');
    const bool fused_will_run = sep_ok && d_res_c1d_
                             && (Br_ == 8 || (Ds_ % 2 == 0))
                             && (!fu_env || fu_env[0] == '1');
    const bool y_transposed = (ytr && ytr[0] == '1') && fused_will_run;
    if (ytr && ytr[0] == '1' && !fused_will_run)
        std::fprintf(stderr,
            "[jhq] JHQ_Y_TRANSPOSED ignored: the encoder that will run here "
            "reads y row-major.\n");

    // The single pass keeps every code beside its list-ordered copy while the
    // lists are gathered, so its peak is twice the code array and change. At
    // Br=8 on 17.8M vectors of 1024 dimensions that is 43 GB against 32 on the
    // card. Past what is free, the build makes two passes over the input
    // instead: the first rotates and assigns, the second rotates and encodes
    // each batch straight into list order. The second pass costs another
    // rotation and another read of the input; the assignment, which is most
    // of the work, is not repeated. JHQ_ADD_TWO_PASS=1|0 forces either.
    // The single pass peaks in the tail, after the pipeline is released, so the
    // pipeline's share is counted back in as free: on the 17.8M set it is 4 GB,
    // and without it the 26 GB Br=4 build was sent the slower way.
    size_t mem_free = 0, mem_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
    const size_t peak_single = (size_t)n * (2 * (size_t)bpv_ + 3 * (size_t)M_ + 5 * sizeof(int))
                             + ((size_t)1 << 30);
    const size_t free_at_tail = mem_free + pipeline_bytes;
    const char* tpe = std::getenv("JHQ_ADD_TWO_PASS");
    const bool two_pass = tpe ? (tpe[0] == '1') : (peak_single > free_at_tail);
    uint8_t* d_pc_b[4] = {nullptr, nullptr, nullptr, nullptr};
    uint8_t* d_rc_b[4] = {nullptr, nullptr, nullptr, nullptr};
    float*   d_co_b[4] = {nullptr, nullptr, nullptr, nullptr};
    int*     d_inv     = nullptr;   // [n] sorted position of each input row
    if (two_pass) {
        std::printf("  [add] two-pass build: single-pass peak %.1f GB, %.1f GB free at the tail\n",
                    peak_single / 1073741824.0, free_at_tail / 1073741824.0);
        for (int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaMalloc(&d_pc_b[b], (size_t)AB_rows * M_));
            CUDA_CHECK(cudaMalloc(&d_rc_b[b], (size_t)AB_rows * bpv_));
            CUDA_CHECK(cudaMalloc(&d_co_b[b], (size_t)AB_rows * sizeof(float)));
        }
    } else {
        CUDA_CHECK(cudaMalloc(&d_pc, (long long)n * M_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_rc, (long long)n * bpv_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_co, (long long)n * sizeof(float)));
    }
    if (add_phases) {
        CUDA_CHECK(cudaDeviceSynchronize());
        t_alloc = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_alloc0).count();
        std::fprintf(stderr, "  [add] device allocations %.1f ms\n", t_alloc);
    }

    const float one = 1.f, zero = 0.f;
    // The first of two passes only assigns, and the assignment does not need
    // the rotated rows: the rotation is orthogonal, so the nearest centroid
    // to Rx is found as well against the rotated-back centroids R^T c, from x
    // as it arrives. That drops the pass's GEMM, 0.6 s of the 17.8M build,
    // and leaves the pass paced by the product and the copy in.
    // JHQ_ASSIGN_XDOMAIN=0 rotates first as before.
    const char* xde = std::getenv("JHQ_ASSIGN_XDOMAIN");
    const bool assign_x = two_pass && (!xde || xde[0] == '1');
    if (assign_x) {
        ensure_assign_centroids(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
    // mode 0: rotate, encode and assign in one pass; 1: assign (in the x
    // domain, or after rotating); 2: rotate, encode, scatter into list order.
    auto run_pass = [&](int mode) {
    int nbatch = 0;
    for (long long start = 0; start < n; start += AB_rows, ++nbatch) {
        const int nb  = (int)std::min(AB_rows, (long long)n - start);
        const int cur = nbatch % NBUF;
        if (nbatch >= NBUF) CUDA_CHECK(cudaEventSynchronize(ev[cur]));
        uint8_t* pc_out = mode == 2 ? d_pc_b[cur] : mode == 0 ? d_pc + start * M_   : nullptr;
        uint8_t* rc_out = mode == 2 ? d_rc_b[cur] : mode == 0 ? d_rc + start * bpv_ : nullptr;
        float*   co_out = mode == 2 ? d_co_b[cur] : mode == 0 ? d_co + start        : nullptr;

        auto t0 = tick();
        // The source is an mmap of the base file, so it cannot be handed to the
        // DMA engine directly and has to pass through the pinned buffer. That
        // copy, single-threaded, was the wall: 12.3 GB at a few GB/s on
        // openai3-3072, more than the GPU work it was supposed to hide behind.
        // Splitting it across the host's cores is the cheapest way to make it
        // small enough to overlap.
        {
            const size_t bytes = (size_t)nb * d_ * sizeof(float);
            const char*  src   = (const char*)(h_x + start * (long long)d_);
            char*        dst   = (char*)h_stage[cur];
#ifdef _OPENMP
            // The host's link is PCIe 5.0 (56 GB/s pinned either way), so the
            // fill is what paces this stage: from the page cache, 8 threads
            // copy at 43 GB/s and 32 at 74 GB/s on the 208-core host. All 208
            // took three times as long as one -- the fork, join and the cores
            // taken from the CUDA driver cost more than the copy they split.
            const char* se = std::getenv("JHQ_STAGE_THREADS");
            const int want = se ? std::atoi(se) : 32;
            const int nthr = std::max(1, std::min(omp_get_max_threads(), want));
#pragma omp parallel num_threads(nthr)
            {
                const int nt = omp_get_num_threads(), ti = omp_get_thread_num();
                const size_t chunk = (bytes + nt - 1) / nt;
                const size_t lo = (size_t)ti * chunk;
                const size_t hi = lo + chunk < bytes ? lo + chunk : bytes;
                if (lo < bytes) std::memcpy(dst + lo, src + lo, hi - lo);
            }
#else
            std::memcpy(dst, src, bytes);
#endif
        }
        CUDA_CHECK(cudaMemcpyAsync(d_xb[cur], h_stage[cur],
                                   (long long)nb * d_ * sizeof(float),
                                   cudaMemcpyHostToDevice, st[cur]));
        lap(t0, t_h2d);

        CUBLAS_CHECK(cublasSetStream(cublas_, st[cur]));
        if (mode == 1 && assign_x) {
            // No rotation: x itself, vector-major as it came, against R^T c.
        } else if (y_transposed) {
            // The encoders read Ds values of one vector per thread, and with
            // vectors contiguous the threads of a warp land d floats apart --
            // every load its own transaction, measured at 378 GB/s of 1792.
            // Storing y dimension-major makes those reads consecutive. The
            // rotation can produce that orientation for free: (Pi X)^T is
            // X^T Pi^T, which is the same GEMM with both operands transposed.
            CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_T,
                                     nb, d_, d_, &one,
                                     d_xb[cur], d_, d_Pi_, d_,
                                     &zero, d_yb[cur], nb));
        } else {
            CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                     d_, nb, d_, &one, d_Pi_, d_,
                                     d_xb[cur], d_, &zero, d_yb[cur], d_));
        }
        lap(t0, t_rot);

        if (mode != 1) {
        // Which form wins depends on the subspace width, and the crossover was
        // measured rather than guessed. The GEMM reduces over Ds, so a narrow
        // subspace makes it a product too thin for cuBLAS to fill; the loop
        // instead holds Ds floats per thread and loses occupancy as Ds grows.
        // Primary encode, milliseconds, loop against GEMM:
        //     Ds=8  (Vogue)          98  vs  247
        //     Ds=16 (OpenAI3-1536)  219  vs  263
        //     Ds=32 (OpenAI3-3072)  439  vs  272
        // JHQ_ENCODE_GEMM forces either for measurement.
        {
            const char* force = std::getenv("JHQ_ENCODE_GEMM");
            const bool use_gemm = force ? (force[0] == '1') : (Ds_ >= 32);
            const char* sep = std::getenv("JHQ_ENCODE_SEPARABLE");
            const bool use_sep = d_levels_ && (!sep || sep[0] == '1');
            const char* fu = std::getenv("JHQ_ENCODE_FUSED");
            const bool use_fused = use_sep && d_res_c1d_ && (Br_ == 8 || (Ds_ % 2 == 0))
                                 && (!fu || fu[0] == '1');
            if (use_fused) {
                // One pass over y for both levels; skips the separate residual
                // encode below.
                launch_encode_fused_cartesian(
                    d_yb[cur], d_levels_, n_levels_, d_res_c1d_,
                    pc_out, rc_out, co_out,
                    nb, d_, M_, Ds_, Kr_, Br_, bpv_, y_transposed ? 1 : 0,
                    st[cur]);
            } else if (use_sep)
                // Ds*L operations against K*Ds; only the product codebook is
                // separable, so this needs the paper's construction.
                launch_primary_encode_cartesian(d_yb[cur], pc_out,
                                                d_levels_, n_levels_,
                                                nb, d_, M_, Ds_, st[cur]);
            else if (use_gemm)
                launch_primary_encode_gemm(cublas_, d_yb[cur], pc_out,
                                           d_cent_, d_cent_sqnorm,
                                           d_dots_enc[cur], dots_rows,
                                           nb, d_, M_, Ds_, K_, st[cur]);
            else
                launch_primary_encode(d_yb[cur], pc_out, d_cent_,
                                      nb, d_, M_, Ds_, K_, st[cur]);
        }
        lap(t0, t_penc);

        {
            const char* fu = std::getenv("JHQ_ENCODE_FUSED");
            const char* sep = std::getenv("JHQ_ENCODE_SEPARABLE");
            const bool sep_on = d_levels_ && (!sep || sep[0] == '1');
            const bool fused = sep_on && d_res_c1d_ && (Br_ == 8 || (Ds_ % 2 == 0))
                             && (!fu || fu[0] == '1');
            if (!fused)
                launch_residual_encode(d_yb[cur], pc_out, rc_out, co_out,
                                       d_cent_, d_res_c1d_,
                                       nb, d_, M_, Ds_, K_, Kr_, Br_, bpv_, st[cur]);
        }
        lap(t0, t_renc);
        }   // mode != 1

        if (mode == 1 && assign_x) {
            assign_into(d_xb[cur], nb, d_assign + start, d_dots[cur],
                        0, st[cur], d_y16[cur], 1);
            lap(t0, t_asg);
        } else if (mode != 2) {
            assign_into(d_yb[cur], nb, d_assign + start, d_dots[cur],
                        y_transposed ? 1 : 0, st[cur], d_y16[cur]);
            lap(t0, t_asg);
        } else {
            scatter_list_storage_kernel<<<(int)(((long long)nb * 32 + 255) / 256), 256, 0, st[cur]>>>(
                d_inv + start, pc_out, rc_out, co_out,
                d_list_primary_t_, d_list_res_, d_list_corr_,
                nb, (long long)n, M_, bpv_);
            CUDA_CHECK(cudaGetLastError());
            lap(t0, t_scatter);
        }
        CUDA_CHECK(cudaEventRecord(ev[cur], st[cur]));
    }
    for (int b = 0; b < NBUF; ++b) CUDA_CHECK(cudaStreamSynchronize(st[b]));
    CUBLAS_CHECK(cublasSetStream(cublas_, 0));
    };  // run_pass
    run_pass(two_pass ? 1 : 0);

    if (add_phases)
        std::fprintf(stderr,
            "  [add] h2d %.1f ms | rotate %.1f | primary encode %.1f | "
            "residual encode %.1f | ivf assign %.1f\n",
            t_h2d, t_rot, t_penc, t_renc, t_asg);

    for (int b = 0; b < NBUF; ++b) {
        cudaFree(d_dots[b]); d_dots[b] = nullptr;
        if (d_y16[b]) { cudaFree(d_y16[b]); d_y16[b] = nullptr; }
    }
    auto free_pipeline = [&]() {
        cudaFree(d_cent_sqnorm);
        for (int b = 0; b < NBUF; ++b) {
            cudaFree(d_dots_enc[b]);
            if (d_pc_b[b]) { cudaFree(d_pc_b[b]); cudaFree(d_rc_b[b]); cudaFree(d_co_b[b]); }
            cudaFreeHost(h_stage[b]); cudaFree(d_xb[b]); cudaFree(d_yb[b]);
            cudaStreamDestroy(st[b]); cudaEventDestroy(ev[b]);
        }
    };
    if (!two_pass) free_pipeline();

    // The tail of add() has never been timed: it is a device sort, a copy of the
    // whole assignment array back to the host, a single-threaded count over it,
    // then two more kernels. At 17.8M rows that copy is 71 MB and that loop is
    // 17.8M iterations.
    double t_sort = 0, t_count = 0, t_gather = 0, t_transpose = 0;
    auto tail = std::chrono::steady_clock::now();
    auto tail_lap = [&](double& acc) {
        if (!add_phases) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - tail).count();
        tail = now;
    };

    int* d_order = nullptr;
    CUDA_CHECK(cudaMalloc(&d_order, (long long)n * sizeof(int)));
    thrust::device_ptr<int> t_assign(d_assign);
    thrust::device_ptr<int> t_order(d_order);
    thrust::sequence(t_order, t_order + n);
    thrust::sort_by_key(t_assign, t_assign + n, t_order);
    tail_lap(t_sort);

    // The assignments are sorted, so list c starts at the first position whose
    // key is >= c: one lower_bound over the keys for each of the nlist + 1
    // boundaries, in place of copying the n keys back and counting them on one
    // host thread (72 ms of the stella-trec24 tail).
    CUDA_CHECK(cudaMalloc(&d_list_offsets_, (long long)(nlist_ + 1) * sizeof(int)));
    if (two_pass) {
        // The sorted order is the id list itself.
        d_list_ids_ = d_order; d_order = nullptr;
    } else {
        CUDA_CHECK(cudaMalloc(&d_list_ids_, (long long)n * sizeof(int)));
    }
    CUDA_CHECK(cudaMalloc(&d_list_res_,     (long long)n * bpv_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_list_corr_,    (long long)n * sizeof(float)));
    {
        thrust::counting_iterator<int> first(0);
        thrust::lower_bound(t_assign, t_assign + n, first, first + nlist_ + 1,
                            thrust::device_ptr<int>(d_list_offsets_));
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaMemcpy(&lo, d_assign, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hi, d_assign + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        if (lo < 0 || hi >= nlist_) throw std::runtime_error("invalid IVF assignment");
    }
    tail_lap(t_count);

    CUDA_CHECK(cudaMalloc(&d_list_primary_t_, (long long)M_ * n * sizeof(uint8_t)));
    if (two_pass) {
        cudaFree(d_assign); d_assign = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inv, (long long)n * sizeof(int)));
        invert_permutation_kernel<<<(int)(((long long)n + 255) / 256), 256>>>(d_list_ids_, d_inv, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        t_h2d = t_rot = t_penc = t_renc = 0;
        run_pass(2);
        if (add_phases)
            std::fprintf(stderr,
                "  [add pass 2] h2d %.1f ms | rotate %.1f | primary encode %.1f | "
                "residual encode %.1f | scatter %.1f\n",
                t_h2d, t_rot, t_penc, t_renc, t_scatter);
        free_pipeline();
        cudaFree(d_inv);
        tail_lap(t_gather);
    } else {
        // Temporary [N, M] primary buffer for gathering, then transposed.
        uint8_t* d_list_primary_nm = nullptr;
        CUDA_CHECK(cudaMalloc(&d_list_primary_nm, (long long)n * M_ * sizeof(uint8_t)));

        gather_list_storage_kernel<<<(int)(((long long)n * 32 + 255) / 256), 256>>>(
            d_order, d_pc, d_rc, d_co,
            d_list_ids_, d_list_primary_nm, d_list_res_, d_list_corr_,
            n, M_, bpv_);
        tail_lap(t_gather);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Transpose [N, M] → [M, N] for coalesced scan access.
        // grid.x = N-tiles, grid.y = M-tiles -- see transpose_uint8_kernel's
        // comment for why N (which can be in the millions) must be on grid.x,
        // not grid.y (CUDA's 65535 cap on grid.y/z).
        constexpr int TILE = 32;
        dim3 grid(((long long)n + TILE - 1) / TILE, (M_ + TILE - 1) / TILE);
        dim3 block(TILE, TILE);
        transpose_uint8_kernel<TILE><<<grid, block>>>(d_list_primary_nm, d_list_primary_t_,
                                                       (long long)n, M_);
        tail_lap(t_transpose);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free temporary [N, M] buffer — scan uses d_list_primary_t_ only.
        cudaFree(d_list_primary_nm);
        cudaFree(d_assign);
        cudaFree(d_order);
        cudaFree(d_pc);
        cudaFree(d_rc);
        cudaFree(d_co);
    }
    if (add_phases)
        std::fprintf(stderr,
            "  [add-tail] sort %.1f ms | offsets %.1f | "
            "%s %.1f | transpose %.1f\n",
            t_sort, t_count, two_pass ? "second pass" : "gather", t_gather, t_transpose);

    ntotal_ = n;
    alloc_workspace(batch_size_);
}

// ── Search ────────────────────────────────────────────────────────────────────
void JHQGpuIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_labels) const {
    if (ntotal_ == 0) throw std::runtime_error("index is empty");

    search_gpu(cublas_,
               d_Pi_, d_cent_, d_res_c1d_,
               d_centroids_, d_cent_norms_,
               d_list_offsets_, d_list_ids_,
               d_list_primary_t_, d_list_res_, d_list_corr_,
               h_q,
               nq, d_, M_, Ds_, K_, Kr_, nlist_, nprobe_,
               Br_, bpv_,
               alpha_, k,
               batch_size_,
               ntotal_,
               ws_,
               h_dists, h_labels);
}

} // namespace jhq_gpu
