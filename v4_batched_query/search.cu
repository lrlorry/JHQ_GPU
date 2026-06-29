#include "v4_batched_query/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace jhq_gpu {

// ── IEEE-754 bit tricks ───────────────────────────────────────────────────────
// Maps any finite/inf float to a uint32 whose ordering matches float ordering:
//   positive floats: set sign bit  (bits | 0x80000000)
//   negative floats: invert all    (~bits)
// Self-inverse is NOT the case; use from_sortable() to recover the float.

__device__ __host__ __forceinline__
uint32_t to_sortable(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __host__ __forceinline__
float from_sortable(uint32_t s) {
    uint32_t b = (s & 0x80000000u) ? (s & 0x7FFFFFFFu) : ~s;
    return __uint_as_float(b);
}

// ── composite key helpers ─────────────────────────────────────────────────────
// key = (uint64)(bqi) << 32 | to_sortable(dist)
// Global sort on uint64 → primary order by bqi, secondary by dist ascending.
// After sort, query bqi's segment is at [bqi*cap, (bqi+1)*cap) since there
// are exactly cap elements tagged with each bqi value.

// ── Kernel: init candidate buffers to sentinel (+inf / -1) ───────────────────
__global__ void init_cand_kernel(
    float*   cand_dist,
    int*     cand_pos,
    long long total)        // B * cap_per_query
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    cand_dist[i] = __int_as_float(0x7F800000);  // +inf
    cand_pos[i]  = -1;
}

// ── Kernel: probe selection ───────────────────────────────────────────────────
// One block per query.  Shared mem = nlist floats (centroid distances) +
// two arrays of 256 floats/ints for the parallel reduction.
// Finds top-nprobe closest centroids via iterative block-wide argmin.
__global__ void select_probes_kernel(
    const float* __restrict__ dots,          // col-major [nlist, B]: dots[c + bqi*nlist]
    const float* __restrict__ cent_norms,    // [nlist]
    const int*   __restrict__ list_offsets,  // [nlist+1]
    int*                      probe_ids,     // [B, nprobe]
    int*                      probe_offsets, // [B, nprobe+1]
    int*                      query_total,   // [B]
    int nlist, int nprobe)
{
    extern __shared__ float s[];  // nlist floats (s_dist), then static portions below

    // Static-size shared arrays for block reduction (256 threads max)
    __shared__ float red_val[256];
    __shared__ int   red_idx[256];
    __shared__ int   sel_idx_s;

    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    // Load centroid distances for this query into shared memory.
    const float* row = dots + (long long)bqi * nlist;  // col-major layout: row for bqi
    for (int c = tid; c < nlist; c += blockDim.x)
        s[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();

    // Iteratively extract the nprobe smallest entries.
    int my_probe_base = bqi * nprobe;
    int my_off_base   = bqi * (nprobe + 1);
    int acc = 0;
    if (tid == 0) probe_offsets[my_off_base] = 0;

    for (int p = 0; p < nprobe; ++p) {
        // Block-wide argmin over s[0..nlist)
        float best_val = __int_as_float(0x7F800000);  // +inf
        int   best_idx = -1;
        for (int c = tid; c < nlist; c += blockDim.x) {
            if (s[c] < best_val) { best_val = s[c]; best_idx = c; }
        }
        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && red_val[tid + stride] < red_val[tid]) {
                red_val[tid] = red_val[tid + stride];
                red_idx[tid] = red_idx[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            int winner = red_idx[0];
            probe_ids[my_probe_base + p] = winner;
            s[winner] = __int_as_float(0x7F800000);  // exclude from next round
            sel_idx_s = winner;
            int sz = list_offsets[winner + 1] - list_offsets[winner];
            acc += sz;
            probe_offsets[my_off_base + p + 1] = acc;
        }
        __syncthreads();
    }

    if (tid == 0)
        query_total[bqi] = acc;
}

// ── Kernel: batched primary LUT ───────────────────────────────────────────────
// grid-strided over B * M * Ds * K1D
__global__ void build_primary_lut_batched_kernel(
    const float* __restrict__ d_q_rot,  // [B, d]
    const float* __restrict__ d_c1d,
    float*                    d_lut,    // [B, M*Ds*K1D]
    int B, int d, int M, int Ds, int K1D)
{
    int lut_size = M * Ds * K1D;
    long long total = (long long)B * lut_size;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / lut_size);
        int local = (int)(i % lut_size);
        int j  = local % K1D;
        int mk = local / K1D;
        int k  = mk % Ds;
        int m  = mk / Ds;
        float qr = d_q_rot[(long long)bqi * d + m * Ds + k];
        float diff = qr - d_c1d[j];
        d_lut[i] = diff * diff;
    }
}

// ── Kernel: batched IVF primary scan ─────────────────────────────────────────
// One block per query.  Each block block-strides over its query_total candidates.
// LUT cached in shared memory (same trick as v3, just per-block now).
__global__ void scan_ivf_batched_kernel(
    const float*   __restrict__ d_lut,           // [B, M*Ds*K1D]
    const int*     __restrict__ probe_ids,       // [B, nprobe]
    const int*     __restrict__ probe_offsets,   // [B, nprobe+1]
    const int*     __restrict__ list_offsets,    // [nlist+1]
    const uint8_t* __restrict__ list_primary,    // [N, M]
    const int*     __restrict__ query_total,     // [B]
    float*                      cand_dist,       // [B, cap_per_query]
    int*                        cand_pos,        // [B, cap_per_query]
    long long cap_per_query,
    int nprobe, int M, int Ds, int K1D, int bits_per_dim)
{
    extern __shared__ float s_lut[];  // M*Ds*K1D floats

    int bqi = blockIdx.x;
    int tid = threadIdx.x;
    int lut_size = M * Ds * K1D;

    // Load this query's LUT slice into shared memory.
    const float* my_lut = d_lut + (long long)bqi * lut_size;
    for (int i = tid; i < lut_size; i += blockDim.x)
        s_lut[i] = my_lut[i];
    __syncthreads();

    int   total   = query_total[bqi];
    const int*   my_probe_ids  = probe_ids     + bqi * nprobe;
    const int*   my_probe_offs = probe_offsets + bqi * (nprobe + 1);
    float*       my_dist       = cand_dist + bqi * cap_per_query;
    int*         my_pos        = cand_pos  + bqi * cap_per_query;

    const int kmask = K1D - 1;

    for (int local_t = tid; local_t < total; local_t += blockDim.x) {
        // Map local_t → (probe slot p, offset within list, abs_pos in storage)
        int p = 0;
        while (p + 1 < nprobe && local_t >= my_probe_offs[p + 1]) ++p;
        int list_id = my_probe_ids[p];
        int local   = local_t - my_probe_offs[p];
        int abs_pos = list_offsets[list_id] + local;

        const uint8_t* code = list_primary + (long long)abs_pos * M;
        float dist = 0.0f;
        for (int m = 0; m < M; ++m) {
            uint8_t cm = code[m];
            const float* lm = s_lut + m * Ds * K1D;
            for (int k = 0; k < Ds; ++k)
                dist += lm[k * K1D + ((cm >> (k * bits_per_dim)) & kmask)];
        }
        my_dist[local_t] = dist;
        my_pos[local_t]  = abs_pos;
    }
}

// ── Kernel: pack (bqi, cand_dist) → composite uint64 key ─────────────────────
// qi = i / cap_per_query  (works because each query has exactly cap_per_query
// entries, padding ones have dist=+inf from init_cand_kernel)
__global__ void pack_keys_kernel(
    const float* __restrict__ cand_dist,
    uint64_t*                 keys,
    long long total,           // B * cap_per_query
    long long cap_per_query)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        uint32_t bqi = (uint32_t)(i / cap_per_query);
        uint32_t s   = to_sortable(cand_dist[i]);
        keys[i] = ((uint64_t)bqi << 32) | (uint64_t)s;
    }
}

// ── Kernel: gather top-ck candidates per query ────────────────────────────────
// After the global sort, query bqi's top-ck entries are at
// [bqi*cap_per_query, bqi*cap_per_query+ck) in the sorted arrays.
__global__ void gather_topck_kernel(
    const uint64_t* __restrict__ sorted_keys,
    const int*      __restrict__ sorted_pos,
    float*                       topck_primary,  // [B, ck]
    int*                         topck_pos,      // [B, ck]
    long long cap_per_query,
    int ck, int B)
{
    long long total = (long long)B * ck;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi  = (int)(i / ck);
        int rank = (int)(i % ck);
        long long src = (long long)bqi * cap_per_query + rank;
        topck_primary[(long long)bqi * ck + rank] = from_sortable((uint32_t)(sorted_keys[src] & 0xFFFFFFFFULL));
        topck_pos    [(long long)bqi * ck + rank] = sorted_pos[src];
    }
}

// ── Kernel: batched residual LUT ──────────────────────────────────────────────
__global__ void build_residual_lut_batched_kernel(
    const float* __restrict__ d_q_rot,  // [B, d]
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,  // [B, d*Kr]
    int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        int j     = local % Kr;
        int dim   = local / Kr;
        float diff = d_q_rot[(long long)bqi * d + dim] - d_res_c1d[j];
        d_lut_r[i] = diff * diff;
    }
}

// ── Kernel: batched residual refinement ──────────────────────────────────────
__global__ void residual_refine_batched_kernel(
    const int*     __restrict__ topck_pos,     // [B, ck]
    const float*   __restrict__ topck_primary, // [B, ck]
    const float*   __restrict__ lut_r,         // [B, d*Kr]
    const uint8_t* __restrict__ list_res,      // [N, bpv]
    const float*   __restrict__ list_corr,     // [N]
    float*                      comp_dists,    // [B, ck]
    int ck, int d, int Kr, int Br, int bpv, int B)
{
    long long total = (long long)B * ck;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi  = (int)(i / ck);
        int rank = (int)(i % ck);
        int pos  = topck_pos[i];

        if (pos < 0) {
            comp_dists[i] = __int_as_float(0x7F800000);  // +inf
            continue;
        }

        const float* my_lut_r = lut_r + (long long)bqi * d * Kr;
        const uint8_t* rc = list_res + (long long)pos * bpv;
        float d_res = 0.0f;
        for (int j = 0; j < d; ++j) {
            int ri = (Br == 4)
                ? ((j % 2 == 0) ? (rc[j / 2] & 0x0F) : (rc[j / 2] >> 4))
                : rc[j];
            d_res += my_lut_r[(long long)j * Kr + ri];
        }
        comp_dists[i] = topck_primary[i] + d_res + list_corr[pos];
    }
}

// ── Kernel: pack (bqi, comp_dist) → composite uint64 key (second sort) ───────
__global__ void pack_keys2_kernel(
    const float* __restrict__ comp_dists,
    uint64_t*                 keys2,
    long long total,    // B * ck
    int ck)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        uint32_t bqi = (uint32_t)((int)(i / ck));
        uint32_t s   = to_sortable(comp_dists[i]);
        keys2[i] = ((uint64_t)bqi << 32) | (uint64_t)s;
    }
}

// ── Kernel: gather final k ids per query ─────────────────────────────────────
// After second sort, query bqi's best-k entries are at [bqi*ck, bqi*ck+k).
__global__ void gather_final_kernel(
    const uint64_t* __restrict__ sorted_keys2,
    const int*      __restrict__ sorted_pos,   // top-ck abs_pos, now sorted by comp_dist
    const int*      __restrict__ list_ids,     // [N] original vector id
    float*                       final_dists,  // [B, k_cap]
    int*                         final_ids,    // [B, k_cap]
    int ck, int k, int B)
{
    long long total = (long long)B * k;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi  = (int)(i / k);
        int rank = (int)(i % k);
        long long src = (long long)bqi * ck + rank;
        int pos = sorted_pos[src];
        if (pos < 0) {
            final_ids  [(long long)bqi * k + rank] = -1;
            final_dists[(long long)bqi * k + rank] = __int_as_float(0x7F800000);
        } else {
            final_ids  [(long long)bqi * k + rank] = list_ids[pos];
            final_dists[(long long)bqi * k + rank] = from_sortable((uint32_t)(sorted_keys2[src] & 0xFFFFFFFFULL));
        }
    }
}

// ── Helper: (re)allocate ck/k-dependent workspace buffers ────────────────────
static void realloc_ck_buffers(SearchWorkspace& ws, int batch_cap, int ck, int k) {
    if (ck > ws.ck_cap) {
        cudaFree(ws.d_topck_pos);      ws.d_topck_pos = nullptr;
        cudaFree(ws.d_topck_primary);  ws.d_topck_primary = nullptr;
        cudaFree(ws.d_comp_dists);     ws.d_comp_dists = nullptr;
        cudaFree(ws.d_keys2);          ws.d_keys2 = nullptr;
        long long n = (long long)batch_cap * ck;
        CUDA_CHECK(cudaMalloc(&ws.d_topck_pos,     n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_topck_primary, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_comp_dists,    n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_keys2,         n * sizeof(uint64_t)));
        ws.ck_cap = ck;
    }
    if (k > ws.k_cap) {
        cudaFree(ws.d_final_ids);    ws.d_final_ids = nullptr;
        cudaFree(ws.d_final_dists);  ws.d_final_dists = nullptr;
        long long n = (long long)batch_cap * k;
        CUDA_CHECK(cudaMalloc(&ws.d_final_ids,   n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_final_dists, n * sizeof(float)));
        ws.k_cap = k;
    }
}

// ── search_gpu ────────────────────────────────────────────────────────────────
void search_gpu(
    cublasHandle_t        cublas,
    const float*          d_Pi,
    const float*          d_c1d,
    const float*          d_res_c1d,
    const float*          d_centroids,
    const float*          d_cent_norms,
    const int*            d_list_offsets,
    const int*            d_list_ids,
    const uint8_t*        d_list_primary,
    const uint8_t*        d_list_res,
    const float*          d_list_corr,
    const float*          d_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids)
{
    if (ws.batch_cap <= 0 || ws.cap_per_query <= 0)
        throw std::runtime_error("v4: workspace not initialised — call add() first");

    int ck = std::min(
                 std::max(k, (int)std::ceil(alpha * (float)k)),
                 (int)ws.cap_per_query);

    realloc_ck_buffers(ws, ws.batch_cap, ck, k);

    const float one = 1.0f, zero = 0.0f;
    const int   BLOCK = 256;
    const int   lut_size = M * Ds * K1D;

    thrust::device_ptr<uint64_t> t_keys (ws.d_keys);
    thrust::device_ptr<int>      t_pos  (ws.d_cand_pos);
    thrust::device_ptr<uint64_t> t_keys2(ws.d_keys2);
    thrust::device_ptr<int>      t_pos2 (ws.d_topck_pos);

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int B = std::min(batch_size, nq - qoff);
        const float* d_q_chunk = d_queries + (long long)qoff * d;

        // ── 1. Rotate batch of B queries ─────────────────────────────────────
        // Q_rot[d, B] = Pi[d,d] * Q[d, B]   (col-major [d,B] ≡ row-major [B,d])
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d, B, d,
                                 &one, d_Pi, d,
                                       d_q_chunk, d,
                                 &zero, ws.d_q_rot, d));

        // ── 2. Centroid dot products: Dots[nlist, B] = Centroids[nlist,d]^T * Q_rot[d,B]
        // (col-major: result d_dots[c + bqi*nlist] = dot(c, q_bqi))
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 nlist, B, d,
                                 &one, d_centroids, d,
                                       ws.d_q_rot, d,
                                 &zero, ws.d_dots, nlist));

        // ── 3. Select nprobe probes per query; compute prefix offsets on GPU ─
        {
            int smem = nlist * (int)sizeof(float);
            select_probes_kernel<<<B, BLOCK, smem>>>(
                ws.d_dots, d_cent_norms, d_list_offsets,
                ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total,
                nlist, nprobe);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 4. Init candidate buffers to +inf / -1 ───────────────────────────
        {
            long long total = (long long)B * ws.cap_per_query;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            init_cand_kernel<<<grid, BLOCK>>>(ws.d_cand_dist, ws.d_cand_pos, total);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 5. Build primary LUTs for all B queries ──────────────────────────
        {
            long long total = (long long)B * lut_size;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            build_primary_lut_batched_kernel<<<grid, BLOCK>>>(
                ws.d_q_rot, d_c1d, ws.d_lut, B, d, M, Ds, K1D);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 6. IVF primary scan for all B queries (one block per query) ──────
        {
            int smem = lut_size * (int)sizeof(float);
            scan_ivf_batched_kernel<<<B, BLOCK, smem>>>(
                ws.d_lut, ws.d_probe_ids, ws.d_probe_offsets,
                d_list_offsets, d_list_primary, ws.d_query_total,
                ws.d_cand_dist, ws.d_cand_pos,
                ws.cap_per_query, nprobe, M, Ds, K1D, bits_per_dim);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 7. Pack keys and sort all B queries' candidates in one shot ───────
        {
            long long total = (long long)B * ws.cap_per_query;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            pack_keys_kernel<<<grid, BLOCK>>>(
                ws.d_cand_dist, ws.d_keys, total, ws.cap_per_query);
            CUDA_CHECK(cudaGetLastError());
            thrust::sort_by_key(t_keys, t_keys + total, t_pos);
        }

        // ── 8. Gather top-ck per query into compact buffers ──────────────────
        {
            long long total = (long long)B * ck;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            gather_topck_kernel<<<grid, BLOCK>>>(
                ws.d_keys, ws.d_cand_pos,
                ws.d_topck_primary, ws.d_topck_pos,
                ws.cap_per_query, ck, B);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 9. Build residual LUTs for all B queries ─────────────────────────
        {
            long long total = (long long)B * d * Kr;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            build_residual_lut_batched_kernel<<<grid, BLOCK>>>(
                ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 10. Residual refinement on top-ck candidates ─────────────────────
        {
            long long total = (long long)B * ck;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            residual_refine_batched_kernel<<<grid, BLOCK>>>(
                ws.d_topck_pos, ws.d_topck_primary,
                ws.d_lut_r, d_list_res, d_list_corr,
                ws.d_comp_dists, ck, d, Kr, Br, bpv, B);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 11. Pack & sort composite distances for final top-k ───────────────
        {
            long long total = (long long)B * ck;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            pack_keys2_kernel<<<grid, BLOCK>>>(
                ws.d_comp_dists, ws.d_keys2, total, ck);
            CUDA_CHECK(cudaGetLastError());
            thrust::sort_by_key(t_keys2, t_keys2 + total, t_pos2);
        }

        // ── 12. Gather final k ids + distances per query ─────────────────────
        {
            long long total = (long long)B * k;
            int grid = (int)std::min((total + BLOCK - 1) / BLOCK, (long long)65535);
            gather_final_kernel<<<grid, BLOCK>>>(
                ws.d_keys2, ws.d_topck_pos,
                d_list_ids,
                ws.d_final_dists, ws.d_final_ids,
                ck, k, B);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── 13. Copy this chunk's results back to host (one sync per chunk) ──
        CUDA_CHECK(cudaMemcpy(h_out_ids   + (long long)qoff * k,
                              ws.d_final_ids,
                              (long long)B * k * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_dists + (long long)qoff * k,
                              ws.d_final_dists,
                              (long long)B * k * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace jhq_gpu
