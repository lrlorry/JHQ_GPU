#include "v4_batched_query/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace jhq_gpu {

__device__ __forceinline__ uint32_t to_sortable(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float from_sortable(uint32_t s) {
    uint32_t b = (s & 0x80000000u) ? (s & 0x7FFFFFFFu) : ~s;
    return __uint_as_float(b);
}

// One block per query. All threads cooperatively load centroid distances into
// shared mem; then ONLY thread 0 does sequential argmin (simple, correct).
__global__ void select_probes_kernel(
    const float* __restrict__ dots,         // [B, nlist] col-major: dots[bqi*nlist + c]
    const float* __restrict__ cent_norms,   // [nlist]
    const int*   __restrict__ list_offsets, // [nlist+1]
    int*                      probe_ids,    // [B, nprobe]
    int*                      probe_offsets,// [B, nprobe+1]
    int*                      query_total,  // [B]
    int nlist, int nprobe)
{
    extern __shared__ float s[];  // [nlist]
    int bqi  = blockIdx.x;
    int tid  = threadIdx.x;
    const int BLOCK = blockDim.x;

    const float* row = dots + (long long)bqi * nlist;
    for (int c = tid; c < nlist; c += BLOCK)
        s[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();

    if (tid == 0) {
        int*  my_ids  = probe_ids     + bqi * nprobe;
        int*  my_offs = probe_offsets + bqi * (nprobe + 1);
        int   acc     = 0;
        my_offs[0] = 0;
        for (int p = 0; p < nprobe; ++p) {
            float best = __int_as_float(0x7F800000);
            int   bidx = 0;
            for (int c = 0; c < nlist; ++c)
                if (s[c] < best) { best = s[c]; bidx = c; }
            s[bidx] = __int_as_float(0x7F800000);
            my_ids[p]  = bidx;
            acc       += list_offsets[bidx + 1] - list_offsets[bidx];
            my_offs[p + 1] = acc;
        }
        query_total[bqi] = acc;
    }
}

__global__ void build_primary_lut_batched_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_c1d,
    float*                    d_lut,
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
        float diff = d_q_rot[(long long)bqi * d + m * Ds + k] - d_c1d[j];
        d_lut[i] = diff * diff;
    }
}

// One block per query. Scan all probed vectors keeping per-thread top-K_LOCAL
// candidates in registers (insertion sort), then gather into shared mem and
// run ck rounds of block-wide argmin to select top-ck directly.
// Eliminates B*cap_per_query global buffer and its thrust sort entirely.
__global__ void scan_ivf_batched_topk_kernel(
    const float*   __restrict__ d_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary,
    const int*     __restrict__ query_total,
    float*                      topck_primary,  // [B, ck]
    int*                        topck_pos,      // [B, ck]
    int nprobe, int M, int Ds, int K1D, int bits_per_dim, int ck, int lut_size)
{
    constexpr int K_LOCAL = 4;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;  // 256
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    // Shared mem layout:
    // [s_lut: lut_size f][s_cdist: K_LOCAL*BLOCK f][s_cpos: K_LOCAL*BLOCK i]
    // [s_red_val: BLOCK f][s_red_idx: BLOCK i]
    // Total = (lut_size + 2*K_LOCAL*BLOCK + 2*BLOCK) * 4 bytes
    extern __shared__ char shm[];
    float* s_lut     = (float*)shm;
    float* s_cdist   = s_lut + lut_size;
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    // Phase 1: cooperative LUT load.
    const float* my_lut = d_lut + (long long)bqi * lut_size;
    for (int i = tid; i < lut_size; i += BLOCK)
        s_lut[i] = my_lut[i];
    __syncthreads();

    // Phase 2: scan with local top-K_LOCAL in registers.
    int total = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);
    const int  kmask   = K1D - 1;

    float ld[K_LOCAL];
    int   lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    for (int local_t = tid; local_t < total; local_t += BLOCK) {
        int p = 0;
        while (p + 1 < nprobe && local_t >= my_offs[p + 1]) ++p;
        int abs_pos = list_offsets[my_ids[p]] + (local_t - my_offs[p]);

        const uint8_t* code = list_primary + (long long)abs_pos * M;
        float dist = 0.0f;
        for (int m = 0; m < M; ++m) {
            uint8_t cm = code[m];
            const float* lm = s_lut + m * Ds * K1D;
            for (int k = 0; k < Ds; ++k)
                dist += lm[k * K1D + ((cm >> (k * bits_per_dim)) & kmask)];
        }

        if (dist < ld[K_LOCAL - 1]) {
            ld[K_LOCAL - 1] = dist;
            lp[K_LOCAL - 1] = abs_pos;
            #pragma unroll
            for (int i = K_LOCAL - 1; i > 0 && ld[i] < ld[i-1]; --i) {
                float td = ld[i-1]; ld[i-1] = ld[i]; ld[i] = td;
                int   tp = lp[i-1]; lp[i-1] = lp[i]; lp[i] = tp;
            }
        }
    }

    // Phase 3: write local candidates to shared mem.
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) {
        s_cdist[tid * K_LOCAL + i] = ld[i];
        s_cpos [tid * K_LOCAL + i] = lp[i];
    }
    __syncthreads();

    // Phase 4: ck rounds of block-wide argmin → write top-ck to output.
    const int n_cands = K_LOCAL * BLOCK;
    float* out_primary = topck_primary + (long long)bqi * ck;
    int*   out_pos     = topck_pos     + (long long)bqi * ck;

    for (int c = 0; c < ck; ++c) {
        float bv = INF;
        int   bi = -1;
        for (int ci = tid; ci < n_cands; ci += BLOCK)
            if (s_cdist[ci] < bv) { bv = s_cdist[ci]; bi = ci; }
        s_red_val[tid] = bv;
        s_red_idx[tid] = bi;
        __syncthreads();

        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_red_val[tid + stride] < s_red_val[tid]) {
                s_red_val[tid] = s_red_val[tid + stride];
                s_red_idx[tid] = s_red_idx[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            int w = s_red_idx[0];
            out_primary[c] = (w >= 0) ? s_cdist[w] : INF;
            out_pos    [c] = (w >= 0) ? s_cpos [w] : -1;
            if (w >= 0) s_cdist[w] = INF;
        }
        __syncthreads();
    }
}

__global__ void build_residual_lut_batched_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
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

__global__ void residual_refine_batched_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ lut_r,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Kr, int Br, int bpv, int B)
{
    long long total = (long long)B * ck;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int pos = topck_pos[i];
        if (pos < 0) {
            comp_dists[i] = __int_as_float(0x7F800000);
            continue;
        }
        int bqi = (int)(i / ck);
        const float*   my_lut_r = lut_r + (long long)bqi * d * Kr;
        const uint8_t* rc       = list_res + (long long)pos * bpv;
        float d_res = 0.0f;
        for (int j = 0; j < d; ++j) {
            int ri = (Br == 4)
                ? ((j % 2 == 0) ? (rc[j/2] & 0x0F) : (rc[j/2] >> 4))
                : rc[j];
            d_res += my_lut_r[(long long)j * Kr + ri];
        }
        comp_dists[i] = topck_primary[i] + d_res + list_corr[pos];
    }
}

__global__ void pack_keys2_kernel(
    const float* __restrict__ comp_dists,
    uint64_t*                 keys2,
    long long total, int ck)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        uint32_t bqi = (uint32_t)(i / ck);
        keys2[i] = ((uint64_t)bqi << 32) | (uint64_t)to_sortable(comp_dists[i]);
    }
}

__global__ void gather_final_kernel(
    const uint64_t* __restrict__ sorted_keys2,
    const int*      __restrict__ sorted_pos,
    const int*      __restrict__ list_ids,
    float*                       final_dists,
    int*                         final_ids,
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
            final_ids  [i] = -1;
            final_dists[i] = __int_as_float(0x7F800000);
        } else {
            final_ids  [i] = list_ids[pos];
            final_dists[i] = from_sortable((uint32_t)(sorted_keys2[src] & 0xFFFFFFFFULL));
        }
    }
}

static void realloc_ck_buffers(SearchWorkspace& ws, int batch_cap, int ck, int k) {
    if (ck > ws.ck_cap) {
        cudaFree(ws.d_topck_pos);      ws.d_topck_pos     = nullptr;
        cudaFree(ws.d_topck_primary);  ws.d_topck_primary = nullptr;
        cudaFree(ws.d_comp_dists);     ws.d_comp_dists    = nullptr;
        cudaFree(ws.d_keys2);          ws.d_keys2         = nullptr;
        long long n = (long long)batch_cap * ck;
        CUDA_CHECK(cudaMalloc(&ws.d_topck_pos,     n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_topck_primary, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_comp_dists,    n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_keys2,         n * sizeof(uint64_t)));
        ws.ck_cap = ck;
    }
    if (k > ws.k_cap) {
        cudaFree(ws.d_final_ids);    ws.d_final_ids   = nullptr;
        cudaFree(ws.d_final_dists);  ws.d_final_dists = nullptr;
        long long n = (long long)batch_cap * k;
        CUDA_CHECK(cudaMalloc(&ws.d_final_ids,   n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_final_dists, n * sizeof(float)));
        ws.k_cap = k;
    }
}

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
    if (ws.batch_cap <= 0)
        throw std::runtime_error("v4: workspace not initialised — call add() first");

    int ck = std::max(k, (int)std::ceil(alpha * (float)k));
    realloc_ck_buffers(ws, ws.batch_cap, ck, k);

    const float one = 1.0f, zero = 0.0f;
    const int   BLOCK    = 256;
    const int   lut_size = M * Ds * K1D;
    // shm = [lut_size + 2*K_LOCAL(4)*BLOCK + 2*BLOCK] * 4 bytes
    const int   scan_smem = (lut_size + 10 * BLOCK) * (int)sizeof(float);

    thrust::device_ptr<uint64_t> t_keys2(ws.d_keys2);
    thrust::device_ptr<int>      t_pos2 (ws.d_topck_pos);

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int B = std::min(batch_size, nq - qoff);
        const float* d_q_chunk = d_queries + (long long)qoff * d;

        // 1. Rotate queries: Q_rot[d,B] = Pi[d,d] * Q[d,B]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d, B, d,
                                 &one, d_Pi, d,
                                       d_q_chunk, d,
                                 &zero, ws.d_q_rot, d));

        // 2. Centroid dots: Dots[nlist,B] = Centroids^T * Q_rot
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 nlist, B, d,
                                 &one, d_centroids, d,
                                       ws.d_q_rot, d,
                                 &zero, ws.d_dots, nlist));

        // 3. Select nprobe probes per query (thread-0-sequential, definitely correct)
        select_probes_kernel<<<B, BLOCK, nlist * (int)sizeof(float)>>>(
            ws.d_dots, d_cent_norms, d_list_offsets,
            ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total,
            nlist, nprobe);
        CUDA_CHECK(cudaGetLastError());

        // 4. Build primary LUTs
        {
            long long tot = (long long)B * lut_size;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_primary_lut_batched_kernel<<<grid, BLOCK>>>(
                ws.d_q_rot, d_c1d, ws.d_lut, B, d, M, Ds, K1D);
            CUDA_CHECK(cudaGetLastError());
        }

        // 5. Scan IVF lists + select top-ck in-block (no large global sort)
        scan_ivf_batched_topk_kernel<<<B, BLOCK, scan_smem>>>(
            ws.d_lut, ws.d_probe_ids, ws.d_probe_offsets,
            d_list_offsets, d_list_primary, ws.d_query_total,
            ws.d_topck_primary, ws.d_topck_pos,
            nprobe, M, Ds, K1D, bits_per_dim, ck, lut_size);
        CUDA_CHECK(cudaGetLastError());

        // 6. Build residual LUTs
        {
            long long tot = (long long)B * d * Kr;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_residual_lut_batched_kernel<<<grid, BLOCK>>>(
                ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }

        // 7. Residual refinement on top-ck
        {
            long long tot = (long long)B * ck;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            residual_refine_batched_kernel<<<grid, BLOCK>>>(
                ws.d_topck_pos, ws.d_topck_primary,
                ws.d_lut_r, d_list_res, d_list_corr,
                ws.d_comp_dists, ck, d, Kr, Br, bpv, B);
            CUDA_CHECK(cudaGetLastError());
        }

        // 8. Pack + sort composite keys for final top-k (only B*ck elements)
        {
            long long tot = (long long)B * ck;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            pack_keys2_kernel<<<grid, BLOCK>>>(
                ws.d_comp_dists, ws.d_keys2, tot, ck);
            CUDA_CHECK(cudaGetLastError());
            thrust::sort_by_key(t_keys2, t_keys2 + tot, t_pos2);
        }

        // 9. Gather final k ids per query
        {
            long long tot = (long long)B * k;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            gather_final_kernel<<<grid, BLOCK>>>(
                ws.d_keys2, ws.d_topck_pos,
                d_list_ids,
                ws.d_final_dists, ws.d_final_ids,
                ck, k, B);
            CUDA_CHECK(cudaGetLastError());
        }

        // 10. One host sync per batch
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
