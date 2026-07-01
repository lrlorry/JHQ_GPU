#include "jhq_v3_ivf/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace jhq_gpu {

namespace {

constexpr int   MAX_FAST_NPROBE = 256;
constexpr int   MAX_FAST_CK = 512;
constexpr int   MAX_FAST_K = 128;
constexpr float HUGE_DIST = 3.4028234663852886e+38F;

__device__ __forceinline__
bool better_pair(float da, int ia, float db, int ib) {
    return da < db || (da == db && ia < ib);
}

} // namespace

__global__ void centroid_distance_kernel(
    const float* __restrict__ q_rot,
    const float* __restrict__ centroids,
    const float* __restrict__ cent_norms,
    float*                    dists,
    int nlist,
    int d)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nlist) return;

    const float* cent = centroids + (long long)c * d;
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) dot += q_rot[j] * cent[j];
    dists[c] = cent_norms[c] - 2.0f * dot; // query norm is constant
}

__global__ void select_probe_offsets_kernel(
    const float* __restrict__ cent_dists,
    const int*   __restrict__ list_offsets,
    int*                    probe_ids,
    int*                    probe_offsets,
    int nlist,
    int nprobe)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float best_d[MAX_FAST_NPROBE];
    int   best_i[MAX_FAST_NPROBE];
    for (int p = 0; p < nprobe; ++p) {
        best_d[p] = HUGE_DIST;
        best_i[p] = -1;
    }

    for (int c = 0; c < nlist; ++c) {
        float dist = cent_dists[c];
        if (!better_pair(dist, c, best_d[nprobe - 1], best_i[nprobe - 1])) continue;

        int pos = nprobe - 1;
        while (pos > 0 && better_pair(dist, c, best_d[pos - 1], best_i[pos - 1])) {
            best_d[pos] = best_d[pos - 1];
            best_i[pos] = best_i[pos - 1];
            --pos;
        }
        best_d[pos] = dist;
        best_i[pos] = c;
    }

    int acc = 0;
    probe_offsets[0] = 0;
    for (int p = 0; p < nprobe; ++p) {
        int list_id = best_i[p];
        probe_ids[p] = list_id;
        acc += list_offsets[list_id + 1] - list_offsets[list_id];
        probe_offsets[p + 1] = acc;
    }
}

__global__ void build_primary_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_c1d,
    float*                    d_lut,
    int M, int Ds, int K1D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * Ds * K1D) return;
    int j = tid % K1D;
    int mk = tid / K1D;
    int k = mk % Ds;
    int m = mk / Ds;
    float diff = d_q_rot[m * Ds + k] - d_c1d[j];
    d_lut[tid] = diff * diff;
}

__global__ void build_probe_offsets_kernel(
    const int* __restrict__ probe_ids,
    const int* __restrict__ list_offsets,
    int*                    probe_offsets,
    int nprobe)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int acc = 0;
    probe_offsets[0] = 0;
    for (int p = 0; p < nprobe; ++p) {
        int list_id = probe_ids[p];
        int sz = list_offsets[list_id + 1] - list_offsets[list_id];
        acc += sz;
        probe_offsets[p + 1] = acc;
    }
}

__global__ void scan_ivf_primary_kernel(
    const float*   __restrict__ lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary,
    float*                      out_dists,
    int*                        out_pos,
    int total,
    int nprobe,
    int M,
    int Ds,
    int K1D,
    int bits_per_dim)
{
    extern __shared__ float s_lut[];
    int lut_size = M * Ds * K1D;
    for (int i = threadIdx.x; i < lut_size; i += blockDim.x)
        s_lut[i] = lut[i];
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    int p = 0;
    while (p + 1 < nprobe && t >= probe_offsets[p + 1]) ++p;
    int list_id = probe_ids[p];
    int local = t - probe_offsets[p];
    int abs_pos = list_offsets[list_id] + local;

    const uint8_t* code = list_primary + (long long)abs_pos * M;
    const int kmask = K1D - 1;
    float dist = 0.0f;
    for (int m = 0; m < M; ++m) {
        uint8_t cm = code[m];
        const float* lm = s_lut + m * Ds * K1D;
        for (int k = 0; k < Ds; ++k) {
            int j = (cm >> (k * bits_per_dim)) & kmask;
            dist += lm[k * K1D + j];
        }
    }
    out_dists[t] = dist;
    out_pos[t] = abs_pos;
}

__global__ void pack_kernel(
    const float* __restrict__ d_dists,
    const int*   __restrict__ d_pos,
    uint64_t*                 d_packed,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_packed[i] = ((uint64_t)__float_as_uint(d_dists[i]) << 32)
                | (uint32_t)d_pos[i];
}

__global__ void unpack_kernel(
    const uint64_t* __restrict__ d_packed,
    float*                       d_dists,
    int*                         d_pos,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_dists[i] = __uint_as_float((uint32_t)(d_packed[i] >> 32));
    d_pos[i] = (int)(d_packed[i] & 0xFFFFFFFFULL);
}

__global__ void select_topk_pairs_kernel(
    const float* __restrict__ in_dists,
    const int*   __restrict__ in_pos,
    float*                    out_dists,
    int*                      out_pos,
    int n,
    int topk)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float best_d[MAX_FAST_CK];
    int   best_i[MAX_FAST_CK];
    for (int i = 0; i < topk; ++i) {
        best_d[i] = HUGE_DIST;
        best_i[i] = -1;
    }

    for (int i = 0; i < n; ++i) {
        float dist = in_dists[i];
        int pos_id = in_pos[i];
        if (!better_pair(dist, pos_id, best_d[topk - 1], best_i[topk - 1])) continue;

        int pos = topk - 1;
        while (pos > 0 && better_pair(dist, pos_id, best_d[pos - 1], best_i[pos - 1])) {
            best_d[pos] = best_d[pos - 1];
            best_i[pos] = best_i[pos - 1];
            --pos;
        }
        best_d[pos] = dist;
        best_i[pos] = pos_id;
    }

    for (int i = 0; i < topk; ++i) {
        out_dists[i] = best_d[i];
        out_pos[i] = best_i[i];
    }
}

__global__ void build_residual_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
    int d,
    int Kr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d * Kr) return;
    int j = tid % Kr;
    int dim = tid / Kr;
    float diff = d_q_rot[dim] - d_res_c1d[j];
    d_lut_r[tid] = diff * diff;
}

__global__ void select_final_topk_kernel(
    const float* __restrict__ comp_dists,
    const int*   __restrict__ cand_pos,
    const int*   __restrict__ list_ids,
    float*                    out_dists,
    int*                      out_ids,
    int ck,
    int k)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float best_d[MAX_FAST_K];
    int   best_id[MAX_FAST_K];
    for (int i = 0; i < k; ++i) {
        best_d[i] = HUGE_DIST;
        best_id[i] = -1;
    }

    for (int i = 0; i < ck; ++i) {
        float dist = comp_dists[i];
        int id = list_ids[cand_pos[i]];
        if (!better_pair(dist, id, best_d[k - 1], best_id[k - 1])) continue;

        int pos = k - 1;
        while (pos > 0 && better_pair(dist, id, best_d[pos - 1], best_id[pos - 1])) {
            best_d[pos] = best_d[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best_d[pos] = dist;
        best_id[pos] = id;
    }

    for (int i = 0; i < k; ++i) {
        out_dists[i] = best_d[i];
        out_ids[i] = best_id[i];
    }
}

__global__ void residual_refine_kernel(
    const int*     __restrict__ cand_pos,
    const float*   __restrict__ cand_primary,
    const float*   __restrict__ lut_r,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck,
    int d,
    int Kr,
    int Br,
    int bpv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ck) return;

    int pos = cand_pos[tid];
    const uint8_t* rc = list_res + (long long)pos * bpv;
    float d_res = 0.0f;
    for (int j = 0; j < d; ++j) {
        int ri = (Br == 4)
            ? ((j % 2 == 0) ? (rc[j / 2] & 0x0F) : (rc[j / 2] >> 4))
            : rc[j];
        d_res += lut_r[(long long)j * Kr + ri];
    }
    comp_dists[tid] = cand_primary[tid] + d_res + list_corr[pos];
}

__global__ void gather_final_ids_kernel(
    const int* __restrict__ cand_pos,
    const int* __restrict__ list_ids,
    int*                 final_ids,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    final_ids[i] = list_ids[cand_pos[i]];
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
    int nq, int N, int d, int M, int Ds, int K1D, int Kr, int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids)
{
    const float inf = std::numeric_limits<float>::infinity();

    float*    d_q_rot = ws.d_q_rot;
    float*    d_lut = ws.d_lut;
    float*    d_dists = ws.d_dists;
    int*      d_indices = ws.d_indices;
    float*    d_lut_r = ws.d_lut_r;
    float*    d_comp_dists = ws.d_comp_dists;
    uint64_t* d_packed = ws.d_packed;
    float*    d_cent_dists = ws.d_cent_dists;
    int*      d_cent_ids = ws.d_cent_ids;
    int*      d_probe_ids = ws.d_probe_ids;
    int*      d_probe_offsets = ws.d_probe_offsets;
    int*      d_final_ids = ws.d_final_ids;

    thrust::device_ptr<float> t_cent_dists(d_cent_dists);
    thrust::device_ptr<int>   t_cent_ids(d_cent_ids);
    thrust::device_ptr<uint64_t> t_packed(d_packed);
    thrust::device_ptr<float> t_comp(d_comp_dists);
    thrust::device_ptr<int> t_indices(d_indices);

    const float one = 1.0f, zero = 0.0f;

    for (int qi = 0; qi < nq; ++qi) {
        const float* d_q = d_queries + (long long)qi * d;

        CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                 d, d, &one, d_Pi, d,
                                 d_q, 1, &zero, d_q_rot, 1));

        centroid_distance_kernel<<<(nlist + 255) / 256, 256>>>(
            d_q_rot, d_centroids, d_cent_norms, d_cent_dists, nlist, d);
        CUDA_CHECK(cudaGetLastError());
        if (nprobe <= MAX_FAST_NPROBE) {
            select_probe_offsets_kernel<<<1, 1>>>(
                d_cent_dists, d_list_offsets, d_probe_ids, d_probe_offsets,
                nlist, nprobe);
            CUDA_CHECK(cudaGetLastError());
        } else {
            thrust::sequence(t_cent_ids, t_cent_ids + nlist);
            thrust::sort_by_key(t_cent_dists, t_cent_dists + nlist, t_cent_ids);
            CUDA_CHECK(cudaMemcpy(d_probe_ids, d_cent_ids,
                                  (long long)nprobe * sizeof(int),
                                  cudaMemcpyDeviceToDevice));

            build_probe_offsets_kernel<<<1, 1>>>(
                d_probe_ids, d_list_offsets, d_probe_offsets, nprobe);
            CUDA_CHECK(cudaGetLastError());
        }

        int total = 0;
        CUDA_CHECK(cudaMemcpy(&total, d_probe_offsets + nprobe, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (total <= 0) {
            for (int i = 0; i < k; ++i) {
                h_out_dists[(long long)qi * k + i] = inf;
                h_out_ids[(long long)qi * k + i] = -1;
            }
            continue;
        }

        int lut_total = M * Ds * K1D;
        build_primary_lut_kernel<<<(lut_total + 255) / 256, 256>>>(
            d_q_rot, d_c1d, d_lut, M, Ds, K1D);
        CUDA_CHECK(cudaGetLastError());

        const int BLOCK = 256;
        int shm = M * Ds * K1D * (int)sizeof(float);
        scan_ivf_primary_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, shm>>>(
            d_lut, d_probe_ids, d_probe_offsets, d_list_offsets,
            d_list_primary, d_dists, d_indices, total,
            nprobe, M, Ds, K1D, bits_per_dim);
        CUDA_CHECK(cudaGetLastError());

        int ck = std::min(std::max(k, (int)std::ceil(alpha * (float)k)), total);
        if (ck <= MAX_FAST_CK) {
            select_topk_pairs_kernel<<<1, 1>>>(
                d_dists, d_indices, d_dists, d_indices, total, ck);
            CUDA_CHECK(cudaGetLastError());
        } else {
            pack_kernel<<<(total + 255) / 256, 256>>>(d_dists, d_indices, d_packed, total);
            CUDA_CHECK(cudaGetLastError());
            thrust::sort(t_packed, t_packed + total);
            unpack_kernel<<<(ck + 255) / 256, 256>>>(d_packed, d_dists, d_indices, ck);
            CUDA_CHECK(cudaGetLastError());
        }

        int res_total = d * Kr;
        build_residual_lut_kernel<<<(res_total + 255) / 256, 256>>>(
            d_q_rot, d_res_c1d, d_lut_r, d, Kr);
        CUDA_CHECK(cudaGetLastError());

        residual_refine_kernel<<<(ck + 255) / 256, 256>>>(
            d_indices, d_dists, d_lut_r, d_list_res, d_list_corr,
            d_comp_dists, ck, d, Kr, Br, bpv);
        CUDA_CHECK(cudaGetLastError());

        int out_k = std::min(k, ck);
        if (out_k <= MAX_FAST_K) {
            select_final_topk_kernel<<<1, 1>>>(
                d_comp_dists, d_indices, d_list_ids,
                d_dists, d_final_ids, ck, out_k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_out_dists + (long long)qi * k,
                                  d_dists, (long long)out_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_out_ids + (long long)qi * k,
                                  d_final_ids, (long long)out_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        } else {
            thrust::sort_by_key(t_comp, t_comp + ck, t_indices);
            gather_final_ids_kernel<<<(out_k + 255) / 256, 256>>>(
                d_indices, d_list_ids, d_final_ids, out_k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_out_dists + (long long)qi * k,
                                  d_comp_dists, (long long)out_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_out_ids + (long long)qi * k,
                                  d_final_ids, (long long)out_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
        for (int i = out_k; i < k; ++i) {
            h_out_dists[(long long)qi * k + i] = inf;
            h_out_ids[(long long)qi * k + i] = -1;
        }
    }
}

} // namespace jhq_gpu
