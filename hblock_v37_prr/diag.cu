// hblock_v37_prr/diag.cu
//
// Exact-seed threshold diagnostic kernels (D1/D2/D3). Additive only — see
// HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md and diag.cuh. These kernels are
// never invoked from search(); they are only reachable through
// HBlockIndex::seed_diagnostic().
#include "hblock_v37_prr/diag.cuh"
#include "common/cuda_utils.cuh"
#include <algorithm>

namespace hblock_v37_prr {

// ══════════════════════════════════════════════════════════════════════════════
// D1: prr_seed_select_kernel
// ══════════════════════════════════════════════════════════════════════════════
//
// Shared memory:
//   s_q[d]            — query
//   s_cent[d]          — cell abs centroid
//   s_u2[leaf_size]    — U2 values (bitonic-sorted per warp, then merged)
//   s_pos[leaf_size]   — leaf position payload, carried alongside s_u2
__global__ void prr_seed_select_kernel(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const uint8_t* __restrict__ leaf_codes,
    const int*     __restrict__ leaf_sizes,
    const int*     __restrict__ leaf_ids_data,
    const int*     __restrict__ d_block_cell_id,
    const float*   __restrict__ d_abs_cents,
    const float*   __restrict__ d_fine_c1d,
    const float*   __restrict__ d_q_batch,
    const float*   __restrict__ d_block_eps,
    int            eps_stride,
    float*         d_diag_l2,
    float*         d_diag_u2,
    int*           d_seed_pos,
    int*           d_seed_id,
    float*         d_seed_u2,
    int d, int Kr, int Br, int bpv, int leaf_size, int max_seed)
{
    const float INF = 1e38f;
    constexpr unsigned FULL = 0xffffffff;
    const int pi   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int n_warps = leaf_size >> 5;

    const int leaf_blk = d_pair_leaf_ids[pi];
    const int qid      = d_pair_qids[pi];
    const int n_vecs   = leaf_sizes[leaf_blk];
    const int cell_id  = d_block_cell_id[leaf_blk];

    extern __shared__ char shm[];
    float* s_q    = (float*)shm;
    float* s_cent = s_q + d;
    float* s_u2   = s_cent + d;
    int*   s_pos  = (int*)(s_u2 + leaf_size);

    const float* q_ptr    = d_q_batch   + (long long)qid     * d;
    const float* cent_ptr = d_abs_cents + (long long)cell_id * d;
    for (int j = tid; j < d; j += blockDim.x) {
        s_q   [j] = q_ptr   [j];
        s_cent[j] = cent_ptr[j];
    }
    __syncthreads();

    // Corrected PQ dist + L2/U2 interval — bit-identical logic to
    // leaf_prr_interval_kernel (same nibble decode, shared Kr-entry codebook).
    const uint8_t* leaf_base = leaf_codes + (long long)leaf_blk * bpv * leaf_size;

    float L2 = INF, U2 = INF;
    if (tid < n_vecs) {
        float pq2 = 0.f;
        if (Br == 4) {
            for (int b = 0; b < bpv; b++) {
                uint8_t byte = __ldg(&leaf_base[b * leaf_size + tid]);
                int j0 = b * 2;
                float qr0 = s_q[j0]   - s_cent[j0];
                float qr1 = s_q[j0+1] - s_cent[j0+1];
                float cw0 = __ldg(&d_fine_c1d[byte & 0x0F]);
                float cw1 = __ldg(&d_fine_c1d[byte >> 4]);
                float d0 = qr0 - cw0, d1 = qr1 - cw1;
                pq2 += d0*d0 + d1*d1;
            }
        } else { // Br == 8
            for (int b = 0; b < bpv; b++) {
                uint8_t byte = __ldg(&leaf_base[b * leaf_size + tid]);
                float qr = s_q[b] - s_cent[b];
                float cw = __ldg(&d_fine_c1d[byte]);
                float d0 = qr - cw;
                pq2 += d0*d0;
            }
        }

        int sub_idx = (eps_stride == 1) ? 0 : (eps_stride == 4) ? (tid >> 5) : tid;
        float eps = d_block_eps[(long long)leaf_blk * eps_stride + sub_idx];

        float pq_d = sqrtf(fmaxf(pq2, 0.f));
        float L = fmaxf(0.f, pq_d - eps);
        float U = pq_d + eps;
        L2 = L * L;
        U2 = U * U;
    }

    d_diag_l2[(long long)pi * leaf_size + tid] = L2;
    d_diag_u2[(long long)pi * leaf_size + tid] = U2;

    // Bitonic sort ascending by U2 within each warp, carrying leaf POSITION (tid)
    // as payload — id is looked up from position at write time.
    float my_u2 = U2; int my_pos = tid;
    for (int k = 2; k <= 32; k <<= 1) {
        bool asc = ((lane & k) == 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            float ou = __shfl_xor_sync(FULL, my_u2,  j);
            int   op = __shfl_xor_sync(FULL, my_pos, j);
            bool lower    = ((lane & j) == 0);
            bool want_min = (asc == lower);
            bool swap     = want_min ? (my_u2 > ou) : (my_u2 < ou);
            if (swap) { my_u2 = ou; my_pos = op; }
        }
    }
    s_u2 [tid] = my_u2;
    s_pos[tid] = my_pos;
    __syncthreads();

    // 4-way merge across the (already per-warp ascending-sorted) segments —
    // same selection pattern as leaf_flat_kernel_v29's per_block_r selection,
    // just capped at max_seed instead of per_block_r.
    if (tid == 0) {
        int ptrs[4] = {0, 32, 64, 96};
        int actual = (max_seed < n_vecs) ? max_seed : n_vecs;
        int r = 0;
        for (; r < actual; r++) {
            float best_u = INF; int best_w = -1;
            for (int w = 0; w < n_warps; w++) {
                int p = ptrs[w];
                if (p < (w+1)*32 && s_u2[p] < best_u) { best_u = s_u2[p]; best_w = w; }
            }
            if (best_w >= 0 && best_u < INF) {
                int pos = s_pos[ptrs[best_w]];
                d_seed_u2 [(long long)pi * max_seed + r] = best_u;
                d_seed_pos[(long long)pi * max_seed + r] = pos;
                d_seed_id [(long long)pi * max_seed + r] =
                    leaf_ids_data[(long long)leaf_blk * leaf_size + pos];
                ptrs[best_w]++;
            } else {
                break; // no more valid (finite-U2) candidates
            }
        }
        for (; r < max_seed; r++) {
            d_seed_u2 [(long long)pi * max_seed + r] = INF;
            d_seed_pos[(long long)pi * max_seed + r] = -1;
            d_seed_id [(long long)pi * max_seed + r] = -1;
        }
    }
}

void launch_prr_seed_select(
    const int* d_pair_leaf_ids, const int* d_pair_qids,
    const uint8_t* d_leaf_codes, const int* d_leaf_sizes, const int* d_leaf_ids_data,
    const int* d_block_cell_id, const float* d_abs_cents,
    const float* d_fine_c1d, const float* d_q_batch,
    const float* d_block_eps, int eps_stride,
    float* d_diag_l2, float* d_diag_u2,
    int* d_seed_pos, int* d_seed_id, float* d_seed_u2,
    int n_pairs, int d, int Kr, int Br, int bpv, int leaf_size, int max_seed,
    cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    // s_q[d] + s_cent[d] + s_u2[leaf_size] + s_pos[leaf_size]
    int smem = 2 * d * (int)sizeof(float) + leaf_size * (int)(sizeof(float) + sizeof(int));
    prr_seed_select_kernel<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_codes, d_leaf_sizes, d_leaf_ids_data,
        d_block_cell_id, d_abs_cents, d_fine_c1d, d_q_batch,
        d_block_eps, eps_stride,
        d_diag_l2, d_diag_u2, d_seed_pos, d_seed_id, d_seed_u2,
        d, Kr, Br, bpv, leaf_size, max_seed);
    CUDA_CHECK(cudaGetLastError());
}

// ══════════════════════════════════════════════════════════════════════════════
// D2: prr_seed_exact_kernel
// ══════════════════════════════════════════════════════════════════════════════
//
// Shared memory: s_query[d] + s_wsum[4]
__global__ void prr_seed_exact_kernel(
    const int*   __restrict__ d_pair_qids,
    const int*   __restrict__ d_seed_id,
    const float* __restrict__ d_base_vecs,
    const float* __restrict__ d_q_batch,
    float* d_seed_exact2,
    int d, int max_seed)
{
    const float INF = __int_as_float(0x7F800000);
    constexpr unsigned FULL = 0xffffffff;
    const int pi   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;

    const int qid = d_pair_qids[pi];
    const float* q_ptr = d_q_batch + (long long)qid * d;

    extern __shared__ char shm[];
    float* s_query = (float*)shm;
    float* s_wsum  = s_query + d;

    for (int j = tid; j < d; j += blockDim.x) s_query[j] = q_ptr[j];
    __syncthreads();

    for (int r = 0; r < max_seed; r++) {
        int sid = d_seed_id[(long long)pi * max_seed + r];
        float partial = 0.f;
        if (sid >= 0) {
            const float* vec = d_base_vecs + (long long)sid * d;
            for (int j = tid; j < d; j += blockDim.x) {
                float diff = s_query[j] - vec[j];
                partial += diff * diff;
            }
        }
        for (int o = 16; o >= 1; o >>= 1)
            partial += __shfl_xor_sync(FULL, partial, o);
        __syncthreads();
        if (lane == 0) s_wsum[wid] = partial;
        __syncthreads();
        if (tid == 0) {
            float total = s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3];
            d_seed_exact2[(long long)pi * max_seed + r] = (sid >= 0) ? total : INF;
        }
        __syncthreads();
    }
}

void launch_prr_seed_exact(
    const int* d_pair_qids, const int* d_seed_id,
    const float* d_base_vecs, const float* d_q_batch,
    float* d_seed_exact2,
    int n_pairs, int d, int max_seed,
    cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    const int BLOCK = 128; // 4 warps, matches s_wsum[4]
    int smem = d * (int)sizeof(float) + 4 * (int)sizeof(float);
    prr_seed_exact_kernel<<<n_pairs, BLOCK, smem, stream>>>(
        d_pair_qids, d_seed_id, d_base_vecs, d_q_batch, d_seed_exact2, d, max_seed);
    CUDA_CHECK(cudaGetLastError());
}

// ══════════════════════════════════════════════════════════════════════════════
// D3: prr_seed_tau_kernel — query-major, reuses the qid-major perm + segment infra
// ══════════════════════════════════════════════════════════════════════════════
__global__ void prr_seed_tau_kernel(
    const float* __restrict__ d_seed_exact2,
    const int*   __restrict__ d_perm,
    const int*   __restrict__ d_query_offsets,
    const int*   __restrict__ d_leaf_cnt,
    float* d_tau_seed2,
    int*   d_insufficient,
    int k, int spb, int max_seed, int batch_size)
{
    const float INF = 1e38f;
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= batch_size) return;
    int seg_start = d_query_offsets[qi];
    int seg_end   = seg_start + d_leaf_cnt[qi];

    float topk[16]; // k <= 16 (matches klocal cap elsewhere in v37_prr)
    for (int r = 0; r < k; r++) topk[r] = INF;
    int cnt = 0;

    for (int p = seg_start; p < seg_end; p++) {
        int pi = d_perm[p];
        int use = spb < max_seed ? spb : max_seed;
        for (int r = 0; r < use; r++) {
            float e = d_seed_exact2[(long long)pi * max_seed + r];
            if (e >= INF) continue; // invalid seed slot
            cnt++;
            float mx = topk[0]; int mi = 0;
            for (int s2 = 1; s2 < k; s2++)
                if (topk[s2] > mx) { mx = topk[s2]; mi = s2; }
            if (e < mx) topk[mi] = e;
        }
    }

    if (cnt < k) {
        // Strict rule: fewer than k valid seeds -> threshold is +INF (prunes nothing).
        d_tau_seed2[qi]    = INF;
        d_insufficient[qi] = 1;
    } else {
        float tau = topk[0];
        for (int r = 1; r < k; r++) tau = fmaxf(tau, topk[r]);
        d_tau_seed2[qi]    = tau;
        d_insufficient[qi] = 0;
    }
}

void launch_prr_seed_tau(
    const float* d_seed_exact2, const int* d_perm,
    const int* d_query_offsets, const int* d_leaf_cnt,
    float* d_tau_seed2, int* d_insufficient,
    int k, int spb, int max_seed, int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return;
    int grid = (batch_size + 127) / 128;
    prr_seed_tau_kernel<<<grid, 128, 0, stream>>>(
        d_seed_exact2, d_perm, d_query_offsets, d_leaf_cnt,
        d_tau_seed2, d_insufficient, k, spb, max_seed, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v37_prr
