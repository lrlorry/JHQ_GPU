#pragma once
// hblock_v37_prr/diag.cuh
//
// Exact-seed threshold diagnostic (go/no-go for a two-wave certified PRR).
// Additive only: does not touch FIXED_PER_BLOCK / CORRECTED_FIXED / CERTIFIED_PRR
// production kernels or result paths. See HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md.
//
// Kernel naming follows the spec:
//   D1 prr_seed_select_kernel — full L2/U2 per candidate + top-8 seeds by smallest U2
//   D2 prr_seed_exact_kernel  — exact squared L2 for each selected seed
//   D3 prr_seed_tau_kernel    — query-major k-th smallest exact seed distance (tau_seed2)
#include <cuda_runtime.h>
#include <cstdint>

namespace hblock_v37_prr {

// Fixed per the spec: seeds are selected once at max_seed_per_block=8, and every
// smaller seed_per_block value (1,2,4,8) is read as a PREFIX of this ascending-by-U2
// seed list. This keeps visited blocks/bounds identical across the spb sweep.
constexpr int PRR_DIAG_MAX_SEED = 8;

// D1: one CUDA block per (query,leaf_block) pair, leaf_size threads.
// Computes the corrected PQ distance + L2/U2 interval exactly like
// leaf_prr_interval_kernel (VECTOR_LEVEL or coarser eps, same nibble decode),
// then:
//   - writes the FULL L2 array  d_diag_l2[pi*leaf_size + tid]
//   - writes the FULL U2 array  d_diag_u2[pi*leaf_size + tid]      (INF if tid >= n_vecs)
//   - selects the up-to-max_seed smallest-U2 candidates (bitonic warp sort on U2,
//     carrying leaf POSITION as payload, then a 4-way merge across the leaf_size/32
//     warps — same pattern as leaf_flat_kernel_v29's per_block_r selection), written
//     ASCENDING by U2 into:
//       d_seed_pos[pi*max_seed + r]  leaf position (0..leaf_size-1) or -1
//       d_seed_id [pi*max_seed + r]  original vector id (looked up via d_leaf_ids_data) or -1
//       d_seed_u2 [pi*max_seed + r]  U2 or +INF
void launch_prr_seed_select(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_sizes,
    const int*     d_leaf_ids_data,
    const int*     d_block_cell_id,
    const float*   d_abs_cents,
    const float*   d_fine_c1d,
    const float*   d_q_batch,
    const float*   d_block_eps,
    int            eps_stride,
    float*         d_diag_l2,     // [n_pairs * leaf_size]
    float*         d_diag_u2,     // [n_pairs * leaf_size]
    int*           d_seed_pos,    // [n_pairs * max_seed]
    int*           d_seed_id,     // [n_pairs * max_seed]
    float*         d_seed_u2,     // [n_pairs * max_seed]
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size, int max_seed,
    cudaStream_t stream);

// D2: one CUDA block per pair, leaf_size threads cooperate. For each of the
// up-to-max_seed seeds, computes the exact squared L2 between the query and
// d_base_vecs_[seed_id] (parallel over d, warp reduce — same pattern as
// prr_exact_rerank_kernel). Writes d_seed_exact2[pi*max_seed + r] (+INF if invalid).
void launch_prr_seed_exact(
    const int*   d_pair_qids,
    const int*   d_seed_id,       // [n_pairs * max_seed]
    const float* d_base_vecs,
    const float* d_q_batch,
    float*       d_seed_exact2,   // [n_pairs * max_seed]
    int n_pairs, int d, int max_seed,
    cudaStream_t stream);

// D3: query-major (one thread per query), using the qid-major pair permutation
// (d_perm) + d_query_offsets + d_leaf_cnt segments built for this batch/ef — the
// same infra launch_prr_tau2 uses. For seed_per_block = spb (a PREFIX of the
// max_seed-sized seed list written by D1), scans the query's visited pairs,
// collects seed_exact2 entries with r < spb and valid (< +INF), computes the
// k-th smallest -> d_tau_seed2[qi]. If fewer than k valid seeds are found for a
// query, d_tau_seed2[qi] = +INF (prunes nothing) and d_insufficient[qi] = 1.
void launch_prr_seed_tau(
    const float* d_seed_exact2,     // [n_pairs * max_seed]
    const int*   d_perm,            // qid-major -> pair index (from launch_prr_tau2)
    const int*   d_query_offsets,
    const int*   d_leaf_cnt,
    float*       d_tau_seed2,       // [batch_size]
    int*         d_insufficient,    // [batch_size]
    int k, int spb, int max_seed, int batch_size,
    cudaStream_t stream);

} // namespace hblock_v37_prr
