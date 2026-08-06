#include "hblock_v39/jhq_gpu_index.cuh"
#include "hblock_v17/encode.cuh"
#include "hblock_v27/search.cuh"   // for gpu_build_block_adj_v27
#include "hblock_v39/search.cuh"   // for align4() and the region-indirected leaf kernel
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using Ms = std::chrono::duration<double, std::milli>;

namespace hblock_v39 {

static std::vector<float>
analytical_fine_c1d(int Kr, float sigma)
{
    std::vector<float> c(Kr);
    for (int i = 0; i < Kr; ++i) {
        float q = (i + 0.5f) / float(Kr);
        c[i] = sigma * float(M_SQRT2) * erfinv_f(2.f * q - 1.f);
    }
    return c;
}

HBlockIndex::HBlockIndex(int d, Params p)
    : d_(d), d_proj_(p.d_proj), Kr_(p.Kr), Br_(p.Br),
      bpv_((d * p.Br + 7) / 8),
      leaf_size_(p.leaf_size),
      K1_(p.K1), K2_(p.K2), K3_(p.K3),
      ck1_(p.ck1), ck2_(p.ck2), ck3_(p.ck3),
      per_block_r_(p.per_block_r), klocal_(p.klocal),
      km_iters_(p.km_iters), batch_size_(p.batch_size),
      graph_degree_(p.graph_degree), max_ef_(p.max_ef),
      entry_per_cell_(p.entry_per_cell),
      n_c2_nbrs_(p.n_c2_nbrs), n_c1_nbrs_(p.n_c1_nbrs),
      max_cand_blocks_(p.max_cand_blocks),
      mini_km_iters_(p.mini_km_iters),
      add_batch_size_(p.add_batch_size),
      region_bytes_(p.region_bytes),
      gpu_code_region_cap_(p.gpu_code_region_cap),
      gpu_raw_region_cap_(p.gpu_raw_region_cap)
{
    if (d <= 0)                  throw std::invalid_argument("d must be positive");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (p.region_bytes <= 0)     throw std::invalid_argument("region_bytes must be positive");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex()
{
    if (ws_.stream) { cublasSetStream(cublas_, nullptr); cudaStreamDestroy(ws_.stream); }
    auto FH = [](void* p){ if (p) cudaFreeHost(p); };
    auto FD = [](void* p){ if (p) cudaFree(p); };
    FH(ws_.h_q_pinned); FH(ws_.h_q_pinned_i8); FH(ws_.h_leaf_cnt);
    FH(ws_.h_final_dists); FH(ws_.h_final_ids); FH(ws_.h_pair_leaf_sorted);
    FH(ws_.h_top1_ids); FH(ws_.h_top2_beam); FH(ws_.h_top3_beam); FH(ws_.h_block_sel);
    FD(ws_.d_q_batch);  FD(ws_.d_q_proj1);  FD(ws_.d_dots1);   FD(ws_.d_q_batch_i8);
    FD(ws_.d_top1_ids); FD(ws_.d_r1_beam);  FD(ws_.d_top2_beam); FD(ws_.d_top3_beam);
    FD(ws_.d_q_r3);     FD(ws_.d_leaf_sel); FD(ws_.d_leaf_cnt);  FD(ws_.d_lut_fine);
    FD(ws_.d_query_offsets);
    FD(ws_.d_pair_leaf_a); FD(ws_.d_pair_qid_a);
    FD(ws_.d_pair_leaf_b); FD(ws_.d_pair_qid_b);
    FD(ws_.d_out_dists); FD(ws_.d_out_ids);
    FD(ws_.d_final_dists); FD(ws_.d_final_ids);
    FD(ws_.d_cub_tmp);
    FD(ws_.d_visited);
    FD(d_Pi1_); FD(d_Pi2_); FD(d_Pi3_);
    FD(d_route1_cents_proj_); FD(d_route1_cents_full_); FD(d_route1_norms_);
    FD(d_route2_cents_proj_); FD(d_route2_cents_full_); FD(d_route2_norms_);
    FD(d_route3_cents_proj_); FD(d_route3_cents_full_); FD(d_route3_norms_);
    FD(d_fine_c1d_);
    FD(d_pair_blk_start_); FD(d_pair_blk_count_);
    FD(d_leaf_sizes_);
    FD(d_block_adj_gpu_); FD(d_blk_proj_gpu_); FD(d_blk_norm_gpu_);
    FD(d_block_code_region_); FD(d_block_code_offset_);
    FD(d_block_raw_region_);  FD(d_block_raw_offset_);
    FD(d_vec_local_pos_);
    FD(d_code_region_pool_); FD(d_raw_region_pool_);
    FD(d_code_region_slot_); FD(d_raw_region_slot_);
    cublasDestroy(cublas_);
}

void HBlockIndex::init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi)
{
    Pi.resize((long long)d_proj * d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f / std::sqrt((float)d_proj));
    for (float& v : Pi) v = g(rng);
}

void HBlockIndex::gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                              int n, int K,
                              std::vector<float>& h_cents_proj,
                              std::vector<float>& h_cents_full,
                              std::vector<int>&   h_assigns)
{
    h_cents_proj.resize((long long)K * d_proj_);
    float *d_x_proj, *d_dots_km; int *d_assigns, *d_counts; float *d_cents, *d_norms;
    CUDA_CHECK(cudaMalloc(&d_x_proj,  (long long)n * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dots_km, (long long)n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assigns, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts,  K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cents,   (long long)K * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms,   K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_proj, h_x_proj, (long long)n*d_proj_*sizeof(float), cudaMemcpyHostToDevice));
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, n-1);
        std::vector<float> ic((long long)K*d_proj_);
        for (int k = 0; k < K; k++)
            std::memcpy(ic.data()+(long long)k*d_proj_, h_x_proj+(long long)uni(rng)*d_proj_, d_proj_*sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_cents, ic.data(), (long long)K*d_proj_*sizeof(float), cudaMemcpyHostToDevice));
    }
    const float one=1.f, zero=0.f;
    for (int iter = 0; iter < km_iters_; iter++) {
        std::vector<float> h_norms(K);
        CUDA_CHECK(cudaMemcpy(h_cents_proj.data(), d_cents, (long long)K*d_proj_*sizeof(float), cudaMemcpyDeviceToHost));
        for (int k=0;k<K;k++){float s=0.f;for(int j=0;j<d_proj_;j++){float v=h_cents_proj[(long long)k*d_proj_+j];s+=v*v;}h_norms[k]=s;}
        CUDA_CHECK(cudaMemcpy(d_norms, h_norms.data(), K*sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K, n, d_proj_, &one, d_cents, d_proj_, d_x_proj, d_proj_, &zero, d_dots_km, K));
        hblock_v17::launch_kmeans_assign(d_dots_km, d_norms, d_assigns, K, n, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_cents, 0, (long long)K*d_proj_*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K*sizeof(int)));
        hblock_v17::launch_kmeans_update(d_x_proj, d_assigns, d_cents, d_counts, n, d_proj_, K, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    h_cents_proj.resize((long long)K*d_proj_);
    h_assigns.resize(n);
    CUDA_CHECK(cudaMemcpy(h_cents_proj.data(), d_cents,   (long long)K*d_proj_*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_assigns.data(),    d_assigns, n*sizeof(int), cudaMemcpyDeviceToHost));
    h_cents_full.assign((long long)K*d_, 0.f);
    std::vector<int> cnt(K,0);
    for (int i=0;i<n;i++){int c=h_assigns[i];cnt[c]++;for(int j=0;j<d_;j++)h_cents_full[(long long)c*d_+j]+=h_x_full[(long long)i*d_+j];}
    for (int k=0;k<K;k++) if(cnt[k]>0) for(int j=0;j<d_;j++) h_cents_full[(long long)k*d_+j]/=(float)cnt[k];
    cudaFree(d_x_proj);cudaFree(d_dots_km);cudaFree(d_assigns);cudaFree(d_counts);cudaFree(d_cents);cudaFree(d_norms);
}

void HBlockIndex::upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                                const std::vector<bool>& h_valid, int K,
                                float*& d_proj_out, float*& d_full_out, float*& d_norms_out)
{
    std::vector<float> h_norms(K);
    for (int k=0;k<K;k++){
        if(!h_valid[k]){h_norms[k]=1e30f;continue;}
        float s=0.f; for(int j=0;j<d_proj_;j++){float v=h_proj[(long long)k*d_proj_+j];s+=v*v;} h_norms[k]=s;
    }
    cudaFree(d_proj_out); cudaFree(d_full_out); cudaFree(d_norms_out);
    CUDA_CHECK(cudaMalloc(&d_proj_out,  (long long)K*d_proj_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_full_out,  (long long)K*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms_out, (long long)K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_out,  h_proj.data(),  (long long)K*d_proj_*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_full_out,  h_full.data(),  (long long)K*d_      *sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norms_out, h_norms.data(), K*sizeof(float),                     cudaMemcpyHostToDevice));
}

static __global__ void local_assign_kernel(
    const float* __restrict__ r_proj,
    const float* __restrict__ C_proj,
    const float* __restrict__ C_norms,
    const int*   __restrict__ outer_id,
    int* assign,
    int n, int d_proj, int K_inner)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int og = outer_id[i];
    const float* r_i    = r_proj  + (long long)i  * d_proj;
    const float* c_base = C_proj  + (long long)og * K_inner * d_proj;
    const float* n_base = C_norms + (long long)og * K_inner;
    float bv = 1e30f; int bi = 0;
    for (int k = 0; k < K_inner; k++) {
        const float* c_k = c_base + (long long)k * d_proj;
        float dot = 0.f;
        for (int j = 0; j < d_proj; j++) dot += r_i[j] * c_k[j];
        float dist = n_base[k] - 2.f * dot;
        if (dist < bv) { bv = dist; bi = k; }
    }
    assign[i] = bi;
}

// Elementwise int8 -> float32 cast. Used by add()'s bulk encode path (GPU
// side, not host) so casting 100M-scale batches doesn't serialize with the
// GPU pipeline -- exact-integer cast, lossless (every int8 value fits
// exactly in a float32).
static __global__ void cast_i8_to_f32_kernel(
    const int8_t* __restrict__ in, float* __restrict__ out, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

void HBlockIndex::train(const float* h_x, int n_train)
{
    using Clock = std::chrono::high_resolution_clock;
    auto T0 = Clock::now();
    const int n_km = std::min(n_train, 200000);
    printf("[v39 train] d=%d d_proj=%d K1=%d K2=%d K3=%d ck1=%d ck2=%d ck3=%d"
           " graph_degree=%d max_ef=%d entry_per_cell=%d\n",
           d_, d_proj_, K1_, K2_, K3_, ck1_, ck2_, ck3_,
           graph_degree_, max_ef_, entry_per_cell_);

    const float one=1.f, zero=0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));

    std::vector<float> Pi1; init_jl_proj(d_, d_proj_, 42, Pi1);
    CUDA_CHECK(cudaMalloc(&d_Pi1_, (long long)d_proj_*d_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi1_, Pi1.data(), (long long)d_proj_*d_*sizeof(float), cudaMemcpyHostToDevice));
    float *d_x_km, *d_y_proj;
    CUDA_CHECK(cudaMalloc(&d_x_km,   (long long)n_km*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_proj, (long long)n_km*d_proj_ *sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_km, h_x, (long long)n_km*d_*sizeof(float), cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj_, n_km, d_, &one, d_Pi1_, d_, d_x_km, d_, &zero, d_y_proj, d_proj_));
    std::vector<float> h_y((long long)n_km*d_proj_), h_x_km((long long)n_km*d_);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_proj, (long long)n_km*d_proj_*sizeof(float), cudaMemcpyDeviceToHost));
    std::memcpy(h_x_km.data(), h_x, (long long)n_km*d_*sizeof(float));
    std::vector<float> h_c1_proj, h_c1_full; std::vector<int> h_assign1;
    gpu_kmeans(h_y.data(), h_x_km.data(), n_km, K1_, h_c1_proj, h_c1_full, h_assign1);
    std::vector<bool> c1_valid(K1_, true);
    upload_cents(h_c1_proj, h_c1_full, c1_valid, K1_,
                 d_route1_cents_proj_, d_route1_cents_full_, d_route1_norms_);

    std::vector<float> Pi2; init_jl_proj(d_, d_proj_, 43, Pi2);
    CUDA_CHECK(cudaMalloc(&d_Pi2_, (long long)d_proj_*d_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi2_, Pi2.data(), (long long)d_proj_*d_*sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> h_r1((long long)n_km*d_);
    for (int i=0;i<n_km;i++){int c1=h_assign1[i];for(int j=0;j<d_;j++)h_r1[(long long)i*d_+j]=h_x_km[(long long)i*d_+j]-h_c1_full[(long long)c1*d_+j];}
    {
        float *d_r1g, *d_r1pg;
        CUDA_CHECK(cudaMalloc(&d_r1g,  (long long)n_km*d_      *sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r1pg, (long long)n_km*d_proj_ *sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_r1g, h_r1.data(), (long long)n_km*d_*sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, n_km, d_, &one, d_Pi2_, d_, d_r1g, d_, &zero, d_y_proj, d_proj_));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_proj, (long long)n_km*d_proj_*sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_r1g); cudaFree(d_r1pg);
    }
    std::vector<float> h_all_c2_proj((long long)K1_*K2_*d_proj_,0.f), h_all_c2_full((long long)K1_*K2_*d_,0.f);
    std::vector<int>   h_assign2(n_km,0); std::vector<bool> h_c2_valid((long long)K1_*K2_,false);
    for (int c1=0;c1<K1_;c1++) {
        std::vector<int> idx; for(int i=0;i<n_km;i++) if(h_assign1[i]==c1) idx.push_back(i);
        int nc1=(int)idx.size(); if(!nc1) continue; int K2e=std::min(K2_,nc1);
        std::vector<float> gp((long long)nc1*d_proj_), gf((long long)nc1*d_);
        for(int k=0;k<nc1;k++){std::memcpy(gp.data()+(long long)k*d_proj_,h_y.data()+(long long)idx[k]*d_proj_,d_proj_*sizeof(float));std::memcpy(gf.data()+(long long)k*d_,h_r1.data()+(long long)idx[k]*d_,d_*sizeof(float));}
        std::vector<float> c2p,c2f; std::vector<int> a2;
        gpu_kmeans(gp.data(),gf.data(),nc1,K2e,c2p,c2f,a2);
        c2p.resize((long long)K2_*d_proj_,0.f); c2f.resize((long long)K2_*d_,0.f);
        std::memcpy(h_all_c2_proj.data()+(long long)c1*K2_*d_proj_,c2p.data(),(long long)K2_*d_proj_*sizeof(float));
        std::memcpy(h_all_c2_full.data()+(long long)c1*K2_*d_,c2f.data(),(long long)K2_*d_*sizeof(float));
        for(int k=0;k<nc1;k++) h_assign2[idx[k]]=a2[k];
        for(int k=0;k<K2e;k++) h_c2_valid[c1*K2_+k]=true;
    }
    upload_cents(h_all_c2_proj,h_all_c2_full,h_c2_valid,K1_*K2_,
                 d_route2_cents_proj_,d_route2_cents_full_,d_route2_norms_);

    std::vector<float> Pi3; init_jl_proj(d_, d_proj_, 44, Pi3);
    CUDA_CHECK(cudaMalloc(&d_Pi3_, (long long)d_proj_*d_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi3_, Pi3.data(), (long long)d_proj_*d_*sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> h_r2((long long)n_km*d_);
    for(int i=0;i<n_km;i++){int c1=h_assign1[i],c2=h_assign2[i];long long c12=(long long)c1*K2_+c2;for(int j=0;j<d_;j++)h_r2[(long long)i*d_+j]=h_r1[(long long)i*d_+j]-h_all_c2_full[c12*d_+j];}
    {
        float *d_r2g, *d_r2pg;
        CUDA_CHECK(cudaMalloc(&d_r2g,  (long long)n_km*d_      *sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r2pg, (long long)n_km*d_proj_ *sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_r2g,h_r2.data(),(long long)n_km*d_*sizeof(float),cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,d_proj_,n_km,d_,&one,d_Pi3_,d_,d_r2g,d_,&zero,d_y_proj,d_proj_));
        CUDA_CHECK(cudaMemcpy(h_y.data(),d_y_proj,(long long)n_km*d_proj_*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(d_r2g); cudaFree(d_r2pg);
    }
    long long n_c12=(long long)K1_*K2_;
    std::vector<float> h_all_c3_proj(n_c12*K3_*d_proj_,0.f), h_all_c3_full(n_c12*K3_*d_,0.f);
    std::vector<int>   h_assign3(n_km,0); std::vector<bool> h_c3_valid(n_c12*K3_,false);
    for(int c1=0;c1<K1_;c1++) for(int c2=0;c2<K2_;c2++){
        long long c12=(long long)c1*K2_+c2;
        std::vector<int> idx; for(int i=0;i<n_km;i++) if(h_assign1[i]==c1&&h_assign2[i]==c2) idx.push_back(i);
        int nc12=(int)idx.size(); if(!nc12) continue; int K3e=std::min(K3_,nc12);
        std::vector<float> gp((long long)nc12*d_proj_),gf((long long)nc12*d_);
        for(int k=0;k<nc12;k++){std::memcpy(gp.data()+(long long)k*d_proj_,h_y.data()+(long long)idx[k]*d_proj_,d_proj_*sizeof(float));std::memcpy(gf.data()+(long long)k*d_,h_r2.data()+(long long)idx[k]*d_,d_*sizeof(float));}
        std::vector<float> c3p,c3f; std::vector<int> a3;
        gpu_kmeans(gp.data(),gf.data(),nc12,K3e,c3p,c3f,a3);
        c3p.resize((long long)K3_*d_proj_,0.f); c3f.resize((long long)K3_*d_,0.f);
        std::memcpy(h_all_c3_proj.data()+(c12*K3_)*d_proj_,c3p.data(),(long long)K3_*d_proj_*sizeof(float));
        std::memcpy(h_all_c3_full.data()+(c12*K3_)*d_,c3f.data(),(long long)K3_*d_*sizeof(float));
        for(int k=0;k<nc12;k++) h_assign3[idx[k]]=a3[k];
        for(int k=0;k<K3e;k++) h_c3_valid[c12*K3_+k]=true;
    }
    upload_cents(h_all_c3_proj,h_all_c3_full,h_c3_valid,n_c12*K3_,
                 d_route3_cents_proj_,d_route3_cents_full_,d_route3_norms_);

    double sum_sq=0.0;
    for(int i=0;i<n_km;i++){int c1=h_assign1[i],c2=h_assign2[i],c3=h_assign3[i];long long c123=((long long)c1*K2_+c2)*K3_+c3;for(int j=0;j<d_;j++){float v=h_r2[(long long)i*d_+j]-h_all_c3_full[c123*d_+j];sum_sq+=(double)v*v;}}
    float sigma=(float)std::sqrt(sum_sq/(double)((long long)n_km*d_));
    auto fine_c1d=analytical_fine_c1d(Kr_,sigma);
    CUDA_CHECK(cudaMalloc(&d_fine_c1d_,Kr_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_,fine_c1d.data(),Kr_*sizeof(float),cudaMemcpyHostToDevice));

    cudaFree(d_x_km); cudaFree(d_y_proj);
    printf("[v39 train] total=%.1f ms  sigma_r3=%.4f\n", Ms(Clock::now()-T0).count(), sigma);
}

template<class Reader>
void HBlockIndex::add_impl(Reader& reader, int n)
{
    if (!d_Pi1_)      throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock supports one add() call");
    if (reader.dim != d_) throw std::runtime_error("reader dim mismatch with index dim");

    // reader.npts is `int` for I8BinReader, `long long` for BVecsReader --
    // widen unconditionally so both instantiations are exact, not just the
    // one that happens to match int.
    long long npts_ll = (long long)reader.npts;
    n = (n < 0) ? (int)npts_ll : (int)std::min((long long)n, npts_ll);
    if (n <= 0) throw std::runtime_error("add(): nothing to add");

    using Clock = std::chrono::high_resolution_clock;
    auto T_add = Clock::now();
    printf("[v39 add] n=%d  d=%d  add_batch_size=%d  mini_km_iters=%d\n",
           n, d_, add_batch_size_, mini_km_iters_);

    const int BATCH = 8192;
    const float one=1.f, zero=0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));

    auto T0 = Clock::now();
    float *d_x,*d_r1,*d_r2,*d_r3,*d_proj1,*d_proj2,*d_proj3;
    int8_t *d_x_i8;
    int *d_c1,*d_c2,*d_c3; uint8_t *d_fc; float *d_dots1;
    CUDA_CHECK(cudaMalloc(&d_x_i8,  (long long)BATCH*d_     *sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_x,     (long long)BATCH*d_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj1, (long long)BATCH*d_proj_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj2, (long long)BATCH*d_proj_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj3, (long long)BATCH*d_proj_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1,    (long long)BATCH*d_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2,    (long long)BATCH*d_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r3,    (long long)BATCH*d_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1,    BATCH*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c2,    BATCH*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c3,    BATCH*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fc,    (long long)BATCH*bpv_));
    CUDA_CHECK(cudaMalloc(&d_dots1, (long long)K1_*BATCH*sizeof(float)));

    std::vector<int>     h_code1(n), h_code2(n), h_code3(n);
    std::vector<uint8_t> h_fc_all((long long)n*bpv_);
    std::vector<float>   h_proj1_all((long long)n*d_proj_);
    // Replaces v36's h_x (a full n*d float array, ~40GB at SPACEV-100M) for
    // everything downstream of the encode loop: this is int8, ~4x smaller,
    // and is the actual raw payload later packed into region stores by
    // build_region_layout() -- never uploaded as a flat GPU-resident array.
    std::vector<int8_t>  h_base_i8_all((long long)n*d_);

    std::vector<int8_t> h_chunk;  // reused reader output buffer

    int n_batches_total = (n + BATCH - 1) / BATCH;
    int report_every = std::max(1, n_batches_total / 10);
    int batch_idx = 0;

    for (long long s = 0; s < n; s += add_batch_size_) {
        int want = (int)std::min((long long)add_batch_size_, (long long)n - s);
        int got = reader.read_batch(s, want, h_chunk);
        if (got != want)
            throw std::runtime_error("reader: short read during add()");
        std::memcpy(h_base_i8_all.data() + s*(long long)d_, h_chunk.data(), (size_t)got*(size_t)d_);

        for (int s2 = 0; s2 < got; s2 += BATCH) {
            int nb = std::min(BATCH, got - s2);
            long long g = s + s2;  // global vector-index offset into [0, n)

            CUDA_CHECK(cudaMemcpy(d_x_i8, h_chunk.data() + (long long)s2*d_,
                                  (long long)nb*d_*sizeof(int8_t), cudaMemcpyHostToDevice));
            {
                long long ne = (long long)nb * d_;
                int threads = 256;
                int blocks  = (int)std::min((ne + threads - 1) / threads, (long long)65535);
                cast_i8_to_f32_kernel<<<blocks, threads>>>(d_x_i8, d_x, ne);
                CUDA_CHECK(cudaGetLastError());
            }

            CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,d_proj_,nb,d_,&one,d_Pi1_,d_,d_x,d_,&zero,d_proj1,d_proj_));
            CUDA_CHECK(cudaMemcpy(h_proj1_all.data()+g*d_proj_,d_proj1,(long long)nb*d_proj_*sizeof(float),cudaMemcpyDeviceToHost));
            CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,K1_,nb,d_proj_,&one,d_route1_cents_proj_,d_proj_,d_proj1,d_proj_,&zero,d_dots1,K1_));
            hblock_v17::launch_assign_from_dots(d_dots1,d_route1_norms_,d_c1,K1_,nb,nullptr);
            hblock_v17::launch_subtract_centroid(d_x,d_c1,d_route1_cents_full_,d_r1,nb,d_,nullptr);
            CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,d_proj_,nb,d_,&one,d_Pi2_,d_,d_r1,d_,&zero,d_proj2,d_proj_));
            local_assign_kernel<<<(nb+255)/256,256>>>(d_proj2,d_route2_cents_proj_,d_route2_norms_,d_c1,d_c2,nb,d_proj_,K2_);
            {
                int *d_c12; CUDA_CHECK(cudaMalloc(&d_c12,nb*sizeof(int)));
                std::vector<int> h_c1b(nb),h_c2b(nb),h_c12(nb);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c1b.data(),d_c1,nb*sizeof(int),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_c2b.data(),d_c2,nb*sizeof(int),cudaMemcpyDeviceToHost));
                for(int i=0;i<nb;i++) h_c12[i]=h_c1b[i]*K2_+h_c2b[i];
                CUDA_CHECK(cudaMemcpy(d_c12,h_c12.data(),nb*sizeof(int),cudaMemcpyHostToDevice));
                hblock_v17::launch_subtract_centroid(d_r1,d_c12,d_route2_cents_full_,d_r2,nb,d_,nullptr);
                CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,d_proj_,nb,d_,&one,d_Pi3_,d_,d_r2,d_,&zero,d_proj3,d_proj_));
                local_assign_kernel<<<(nb+255)/256,256>>>(d_proj3,d_route3_cents_proj_,d_route3_norms_,d_c12,d_c3,nb,d_proj_,K3_);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<int> h_c3b(nb),h_c123(nb);
                CUDA_CHECK(cudaMemcpy(h_c3b.data(),d_c3,nb*sizeof(int),cudaMemcpyDeviceToHost));
                for(int i=0;i<nb;i++) h_c123[i]=h_c12[i]*K3_+h_c3b[i];
                std::memcpy(h_code1.data()+g,h_c1b.data(),nb*sizeof(int));
                std::memcpy(h_code2.data()+g,h_c2b.data(),nb*sizeof(int));
                std::memcpy(h_code3.data()+g,h_c3b.data(),nb*sizeof(int));
                int *d_c123; CUDA_CHECK(cudaMalloc(&d_c123,nb*sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_c123,h_c123.data(),nb*sizeof(int),cudaMemcpyHostToDevice));
                hblock_v17::launch_subtract_centroid(d_r2,d_c123,d_route3_cents_full_,d_r3,nb,d_,nullptr);
                hblock_v17::launch_fine_encode(d_r3,d_fine_c1d_,d_fc,nb,d_,Kr_,Br_,bpv_,nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_fc_all.data()+g*bpv_,d_fc,(long long)nb*bpv_,cudaMemcpyDeviceToHost));
                cudaFree(d_c12); cudaFree(d_c123);
            }
            batch_idx++;
            if(batch_idx % report_every == 0 || batch_idx == n_batches_total)
                printf("  [encode] %d/%d batches  %.1f ms\n",
                       batch_idx, n_batches_total, Ms(Clock::now()-T0).count());
        }
    }
    cudaFree(d_x_i8);
    cudaFree(d_x);cudaFree(d_proj1);cudaFree(d_proj2);cudaFree(d_proj3);
    cudaFree(d_r1);cudaFree(d_r2);cudaFree(d_r3);
    cudaFree(d_c1);cudaFree(d_c2);cudaFree(d_c3);cudaFree(d_fc);cudaFree(d_dots1);
    printf("  [encode total] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    long long K2K3=(long long)K2_*K3_;
    std::vector<int> order(n);
    std::iota(order.begin(),order.end(),0);
    // Build flat cell-code keys on host, then sort on GPU with CUB radix sort
    std::vector<long long> h_keys(n);
    for (int i = 0; i < n; i++)
        h_keys[i] = (long long)h_code1[i]*K2K3 + h_code2[i]*K3_ + h_code3[i];
    {
        long long *d_keys_in=nullptr, *d_keys_out=nullptr;
        int       *d_vals_in=nullptr, *d_vals_out=nullptr;
        CUDA_CHECK(cudaMalloc(&d_keys_in,  (size_t)n*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_keys_out, (size_t)n*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_vals_in,  (size_t)n*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vals_out, (size_t)n*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys.data(), (size_t)n*sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals_in, order.data(),  (size_t)n*sizeof(int),       cudaMemcpyHostToDevice));
        void *d_tmp=nullptr; size_t tmp_bytes=0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
        cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(order.data(), d_vals_out, (size_t)n*sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_keys_in); cudaFree(d_keys_out);
        cudaFree(d_vals_in); cudaFree(d_vals_out);
        cudaFree(d_tmp);
    }
    printf("  [sort GPU radix] %.1f ms\n", Ms(Clock::now()-T0).count());

    // per_cell_blk_sizes[c123] = actual cluster sizes; empty -> single block or not reordered
    std::vector<std::vector<int>> per_cell_blk_sizes((long long)K1_*K2_*K3_);

    if (mini_km_iters_ > 0) {
        T0 = Clock::now();
        int n_cells_reordered = 0;
        for (int i = 0, j; i < n; i = j) {
            int c1=h_code1[order[i]], c2=h_code2[order[i]], c3=h_code3[order[i]];
            for (j=i; j<n && h_code1[order[j]]==c1 && h_code2[order[j]]==c2 && h_code3[order[j]]==c3; ++j) {}
            int N_cell = j - i;
            if (N_cell <= leaf_size_) continue;
            int K = (N_cell + leaf_size_ - 1) / leaf_size_;
            std::vector<float> cell_proj((long long)N_cell * d_proj_);
            for (int vi = 0; vi < N_cell; vi++) {
                int oid = order[i + vi];
                std::memcpy(cell_proj.data() + (long long)vi*d_proj_,
                            h_proj1_all.data() + (long long)oid*d_proj_,
                            d_proj_ * sizeof(float));
            }
            std::vector<float> cents((long long)K * d_proj_, 0.f);
            for (int k = 0; k < K; k++) {
                int idx = (long long)k * N_cell / K;
                std::memcpy(cents.data() + (long long)k*d_proj_,
                            cell_proj.data() + (long long)idx*d_proj_,
                            d_proj_ * sizeof(float));
            }
            std::vector<int> assign(N_cell, 0);
            for (int iter = 0; iter < mini_km_iters_; iter++) {
                for (int vi = 0; vi < N_cell; vi++) {
                    const float* xi = cell_proj.data() + (long long)vi*d_proj_;
                    float best = 1e30f; int bi = 0;
                    for (int k = 0; k < K; k++) {
                        const float* ck = cents.data() + (long long)k*d_proj_;
                        float dist = 0.f;
                        for (int jp = 0; jp < d_proj_; jp++) { float v = xi[jp]-ck[jp]; dist += v*v; }
                        if (dist < best) { best = dist; bi = k; }
                    }
                    assign[vi] = bi;
                }
                std::fill(cents.begin(), cents.end(), 0.f);
                std::vector<int> cnt(K, 0);
                for (int vi = 0; vi < N_cell; vi++) {
                    int k = assign[vi]; cnt[k]++;
                    const float* xi = cell_proj.data() + (long long)vi*d_proj_;
                    float* ck = cents.data() + (long long)k*d_proj_;
                    for (int jp = 0; jp < d_proj_; jp++) ck[jp] += xi[jp];
                }
                for (int k = 0; k < K; k++)
                    if (cnt[k] > 0)
                        for (int jp = 0; jp < d_proj_; jp++) cents[(long long)k*d_proj_+jp] /= cnt[k];
            }
            {
                std::vector<std::tuple<float,int,int>> all_pairs;
                all_pairs.reserve((size_t)N_cell * K);
                for (int vi = 0; vi < N_cell; vi++) {
                    const float* xi = cell_proj.data() + (long long)vi*d_proj_;
                    for (int k = 0; k < K; k++) {
                        const float* ck = cents.data() + (long long)k*d_proj_;
                        float dist = 0.f;
                        for (int jp = 0; jp < d_proj_; jp++) { float v=xi[jp]-ck[jp]; dist+=v*v; }
                        all_pairs.emplace_back(dist, vi, k);
                    }
                }
                std::sort(all_pairs.begin(), all_pairs.end());
                std::vector<int>  cap(K, leaf_size_);
                std::vector<bool> done(N_cell, false);
                int n_done = 0;
                for (auto& [dd, vi, k] : all_pairs) {
                    if (!done[vi] && cap[k] > 0) {
                        assign[vi] = k; done[vi] = true; cap[k]--;
                        if (++n_done == N_cell) break;
                    }
                }
            }
            {
                long long c123 = (long long)c1*K2_*K3_ + c2*K3_ + c3;
                std::vector<int> cnt(K, 0);
                for (int vi = 0; vi < N_cell; vi++) cnt[assign[vi]]++;
                per_cell_blk_sizes[c123] = cnt;
            }
            std::vector<int> cell_local(N_cell);
            std::iota(cell_local.begin(), cell_local.end(), 0);
            std::stable_sort(cell_local.begin(), cell_local.end(),
                             [&](int a, int b){ return assign[a] < assign[b]; });
            std::vector<int> tmp(N_cell);
            for (int vi = 0; vi < N_cell; vi++) tmp[vi] = order[i + cell_local[vi]];
            for (int vi = 0; vi < N_cell; vi++) order[i + vi] = tmp[vi];
            n_cells_reordered++;
        }
        printf("  [balanced kmeans] %d cells  %.1f ms\n",
               n_cells_reordered, Ms(Clock::now()-T0).count());
    }

    T0 = Clock::now();
    long long n_cells=(long long)K1_*K2_*K3_;
    std::vector<int> pair_cnt(n_cells,0);
    for(int i=0,j;i<n;i=j){
        int c1=h_code1[order[i]],c2=h_code2[order[i]],c3=h_code3[order[i]];
        for(j=i;j<n&&h_code1[order[j]]==c1&&h_code2[order[j]]==c2&&h_code3[order[j]]==c3;++j){}
        pair_cnt[(long long)c1*K2K3+c2*K3_+c3]=(j-i+leaf_size_-1)/leaf_size_;
    }
    std::vector<int> pair_start(n_cells,0);
    int total_blocks=0;
    for(long long p=0;p<n_cells;++p){pair_start[p]=total_blocks;total_blocks+=pair_cnt[p];}
    n_leaf_blocks_=total_blocks;
    max_blk_per_cell_=*std::max_element(pair_cnt.begin(),pair_cnt.end());
    printf("  [leaf layout] %d blocks  max_per_cell=%d  %.1f ms\n",
           total_blocks, max_blk_per_cell_, Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    std::vector<uint8_t> h_leaf_codes((long long)total_blocks*bpv_*leaf_size_,0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks*leaf_size_,-1);
    std::vector<int>     h_leaf_sizes(total_blocks,0);
    h_block_cell_id_.assign(total_blocks,0);
    h_block_cent_.assign((long long)total_blocks*d_, 0.f);
    h_vec_local_pos_.assign(n, -1);

    for(int i=0,j;i<n;i=j){
        int c1=h_code1[order[i]],c2=h_code2[order[i]],c3=h_code3[order[i]];
        for(j=i;j<n&&h_code1[order[j]]==c1&&h_code2[order[j]]==c2&&h_code3[order[j]]==c3;++j){}
        long long cidx=(long long)c1*K2K3+c2*K3_+c3;
        int base_blk=pair_start[cidx];
        const auto& blk_szs = per_cell_blk_sizes[cidx];
        auto pack_one = [&](int blk, int pos, int oid) {
            h_leaf_ids[(long long)blk*leaf_size_+pos]=oid;
            h_vec_local_pos_[oid] = pos;
            const uint8_t* src=h_fc_all.data()+(long long)oid*bpv_;
            uint8_t* dst_base=h_leaf_codes.data()+(long long)blk*bpv_*leaf_size_;
            for(int b=0;b<bpv_;b++) dst_base[(long long)b*leaf_size_+pos]=src[b];
            h_leaf_sizes[blk]=std::max(h_leaf_sizes[blk],pos+1);
            h_block_cell_id_[blk]=(int)cidx;
            // v39: block centroid accumulated from the int8 region source
            // (h_base_i8_all), not a resident float base array.
            const int8_t* xv=h_base_i8_all.data()+(long long)oid*d_;
            float* bc=h_block_cent_.data()+(long long)blk*d_;
            for(int dim=0;dim<d_;dim++) bc[dim]+=(float)xv[dim];
        };
        if (blk_szs.empty()) {
            // Single-block cell: fixed-128 cut (original behavior)
            for(int vi=i;vi<j;++vi){
                int in_blk=vi-i;
                pack_one(base_blk+in_blk/leaf_size_, in_blk%leaf_size_, order[vi]);
            }
        } else {
            // Multi-block cell: cluster boundary = block boundary
            int vi = i;
            for (int k=0; k<(int)blk_szs.size(); k++)
                for (int pos=0; pos<blk_szs[k]; pos++, vi++)
                    pack_one(base_blk+k, pos, order[vi]);
        }
    }
    for(int b=0;b<total_blocks;b++){
        int sz=h_leaf_sizes[b]; if(sz==0) continue;
        float inv=1.0f/sz;
        float* bc=h_block_cent_.data()+(long long)b*d_;
        for(int dim=0;dim<d_;dim++) bc[dim]*=inv;
    }
    printf("  [pack+centroids] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    {
        float *d_cent=nullptr, *d_proj_buf=nullptr;
        CUDA_CHECK(cudaMalloc(&d_cent,     (long long)total_blocks*d_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_proj_buf, (long long)total_blocks*d_proj_*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent, h_block_cent_.data(),
                              (long long)total_blocks*d_*sizeof(float), cudaMemcpyHostToDevice));
        const float one=1.f, zero=0.f;
        CUBLAS_CHECK(cublasSgemm(cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N, d_proj_, total_blocks, d_,
            &one, d_Pi1_, d_, d_cent, d_, &zero, d_proj_buf, d_proj_));
        h_block_cent_proj_.resize((long long)total_blocks*d_proj_);
        CUDA_CHECK(cudaMemcpy(h_block_cent_proj_.data(), d_proj_buf,
                              (long long)total_blocks*d_proj_*sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_cent); cudaFree(d_proj_buf);
    }
    h_block_cent_norm_.resize(total_blocks);
    for(int b=0;b<total_blocks;b++){
        const float* p=h_block_cent_proj_.data()+(long long)b*d_proj_;
        float norm=0.f;
        for(int jp=0;jp<d_proj_;jp++) norm+=p[jp]*p[jp];
        h_block_cent_norm_[b]=norm;
    }
    printf("  [proj GEMM + norms] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_,n_cells*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_,n_cells*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_,pair_start.data(),n_cells*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_,pair_cnt.data(),  n_cells*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_,total_blocks*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_,h_leaf_sizes.data(),total_blocks*sizeof(int),cudaMemcpyHostToDevice));
    h_pair_blk_start_cpu_=pair_start;
    h_pair_blk_count_cpu_=pair_cnt;
    h_leaf_ids_cpu_   = h_leaf_ids;
    h_leaf_sizes_cpu_ = h_leaf_sizes;
    printf("  [upload block metadata] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    {
        std::vector<float> h_c1f((long long)K1_*d_),h_c2f((long long)K1_*K2_*d_),h_c3f((long long)K1_*K2_*K3_*d_);
        CUDA_CHECK(cudaMemcpy(h_c1f.data(),d_route1_cents_full_,(long long)K1_*d_*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_c2f.data(),d_route2_cents_full_,(long long)K1_*K2_*d_*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_c3f.data(),d_route3_cents_full_,(long long)K1_*K2_*K3_*d_*sizeof(float),cudaMemcpyDeviceToHost));
        h_leaf_abs_cents_.resize((long long)n_cells*d_);
        for(int c1=0;c1<K1_;c1++) for(int c2=0;c2<K2_;c2++){
            long long c12=(long long)c1*K2_+c2;
            for(int c3=0;c3<K3_;c3++){
                long long c123=c12*K3_+c3;
                float* abs=h_leaf_abs_cents_.data()+c123*d_;
                const float* a1=h_c1f.data()+(long long)c1*d_;
                const float* a2=h_c2f.data()+c12*d_;
                const float* a3=h_c3f.data()+c123*d_;
                for(int dim=0;dim<d_;dim++) abs[dim]=a1[dim]+a2[dim]+a3[dim];
            }
        }
    }
    printf("  [abs centroids] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    {
        long long proj_b = (long long)total_blocks * d_proj_ * sizeof(float);
        long long norm_b = (long long)total_blocks * sizeof(float);
        if(d_blk_proj_gpu_) cudaFree(d_blk_proj_gpu_);
        if(d_blk_norm_gpu_) cudaFree(d_blk_norm_gpu_);
        CUDA_CHECK(cudaMalloc(&d_blk_proj_gpu_, proj_b));
        CUDA_CHECK(cudaMalloc(&d_blk_norm_gpu_, norm_b));
        CUDA_CHECK(cudaMemcpy(d_blk_proj_gpu_, h_block_cent_proj_.data(), proj_b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_blk_norm_gpu_, h_block_cent_norm_.data(), norm_b, cudaMemcpyHostToDevice));
    }
    printf("  [upload proj/norm GPU] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    printf("  [graph build] n_blocks=%d degree=%d n_c2_nbrs=%d n_c1_nbrs=%d ...\n",
           total_blocks, graph_degree_, n_c2_nbrs_, n_c1_nbrs_);
    build_block_graph(pair_start, pair_cnt);
    printf("  [graph build total] %.1f ms\n", Ms(Clock::now()-T0).count());

    T0 = Clock::now();
    build_region_layout(h_leaf_codes, h_leaf_ids, h_leaf_sizes, h_base_i8_all, total_blocks);
    printf("  [region layout total] %.1f ms\n", Ms(Clock::now()-T0).count());

    ntotal_=n;
    h_block_cent_.clear(); h_block_cent_.shrink_to_fit();

    T0 = Clock::now();
    alloc_workspace();
    printf("  [alloc workspace] %.1f ms\n", Ms(Clock::now()-T0).count());

    printf("[v39 add total] %.1f ms  blocks=%d  code_regions=%d  raw_regions=%d\n",
           Ms(Clock::now()-T_add).count(), total_blocks, n_code_regions_, n_raw_regions_);
}

void HBlockIndex::add(I8BinReader& reader, int n) { add_impl(reader, n); }
void HBlockIndex::add(BVecsReader& reader, int n) { add_impl(reader, n); }

void HBlockIndex::build_region_layout(
    const std::vector<uint8_t>& h_leaf_codes,
    const std::vector<int>& h_leaf_ids,
    const std::vector<int>& h_leaf_sizes,
    const std::vector<int8_t>& h_raw_all,
    int total_blocks)
{
    // ── Code regions: pack each block's (codes + vector ids) contiguously,
    // filling regions up to region_bytes_ before starting a new one. v39
    // uses physical block order (no graph-aware repacking yet -- matches
    // the design-doc staging: get correctness + real reuse measurement
    // first, layout locality is a later optimization on top of this).
    const size_t code_rec_codes   = (size_t)bpv_ * leaf_size_;
    const size_t code_rec_ids_off = align4(code_rec_codes);
    const size_t code_rec_bytes   = code_rec_ids_off + (size_t)leaf_size_ * sizeof(int);
    if (code_rec_bytes > (size_t)region_bytes_)
        throw std::runtime_error("build_region_layout: region_bytes too small for one block's code record");

    h_block_code_region_.assign(total_blocks, -1);
    h_block_code_offset_.assign(total_blocks, -1);
    {
        std::vector<uint8_t> code_store;
        size_t cur_off = (size_t)region_bytes_;  // forces a new region on block 0
        int cur_region = -1;
        for (int b = 0; b < total_blocks; b++) {
            if (cur_off + code_rec_bytes > (size_t)region_bytes_) {
                cur_region++;
                code_store.resize(code_store.size() + region_bytes_, 0);
                cur_off = 0;
            }
            h_block_code_region_[b] = cur_region;
            h_block_code_offset_[b] = (int)cur_off;
            uint8_t* rec = code_store.data() + (size_t)cur_region * region_bytes_ + cur_off;
            std::memcpy(rec, h_leaf_codes.data() + (size_t)b * code_rec_codes, code_rec_codes);
            std::memcpy(rec + code_rec_ids_off, h_leaf_ids.data() + (size_t)b * leaf_size_,
                        (size_t)leaf_size_ * sizeof(int));
            cur_off += code_rec_bytes;
        }
        n_code_regions_ = cur_region + 1;
        h_code_region_store_ = std::move(code_store);
    }

    // ── Raw regions: one block's leaf_size_*d_ raw int8 bytes per record,
    // same block-major grouping so the leaf kernel can locate a candidate's
    // raw vector via (leaf_blk, local_pos) without a second id-indexed
    // region table.
    const size_t raw_rec_bytes = (size_t)leaf_size_ * d_;
    if (raw_rec_bytes > (size_t)region_bytes_)
        throw std::runtime_error("build_region_layout: region_bytes too small for one block's raw record");

    h_block_raw_region_.assign(total_blocks, -1);
    h_block_raw_offset_.assign(total_blocks, -1);
    {
        std::vector<uint8_t> raw_store;
        size_t cur_off = (size_t)region_bytes_;
        int cur_region = -1;
        for (int b = 0; b < total_blocks; b++) {
            if (cur_off + raw_rec_bytes > (size_t)region_bytes_) {
                cur_region++;
                raw_store.resize(raw_store.size() + region_bytes_, 0);
                cur_off = 0;
            }
            h_block_raw_region_[b] = cur_region;
            h_block_raw_offset_[b] = (int)cur_off;
            uint8_t* rec = raw_store.data() + (size_t)cur_region * region_bytes_ + cur_off;
            int sz = h_leaf_sizes[b];
            for (int pos = 0; pos < sz; pos++) {
                int oid = h_leaf_ids[(size_t)b * leaf_size_ + pos];
                if (oid < 0) continue;
                std::memcpy(rec + (size_t)pos * d_, h_raw_all.data() + (size_t)oid * d_, d_);
            }
            cur_off += raw_rec_bytes;
        }
        n_raw_regions_ = cur_region + 1;
        h_raw_region_store_ = std::move(raw_store);
    }

    printf("  [region layout] %d code regions (%.2f GB), %d raw regions (%.2f GB), region_bytes=%d\n",
           n_code_regions_, (double)h_code_region_store_.size()/1e9,
           n_raw_regions_,  (double)h_raw_region_store_.size()/1e9,
           region_bytes_);

    auto up_i = [](int*& d_ptr, const std::vector<int>& h) {
        if (d_ptr) cudaFree(d_ptr);
        CUDA_CHECK(cudaMalloc(&d_ptr, h.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
    };
    up_i(d_block_code_region_, h_block_code_region_);
    up_i(d_block_code_offset_, h_block_code_offset_);
    up_i(d_block_raw_region_,  h_block_raw_region_);
    up_i(d_block_raw_offset_,  h_block_raw_offset_);
    up_i(d_vec_local_pos_,     h_vec_local_pos_);

    // Bounded GPU region pool + indirection, empty until search() stages
    // regions on demand.
    if (d_code_region_pool_) cudaFree(d_code_region_pool_);
    if (d_raw_region_pool_)  cudaFree(d_raw_region_pool_);
    CUDA_CHECK(cudaMalloc(&d_code_region_pool_, (long long)gpu_code_region_cap_ * region_bytes_));
    CUDA_CHECK(cudaMalloc(&d_raw_region_pool_,  (long long)gpu_raw_region_cap_  * region_bytes_));

    h_code_region_slot_.assign(n_code_regions_, -1);
    h_raw_region_slot_.assign(n_raw_regions_, -1);
    code_region_last_call_.assign(n_code_regions_, -1);
    raw_region_last_call_.assign(n_raw_regions_, -1);
    call_id_ = 0;
    code_pool_region_of_slot_.assign(gpu_code_region_cap_, -1);
    raw_pool_region_of_slot_.assign(gpu_raw_region_cap_, -1);
    code_lru_.clear();
    raw_lru_.clear();
    code_lru_pos_.assign(n_code_regions_, code_lru_.end());
    raw_lru_pos_.assign(n_raw_regions_, raw_lru_.end());

    if (d_code_region_slot_) cudaFree(d_code_region_slot_);
    if (d_raw_region_slot_)  cudaFree(d_raw_region_slot_);
    CUDA_CHECK(cudaMalloc(&d_code_region_slot_, (long long)n_code_regions_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_raw_region_slot_,  (long long)n_raw_regions_  * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_code_region_slot_, 0xFF, (long long)n_code_regions_ * sizeof(int))); // all -1
    CUDA_CHECK(cudaMemset(d_raw_region_slot_,  0xFF, (long long)n_raw_regions_  * sizeof(int)));

    if (gpu_code_region_cap_ >= n_code_regions_)
        fprintf(stderr, "[v39 add] warning: gpu_code_region_cap (%d) >= n_code_regions (%d) "
                "-- the GPU pool can hold the whole code store; the out-of-core fetch/evict "
                "path won't actually be exercised at this scale\n",
                gpu_code_region_cap_, n_code_regions_);
    if (gpu_raw_region_cap_ >= n_raw_regions_)
        fprintf(stderr, "[v39 add] warning: gpu_raw_region_cap (%d) >= n_raw_regions (%d) "
                "-- the GPU pool can hold the whole raw store; the out-of-core fetch/evict "
                "path won't actually be exercised at this scale\n",
                gpu_raw_region_cap_, n_raw_regions_);
}

long long HBlockIndex::fetch_code_regions(const std::vector<int>& needed_region_ids) const
{
    stat_code_region_reqs_ += (long long)needed_region_ids.size();
    std::vector<int> uniq = needed_region_ids;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    stat_code_region_unique_ += (long long)uniq.size();

    // Correctness guard, not a soft warning: every region this batch's
    // kernel launch will touch must stay resident for the *entire* launch.
    // If the batch's distinct-region working set exceeds the pool, a later
    // region in this same call would LRU-evict one still needed by an
    // earlier-loaded (but not yet consumed) pair -- silent wrong-data
    // corruption, not a crash, since the kernel only trusts the slot table.
    // v39 does not split a batch into region-fitting waves yet (that is
    // the natural next step, mirroring the plan docs' wave-based
    // streaming); fail loudly here instead of risking that corruption.
    if ((int)uniq.size() > gpu_code_region_cap_)
        throw std::runtime_error(
            "fetch_code_regions: batch needs " + std::to_string(uniq.size()) +
            " distinct code regions but gpu_code_region_cap is " +
            std::to_string(gpu_code_region_cap_) + " -- increase gpu_code_region_cap, "
            "raise region_bytes (fewer, bigger regions), or shrink batch_size_/ef");

    long long bytes_moved = 0;
    cudaStream_t s = ws_.stream;
    for (int region : uniq) {
        if (region < 0 || region >= n_code_regions_) continue;

        if (code_region_last_call_[region] != call_id_) {
            code_region_last_call_[region] = call_id_;
            stat_code_region_true_unique_++;
        }

        if (h_code_region_slot_[region] >= 0) {
            code_lru_.erase(code_lru_pos_[region]);
            code_lru_.push_front(region);
            code_lru_pos_[region] = code_lru_.begin();
            continue;
        }

        int slot;
        if ((int)code_lru_.size() < gpu_code_region_cap_) {
            slot = (int)code_lru_.size();
        } else {
            int victim = code_lru_.back();
            code_lru_.pop_back();
            slot = h_code_region_slot_[victim];
            h_code_region_slot_[victim] = -1;
            code_pool_region_of_slot_[slot] = -1;
            int neg1 = -1;
            CUDA_CHECK(cudaMemcpyAsync(d_code_region_slot_ + victim, &neg1, sizeof(int),
                                       cudaMemcpyHostToDevice, s));
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_code_region_pool_ + (long long)slot * region_bytes_,
            h_code_region_store_.data() + (long long)region * region_bytes_,
            region_bytes_, cudaMemcpyHostToDevice, s));
        bytes_moved += region_bytes_;

        h_code_region_slot_[region] = slot;
        code_pool_region_of_slot_[slot] = region;
        code_lru_.push_front(region);
        code_lru_pos_[region] = code_lru_.begin();
        CUDA_CHECK(cudaMemcpyAsync(d_code_region_slot_ + region, &slot, sizeof(int),
                                   cudaMemcpyHostToDevice, s));
    }
    stat_code_bytes_h2d_ += bytes_moved;
    return bytes_moved;
}

long long HBlockIndex::fetch_raw_regions(const std::vector<int>& needed_region_ids) const
{
    stat_raw_region_reqs_ += (long long)needed_region_ids.size();
    std::vector<int> uniq = needed_region_ids;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    stat_raw_region_unique_ += (long long)uniq.size();

    // See the identical guard in fetch_code_regions() -- same correctness
    // argument, applied to the raw-vector pool.
    if ((int)uniq.size() > gpu_raw_region_cap_)
        throw std::runtime_error(
            "fetch_raw_regions: batch needs " + std::to_string(uniq.size()) +
            " distinct raw regions but gpu_raw_region_cap is " +
            std::to_string(gpu_raw_region_cap_) + " -- increase gpu_raw_region_cap, "
            "raise region_bytes (fewer, bigger regions), or shrink batch_size_/ef");

    long long bytes_moved = 0;
    cudaStream_t s = ws_.stream;
    for (int region : uniq) {
        if (region < 0 || region >= n_raw_regions_) continue;

        if (raw_region_last_call_[region] != call_id_) {
            raw_region_last_call_[region] = call_id_;
            stat_raw_region_true_unique_++;
        }

        if (h_raw_region_slot_[region] >= 0) {
            raw_lru_.erase(raw_lru_pos_[region]);
            raw_lru_.push_front(region);
            raw_lru_pos_[region] = raw_lru_.begin();
            continue;
        }

        int slot;
        if ((int)raw_lru_.size() < gpu_raw_region_cap_) {
            slot = (int)raw_lru_.size();
        } else {
            int victim = raw_lru_.back();
            raw_lru_.pop_back();
            slot = h_raw_region_slot_[victim];
            h_raw_region_slot_[victim] = -1;
            raw_pool_region_of_slot_[slot] = -1;
            int neg1 = -1;
            CUDA_CHECK(cudaMemcpyAsync(d_raw_region_slot_ + victim, &neg1, sizeof(int),
                                       cudaMemcpyHostToDevice, s));
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_raw_region_pool_ + (long long)slot * region_bytes_,
            h_raw_region_store_.data() + (long long)region * region_bytes_,
            region_bytes_, cudaMemcpyHostToDevice, s));
        bytes_moved += region_bytes_;

        h_raw_region_slot_[region] = slot;
        raw_pool_region_of_slot_[slot] = region;
        raw_lru_.push_front(region);
        raw_lru_pos_[region] = raw_lru_.begin();
        CUDA_CHECK(cudaMemcpyAsync(d_raw_region_slot_ + region, &slot, sizeof(int),
                                   cudaMemcpyHostToDevice, s));
    }
    stat_raw_bytes_h2d_ += bytes_moved;
    return bytes_moved;
}

static inline float l2sq(const float* a, const float* b, int d)
{
    float s=0.f;
    for(int i=0;i<d;i++){float v=a[i]-b[i];s+=v*v;}
    return s;
}

void HBlockIndex::build_block_graph(
    const std::vector<int>& h_pair_blk_start,
    const std::vector<int>& h_pair_blk_count)
{
    int n_cells  = K1_*K2_*K3_;
    int n_blocks = n_leaf_blocks_;
    int deg      = graph_degree_;

    std::vector<std::vector<int>> nearest_c2(K1_*K2_);
    for(int c1=0;c1<K1_;c1++){
        for(int c2=0;c2<K2_;c2++){
            const float* ref=h_leaf_abs_cents_.data()+(long long)(c1*K2_+c2)*K3_*d_;
            std::vector<std::pair<float,int>> row;
            for(int c2p=0;c2p<K2_;c2p++){
                if(c2p==c2){row.push_back({1e30f,c2p});continue;}
                const float* cp=h_leaf_abs_cents_.data()+(long long)(c1*K2_+c2p)*K3_*d_;
                row.push_back({l2sq(ref,cp,d_),c2p});
            }
            std::sort(row.begin(),row.end());
            nearest_c2[c1*K2_+c2].clear();
            for(auto& [d2,idx]:row) nearest_c2[c1*K2_+c2].push_back(idx);
        }
    }

    std::vector<std::vector<int>> nearest_c1(K1_);
    for(int c1=0;c1<K1_;c1++){
        const float* ref=h_leaf_abs_cents_.data()+(long long)c1*K2_*K3_*d_;
        std::vector<std::pair<float,int>> row;
        for(int c1p=0;c1p<K1_;c1p++){
            if(c1p==c1){row.push_back({1e30f,c1p});continue;}
            const float* cp=h_leaf_abs_cents_.data()+(long long)c1p*K2_*K3_*d_;
            row.push_back({l2sq(ref,cp,d_),c1p});
        }
        std::sort(row.begin(),row.end());
        for(auto& [d2,idx]:row) nearest_c1[c1].push_back(idx);
    }

    using Clock = std::chrono::high_resolution_clock;
    auto Tg = Clock::now();

    std::vector<int> h_csr_start(n_cells+1, 0);
    std::vector<int> h_csr_list;
    std::vector<int> h_cell_id(n_blocks, 0);
    h_csr_list.reserve((size_t)n_cells * 256);

    for(int c1=0;c1<K1_;c1++) for(int c2=0;c2<K2_;c2++) for(int c3=0;c3<K3_;c3++){
        long long c123=(long long)(c1*K2_+c2)*K3_+c3;
        int blk_base=h_pair_blk_start[c123], blk_cnt=h_pair_blk_count[c123];
        for(int bi=0;bi<blk_cnt;bi++) h_cell_id[blk_base+bi]=(int)c123;
        if(blk_cnt==0){ h_csr_start[c123+1]=h_csr_start[c123]; continue; }

        std::vector<int> cand_blocks;
        cand_blocks.reserve(512);

        for(int c3p=0;c3p<K3_;c3p++){
            long long cp=(long long)(c1*K2_+c2)*K3_+c3p;
            int bs=h_pair_blk_start[cp], bc=h_pair_blk_count[cp];
            for(int bi=0;bi<bc;bi++) cand_blocks.push_back(bs+bi);
        }

        auto& nc2=nearest_c2[c1*K2_+c2];
        int n_c2_use=std::min(n_c2_nbrs_,(int)nc2.size());
        for(int ni=0;ni<n_c2_use;ni++){
            int c2p=nc2[ni];
            for(int c3p=0;c3p<K3_;c3p++){
                long long cp=(long long)(c1*K2_+c2p)*K3_+c3p;
                int bs=h_pair_blk_start[cp], bc=h_pair_blk_count[cp];
                for(int bi=0;bi<bc;bi++) cand_blocks.push_back(bs+bi);
            }
        }

        auto& nc1=nearest_c1[c1];
        int n_c1_use=std::min(n_c1_nbrs_,(int)nc1.size());
        for(int ni=0;ni<n_c1_use;ni++){
            int c1p=nc1[ni];
            const float* ref_c2=h_leaf_abs_cents_.data()+(long long)(c1*K2_+c2)*K3_*d_;
            std::vector<std::pair<float,int>> c2p_dists;
            for(int c2p=0;c2p<K2_;c2p++){
                const float* cp2=h_leaf_abs_cents_.data()+(long long)(c1p*K2_+c2p)*K3_*d_;
                c2p_dists.push_back({l2sq(ref_c2,cp2,d_),c2p});
            }
            std::partial_sort(c2p_dists.begin(),c2p_dists.begin()+2,c2p_dists.end());
            for(int r=0;r<2;r++){
                int c2p=c2p_dists[r].second;
                for(int c3p=0;c3p<K3_;c3p++){
                    long long cp=(long long)(c1p*K2_+c2p)*K3_+c3p;
                    int bs=h_pair_blk_start[cp], bc=h_pair_blk_count[cp];
                    for(int bi=0;bi<bc;bi++) cand_blocks.push_back(bs+bi);
                }
            }
        }

        std::sort(cand_blocks.begin(), cand_blocks.end());
        cand_blocks.erase(std::unique(cand_blocks.begin(), cand_blocks.end()),
                          cand_blocks.end());
        if((int)cand_blocks.size() > max_cand_blocks_){
            const float* cell_proj=h_block_cent_proj_.data()+(long long)blk_base*d_proj_;
            float cell_norm=h_block_cent_norm_[blk_base];
            std::vector<std::pair<float,int>> ranked;
            ranked.reserve(cand_blocks.size());
            for(int bp:cand_blocks){
                const float* pbp=h_block_cent_proj_.data()+(long long)bp*d_proj_;
                float dot=0.f;
                for(int jp=0;jp<d_proj_;jp++) dot+=cell_proj[jp]*pbp[jp];
                ranked.push_back({cell_norm+h_block_cent_norm_[bp]-2.f*dot, bp});
            }
            std::partial_sort(ranked.begin(),ranked.begin()+max_cand_blocks_,ranked.end());
            cand_blocks.resize(max_cand_blocks_);
            for(int i=0;i<max_cand_blocks_;i++) cand_blocks[i]=ranked[i].second;
        }

        h_csr_start[c123+1] = h_csr_start[c123] + (int)cand_blocks.size();
        for(int bp : cand_blocks) h_csr_list.push_back(bp);
    }

    int total_cands = (int)h_csr_list.size();
    printf("    [graph/csr_build] %.1f ms  cells=%d  total_cands=%d  avg_cands/block=%.0f\n",
           Ms(Clock::now()-Tg).count(), n_cells, total_cands,
           (double)total_cands/n_blocks);

    auto Tup = Clock::now();
    int *d_csr_start=nullptr, *d_csr_list_g=nullptr, *d_cell_id_g=nullptr;
    CUDA_CHECK(cudaMalloc(&d_csr_start,    (long long)(n_cells+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_list_g,   (long long)total_cands*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_id_g,    (long long)n_blocks*sizeof(int)));
    if(d_block_adj_gpu_) cudaFree(d_block_adj_gpu_);
    CUDA_CHECK(cudaMalloc(&d_block_adj_gpu_, (long long)n_blocks*deg*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_csr_start,  h_csr_start.data(), (long long)(n_cells+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_list_g, h_csr_list.data(),  (long long)total_cands*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_id_g,  h_cell_id.data(),   (long long)n_blocks*sizeof(int),    cudaMemcpyHostToDevice));
    printf("    [graph/upload_csr] %.1f ms\n", Ms(Clock::now()-Tup).count());

    auto Tkernel = Clock::now();
    hblock_v27::gpu_build_block_adj_v27(
        d_blk_proj_gpu_, d_blk_norm_gpu_,
        d_csr_start, d_csr_list_g, d_cell_id_g,
        d_block_adj_gpu_, n_blocks, d_proj_, deg, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("    [graph/GPU_kernel] %.1f ms\n", Ms(Clock::now()-Tkernel).count());

    cudaFree(d_csr_start); cudaFree(d_csr_list_g); cudaFree(d_cell_id_g);

    // Keep CPU copy for BFS diagnostic
    h_block_adj_.resize((long long)n_blocks * deg);
    CUDA_CHECK(cudaMemcpy(h_block_adj_.data(), d_block_adj_gpu_,
                          (long long)n_blocks * deg * sizeof(int),
                          cudaMemcpyDeviceToHost));

    printf("    [graph total] %.1f ms  %d blocks x %d degree\n",
           Ms(Clock::now()-Tg).count(), n_blocks, deg);
}

void HBlockIndex::alloc_workspace()
{
    const int B=batch_size_, max_ls=max_ef_;
    const int max_pairs=B*max_ls;

    if(ws_.stream){cublasSetStream(cublas_,nullptr);cudaStreamDestroy(ws_.stream);}
    CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_,ws_.stream));

#define FH(p) do{if(ws_.p){cudaFreeHost(ws_.p);ws_.p=nullptr;}}while(0)
#define FD(p) do{if(ws_.p){cudaFree(ws_.p);ws_.p=nullptr;}}while(0)
    FH(h_q_pinned);FH(h_q_pinned_i8);FH(h_leaf_cnt);FH(h_final_dists);FH(h_final_ids);FH(h_pair_leaf_sorted);
    FH(h_top1_ids);FH(h_top2_beam);FH(h_top3_beam);FH(h_block_sel);
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned,    (long long)B*d_*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned_i8, (long long)B*d_*sizeof(int8_t)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_leaf_cnt,    (long long)B*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_dists, (long long)B*K_MAX*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_ids,   (long long)B*K_MAX*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_top1_ids,    (long long)B*ck1_*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_top2_beam,   (long long)B*ck1_*ck2_*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_top3_beam,   (long long)B*ck1_*ck2_*ck3_*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_block_sel,   (long long)B*max_ls*sizeof(int)));
    int bmap_wds  = (n_leaf_blocks_ + 31) / 32;

    FD(d_visited);
    CUDA_CHECK(cudaMalloc(&ws_.d_visited, (long long)B*bmap_wds*sizeof(int)));
    ws_.bitmap_words = bmap_wds;

    FD(d_q_batch);FD(d_q_proj1);FD(d_dots1);FD(d_top1_ids);FD(d_q_batch_i8);
    FD(d_r1_beam);FD(d_top2_beam);FD(d_top3_beam);FD(d_q_r3);
    FD(d_leaf_sel);FD(d_leaf_cnt);FD(d_lut_fine);
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,    (long long)B*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch_i8, (long long)B*d_      *sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_proj1,    (long long)B*d_proj_ *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,      (long long)B*K1_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,   (long long)B*ck1_    *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_r1_beam,    (long long)B*ck1_*d_ *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_beam,  (long long)B*ck1_*ck2_*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top3_beam,  (long long)B*ck1_*ck2_*ck3_*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r3,       (long long)B*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,   (long long)B*max_ls  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,   (long long)B         *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,   (long long)B*d_*Kr_  *sizeof(float)));

    FD(d_query_offsets);FD(d_pair_leaf_a);FD(d_pair_qid_a);FD(d_pair_leaf_b);FD(d_pair_qid_b);
    CUDA_CHECK(cudaMalloc(&ws_.d_query_offsets,(long long)(B+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_a,  (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_a,   (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_b,  (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_b,   (long long)max_pairs*sizeof(int)));

    // Host mirror of the sorted pair_leaf array, for region fetch planning
    // between gpu_build_and_sort_pairs_v29() and launch_leaf_flat_v39().
    CUDA_CHECK(cudaMallocHost(&ws_.h_pair_leaf_sorted, (long long)max_pairs*sizeof(int)));

    // Per-block exact results: [max_pairs × klocal_] (no global gather buffer needed)
    FD(d_out_dists);FD(d_out_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_out_dists,(long long)max_pairs*klocal_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_ids,  (long long)max_pairs*klocal_*sizeof(int)));

    FD(d_final_dists);FD(d_final_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists,(long long)B*K_MAX*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,  (long long)B*K_MAX*sizeof(int)));

    size_t scan_bytes=0,sort_leaf=0,sort_qid=0;
    cub::DeviceScan::ExclusiveSum(nullptr,scan_bytes,(int*)nullptr,(int*)nullptr,B);
    cub::DeviceRadixSort::SortPairs(nullptr,sort_leaf,(int*)nullptr,(int*)nullptr,(int*)nullptr,(int*)nullptr,max_pairs,0,20);
    cub::DeviceRadixSort::SortPairs(nullptr,sort_qid, (int*)nullptr,(int*)nullptr,(int*)nullptr,(int*)nullptr,max_pairs,0,15);
    ws_.cub_bytes=std::max({scan_bytes,sort_leaf,sort_qid});
    FD(d_cub_tmp);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp,ws_.cub_bytes));

    ws_.batch_cap=B; ws_.max_pairs=max_pairs; ws_.max_leaf_sel=max_ls;
    ws_.per_block_r=per_block_r_; ws_.klocal=klocal_;
    ws_.d_proj=d_proj_;
    ws_.ck1=ck1_; ws_.ck2=ck2_; ws_.ck3=ck3_;
    ws_.beam_size=max_ef_;  // overwritten per-call in search()
#undef FH
#undef FD
}

void HBlockIndex::search(const int8_t* h_q, int nq, int k,
                          float* h_dists, int* h_ids, int ef) const
{
    if(ntotal_==0)  throw std::runtime_error("empty index");
    if(k>K_MAX)     throw std::runtime_error("k exceeds K_MAX");
    if(k>klocal_)   throw std::runtime_error("k must be <= klocal");
    if(ef>max_ef_)  throw std::runtime_error("ef exceeds max_ef; rebuild with larger max_ef");
    ws_.beam_size    = ef;                 // v35/v38: beam = ef, no cap
    ws_.max_leaf_sel = ef;                 // d_leaf_sel holds up to ef expanded blocks

    using Clock=std::chrono::high_resolution_clock;
    double ms_route=0,ms_trav=0,ms_pairs=0,ms_plan=0,ms_pq=0,ms_merge=0,ms_d2h=0;
    long long stat_visited=0, stat_pairs=0;
    cudaStream_t s=ws_.stream;

    call_id_++;
    stat_code_bytes_h2d_ = 0; stat_raw_bytes_h2d_ = 0;
    stat_code_region_reqs_ = 0; stat_code_region_unique_ = 0;
    stat_raw_region_reqs_ = 0; stat_raw_region_unique_ = 0;
    stat_code_region_true_unique_ = 0; stat_raw_region_true_unique_ = 0;

    std::vector<float> h_q_cast((long long)batch_size_ * d_);   // reused int8->float scratch for routing

    for(int qstart=0;qstart<nq;qstart+=batch_size_){
        int nb=std::min(batch_size_,nq-qstart);
        const int8_t* h_qb_i8 = h_q + (long long)qstart*d_;

        auto t0=Clock::now();
        // Host-side cast for the (unchanged, float-based) routing pipeline.
        // Exact/lossless: every int8 value fits in a float32.
        for (long long i = 0; i < (long long)nb*d_; i++) h_q_cast[i] = (float)h_qb_i8[i];
        route_gpu_v29(cublas_,
                      d_Pi1_,d_Pi2_,d_Pi3_,
                      d_route1_cents_proj_,d_route1_cents_full_,d_route1_norms_,
                      d_route2_cents_proj_,d_route2_cents_full_,d_route2_norms_,
                      d_route3_cents_proj_,d_route3_cents_full_,d_route3_norms_,
                      d_fine_c1d_,h_q_cast.data(),nb,d_,d_proj_,
                      K1_,K2_,K3_,Kr_,ck1_,ck2_,ck3_,batch_size_,ws_);

        // Raw int8 query batch, for exact rerank (launch_leaf_flat_v39).
        std::memcpy(ws_.h_q_pinned_i8, h_qb_i8, (long long)nb*d_*sizeof(int8_t));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_q_batch_i8, ws_.h_q_pinned_i8,
                                   (long long)nb*d_*sizeof(int8_t), cudaMemcpyHostToDevice, s));
        ms_route+=Ms(Clock::now()-t0).count();

        auto t1=Clock::now();
        gpu_block_search_v35(nb, n_leaf_blocks_, d_proj_,
                             K2_, K3_, ck1_, ck2_, ck3_,
                             graph_degree_, ef, ef,
                             entry_per_cell_,
                             d_block_adj_gpu_, d_blk_proj_gpu_, d_blk_norm_gpu_,
                             d_pair_blk_start_, d_pair_blk_count_, ws_);
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_leaf_cnt, ws_.d_leaf_cnt,
                                   (long long)nb*sizeof(int),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        ms_trav+=Ms(Clock::now()-t1).count();

        int n_pairs=0;
        for(int qi=0;qi<nb;qi++){
            n_pairs+=ws_.h_leaf_cnt[qi];
            stat_visited+=ws_.h_leaf_cnt[qi];
        }
        stat_pairs+=n_pairs;

        auto t3=Clock::now();
        gpu_build_and_sort_pairs_v29(nb,n_pairs,n_leaf_blocks_,ws_.max_leaf_sel,ws_);
        ms_pairs+=Ms(Clock::now()-t3).count();

        // ── v39 region fetch planning ────────────────────────────────────
        // Every leaf_blk touched by this batch needs both its code region
        // (PQ scan) and its raw region (exact rerank of its own selected
        // candidates -- always the same block, see leaf_flat_kernel_v39).
        // Copy the sorted pair_leaf array back, dedupe host-side (cheap: at
        // most batch_size_*ef ints), map through the block->region tables,
        // and stage anything missing into the GPU pool before launching.
        auto t3b=Clock::now();
        if (n_pairs > 0) {
            CUDA_CHECK(cudaMemcpyAsync(ws_.h_pair_leaf_sorted, ws_.d_pair_leaf_b,
                                       (long long)n_pairs*sizeof(int),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));

            std::vector<int> uniq_blocks;
            uniq_blocks.reserve(256);
            for (int i = 0; i < n_pairs; i++) {
                int blk = ws_.h_pair_leaf_sorted[i];
                if (uniq_blocks.empty() || uniq_blocks.back() != blk) uniq_blocks.push_back(blk);
            }
            std::vector<int> code_regions, raw_regions;
            code_regions.reserve(uniq_blocks.size());
            raw_regions.reserve(uniq_blocks.size());
            for (int blk : uniq_blocks) {
                if (blk < 0 || blk >= n_leaf_blocks_) continue;
                code_regions.push_back(h_block_code_region_[blk]);
                raw_regions.push_back(h_block_raw_region_[blk]);
            }
            fetch_code_regions(code_regions);
            fetch_raw_regions(raw_regions);
        }
        ms_plan+=Ms(Clock::now()-t3b).count();

        // Per-block: PQ filter -> exact L2 -> klocal top candidates
        auto t4=Clock::now();
        launch_leaf_flat_v39(
            ws_.d_pair_leaf_b, ws_.d_pair_qid_b,
            d_code_region_pool_, d_code_region_slot_, d_block_code_region_, d_block_code_offset_,
            d_raw_region_pool_,  d_raw_region_slot_,  d_block_raw_region_,  d_block_raw_offset_,
            d_vec_local_pos_,
            d_leaf_sizes_,
            ws_.d_lut_fine, ws_.d_q_batch_i8,
            ws_.d_out_dists, ws_.d_out_ids,
            region_bytes_,
            n_pairs, d_, Kr_, Br_, bpv_, leaf_size_,
            per_block_r_, klocal_, s);
        ms_pq+=Ms(Clock::now()-t4).count();

        // Merge per-block exact results -> global top-k
        auto t5=Clock::now();
        launch_final_merge_v29(nb, n_pairs, klocal_, k, ws_);
        ms_merge+=Ms(Clock::now()-t5).count();

        auto t8=Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_dists,ws_.d_final_dists,
                                   (long long)nb*k*sizeof(float),cudaMemcpyDeviceToHost,s));
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_ids,ws_.d_final_ids,
                                   (long long)nb*k*sizeof(int),cudaMemcpyDeviceToHost,s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        ms_d2h+=Ms(Clock::now()-t8).count();

        std::memcpy(h_dists+(long long)qstart*k,ws_.h_final_dists,(long long)nb*k*sizeof(float));
        std::memcpy(h_ids  +(long long)qstart*k,ws_.h_final_ids,  (long long)nb*k*sizeof(int));
    }

    printf("  [v39] Route=%.2f  Traverse=%.2f  Pairs=%.2f  RegionPlan=%.2f  PerBlkExact=%.2f  Merge=%.2f  D2H=%.2f ms\n",
           ms_route,ms_trav,ms_pairs,ms_plan,ms_pq,ms_merge,ms_d2h);
    printf("  [v39 stats] avg_visited=%.1f  avg_pairs=%.1f  (over %d queries)\n",
           (double)stat_visited/nq, (double)stat_pairs/nq, nq);
    printf("  [v39 region] code: %lld req -> %lld uniq_per_batch_sum -> %lld uniq_this_call, "
           "%.2f MB H2D (region_reuse=%.2f) | "
           "raw: %lld req -> %lld uniq_per_batch_sum -> %lld uniq_this_call, "
           "%.2f MB H2D (region_reuse=%.2f)\n",
           stat_code_region_reqs_, stat_code_region_unique_, stat_code_region_true_unique_,
           (double)stat_code_bytes_h2d_/1e6,
           stat_code_region_true_unique_ ? (double)stat_code_region_reqs_/stat_code_region_true_unique_ : 0.0,
           stat_raw_region_reqs_,  stat_raw_region_unique_,  stat_raw_region_true_unique_,
           (double)stat_raw_bytes_h2d_/1e6,
           stat_raw_region_true_unique_ ? (double)stat_raw_region_reqs_/stat_raw_region_true_unique_ : 0.0);
}

HBlockIndex::RoutingDiag HBlockIndex::diagnose_missed_gt(
    const int8_t* h_q, int nq, int k,
    const int32_t* h_gt, int gt_k, int ef,
    bool verbose) const
{
    (void)h_q; (void)nq; (void)k; (void)h_gt; (void)gt_k; (void)ef; (void)verbose;
    // Deliberately deferred: v39's scope is add()/search() payload
    // residency (region partitioning + bounded GPU pool), not routing
    // diagnostics. routing_recall/graph_coverage don't depend on the
    // region layer at all -- port v36's vec_to_block + BFS logic here
    // unchanged (adjusted for the int8 query type) if/when it's needed.
    throw std::runtime_error(
        "HBlockIndex::diagnose_missed_gt: not implemented in v39 (out of scope -- "
        "see comment at this function's definition)");
}

} // namespace hblock_v39
