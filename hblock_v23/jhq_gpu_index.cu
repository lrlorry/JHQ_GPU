#include "hblock_v23/jhq_gpu_index.cuh"
#include "hblock_v17/encode.cuh"
#include "hblock_v23/search.cuh"
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

namespace hblock_v23 {

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
      rerank_r_(p.rerank_r), km_iters_(p.km_iters),
      batch_size_(p.batch_size),
      graph_degree_(p.graph_degree), graph_budget_(p.graph_budget),
      entry_per_cell_(p.entry_per_cell),
      n_c2_nbrs_(p.n_c2_nbrs), n_c1_nbrs_(p.n_c1_nbrs),
      max_cand_blocks_(p.max_cand_blocks)
{
    if (d <= 0)                  throw std::invalid_argument("d must be positive");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex()
{
    if (ws_.stream) { cublasSetStream(cublas_, nullptr); cudaStreamDestroy(ws_.stream); }
    auto FH = [](void* p){ if (p) cudaFreeHost(p); };
    auto FD = [](void* p){ if (p) cudaFree(p); };
    FH(ws_.h_q_pinned); FH(ws_.h_leaf_cnt); FH(ws_.h_final_dists); FH(ws_.h_final_ids);
    FH(ws_.h_top1_ids); FH(ws_.h_top2_beam); FH(ws_.h_top3_beam); FH(ws_.h_block_sel);
    FD(ws_.d_q_batch);  FD(ws_.d_q_proj1);  FD(ws_.d_dots1);
    FD(ws_.d_top1_ids); FD(ws_.d_r1_beam);  FD(ws_.d_top2_beam); FD(ws_.d_top3_beam);
    FD(ws_.d_q_r3);     FD(ws_.d_leaf_sel); FD(ws_.d_leaf_cnt);  FD(ws_.d_lut_fine);
    FD(ws_.d_query_offsets);
    FD(ws_.d_pair_leaf_a); FD(ws_.d_pair_qid_a);
    FD(ws_.d_pair_leaf_b); FD(ws_.d_pair_qid_b);
    FD(ws_.d_out_dists); FD(ws_.d_out_ids);
    FD(ws_.d_pq_dists);  FD(ws_.d_pq_ids);
    FD(ws_.d_cand_vecs); FD(ws_.d_exact_dists);
    FD(ws_.d_final_dists); FD(ws_.d_final_ids);
    FD(ws_.d_cub_tmp);
    FD(ws_.d_visited);
    FD(d_Pi1_); FD(d_Pi2_); FD(d_Pi3_);
    FD(d_route1_cents_proj_); FD(d_route1_cents_full_); FD(d_route1_norms_);
    FD(d_route2_cents_proj_); FD(d_route2_cents_full_); FD(d_route2_norms_);
    FD(d_route3_cents_proj_); FD(d_route3_cents_full_); FD(d_route3_norms_);
    FD(d_fine_c1d_);
    FD(d_pair_blk_start_); FD(d_pair_blk_count_);
    FD(d_leaf_codes_); FD(d_leaf_ids_); FD(d_leaf_sizes_);
    FD(d_base_vecs_);
    FD(d_block_adj_gpu_); FD(d_blk_proj_gpu_); FD(d_blk_norm_gpu_);
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

void HBlockIndex::train(const float* h_x, int n_train)
{
    using Clock = std::chrono::high_resolution_clock;
    auto T0 = Clock::now();
    const int n_km = std::min(n_train, 200000);
    printf("[v23 train] d=%d d_proj=%d K1=%d K2=%d K3=%d ck1=%d ck2=%d ck3=%d"
           " graph_degree=%d graph_budget=%d entry_per_cell=%d\n",
           d_, d_proj_, K1_, K2_, K3_, ck1_, ck2_, ck3_,
           graph_degree_, graph_budget_, entry_per_cell_);

    const float one=1.f, zero=0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));

    // ── L1 ────────────────────────────────────────────────────────────────────
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

    // ── L2 ────────────────────────────────────────────────────────────────────
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

    // ── L3 ────────────────────────────────────────────────────────────────────
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

    // ── Fine PQ codebook ─────────────────────────────────────────────────────
    double sum_sq=0.0;
    for(int i=0;i<n_km;i++){int c1=h_assign1[i],c2=h_assign2[i],c3=h_assign3[i];long long c123=((long long)c1*K2_+c2)*K3_+c3;for(int j=0;j<d_;j++){float v=h_r2[(long long)i*d_+j]-h_all_c3_full[c123*d_+j];sum_sq+=(double)v*v;}}
    float sigma=(float)std::sqrt(sum_sq/(double)((long long)n_km*d_));
    auto fine_c1d=analytical_fine_c1d(Kr_,sigma);
    CUDA_CHECK(cudaMalloc(&d_fine_c1d_,Kr_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_,fine_c1d.data(),Kr_*sizeof(float),cudaMemcpyHostToDevice));

    cudaFree(d_x_km); cudaFree(d_y_proj);
    printf("[v23 train] total=%.1f ms  sigma_r3=%.4f\n", Ms(Clock::now()-T0).count(), sigma);
}

void HBlockIndex::add(const float* h_x, int n)
{
    if (!d_Pi1_)      throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock supports one add() call");
    using Clock = std::chrono::high_resolution_clock;
    auto T_add = Clock::now();
    printf("[v23 add] n=%d  d=%d  batch=%d\n", n, d_, 8192);

    const int BATCH=8192;
    const float one=1.f, zero=0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));

    // ── Encoding loop ─────────────────────────────────────────────────────────
    auto T0 = Clock::now();
    float *d_x,*d_r1,*d_r2,*d_r3,*d_proj1,*d_proj2,*d_proj3;
    int *d_c1,*d_c2,*d_c3; uint8_t *d_fc; float *d_dots1;
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

    int n_batches = (n + BATCH - 1) / BATCH;
    int report_every = std::max(1, n_batches / 10);
    for (int s=0;s<n;s+=BATCH) {
        int nb=std::min(BATCH,n-s);
        CUDA_CHECK(cudaMemcpy(d_x,h_x+(long long)s*d_,(long long)nb*d_*sizeof(float),cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_,CUBLAS_OP_T,CUBLAS_OP_N,d_proj_,nb,d_,&one,d_Pi1_,d_,d_x,d_,&zero,d_proj1,d_proj_));
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
            std::memcpy(h_code1.data()+s,h_c1b.data(),nb*sizeof(int));
            std::memcpy(h_code2.data()+s,h_c2b.data(),nb*sizeof(int));
            std::memcpy(h_code3.data()+s,h_c3b.data(),nb*sizeof(int));
            int *d_c123; CUDA_CHECK(cudaMalloc(&d_c123,nb*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_c123,h_c123.data(),nb*sizeof(int),cudaMemcpyHostToDevice));
            hblock_v17::launch_subtract_centroid(d_r2,d_c123,d_route3_cents_full_,d_r3,nb,d_,nullptr);
            hblock_v17::launch_fine_encode(d_r3,d_fine_c1d_,d_fc,nb,d_,Kr_,Br_,bpv_,nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_fc_all.data()+(long long)s*bpv_,d_fc,(long long)nb*bpv_,cudaMemcpyDeviceToHost));
            cudaFree(d_c12); cudaFree(d_c123);
        }
        int batch_idx = s/BATCH + 1;
        if(batch_idx % report_every == 0 || batch_idx == n_batches)
            printf("  [encode] %d/%d batches  %.1f ms\n",
                   batch_idx, n_batches, Ms(Clock::now()-T0).count());
    }
    cudaFree(d_x);cudaFree(d_proj1);cudaFree(d_proj2);cudaFree(d_proj3);
    cudaFree(d_r1);cudaFree(d_r2);cudaFree(d_r3);
    cudaFree(d_c1);cudaFree(d_c2);cudaFree(d_c3);cudaFree(d_fc);cudaFree(d_dots1);
    printf("  [encode total] %.1f ms\n", Ms(Clock::now()-T0).count());

    // ── Sort by (c1,c2,c3) ────────────────────────────────────────────────────
    T0 = Clock::now();
    long long K2K3=(long long)K2_*K3_;
    std::vector<int> order(n);
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){
        long long ka=(long long)h_code1[a]*K2K3+h_code2[a]*K3_+h_code3[a];
        long long kb=(long long)h_code1[b]*K2K3+h_code2[b]*K3_+h_code3[b];
        return ka<kb;
    });
    printf("  [sort] %.1f ms\n", Ms(Clock::now()-T0).count());

    // ── Build leaf block layout ───────────────────────────────────────────────
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

    // ── Pack leaf codes + block centroids ─────────────────────────────────────
    T0 = Clock::now();
    std::vector<uint8_t> h_leaf_codes((long long)total_blocks*bpv_*leaf_size_,0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks*leaf_size_,-1);
    std::vector<int>     h_leaf_sizes(total_blocks,0);
    h_block_cell_id_.assign(total_blocks,0);
    h_block_cent_.assign((long long)total_blocks*d_, 0.f);

    for(int i=0,j;i<n;i=j){
        int c1=h_code1[order[i]],c2=h_code2[order[i]],c3=h_code3[order[i]];
        for(j=i;j<n&&h_code1[order[j]]==c1&&h_code2[order[j]]==c2&&h_code3[order[j]]==c3;++j){}
        long long cidx=(long long)c1*K2K3+c2*K3_+c3;
        int base_blk=pair_start[cidx];
        for(int vi=i;vi<j;++vi){
            int in_blk=vi-i, blk=base_blk+in_blk/leaf_size_, pos=in_blk%leaf_size_;
            int oid=order[vi];
            h_leaf_ids[(long long)blk*leaf_size_+pos]=oid;
            const uint8_t* src=h_fc_all.data()+(long long)oid*bpv_;
            uint8_t* dst_base=h_leaf_codes.data()+(long long)blk*bpv_*leaf_size_;
            for(int b=0;b<bpv_;b++) dst_base[(long long)b*leaf_size_+pos]=src[b];
            h_leaf_sizes[blk]=std::max(h_leaf_sizes[blk],pos+1);
            h_block_cell_id_[blk]=(int)cidx;
            const float* xv=h_x+(long long)oid*d_;
            float* bc=h_block_cent_.data()+(long long)blk*d_;
            for(int dim=0;dim<d_;dim++) bc[dim]+=xv[dim];
        }
    }
    for(int b=0;b<total_blocks;b++){
        int sz=h_leaf_sizes[b]; if(sz==0) continue;
        float inv=1.0f/sz;
        float* bc=h_block_cent_.data()+(long long)b*d_;
        for(int dim=0;dim<d_;dim++) bc[dim]*=inv;
    }
    printf("  [pack+centroids] %.1f ms  (%.2f GB leaf codes)\n",
           Ms(Clock::now()-T0).count(),
           (double)total_blocks*bpv_*leaf_size_/1e9);

    // ── Project block centroids via cuBLAS GEMM ───────────────────────────────
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

    // ── Upload leaf structures to GPU ─────────────────────────────────────────
    T0 = Clock::now();
    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_,n_cells*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_,n_cells*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_,pair_start.data(),n_cells*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_,pair_cnt.data(),  n_cells*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_codes_,(long long)total_blocks*bpv_*leaf_size_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,  (long long)total_blocks*leaf_size_*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_,total_blocks*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_codes_,h_leaf_codes.data(),(long long)total_blocks*bpv_*leaf_size_,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_,  h_leaf_ids.data(),  (long long)total_blocks*leaf_size_*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_,h_leaf_sizes.data(),total_blocks*sizeof(int),cudaMemcpyHostToDevice));
    h_pair_blk_start_cpu_=pair_start;
    h_pair_blk_count_cpu_=pair_cnt;
    printf("  [upload leaf GPU] %.1f ms  (%.2f GB)\n",
           Ms(Clock::now()-T0).count(),
           (double)(total_blocks*bpv_*leaf_size_ + total_blocks*leaf_size_*4)/1e9);

    // ── Abs centroids for graph candidate generation ──────────────────────────
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

    // ── Upload proj/norm to GPU (needed by GPU graph build kernel) ────────────
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

    // ── Build block graph ─────────────────────────────────────────────────────
    T0 = Clock::now();
    printf("  [graph build] n_blocks=%d degree=%d n_c2_nbrs=%d n_c1_nbrs=%d ...\n",
           total_blocks, graph_degree_, n_c2_nbrs_, n_c1_nbrs_);
    build_block_graph(pair_start, pair_cnt);
    printf("  [graph build total] %.1f ms\n", Ms(Clock::now()-T0).count());

    // ── Upload base vectors ───────────────────────────────────────────────────
    T0 = Clock::now();
    printf("  [upload base vecs] %.2f GB ...\n",(double)n*d_*4/1e9);
    CUDA_CHECK(cudaMalloc(&d_base_vecs_,(long long)n*d_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base_vecs_,h_x,(long long)n*d_*sizeof(float),cudaMemcpyHostToDevice));
    printf("  [upload base vecs] %.1f ms\n", Ms(Clock::now()-T0).count());

    // ── Finalize ──────────────────────────────────────────────────────────────
    ntotal_=n;
    h_block_cent_.clear(); h_block_cent_.shrink_to_fit();

    T0 = Clock::now();
    alloc_workspace();
    printf("  [alloc workspace] %.1f ms\n", Ms(Clock::now()-T0).count());

    printf("[v23 add total] %.1f ms  blocks=%d\n",
           Ms(Clock::now()-T_add).count(), total_blocks);
}

// Helper: squared L2 distance between two float vectors
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

    // For each cell c, we need its candidate neighbor cells.
    // Use h_leaf_abs_cents_ to find nearest sibling cells at each level.
    //
    // C2 representative for (c1,c2) group: abs_cent of (c1,c2,0)
    // C1 representative for c1 group:      abs_cent of (c1,0,0)
    // These are only used for cell-level ordering within a level, not exact distances.

    // Precompute which c2 groups are nearest within each c1
    // nearest_c2[c1*K2_+c2] = sorted list of c2 siblings by C2 centroid dist
    std::vector<std::vector<int>> nearest_c2(K1_*K2_);
    for(int c1=0;c1<K1_;c1++){
        // C2 representative: abs_cent[(c1,c2,0)]
        std::vector<std::pair<float,int>> c2_dists;
        for(int c2=0;c2<K2_;c2++){
            long long ref_c123=(long long)(c1*K2_+c2)*K3_;
            c2_dists.push_back({0.f, c2}); // placeholder
        }
        // For each c2, distance from (c1,c2_ref,0) to (c1,c2',0) for ranking
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

    // Precompute which c1 groups are nearest
    // nearest_c1[c1] = sorted list of c1 siblings by C1 centroid dist
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

        // Collect candidate cells and their blocks
        std::vector<int> cand_blocks;
        cand_blocks.reserve(512);

        // 1. All c3 siblings (same (c1,c2))
        for(int c3p=0;c3p<K3_;c3p++){
            long long cp=(long long)(c1*K2_+c2)*K3_+c3p;
            int bs=h_pair_blk_start[cp], bc=h_pair_blk_count[cp];
            for(int bi=0;bi<bc;bi++) cand_blocks.push_back(bs+bi);
        }

        // 2. Top n_c2_nbrs_ c2 siblings — all their c3 cells
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

        // 3. Top n_c1_nbrs_ c1 siblings — top 2 c2 of each, all their c3
        auto& nc1=nearest_c1[c1];
        int n_c1_use=std::min(n_c1_nbrs_,(int)nc1.size());
        for(int ni=0;ni<n_c1_use;ni++){
            int c1p=nc1[ni];
            // Use the 2 nearest c2 in this c1p group
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

        // Deduplicate
        std::sort(cand_blocks.begin(), cand_blocks.end());
        cand_blocks.erase(std::unique(cand_blocks.begin(), cand_blocks.end()),
                          cand_blocks.end());
        // Cap by projected distance from cell centroid (blk_base as proxy), not block ID
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
    gpu_build_block_adj_v23(d_blk_proj_gpu_, d_blk_norm_gpu_,
                            d_csr_start, d_csr_list_g, d_cell_id_g,
                            d_block_adj_gpu_, n_blocks, d_proj_, deg, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("    [graph/GPU_kernel] %.1f ms\n", Ms(Clock::now()-Tkernel).count());

    cudaFree(d_csr_start); cudaFree(d_csr_list_g); cudaFree(d_cell_id_g);
    printf("    [graph total] %.1f ms  %d blocks x %d degree\n",
           Ms(Clock::now()-Tg).count(), n_blocks, deg);
}


void HBlockIndex::alloc_workspace()
{
    const int B=batch_size_, max_ls=graph_budget_, R=rerank_r_;
    const int max_pairs=B*max_ls;

    if(ws_.stream){cublasSetStream(cublas_,nullptr);cudaStreamDestroy(ws_.stream);}
    CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_,ws_.stream));

#define FH(p) do{if(ws_.p){cudaFreeHost(ws_.p);ws_.p=nullptr;}}while(0)
#define FD(p) do{if(ws_.p){cudaFree(ws_.p);ws_.p=nullptr;}}while(0)
    FH(h_q_pinned);FH(h_leaf_cnt);FH(h_final_dists);FH(h_final_ids);
    FH(h_top1_ids);FH(h_top2_beam);FH(h_top3_beam);FH(h_block_sel);
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned,    (long long)B*d_*sizeof(float)));
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

    FD(d_q_batch);FD(d_q_proj1);FD(d_dots1);FD(d_top1_ids);
    FD(d_r1_beam);FD(d_top2_beam);FD(d_top3_beam);FD(d_q_r3);
    FD(d_leaf_sel);FD(d_leaf_cnt);FD(d_lut_fine);
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,   (long long)B*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_proj1,   (long long)B*d_proj_ *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,     (long long)B*K1_     *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,  (long long)B*ck1_    *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_r1_beam,   (long long)B*ck1_*d_ *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_beam, (long long)B*ck1_*ck2_*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top3_beam, (long long)B*ck1_*ck2_*ck3_*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r3,      (long long)B*d_      *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,  (long long)B*max_ls  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,  (long long)B         *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,  (long long)B*d_*Kr_  *sizeof(float)));

    FD(d_query_offsets);FD(d_pair_leaf_a);FD(d_pair_qid_a);FD(d_pair_leaf_b);FD(d_pair_qid_b);
    CUDA_CHECK(cudaMalloc(&ws_.d_query_offsets,(long long)(B+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_a,  (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_a,   (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_b,  (long long)max_pairs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_b,   (long long)max_pairs*sizeof(int)));

    FD(d_out_dists);FD(d_out_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_out_dists,(long long)max_pairs*TOP_P*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_ids,  (long long)max_pairs*TOP_P*sizeof(int)));

    FD(d_pq_dists);FD(d_pq_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_pq_dists,(long long)B*K_MAX*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pq_ids,  (long long)B*K_MAX*sizeof(int)));

    FD(d_cand_vecs);FD(d_exact_dists);
    CUDA_CHECK(cudaMalloc(&ws_.d_cand_vecs,  (long long)B*R*d_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_exact_dists,(long long)B*R   *sizeof(float)));

    FD(d_final_dists);FD(d_final_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists,(long long)B*K_MAX*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,  (long long)B*K_MAX*sizeof(int)));

    size_t scan_bytes=0,sort_leaf=0,sort_qid=0;
    cub::DeviceScan::ExclusiveSum(nullptr,scan_bytes,(int*)nullptr,(int*)nullptr,B);
    cub::DeviceRadixSort::SortPairs(nullptr,sort_leaf,(int*)nullptr,(int*)nullptr,(int*)nullptr,(int*)nullptr,max_pairs,0,20);
    cub::DeviceRadixSort::SortPairs(nullptr,sort_qid, (int*)nullptr,(int*)nullptr,(int*)nullptr,(int*)nullptr,max_pairs,0,10);
    ws_.cub_bytes=std::max({scan_bytes,sort_leaf,sort_qid});
    FD(d_cub_tmp);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp,ws_.cub_bytes));

    ws_.batch_cap=B; ws_.max_pairs=max_pairs; ws_.max_leaf_sel=max_ls;
    ws_.rerank_r=R; ws_.d_proj=d_proj_;
    ws_.ck1=ck1_; ws_.ck2=ck2_; ws_.ck3=ck3_;
#undef FH
#undef FD
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if(ntotal_==0)   throw std::runtime_error("empty index");
    if(k>K_MAX)      throw std::runtime_error("k exceeds K_MAX");
    if(k>rerank_r_)  throw std::runtime_error("k must be <= rerank_r");

    using Clock=std::chrono::high_resolution_clock;
    double ms_route=0,ms_trav=0,ms_pairs=0,
           ms_pq=0,ms_merge=0,ms_gather=0,ms_rerank=0,ms_d2h=0;
    long long stat_visited=0, stat_pairs=0;
    cudaStream_t s=ws_.stream;

    for(int qstart=0;qstart<nq;qstart+=batch_size_){
        int nb=std::min(batch_size_,nq-qstart);
        const float* h_qb=h_q+(long long)qstart*d_;

        auto t0=Clock::now();
        route_gpu_v23(cublas_,
                      d_Pi1_,d_Pi2_,d_Pi3_,
                      d_route1_cents_proj_,d_route1_cents_full_,d_route1_norms_,
                      d_route2_cents_proj_,d_route2_cents_full_,d_route2_norms_,
                      d_route3_cents_proj_,d_route3_cents_full_,d_route3_norms_,
                      d_fine_c1d_,h_qb,nb,d_,d_proj_,
                      K1_,K2_,K3_,Kr_,ck1_,ck2_,ck3_,batch_size_,ws_);
        ms_route+=Ms(Clock::now()-t0).count();

        // ── GPU: fused entry-selection + beam search (one warp per query) ───────
        auto t1=Clock::now();
        gpu_block_search_v23(nb, n_leaf_blocks_, d_proj_,
                             K2_, K3_, ck1_, ck2_, ck3_,
                             graph_degree_, graph_budget_, ws_.max_leaf_sel,
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
        gpu_build_and_sort_pairs_v23(nb,n_pairs,n_leaf_blocks_,ws_.max_leaf_sel,ws_);
        ms_pairs+=Ms(Clock::now()-t3).count();

        auto t4=Clock::now();
        launch_leaf_flat_v23(ws_.d_pair_leaf_b,ws_.d_pair_qid_b,
                             d_leaf_codes_,d_leaf_ids_,d_leaf_sizes_,ws_.d_lut_fine,
                             ws_.d_out_dists,ws_.d_out_ids,
                             n_pairs,d_,Kr_,Br_,bpv_,leaf_size_,s);
        ms_pq+=Ms(Clock::now()-t4).count();

        auto t5=Clock::now();
        gpu_merge_pq_topk_v23(nb,n_pairs,rerank_r_,ws_);
        ms_merge+=Ms(Clock::now()-t5).count();

        auto t6=Clock::now();
        launch_gather_vecs_v23(d_base_vecs_,ws_.d_pq_ids,ws_.d_cand_vecs,nb,rerank_r_,d_,s);
        ms_gather+=Ms(Clock::now()-t6).count();

        auto t7=Clock::now();
        launch_exact_rerank_v23(ws_.d_q_batch,ws_.d_cand_vecs,ws_.d_pq_ids,
                                ws_.d_exact_dists,ws_.d_final_dists,ws_.d_final_ids,
                                nb,rerank_r_,d_,k,s);
        ms_rerank+=Ms(Clock::now()-t7).count();

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

    printf("  [v23] Route=%.2f  Traverse(fused)=%.2f"
           "  Pairs=%.2f  PQ=%.2f  Merge=%.2f  Gather=%.2f  Rerank=%.2f  D2H=%.2f ms\n",
           ms_route,ms_trav,ms_pairs,ms_pq,ms_merge,ms_gather,ms_rerank,ms_d2h);
    printf("  [v23 stats] avg_visited=%.1f  avg_pairs=%.1f  (over %d queries)\n",
           (double)stat_visited/nq, (double)stat_pairs/nq, nq);
}

} // namespace hblock_v23
