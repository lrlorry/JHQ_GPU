#!/usr/bin/env python3
"""Bit-exactness test and microbenchmark of v69's probe selection against v65's.

Generates one CUDA file holding both select_probes_kernel sections, cut verbatim
out of jhq_v65_schedule/search.cu and jhq_v69_probesel/search.cu (so the test
runs the code the index compiles, not a copy of it), plus a driver that runs
both on the same inputs and compares probe ids, probe offsets and per-query
totals word for word. The inputs are built to break a wrong selection: exact
ties at the boundary, both zeros, infinities, nlist < nprobe, nprobe > BLOCK.

    python3 probe_select_test.py <out.cu>
    nvcc -O3 --expt-relaxed-constexpr --use_fast_math \
         --generate-code=arch=compute_80,code=[compute_80,sm_80] -o <bin> <out.cu>
    <bin>      # prints one line per failing case, then the timings, then PROBE_TEST_OK / PROBE_TEST_FAIL
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEGIN = "// ── select_probes_kernel"
END = "// ── build_byte_lut_kernel"


def section(path):
    s = open(path).read()
    a = s.index(BEGIN)
    b = s.index(END, a)
    return s[a:b]


DRIVER = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

// v = cent_norms[c] - 2 dots[b][c]; mode picks the distribution
static void make_inputs(int mode, int B, int nlist, std::mt19937& rng,
                        std::vector<float>& dots, std::vector<float>& cn, std::vector<int>& offs)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 1.f);
    dots.assign((size_t)B * nlist, 0.f);
    cn.assign(nlist, 0.f);
    for (int c = 0; c < nlist; ++c) {
        switch (mode) {
        case 0: cn[c] = U(rng); break;                                     // continuous
        case 1: cn[c] = 0.25f * (float)(rng() % 3); break;                 // few levels: heavy ties
        case 2: cn[c] = 1.0f; break;                                       // all equal
        case 3: { const float z[4] = {0.0f, -0.0f, 1.0f, 2.0f}; cn[c] = z[rng() % 4]; } break;   // both zeros
        case 4: cn[c] = 0.3f + 0.7f * U(rng); break;                       // like real centroids
        case 5: cn[c] = (rng() % 4 == 0) ? 0.5f * (float)(rng() % 3) : __builtin_inff(); break; // mostly +inf
        default: cn[c] = 1e30f * (U(rng) - 0.5f); break;                   // huge magnitudes, both signs
        }
    }
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < nlist; ++c) {
            float x;
            switch (mode) {
            case 0: x = U(rng) - 0.5f; break;
            case 1: x = 0.125f * (float)(rng() % 4); break;
            case 2: x = 0.0f; break;
            case 3: { const float z[3] = {0.0f, 0.5f, 1.0f}; x = z[rng() % 3]; } break;
            case 4: x = 0.1f * N(rng); break;
            case 5: x = 0.25f * (float)(rng() % 2); break;
            default: x = 1e30f * (U(rng) - 0.5f); break;
            }
            dots[(size_t)b * nlist + c] = x;
        }
    offs.assign(nlist + 1, 0);
    for (int c = 0; c < nlist; ++c) offs[c + 1] = offs[c] + (int)(rng() % 50);
}

struct Dev {
    float *dots = nullptr, *cn = nullptr; int *offs = nullptr;
    int *ids = nullptr, *poffs = nullptr, *tot = nullptr;
};

static void alloc(Dev& d, int B, int nlist, int nprobe) {
    CK(cudaMalloc(&d.dots, sizeof(float) * (size_t)B * nlist));
    CK(cudaMalloc(&d.cn, sizeof(float) * nlist));
    CK(cudaMalloc(&d.offs, sizeof(int) * (nlist + 1)));
    CK(cudaMalloc(&d.ids, sizeof(int) * (size_t)B * nprobe));
    CK(cudaMalloc(&d.poffs, sizeof(int) * (size_t)B * (nprobe + 1)));
    CK(cudaMalloc(&d.tot, sizeof(int) * B));
}
static void release(Dev& d) {
    cudaFree(d.dots); cudaFree(d.cn); cudaFree(d.offs); cudaFree(d.ids); cudaFree(d.poffs); cudaFree(d.tot);
}

static void run65(const Dev& d, int B, int nlist, int nprobe, int block) {
    int cap = 1; while (cap < nprobe + block) cap <<= 1;
    const int smem = cap * 8 + 8;
    if (smem > 48 * 1024)
        CK(cudaFuncSetAttribute(ref65::select_probes_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    ref65::select_probes_kernel<<<B, block, smem>>>(d.dots, d.cn, d.offs, d.ids, d.poffs, d.tot, nlist, nprobe, cap);
}
static void run69(const Dev& d, int B, int nlist, int nprobe, int block) {
    const int smem = new69::probe_select_smem(nprobe);
    if (smem > 48 * 1024)
        CK(cudaFuncSetAttribute(new69::select_probes_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    new69::select_probes_kernel<<<B, block, smem>>>(d.dots, d.cn, d.offs, d.ids, d.poffs, d.tot, nlist, nprobe);
}

struct Out { std::vector<int> ids, poffs, tot; };
static Out fetch(const Dev& d, int B, int nprobe) {
    Out o; o.ids.resize((size_t)B * nprobe); o.poffs.resize((size_t)B * (nprobe + 1)); o.tot.resize(B);
    CK(cudaMemcpy(o.ids.data(), d.ids, sizeof(int) * o.ids.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.poffs.data(), d.poffs, sizeof(int) * o.poffs.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.tot.data(), d.tot, sizeof(int) * o.tot.size(), cudaMemcpyDeviceToHost));
    return o;
}

int main() {
    std::mt19937 rng(20261014);
    const int nlists[]  = {1, 7, 100, 1000, 4096, 8192, 10007, 32768};
    const int nprobes[] = {1, 8, 31, 64, 128, 256, 1000, 2000};
    const int blocks[]  = {128, 256, 512, 1024};
    const int modes = 7;
    long cases = 0, fails = 0;
    for (int nlist : nlists) for (int nprobe : nprobes) for (int block : blocks) for (int mode = 0; mode < modes; ++mode) {
        const int B = 4;
        std::vector<float> dots, cn; std::vector<int> offs;
        make_inputs(mode, B, nlist, rng, dots, cn, offs);
        Dev d; alloc(d, B, nlist, nprobe);
        CK(cudaMemcpy(d.dots, dots.data(), sizeof(float) * dots.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.cn, cn.data(), sizeof(float) * cn.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.offs, offs.data(), sizeof(int) * offs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(d.ids, 0xff, sizeof(int) * (size_t)B * nprobe));
        run65(d, B, nlist, nprobe, block); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const Out a = fetch(d, B, nprobe);
        CK(cudaMemset(d.ids, 0xff, sizeof(int) * (size_t)B * nprobe));
        CK(cudaMemset(d.poffs, 0xff, sizeof(int) * (size_t)B * (nprobe + 1)));
        CK(cudaMemset(d.tot, 0xff, sizeof(int) * B));
        run69(d, B, nlist, nprobe, block); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const Out b = fetch(d, B, nprobe);
        ++cases;
        if (a.ids != b.ids || a.poffs != b.poffs || a.tot != b.tot) {
            ++fails;
            int first = -1;
            for (size_t i = 0; i < a.ids.size(); ++i) if (a.ids[i] != b.ids[i]) { first = (int)i; break; }
            std::printf("FAIL nlist=%d nprobe=%d block=%d mode=%d ids_differ_at=%d offs_same=%d tot_same=%d\n",
                        nlist, nprobe, block, mode, first, (int)(a.poffs == b.poffs), (int)(a.tot == b.tot));
        }
        release(d);
    }
    std::printf("exactness: %ld cases, %ld failing\n", cases, fails);

    // kernel time at the paper's shapes: nlist 4096 / 8192, nprobe 128, BLOCK 512 / 1024, b = 8 .. 1000
    std::printf("timing (us per launch, median of 9 x 50 launches):\n");
    for (int nlist : {4096, 8192}) for (int block : {512, 1024}) for (int B : {8, 32, 256, 1000}) {
        const int nprobe = 128;
        std::vector<float> dots, cn; std::vector<int> offs;
        make_inputs(4, B, nlist, rng, dots, cn, offs);
        Dev d; alloc(d, B, nlist, nprobe);
        CK(cudaMemcpy(d.dots, dots.data(), sizeof(float) * dots.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.cn, cn.data(), sizeof(float) * cn.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.offs, offs.data(), sizeof(int) * offs.size(), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        double t[2];
        for (int which = 0; which < 2; ++which) {
            std::vector<float> rep;
            for (int r = 0; r < 9; ++r) {
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 50; ++i) { if (which == 0) run65(d, B, nlist, nprobe, block); else run69(d, B, nlist, nprobe, block); }
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
                rep.push_back(ms * 1000.f / 50.f);
            }
            std::sort(rep.begin(), rep.end());
            t[which] = rep[4];
        }
        std::printf("  nlist=%5d block=%4d b=%4d  v65 %8.2f  v69 %8.2f  x%.2f\n", nlist, block, B, t[0], t[1], t[0] / t[1]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        release(d);
    }
    std::printf(fails ? "PROBE_TEST_FAIL\n" : "PROBE_TEST_OK\n");
    return fails ? 1 : 0;
}
'''


def main():
    out = sys.argv[1]
    v65 = section(os.path.join(ROOT, "jhq_v65_schedule", "search.cu"))
    v69 = section(os.path.join(ROOT, "jhq_v69_probesel", "search.cu"))
    with open(out, "w") as f:
        f.write("// generated by experiments/practical_opt/probe_select_test.py -- do not edit\n")
        f.write("#include <cuda_runtime.h>\n\n")
        f.write("namespace ref65 {\n" + v65 + "\n}  // namespace ref65\n\n")
        f.write("namespace new69 {\n" + v69 + "\n}  // namespace new69\n")
        f.write(DRIVER)
    print("wrote", out)


if __name__ == "__main__":
    main()
