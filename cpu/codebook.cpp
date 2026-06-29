#include "cpu/codebook.h"
#include "cpu/erfinv.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

std::vector<float> train_1d_kmeans(const float* vals, int n, int Kr, int max_iter) {
    if (n <= 0 || Kr <= 0) return {};
    if (n < Kr) Kr = n;

    // Quantile-based initialization — evenly spaced percentiles of the data
    std::vector<float> sorted(vals, vals + n);
    std::sort(sorted.begin(), sorted.end());

    std::vector<float> c(Kr);
    for (int i = 0; i < Kr; i++) {
        int idx = std::min((int)((i + 0.5f) / Kr * n), n - 1);
        c[i] = sorted[idx];
    }

    std::vector<double> sum(Kr);
    std::vector<int>    cnt(Kr);

    for (int iter = 0; iter < max_iter; iter++) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);

        // Assignment: binary search on sorted centroids → O(n log Kr)
        for (int i = 0; i < n; i++) {
            float v = vals[i];
            int lo = 0, hi = Kr - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (v < 0.5f * (c[mid] + c[mid + 1])) hi = mid;
                else lo = mid + 1;
            }
            sum[lo] += v;
            cnt[lo]++;
        }

        // Update: mean of each cluster; empty cluster → reuse quantile seed
        bool changed = false;
        for (int i = 0; i < Kr; i++) {
            float new_c;
            if (cnt[i] == 0) {
                int idx = std::min((int)((i + 0.5f) / Kr * n), n - 1);
                new_c = sorted[idx];
            } else {
                new_c = (float)(sum[i] / cnt[i]);
            }
            if (new_c != c[i]) { c[i] = new_c; changed = true; }
        }
        std::sort(c.begin(), c.end());  // maintain monotone property for binary search
        if (!changed) break;
    }
    return c;
}

LloydMaxCodebook::LloydMaxCodebook(int d, int M, int B, float sigma)
    : d_(d), M_(M), B_(B), Ds_(d / M), sigma_(sigma)
{
    if (d % M != 0)
        throw std::invalid_argument("d must be divisible by M");
    if (B % Ds_ != 0)
        throw std::invalid_argument("B must be divisible by Ds = d/M");

    bits_per_dim_ = B / Ds_;
    K1D_ = 1 << bits_per_dim_;  // 2^(B/Ds) codewords per 1D dimension

    // Analytical 1D Lloyd-Max codewords (paper Eq. 3 / 4):
    //   c_i = σ·√2 · erfinv((2i−1)/K_1D),   i = 1,…,K_1D   (1-indexed in paper)
    // In 0-indexed form: q_i = (i + 0.5) / K_1D
    c1d_.resize(K1D_);
    for (int i = 0; i < K1D_; i++) {
        float q_i = (i + 0.5f) / (float)K1D_;
        c1d_[i] = sigma_ * float(M_SQRT2) * erfinv_f(2.0f * q_i - 1.0f);
    }
    // c1d_ is already sorted ascending (erfinv is monotone increasing)
}

int LloydMaxCodebook::nearest_1d(float v) const {
    // Binary-search the midpoints between adjacent codewords.
    if (K1D_ == 1) return 0;
    int lo = 0, hi = K1D_ - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (v < 0.5f * (c1d_[mid] + c1d_[mid + 1]))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

void LloydMaxCodebook::encode(const float* y, uint8_t* codes, int n) const {
    for (int i = 0; i < n; i++) {
        const float* yi = y + (size_t)i * d_;
        uint8_t* ci     = codes + (size_t)i * M_;
        for (int m = 0; m < M_; m++) {
            const float* ym = yi + m * Ds_;
            uint8_t code = 0;
            for (int k = 0; k < Ds_; k++) {
                int j = nearest_1d(ym[k]);
                code |= (uint8_t)(j << (k * bits_per_dim_));
            }
            ci[m] = code;
        }
    }
}

void LloydMaxCodebook::build_flat_lut(const float* q_rot, float* flat_lut) const {
    // Expand Cartesian-product LUT → standard 256-entry-per-subspace format.
    // flat_lut[m*256 + c] = Σ_k (q_rot[m*Ds+k] − c1d_[(c >> k*bpd) & (K1D-1)])²
    const int mask = K1D_ - 1;
    for (int m = 0; m < M_; m++) {
        const float* qm = q_rot + m * Ds_;
        float* lm = flat_lut + (size_t)m * 256;
        for (int c = 0; c < 256; c++) {
            float dist = 0.0f;
            for (int k = 0; k < Ds_; k++) {
                int j = (c >> (k * bits_per_dim_)) & mask;
                float diff = qm[k] - c1d_[j];
                dist += diff * diff;
            }
            lm[c] = dist;
        }
    }
}

void LloydMaxCodebook::build_lut(const float* q_rot, float* lut) const {
    // lut layout: [m * Ds_ * K1D_  +  k * K1D_  +  i]
    for (int m = 0; m < M_; m++) {
        const float* qm  = q_rot + m * Ds_;
        float*       lm  = lut   + m * Ds_ * K1D_;
        for (int k = 0; k < Ds_; k++) {
            float*       lmk = lm + k * K1D_;
            float        qmk = qm[k];
            for (int i = 0; i < K1D_; i++) {
                float diff = qmk - c1d_[i];
                lmk[i] = diff * diff;
            }
        }
    }
}

float LloydMaxCodebook::adc_distance(const uint8_t* code, const float* lut) const {
    const int mask = K1D_ - 1;
    float dist = 0.0f;
    for (int m = 0; m < M_; m++) {
        const uint8_t code_m = code[m];
        const float*  lm     = lut + m * Ds_ * K1D_;
        for (int k = 0; k < Ds_; k++) {
            int j = (code_m >> (k * bits_per_dim_)) & mask;
            dist += lm[k * K1D_ + j];
        }
    }
    return dist;
}

void LloydMaxCodebook::reconstruct(const uint8_t* code, float* out) const {
    const int mask = K1D_ - 1;
    for (int m = 0; m < M_; m++) {
        const uint8_t code_m = code[m];
        float* outm = out + m * Ds_;
        for (int k = 0; k < Ds_; k++) {
            int j = (code_m >> (k * bits_per_dim_)) & mask;
            outm[k] = c1d_[j];
        }
    }
}
