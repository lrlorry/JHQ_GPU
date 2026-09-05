#pragma once
#include <cstdint>
#include <vector>

// 1D Lloyd-Max k-means on n scalar residual values.
// Returns Kr sorted codewords. Used for the JHQ residual scalar codebook (§4.2).
// The paper explicitly uses 1D k-means — not the analytical Gaussian formula —
// because the residual distribution is a mixture of truncated Gaussians.
std::vector<float> train_1d_kmeans(const float* vals, int n, int Kr, int max_iter = 25);

// The same 1-D Lloyd, except that a cell which draws nothing keeps the centroid
// it had rather than being reseeded to a quantile of the data. The reseed is a
// jump into a dense region, so the cell takes points from a neighbour, empties
// another, and is thrown back again: on GPU that never settled in 20,000
// iterations. Held in place, distortion cannot increase and the iteration
// reaches a fixed point, which is what a Lloyd-Max quantizer is defined as.
// train_1d_kmeans keeps the old rule because jhq_v23_faithful is frozen against
// it; from jhq_v34_empty_hold on, this is the one to call.
std::vector<float> train_1d_kmeans_hold(const float* vals, int n, int Kr,
                                        int max_iter, float tol = 1e-6f);

// Lloyd-Max codebook for JQ (paper §3.2, Equations 3 & 4).
//
// Because the JL transform makes each dimension independently N(0,σ²),
// we can compute codewords analytically from the Gaussian inverse CDF —
// no k-means training needed.
//
// Codebook structure:
//   M subspaces, each of dimension Ds = d/M.
//   K = 2^B codewords per subspace.
//   All subspaces share the same 1D codewords (same Gaussian distribution).
//   Ds-dimensional codewords = Cartesian product of 1D codewords.
//   K_1D = 2^(B/Ds) codewords per 1D dimension.
//
// Encoding one subvector (Ds dims) → B-bit integer:
//   For each dimension k: j_k = nearest_1d(y^(m)_k)   ∈ {0,…,K_1D-1}
//   code = Σ_k  j_k · K_1D^k            (packed into B bits)
//
// ADC distance (Asymmetric Distance Computation):
//   Precompute LUT[m][k][i] = (q_rot[m·Ds+k] − c1d[i])²  per query
//   d_approx(q,x) = Σ_m Σ_k  LUT[m][k][ j_k(code_m) ]
class LloydMaxCodebook {
public:
    // Requires: d % M == 0  and  B % (d/M) == 0
    LloydMaxCodebook(int d, int M, int B, float sigma);

    // Encode n rotated vectors y (n×d) → codes (n×M, each B bits as uint8_t)
    void encode(const float* y, uint8_t* codes, int n) const;

    // Build LUT for one rotated query q_rot (1×d).
    // lut must point to M × Ds × K_1D floats.
    void build_lut(const float* q_rot, float* lut) const;

    // Build flat (256-entry per subspace) LUT — O(1) lookup per code byte.
    // flat_lut must point to M × 256 floats.
    void build_flat_lut(const float* q_rot, float* flat_lut) const;

    // ADC distance for one database vector (code points to its M-byte code).
    float adc_distance(const uint8_t* code, const float* lut) const;

    int M()   const { return M_; }
    int Ds()  const { return Ds_; }
    int K1D() const { return K1D_; }
    int B()   const { return B_; }

    // Reconstruct the quantised approximation ŷ for one code (M bytes → d floats)
    void reconstruct(const uint8_t* code, float* out) const;

    const std::vector<float>& c1d() const { return c1d_; }

private:
    int d_, M_, B_, Ds_, K1D_, bits_per_dim_;
    float sigma_;
    std::vector<float> c1d_;  // (K1D_,) shared 1D codewords, ascending

    int nearest_1d(float v) const;
};
