#pragma once
#include <vector>

// Johnson-Lindenstrauss orthogonal transformation.
//
// Generates a random d×d orthogonal matrix Π via QR decomposition of a
// Gaussian random matrix (paper §3.1, Definition 1 and Example 3).
//
// Key property (Lemma 1): Π preserves Euclidean/inner-product/cosine distances,
// so the nearest-neighbour ordering is unchanged after rotation.
//
// Key consequence (Lemma 2): after rotation, each dimension y_i ~ N(0, σ²)
// independently, enabling the analytical Lloyd-Max codebook.
class JLTransform {
public:
    explicit JLTransform(int d, int seed = 42);

    // Rotate n row-vectors x (n×d, row-major) into y (n×d): y = x · Πᵀ
    void apply(const float* x, float* y, int n) const;

    int   dim()     const { return d_; }
    float sigma()   const { return sigma_; }
    // Raw Π in column-major order (d×d), as produced by LAPACK sorgqr.
    const float* pi_data() const { return Pi_.data(); }

    // Estimate per-dimension std σ from n sample vectors (original space).
    // Must be called before building the codebook.
    void estimate_sigma(const float* x, int n_samples);

private:
    int d_;
    float sigma_ = 1.0f;
    std::vector<float> Pi_;  // d×d, column-major (LAPACK convention)

    void build_rotation(int seed);
};
