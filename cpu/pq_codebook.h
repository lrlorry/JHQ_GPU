#pragma once
#include <cstddef>
#include <iosfwd>
#include <cstdint>
#include <vector>

// Product-quantizer primary codebook — the level-0 quantizer the JHQ paper's
// own implementation uses (JHQ_official jhqlib/IndexJHQ: primary_pq_ is a
// faiss::ProductQuantizer, primary_ksub() = 1 << level_bits[0]).
//
// This replaces cpu/codebook.h's LloydMaxCodebook for the primary level. The
// two differ in what a subspace codeword is:
//
//   LloydMaxCodebook  K_1D = 2^(B/Ds) codewords per *dimension*, and a
//                     subspace codeword is their Cartesian product. That needs
//                     B % Ds == 0 with B <= 8, i.e. Ds <= 8, i.e. at least one
//                     bit per dimension -- so a 3072-d vector cannot have a
//                     primary code shorter than 3072 bits.
//
//   PQCodebook        K = 2^B free codewords in the Ds-dimensional subspace,
//                     with Ds = d/M unconstrained. A 3072-d vector at M=16
//                     gives Ds=192 and a 128-bit primary code -- which is the
//                     operating point the paper reports for JHQ on
//                     OpenAI3-3072 (§5.1, and Table 3's ECL of 129 bit).
//
// The paper's §5.3 ablation ("replacing JQ's vector quantizer with independent
// per-dimension scalar quantizers ... at 99% recall, scalar quantization is
// about 285x slower") is precisely the difference between the two: the
// product-of-scalars variant needs far longer codes for the same accuracy.
//
// Training follows the official train_primary_level: analytical Gaussian
// placement as the initial centroids, then a few Lloyd iterations. The
// analytical part is what makes JQ's index build cheap; the refinement is
// short (5 iterations by default there) and runs only on the training sample.
class PQCodebook {
public:
    // d must be divisible by M. B is bits per subspace, so K = 2^B codewords;
    // B <= 8 keeps one code byte per subspace, which is what the GPU scan and
    // the [B, M, 256] byte LUT assume.
    PQCodebook(int d, int M, int B);

    // y: n rotated training vectors, row-major n x d.
    void train(const float* y, int n, int kmeans_iters = 5, int seed = 1234);

    // Centroids for subspace m: K * Ds floats, row-major [K][Ds].
    const float* centroids(int m) const {
        return cent_.data() + (size_t)m * (size_t)K_ * (size_t)Ds_;
    }
    const float* data() const { return cent_.data(); }
    // Writable view of the same buffer. Exists so the Lloyd iterations can run
    // on the device and hand the result back; the analytical initialisation
    // stays on the host either way, so the starting point does not change.
    float*       mutable_data() { return cent_.data(); }

    // Seed every subspace from per-subspace moments, both laid out [M][Ds].
    // Equivalent to train(y, n, 0, seed) but without needing y on the host.
    void init_from_stats(const float* mean_all, const float* var_all, int seed);
    size_t       size() const { return cent_.size(); }

    int d()  const { return d_; }
    int M()  const { return M_; }
    int Ds() const { return Ds_; }
    int K()  const { return K_; }

    // Nearest centroid in subspace m to the Ds-dim subvector ym.
    // Round-trip the trained centroids; see JLTransform::write_state.
    void write_state(std::ostream& os) const;
    bool read_state(std::istream& is);

    int encode_subspace(int m, const float* ym) const;

    // Reconstruct the full d-dim approximation from an M-byte code.
    void reconstruct(const uint8_t* code, float* out) const;

private:
    int d_, M_, Ds_, K_;
    std::vector<float> cent_;   // [M][K][Ds]

    // Mirrors IndexJHQ::analytical_gaussian_init: K centroids placed along
    // Gram-Schmidt-orthogonalised random directions at Gaussian-quantile
    // radii, scaled per dimension by that dimension's variance. After the JL
    // rotation every dimension is near-N(0, sigma^2), which is what makes
    // these positions a good starting point without looking at the data
    // beyond its mean and variance.
    void analytical_init(const float* sub, int n, float* out, int seed) const;
    // Same construction, from the two moments rather than from the data.
    void analytical_init_from_stats(const float* mean, const float* var,
                                    float* out, int seed) const;
};
