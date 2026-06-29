#pragma once
#include <cmath>
#include <limits>

// Inverse error function via Winitzki (2008) approximation + one Newton step.
// Max relative error ~1e-5 after refinement.
inline float erfinv_f(float x) {
    if (x >=  1.0f) return  std::numeric_limits<float>::infinity();
    if (x <= -1.0f) return -std::numeric_limits<float>::infinity();
    if (x ==  0.0f) return  0.0f;

    const float sign = (x > 0.0f) ? 1.0f : -1.0f;
    const float a = std::abs(x);

    // Winitzki approximation: c = 0.147
    const float kC   = 0.147f;
    const float k2Pi = 2.0f / 3.14159265358979f;
    const float ln1  = std::log(1.0f - a * a);
    const float t    = k2Pi / kC + 0.5f * ln1;
    float z = sign * std::sqrt(std::sqrt(t * t - ln1 / kC) - t);

    // Newton step: z -= (erf(z) - x) / (2/√π · exp(-z²))
    const float err = std::erf(z) - x;
    const float der = (2.0f / std::sqrt(float(M_PI))) * std::exp(-z * z);
    z -= err / der;
    return z;
}
