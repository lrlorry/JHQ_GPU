#!/usr/bin/env python3
"""Numerical checks behind THEORY.md (no data, model only). Prints every number THEORY.md quotes.

  python3 theory_numerics.py > raw/theory_numerics.txt

Model (unit per-coordinate scale): query coordinate q ~ N(0, 1); a candidate at correlation rho has
y = rho q + sqrt(1 - rho^2) z, z ~ N(0, 1), so P(s = +1 | q) = Phi(kappa q), kappa = rho / sqrt(1 - rho^2).
A score sum_j h(q_j) s_j separating correlation rho1 from rho0 has
  SNR(h) = d (E[h g])^2 / E[h^2 v],  g = 2(Phi(k1 q) - Phi(k0 q)),  v = 4(Phi1 (1-Phi1) + Phi0 (1-Phi0)),
maximised (Cauchy-Schwarz) by h* = g / v with SNR* = d E[g^2 / v]. JHQ is h = q.
"""
import numpy as np
from scipy import integrate, optimize
from scipy.special import log_ndtr, ndtr

phi = lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)


def psi(t):
    """log Phi(t) - log Phi(-t), stable."""
    return log_ndtr(t) - log_ndtr(-t)


def gauss_E(f, lo=-9, hi=9):
    return integrate.quad(lambda q: f(q) * phi(q), lo, hi, limit=400)[0]


def snr_ratio(h, k1, k0):
    g = lambda q: 2 * (ndtr(k1 * q) - ndtr(k0 * q))
    v = lambda q: 4 * (ndtr(k1 * q) * ndtr(-k1 * q) + ndtr(k0 * q) * ndtr(-k0 * q))
    snr = lambda hh: gauss_E(lambda q: hh(q) * g(q)) ** 2 / gauss_E(lambda q: hh(q) ** 2 * v(q))
    opt = gauss_E(lambda q: g(q) ** 2 / v(q))
    return snr(h) / snr(lambda q: q), opt / snr(lambda q: q)


def main():
    print("== 1. psi(t) = log Phi(t) - log Phi(-t): series and limits")
    for t in (0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
        lin = 4 * phi(0) * t
        cub = lin + (16 * phi(0) ** 3 / 3 - 2 * phi(0) / 3) * t ** 3
        large = t * t / 2 + np.log(t * np.sqrt(2 * np.pi))
        print("  t=%5.2f psi=%10.5f  4phi(0)t=%10.5f (ratio %.4f)  +cubic=%10.5f  large-t=%10.5f (ratio %.4f)"
              % (t, psi(t), lin, psi(t) / lin, cub, large, psi(t) / large))
    print("  cubic coefficient (16 phi0^3 - 2 phi0)/3 = %.5f" % ((16 * phi(0) ** 3 - 2 * phi(0)) / 3))

    print("== 2. ranking SNR of a reweighted sign inner product, relative to JHQ (h = q)")
    print("  rho1 rho0 | locally optimal h* | LLR h = psi(lambda q)/2 at lambda = kappa1 | cos(h*, q)")
    for r1, r0 in ((0.9, 0.85), (0.88, 0.80), (0.85, 0.6), (0.8, 0.5), (0.7, 0.6), (0.95, 0.9)):
        k1, k0 = r1 / np.sqrt(1 - r1 ** 2), r0 / np.sqrt(1 - r0 ** 2)
        llr, opt = snr_ratio(lambda q: 0.5 * psi(k1 * q), k1, k0)
        g = lambda q: 2 * (ndtr(k1 * q) - ndtr(k0 * q))
        v = lambda q: 4 * (ndtr(k1 * q) * ndtr(-k1 * q) + ndtr(k0 * q) * ndtr(-k0 * q))
        hs = lambda q: g(q) / v(q)
        cos = gauss_E(lambda q: hs(q) * q) / np.sqrt(gauss_E(lambda q: hs(q) ** 2) * gauss_E(lambda q: q * q))
        print("  %.2f %.2f |  x%.4f            |  x%.4f                                    | %.5f" % (r1, r0, opt, llr, cos))

    print("== 3. rank-optimal threshold: argmin_t E_q log Bhattacharyya(t, q), biased background")
    print("  background y ~ N(m, 1); neighbour y | q ~ N(m + rho (q - m), 1 - rho^2); q ~ N(m, 1)")
    for m in (0.0, 0.5, 1.0):
        for rho in (0.8, 0.9):
            s1 = np.sqrt(1 - rho ** 2)

            def g_of(t):
                def lb(q):
                    p0 = ndtr(m - t)
                    p1 = ndtr((m + rho * (q - m) - t) / s1)
                    return np.log(np.sqrt(p0 * p1) + np.sqrt((1 - p0) * (1 - p1)))
                return integrate.quad(lambda q: lb(q) * phi(q - m), m - 9, m + 9, limit=400)[0]
            res = optimize.minimize_scalar(g_of, bounds=(m - 3, m + 3), method="bounded", options={"xatol": 1e-6})
            print("  m=%.1f rho=%.1f: argmin t = %+.5f (m = %+.1f); exponent per coordinate -2 g: at t=m %.5f, at t=0 %.5f (ratio %.3f)"
                  % (m, rho, res.x, m, -2 * g_of(m), -2 * g_of(0.0), g_of(0.0) / g_of(m)))

    print("== 4. information in a sign bit whose threshold sits a std away from the centre")
    print("  Fisher information about a location shift, I(a) = phi(a)^2 / (Phi(a)(1 - Phi(a))), relative to I(0) = 2/pi")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        I = phi(a) ** 2 / (ndtr(a) * ndtr(-a))
        print("  a=%.2f  P(bit=1)=%.3f |E s|=%.3f  I(a)/I(0)=%.3f" % (a, ndtr(a), abs(2 * ndtr(a) - 1), I / (2 / np.pi)))

    print("== 5. half-normal conditional mean and the JHQ level")
    print("  E[X | X > 0] / sigma = sqrt(2/pi) = %.6f ; JHQ level / sigma = sqrt(2) erfinv(1/2) = %.6f (median of |X|)"
          % (np.sqrt(2 / np.pi), np.sqrt(2) * 0.4769362762044699))


if __name__ == "__main__":
    main()
